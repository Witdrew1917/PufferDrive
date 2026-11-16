#ifndef PUFFERLIB_OCEAN_DRIVE_NAVI_PID_H
#define PUFFERLIB_OCEAN_DRIVE_NAVI_PID_H

#define PID_ROUTE_POINTS 40
#define PID_ROUTE_SPACING 2.0f
#define PID_LEAD_CUTOFF 3.5f
#define PID_MAX_LATERAL_ACCELERATION 4.0f

typedef struct {
    float x[PID_ROUTE_POINTS + 1];
    float y[PID_ROUTE_POINTS + 1];
    int count;
} PIDRoute;

typedef struct {
    int found;
    float distance;
    float speed;
    float length;
} PIDLead;

static PIDRoute navi_pid_route(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    PIDRoute route = {0};
    if (!navigation_loaded(env) || agent->navi_goal_idx < 0)
        return route;
    update_navigation_state(env, agent_idx);
    if (agent->current_lane_idx < 0 || agent->navi_off_route)
        return route;

    int count;
    if (env->znavigation != NULL) {
        if (agent->znavi_entity_idx < 0)
            return route;
        route.x[0] = agent->znavi_proj_x;
        route.y[0] = agent->znavi_proj_y;
        count = znavi_lookahead_points(env, agent, PID_ROUTE_POINTS, PID_ROUTE_POINTS * PID_ROUTE_SPACING, route.x + 1,
                                       route.y + 1);
    } else {
        if (agent->navi_lane_idx < 0)
            return route;
        const Entity *lane = &env->entities[env->navigation->lane_entity_idx[agent->navi_lane_idx]];
        int segment;
        lane_project(lane, agent->x, agent->y, &segment, &route.x[0], &route.y[0]);
        count = navigation_route_waypoints(env, agent, PID_ROUTE_POINTS, PID_ROUTE_SPACING, route.x + 1, route.y + 1);
    }
    route.count = count + 1;
    return route;
}

static void navi_pid_consider_lead(const Entity *ego, const Entity *other, const PIDRoute *route, PIDLead *lead) {
    if (other == ego || other->removed || !other->valid || other->x == INVALID_POSITION)
        return;
    if (fabsf(other->z - ego->z) > 0.5f * (ego->height + other->height) + 1.0f)
        return;

    float dx = other->x - ego->x, dy = other->y - ego->y;
    float longitudinal = dx * ego->heading_x + dy * ego->heading_y;
    float lateral = -dx * ego->heading_y + dy * ego->heading_x;
    float heading_dot = other->heading_x * ego->heading_x + other->heading_y * ego->heading_y;
    if (longitudinal > 0.0f && fabsf(lateral) < PID_LEAD_CUTOFF && heading_dot > 0.0f &&
        longitudinal < lead->distance) {
        lead->found = 1;
        lead->distance = longitudinal;
        lead->speed = other->vx * ego->heading_x + other->vy * ego->heading_y;
        lead->length = other->length;
    }

    float accumulated = 0.0f;
    float prev_x = ego->heading_x, prev_y = ego->heading_y;
    for (int i = 1; i < route->count; i++) {
        float ex = route->x[i] - route->x[i - 1], ey = route->y[i] - route->y[i - 1];
        float length = sqrtf(ex * ex + ey * ey);
        if (length < 1e-4f)
            continue;
        float dir_x = ex / length, dir_y = ey / length;
        if (dir_x * prev_x + dir_y * prev_y < 0.5f)
            break;
        prev_x = dir_x;
        prev_y = dir_y;
        if (accumulated >= lead->distance)
            break;
        float ox = other->x - route->x[i - 1], oy = other->y - route->y[i - 1];
        float t = (ox * ex + oy * ey) / (length * length);
        float clamped_t = clip(t, 0.0f, 1.0f);
        float rx = ox - clamped_t * ex, ry = oy - clamped_t * ey;
        float distance = accumulated + clamped_t * length;
        if ((i != 1 || t >= 0.0f) && rx * rx + ry * ry < PID_LEAD_CUTOFF * PID_LEAD_CUTOFF &&
            other->heading_x * dir_x + other->heading_y * dir_y > 0.0f && distance < lead->distance) {
            lead->found = 1;
            lead->distance = distance;
            lead->speed = other->vx * dir_x + other->vy * dir_y;
            lead->length = other->length;
        }
        accumulated += length;
    }
}

static PIDLead navi_pid_find_lead(Drive *env, int agent_idx, const PIDRoute *route) {
    Entity *agent = &env->entities[agent_idx];
    PIDLead lead = {.distance = INFINITY};
    for (int i = 0; i < env->active_agent_count; i++)
        navi_pid_consider_lead(agent, &env->entities[env->active_agent_indices[i]], route, &lead);
    for (int i = 0; i < env->static_agent_count; i++)
        navi_pid_consider_lead(agent, &env->entities[env->static_agent_indices[i]], route, &lead);
    return lead;
}

static float navi_pid_curvature(const PIDRoute *route) {
    if (route->count < 4)
        return 0.0f;
    float ax = route->x[1] - route->x[0], ay = route->y[1] - route->y[0];
    float bx = route->x[3] - route->x[1], by = route->y[3] - route->y[1];
    float cx = route->x[3] - route->x[0], cy = route->y[3] - route->y[0];
    float denominator = hypotf(ax, ay) * hypotf(bx, by) * hypotf(cx, cy);
    return denominator > 1e-4f ? 2.0f * (ax * by - ay * bx) / denominator : 0.0f;
}

static void move_navi_pid(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed || agent->x == INVALID_POSITION)
        return;
    if (agent->stopped) {
        agent->vx = agent->vy = agent->vz = 0.0f;
        return;
    }
    PIDRoute route = navi_pid_route(env, agent_idx);
    float speed = fmaxf(0.0f, agent->vx * agent->heading_x + agent->vy * agent->heading_y);
    float desired_speed = env->navi_pid_default_speed;
    if (env->znavigation != NULL && agent->navi_speed_limit > 0.0f && isfinite(agent->navi_speed_limit))
        desired_speed = agent->navi_speed_limit * env->navi_pid_speed_limit_fraction;
    desired_speed = fminf(desired_speed, MAX_SPEED);

    if (route.count < 2) {
        desired_speed = 0.0f;
        if (route.count == 0 && !agent->navi_pid_route_warning) {
            fprintf(stderr, "[drive/navi_pid] agent %d has no reachable navigation route; braking\n", agent_idx);
            agent->navi_pid_route_warning = 1;
        }
    } else if (route.count <= PID_ROUTE_POINTS) {
        float remaining = (route.count - 1) * PID_ROUTE_SPACING;
        desired_speed = fminf(desired_speed, sqrtf(2.0f * (-ACCEL_MIN) * remaining));
    }
    float route_curvature = navi_pid_curvature(&route);
    float curvature_magnitude = fabsf(route_curvature);
    if (curvature_magnitude > 1e-5f) {
        float curvature_speed = sqrtf(PID_MAX_LATERAL_ACCELERATION / curvature_magnitude);
        desired_speed = fminf(desired_speed, curvature_speed);
    }

    PIDLead lead = navi_pid_find_lead(env, agent_idx, &route);
    float acceleration = env->navi_pid_speed_gain * (desired_speed - speed);
    if (lead.found) {
        float desired_gap = fmaxf(fmaxf(speed, lead.speed), 5.0f);
        float gap = lead.distance - 0.5f * (agent->length + lead.length);
        float gap_acceleration =
            env->navi_pid_gap_position_gain * (gap - desired_gap) +
            env->navi_pid_gap_velocity_gain * (lead.speed - speed);
        acceleration = fminf(acceleration, gap_acceleration);
    }
    acceleration = clip(acceleration, ACCEL_MIN, ACCEL_MAX);
    float new_speed = clip(speed + acceleration * env->dt, 0.0f, MAX_SPEED);
    acceleration = (new_speed - speed) / env->dt;

    float steering = 0.0f;
    if (route.count >= 2 && agent->wheelbase > 0.0f) {
        float dx = route.x[1] - route.x[0], dy = route.y[1] - route.y[0];
        float route_heading = atan2f(dy, dx);
        float offset = -sinf(route_heading) * (route.x[0] - agent->x) + cosf(route_heading) * (route.y[0] - agent->y);
        float heading_error = normalize_heading(route_heading - agent->heading);
        float feedback = env->navi_pid_lateral_offset_gain * clip(offset, -5.0f, 5.0f) +
                         env->navi_pid_heading_gain * heading_error;
        float feedback_limit = PID_MAX_LATERAL_ACCELERATION / fmaxf(speed * speed, 4.0f);
        float curvature =
            clip(route_curvature, -0.01f, 0.01f) + clip(feedback, -feedback_limit, feedback_limit);
        float target = atanf(curvature * agent->wheelbase);
        steering = agent->steering_angle + env->dt / (0.02f + env->dt) * (target - agent->steering_angle);
    }
    steering = clip(steering, -0.55f, 0.55f);
    float curvature = agent->wheelbase > 0.0f ? tanf(steering) / agent->wheelbase : 0.0f;
    float distance = 0.5f * (speed + new_speed) * env->dt;
    float yaw = distance * curvature;
    float dx = fabsf(curvature) < 1e-5f ? distance : sinf(yaw) / curvature;
    float dy = fabsf(curvature) < 1e-5f ? 0.0f : (1.0f - cosf(yaw)) / curvature;
    agent->x += dx * agent->heading_x - dy * agent->heading_y;
    agent->y += dx * agent->heading_y + dy * agent->heading_x;
    agent->heading = normalize_heading(agent->heading + yaw);
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
    agent->vx = new_speed * agent->heading_x;
    agent->vy = new_speed * agent->heading_y;
    agent->vz = 0.0f;
    float lateral_acceleration = new_speed * new_speed * curvature;
    agent->jerk_long = (acceleration - agent->a_long) / env->dt;
    agent->jerk_lat = (lateral_acceleration - agent->a_lat) / env->dt;
    agent->a_long = acceleration;
    agent->a_lat = lateral_acceleration;
    agent->steering_angle = steering;
}

#endif
