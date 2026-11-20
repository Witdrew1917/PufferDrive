#ifndef PUFFERLIB_OCEAN_DRIVE_NAVI_PID_H
#define PUFFERLIB_OCEAN_DRIVE_NAVI_PID_H

#define PID_ROUTE_POINTS 40
#define PID_ROUTE_SPACING 2.0f
#define PID_ROUTE_DISTANCE (PID_ROUTE_POINTS * PID_ROUTE_SPACING)
#define PID_LEAD_CUTOFF 3.5f
#define PID_MAX_LATERAL_ACCELERATION 4.0f

typedef struct {
    float x[PID_ROUTE_POINTS + 3];
    float y[PID_ROUTE_POINTS + 3];
    float remaining;
    int count;
} PIDRoute;

typedef struct {
    int found;
    float distance;
    float speed;
    float length;
} PIDLead;

typedef union {
    float continuous[3];
    int discrete[3];
} NaviPIDAction;

static PIDRoute navi_pid_goal_segment(const Entity *agent) {
    float start_x = agent->navi_pid_terminal_x, start_y = agent->navi_pid_terminal_y;
    float dx = agent->init_goal_x - start_x, dy = agent->init_goal_y - start_y;
    float length = hypotf(dx, dy);
    float progress = clip(((agent->x - start_x) * dx + (agent->y - start_y) * dy) / length, 0.0f, length);
    PIDRoute route = {.remaining = length - progress, .count = 1};
    route.x[0] = start_x + progress * (dx / length);
    route.y[0] = start_y + progress * (dy / length);
    if (route.remaining > 1e-4f) {
        route.x[1] = agent->init_goal_x;
        route.y[1] = agent->init_goal_y;
        route.count++;
    }
    return route;
}

static PIDRoute navi_pid_route(Drive *env, int agent_idx, bool refresh_navigation) {
    Entity *agent = &env->entities[agent_idx];
    PIDRoute route = {.remaining = -1.0f};
    if (!navigation_loaded(env) || agent->navi_goal_idx < 0)
        return route;
    if (refresh_navigation)
        update_navigation_state(env, agent_idx);
    if (env->append_goal_to_navi_lookahead && agent->navi_pid_on_goal_segment)
        return navi_pid_goal_segment(agent);
    if (agent->current_lane_idx < 0 || agent->navi_off_route)
        return route;

    int count;
    if (env->znavigation != NULL) {
        if (agent->znavi_entity_idx < 0)
            return route;
        route.x[0] = agent->znavi_proj_x;
        route.y[0] = agent->znavi_proj_y;
        const ZNavigation *zn = env->znavigation;
        ZNaviWalk walk = znavi_walk_init(agent->znavi_entity_idx, agent->znavi_point_idx, route.x[0], route.y[0]);
        count = znavi_sample_route(zn, &walk, PID_ROUTE_POINTS, PID_ROUTE_DISTANCE, route.x + 1, route.y + 1);
        if (env->append_goal_to_navi_lookahead && znavi_can_append_mission_goal(zn, agent, &walk, PID_ROUTE_DISTANCE)) {
            float terminal_x, terminal_y;
            znavi_point(zn, walk.navi_e, walk.navi_p, &terminal_x, &terminal_y);
            float extension = hypotf(agent->init_goal_x - terminal_x, agent->init_goal_y - terminal_y);
            float speed = fmaxf(0.0f, agent->vx * agent->heading_x + agent->vy * agent->heading_y);
            float step_distance = (speed + 0.5f * ACCEL_MAX * env->dt) * env->dt;
            if (extension > 1e-4f && walk.accum <= fmaxf(step_distance, 1e-4f)) {
                // Keep tracking the final leg after leaving the lane graph.
                agent->navi_pid_on_goal_segment = true;
                agent->navi_pid_terminal_x = terminal_x;
                agent->navi_pid_terminal_y = terminal_y;
                if (walk.accum <= 1e-4f)
                    return navi_pid_goal_segment(agent);
            }
            if (walk.accum - count * PID_ROUTE_SPACING > 1e-4f) {
                route.x[++count] = terminal_x;
                route.y[count] = terminal_y;
            }
            if (extension > 1e-4f) {
                route.x[++count] = agent->init_goal_x;
                route.y[count] = agent->init_goal_y;
            }
            route.remaining = walk.accum + extension;
        }
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

static float navi_pid_curvature(const PIDRoute *route, int i) {
    if (i + 2 >= route->count)
        return 0.0f;
    float ax = route->x[i] - route->x[i - 1], ay = route->y[i] - route->y[i - 1];
    float bx = route->x[i + 2] - route->x[i], by = route->y[i + 2] - route->y[i];
    float cx = ax + bx, cy = ay + by;
    float denominator = hypotf(ax, ay) * hypotf(bx, by) * hypotf(cx, cy);
    return denominator > 1e-4f ? 2.0f * (ax * by - ay * bx) / denominator : 0.0f;
}

static float navi_pid_preview_speed(const Drive *env, const Entity *agent, float distance, float target) {
    const ZNavigation *zn = env->znavigation;
    if (distance <= 0.0f || agent->znavi_entity_idx < 0)
        return target;

    ZNaviWalk walk =
        znavi_walk_init(agent->znavi_entity_idx, agent->znavi_point_idx, agent->znavi_proj_x, agent->znavi_proj_y);
    for (int step = 0; step < ZNAVI_LOOKAHEAD_MAX_WALK_STEPS; step++) {
        int previous_line = walk.navi_e % zn->num_lines;
        if (!znavi_walk_step(zn, &walk) || walk.accum > distance)
            break;
        int road = zn->line_road[walk.navi_e % zn->num_lines];
        if (road < 0 || road == zn->line_road[previous_line])
            continue;
        float next_limit = env->entities[env->num_objects + road].road_speed_limit;
        if (next_limit > 0.0f && isfinite(next_limit)) {
            target = fminf(target, next_limit * env->navi_pid_speed_limit_fraction);
        }
    }
    return target;
}

static int navi_pid_action_bin(float value, const float *bins, int count) {
    int best = 0;
    for (int i = 1; i < count; i++) {
        if (fabsf(value - bins[i]) < fabsf(value - bins[best]))
            best = i;
    }
    return best;
}

static NaviPIDAction navi_pid_action(Drive *env, int agent_idx, bool refresh_navigation) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->stopped) {
        if (env->action_type == 1)
            return (NaviPIDAction){.continuous = {0}};
        if (env->dynamics_model == DELTA_LOCAL)
            return (NaviPIDAction){.discrete = {NUM_DX_BINS / 2, NUM_DY_BINS / 2, NUM_YAW_BINS / 2}};
        if (env->dynamics_model == JERK)
            return (NaviPIDAction){.discrete = {2 * 3 + 1}};
        return (NaviPIDAction){.discrete = {(NUM_ACCEL_BINS / 2) * NUM_STEER_BINS + NUM_STEER_BINS / 2}};
    }
    PIDRoute route = navi_pid_route(env, agent_idx, refresh_navigation);
    float signed_speed =
        copysignf(hypotf(agent->vx, agent->vy), agent->vx * agent->heading_x + agent->vy * agent->heading_y);
    float speed = fmaxf(0.0f, signed_speed);
    float desired_speed = env->navi_pid_default_speed;
    float preview_distance = env->navi_pid_speed_lookahead_distance;
    if (env->znavigation != NULL && agent->navi_speed_limit > 0.0f && isfinite(agent->navi_speed_limit))
        desired_speed = agent->navi_speed_limit * env->navi_pid_speed_limit_fraction;
    desired_speed = fminf(desired_speed, MAX_SPEED);

    if (route.count < 2) {
        desired_speed = 0.0f;
        if (route.count == 0 && !agent->navi_pid_route_warning) {
            agent->navi_pid_route_warning = 1;
        }
    } else {
        if (route.remaining >= 0.0f || route.count <= PID_ROUTE_POINTS) {
            float remaining = route.remaining >= 0.0f ? route.remaining : (route.count - 1) * PID_ROUTE_SPACING;
            desired_speed = fminf(desired_speed, sqrtf(2.0f * (-ACCEL_MIN) * remaining));
        }
        if (env->znavigation != NULL && !agent->navi_pid_on_goal_segment)
            desired_speed = navi_pid_preview_speed(env, agent, preview_distance, desired_speed);
    }
    float route_curvature = navi_pid_curvature(&route, 1);
    float curvature_magnitude = fabsf(route_curvature);
    for (int i = 2; i + 2 < route.count && i * PID_ROUTE_SPACING <= preview_distance; i++)
        curvature_magnitude = fmaxf(curvature_magnitude, fabsf(navi_pid_curvature(&route, i)));
    if (curvature_magnitude > 1e-5f) {
        float curvature_speed = sqrtf(PID_MAX_LATERAL_ACCELERATION / curvature_magnitude);
        desired_speed = fminf(desired_speed, curvature_speed);
    }

    PIDLead lead = navi_pid_find_lead(env, agent_idx, &route);
    float acceleration = env->navi_pid_speed_gain * (desired_speed - signed_speed);
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
    if (signed_speed >= 0.0f)
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
        float previous = env->dynamics_model == JERK ? agent->steering_angle : agent->navi_pid_steering;
        steering = previous + env->dt / (0.02f + env->dt) * (target - previous);
    }
    steering = clip(steering, -0.55f, 0.55f);
    agent->navi_pid_steering = steering;
    if (env->dynamics_model == CLASSIC) {
        if (env->action_type == 1)
            return (NaviPIDAction){.continuous = {clip(acceleration / ACCEL_MAX, -1.0f, 1.0f), steering / STEER_MAX}};
        int accel_bin = navi_pid_action_bin(acceleration, ACCELERATION_VALUES, NUM_ACCEL_BINS);
        if (signed_speed >= 0.0f && signed_speed + ACCELERATION_VALUES[accel_bin] * env->dt < 0.0f)
            accel_bin++;
        int steer_bin = navi_pid_action_bin(steering, STEERING_VALUES, NUM_STEER_BINS);
        return (NaviPIDAction){.discrete = {accel_bin * NUM_STEER_BINS + steer_bin}};
    }

    float curvature = agent->wheelbase > 0.0f ? tanf(steering) / agent->wheelbase : 0.0f;
    if (env->dynamics_model == JERK) {
        float longitudinal = (acceleration - agent->a_long) / env->dt;
        float lateral = (new_speed * new_speed * curvature - agent->a_lat) / env->dt;
        if (env->action_type == 1) {
            float scale = longitudinal < 0.0f ? -JERK_LONG[0] : JERK_LONG[3];
            return (NaviPIDAction){
                .continuous = {clip(longitudinal / scale, -1.0f, 1.0f), clip(lateral / JERK_LAT[2], -1.0f, 1.0f)}};
        }
        int long_bin = navi_pid_action_bin(longitudinal, JERK_LONG, 4);
        int lat_bin = navi_pid_action_bin(lateral, JERK_LAT, 3);
        return (NaviPIDAction){.discrete = {long_bin * 3 + lat_bin}};
    }

    float distance = 0.5f * (speed + new_speed) * env->dt;
    float yaw = distance * curvature;
    float dx = fabsf(curvature) < 1e-5f ? distance : sinf(yaw) / curvature;
    float dy = fabsf(curvature) < 1e-5f ? 0.0f : (1.0f - cosf(yaw)) / curvature;
    if (env->action_type == 1) {
        return (NaviPIDAction){.continuous = {clip(dx / DELTA_MAX_DX, -1.0f, 1.0f),
                                              clip(dy / DELTA_MAX_DY, -1.0f, 1.0f),
                                              clip(yaw / DELTA_MAX_DYAW, -1.0f, 1.0f)}};
    }

    // Carry sub-bin forward motion so a PID starting from rest can escape the zero bin.
    dx = clip(dx + agent->navi_pid_dx_residual, 0.0f, DELTA_MAX_DX);
    int dx_bin = navi_pid_action_bin(dx, DELTA_DX_VALUES, NUM_DX_BINS);
    // move_dynamics clamps the executed step to the longitudinal acceleration bound; carry the residual
    // against what will actually execute (bounded to half a bin, so clamp losses are not caught up later),
    // otherwise the phantom residual pins agents to their speed bin.
    float max_dx_change = A_LONG_MAX * env->dt * env->dt;
    float executed_dx =
        clip(DELTA_DX_VALUES[dx_bin], agent->prev_action_dx - max_dx_change, agent->prev_action_dx + max_dx_change);
    float half_bin = 0.5f * (DELTA_DX_VALUES[1] - DELTA_DX_VALUES[0]);
    agent->navi_pid_dx_residual = clip(dx - executed_dx, -half_bin, half_bin);
    int dy_bin = navi_pid_action_bin(dy, DELTA_DY_VALUES, NUM_DY_BINS);
    int yaw_bin = navi_pid_action_bin(yaw, DELTA_YAW_VALUES, NUM_YAW_BINS);
    return (NaviPIDAction){.discrete = {dx_bin, dy_bin, yaw_bin}};
}

#endif
