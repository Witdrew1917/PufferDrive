#include "drive.h"

static float line_width, route_min_y, route_max_x;
static int drawn_goals;

void rlSetLineWidth(float width) { line_width = width; }
void rlDrawRenderBatchActive(void) {}
Color Fade(Color color, float alpha) {
    color.a = (unsigned char)(255 * alpha);
    return color;
}
void DrawLine3D(Vector3 start, Vector3 end, Color color) {
    if (line_width == NAVI_RENDER_ROUTE_WIDTH_PX) {
        route_min_y = fminf(route_min_y, fminf(start.y, end.y));
        route_max_x = fmaxf(route_max_x, fmaxf(start.x, end.x));
    }
}
void DrawSphere(Vector3 center, float radius, Color color) {
    if (radius == NAVI_RENDER_GOAL_RADIUS_M)
        drawn_goals++;
}
void DrawCylinderEx(Vector3 start, Vector3 end, float r1, float r2, int sides, Color color) {}
void DrawCircle3D(Vector3 center, float radius, Vector3 axis, float angle, Color color) {}

static void run_reward_steps(Drive *env, GridMap *grid, const char *scenario) {
    Entity *agent = &env->entities[0];
    int active[] = {0}, background[] = {2};
    float rewards[1] = {0};
    unsigned char terminals[1] = {0}, truncations[1] = {0};
    Log logs[1] = {{0}};
    int cache_count[] = {0};
    GridMapEntity *cache[] = {grid->cells[0]};
    grid->neighbor_cache_count = cache_count;
    grid->neighbor_cache_entities = cache;
    env->active_agent_indices = active;
    env->active_agent_count = 1;
    env->rewards = rewards;
    env->terminals = terminals;
    env->truncations = truncations;
    env->logs = logs;
    env->episode_length = 100;
    env->goal_radius = 0.2f;
    env->goal_speed = 1.0f;
    env->reward_progress_navi = 0.25f;
    env->navigation_enabled = 1;
    agent->reward_goal_cond = 2.0f;
    agent->controller = CONTROLLER_STATIC;
    agent->heading_x = 1.0f;
    agent->heading_y = 0.0f;
    agent->goal_position_x = 95.0f;
    agent->goal_position_y = 0.0f;
    float observations[observation_size(env)];
    env->observations = observations;
    reset_navigation_state(env, 0);
    update_navigation_state(env, 0);

    if (strcmp(scenario, "branch") == 0)
        agent->navi_lane_idx = 1;
    if (strcmp(scenario, "goal") == 0 || strcmp(scenario, "newgoal") == 0 || strcmp(scenario, "newgoal_reset") == 0 ||
        strcmp(scenario, "removegoal") == 0 || strcmp(scenario, "collisiongoal") == 0) {
        agent->goal_position_x = 6.0f;
        env->goal_behavior = strncmp(scenario, "newgoal", 7) == 0  ? GOAL_GENERATE_NEW
                             : strcmp(scenario, "removegoal") == 0 ? GOAL_REMOVE
                                                                   : GOAL_STOP;
    }

    Entity saved = env->entities[2];
    float saved_distances[env->navigation->num_lanes];
    int table = agent->navi_goal_idx * env->navigation->num_lanes;
    memcpy(saved_distances, &env->navigation->distance_to_goal_m[table], sizeof(saved_distances));
    float step_rewards[6];
    for (int step = 0; step < 6; step++) {
        if (step == 2 && strcmp(scenario, "collision_reset") == 0) {
            env->init_steps = 10;
            env->fix_rewards = 1;
            env->reward_goal = 2.0f;
            c_reset(env);
        }
        if (!agent->removed)
            agent->x = step == 2 ? 5.5f : 5.5f + 0.5f * (step == 5 ? 4 : step);
        if (strcmp(scenario, "retrace") == 0) {
            const float positions[] = {5.5f, 6.0f, 5.5f, 6.0f, 6.25f, 6.5f};
            agent->x = positions[step];
        }
        if (step == 1 && strcmp(scenario, "newgoal_reset") == 0)
            agent->closest_navi_distance_to_goal_m = 0.0f;
        if (step == 1 && (strcmp(scenario, "collision") == 0 || strcmp(scenario, "collisiongoal") == 0 ||
                          strcmp(scenario, "collision_reset") == 0)) {
            env->static_agent_indices = background;
            env->static_agent_count = 1;
            env->num_actors = 2;
            Entity *other = &env->entities[2];
            other->type = VEHICLE;
            other->controller = CONTROLLER_STATIC;
            other->x = agent->x;
            other->y = agent->y;
            other->heading_x = 1.0f;
            other->width = 2.0f;
            other->length = 4.0f;
        }
        if (step == 1 && strcmp(scenario, "offroad") == 0)
            env->entities[1].type = ROAD_EDGE;
        if (step == 1 && strcmp(scenario, "offroute") == 0)
            for (int lane = 0; lane < env->navigation->num_lanes; lane++)
                env->navigation->distance_to_goal_m[table + lane] = NAVI_UNREACHABLE_M;

        c_step(env);
        step_rewards[step] = rewards[0];
        env->static_agent_count = 0;
        env->num_actors = 1;
        env->entities[2] = saved;
        env->entities[1].type = ROAD_LANE;
        memcpy(&env->navigation->distance_to_goal_m[table], saved_distances, sizeof(saved_distances));
    }
    printf("RESULT {\"rewards\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],\"episode_return\":%.6f}\n", step_rewards[0],
           step_rewards[1], step_rewards[2], step_rewards[3], step_rewards[4], step_rewards[5], logs[0].episode_return);
}

int main(int argc, char **argv) {
    if (argc < 5)
        return 2;
    Drive env = {0};
    env.map_name = argv[1];
    env.entities = load_map_binary(env.map_name, &env);
    if (env.entities == NULL)
        return 3;
    env.navigation = load_navigation_binary(&env, 1);
    Entity *agent = &env.entities[0];
    reset_navigation_state(&env, 0);
    agent->x = strtof(argv[2], NULL);
    agent->y = strtof(argv[3], NULL);
    agent->navi_position_x = agent->x;
    agent->navi_position_y = agent->y;
    agent->heading = strtof(argv[4], NULL);
    if (argc > 5)
        agent->navi_lane_idx = atoi(argv[5]);
    if (argc > 6)
        agent->navi_goal_idx = atoi(argv[6]);
    agent->navi_distance_to_goal_m = 500.0f;

    // One populated cell exercises the real matcher without initializing unrelated simulation systems.
    GridMapEntity cell[MAX_ENTITIES_PER_CELL * 25];
    int count = 0;
    for (int i = 0; i < env.num_entities; i++) {
        if (env.entities[i].type != ROAD_LANE)
            continue;
        for (int j = 0; j + 1 < env.entities[i].array_size; j++) {
            if (count == MAX_ENTITIES_PER_CELL * 25)
                return 4;
            cell[count++] = (GridMapEntity){.entity_idx = i, .geometry_idx = j};
        }
    }
    GridMapEntity *cells[] = {cell};
    GridMap grid = {.top_left_x = agent->x - 1,
                    .bottom_right_x = agent->x + 1,
                    .top_left_y = agent->y + 1,
                    .bottom_right_y = agent->y - 1,
                    .grid_cols = 1,
                    .grid_rows = 1,
                    .cells = cells,
                    .cell_entities_count = &count};
    env.grid_map = &grid;
    if (argc > 7) {
        run_reward_steps(&env, &grid, argv[7]);
        free_navigation(env.navigation);
        for (int i = 0; i < env.num_entities; i++)
            free_entity(&env.entities[i]);
        free(env.entities);
        return 0;
    }
    update_navigation_state(&env, 0);
    int physical = agent->current_lane_idx == -1 ? -1 : env.navigation->entity_to_navi_lane[agent->current_lane_idx];
    int selected = agent->navi_lane_idx;
    env.goal_target_distance = 95.0f;
    int sampled_goal = closest_navigation_goal(&env, 0);
    if (agent->navi_lane_idx != selected)
        return 5;
    float previous_distance = agent->prev_navi_distance_to_goal_m;
    float xs[8], ys[8];
    int n = navigation_route_waypoints(&env, agent, 8, 5.0f, xs, ys);
    int active[] = {0};
    env.active_agent_indices = active;
    env.navigation_waypoints = 8;
    env.navigation_waypoint_spacing = 5.0f;
    route_min_y = INFINITY;
    route_max_x = -INFINITY;
    drawn_goals = 0;
    if (agent->navi_goal_idx != -1)
        gui_draw_route(&env, 0, PUFF_CYAN, 1);
    float gui_min_y = isfinite(route_min_y) ? route_min_y : 0.0f;
    float gui_max_x = isfinite(route_max_x) ? route_max_x : 0.0f;
    int gui_goals = drawn_goals;
    route_min_y = INFINITY;
    route_max_x = -INFINITY;
    drawn_goals = 0;
    draw_navigation_overlay(&env);
    float legacy_min_y = isfinite(route_min_y) ? route_min_y : 0.0f;
    float legacy_max_x = isfinite(route_max_x) ? route_max_x : 0.0f;
    int legacy_goals = drawn_goals;
    update_navigation_state(&env, 0);
    printf("RESULT {\"physical\":%d,\"selected\":%d,\"sampled_goal\":%d,\"off_route\":%d,"
           "\"distance\":%.6f,\"previous_distance\":%.6f,\"stationary_delta\":%.6f,"
           "\"center_feasibility\":%.6f,\"gui_min_y\":%.6f,\"legacy_min_y\":%.6f,"
           "\"gui_max_x\":%.6f,\"legacy_max_x\":%.6f,\"gui_goals\":%d,\"legacy_goals\":%d,\"waypoints\":[",
           physical, selected, sampled_goal, agent->navi_off_route, agent->navi_distance_to_goal_m, previous_distance,
           agent->prev_navi_distance_to_goal_m - agent->navi_distance_to_goal_m, agent->navi_feasibility_center_m,
           gui_min_y, legacy_min_y, gui_max_x, legacy_max_x, gui_goals, legacy_goals);
    for (int i = 0; i < n; i++)
        printf("%s[%.6f,%.6f]", i ? "," : "", xs[i], ys[i]);
    puts("]}");
    free_navigation(env.navigation);
    for (int i = 0; i < env.num_entities; i++)
        free_entity(&env.entities[i]);
    free(env.entities);
    return 0;
}
