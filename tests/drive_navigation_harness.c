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

int main(int argc, char **argv) {
    if (argc < 5)
        return 2;
    Drive env = {0};
    env.map_name = argv[1];
    env.entities = load_map_binary(env.map_name, &env);
    if (env.entities == NULL)
        return 3;
    env.navigation = load_navigation_binary(&env);
    Entity *agent = &env.entities[0];
    reset_navigation_state(&env, 0);
    agent->x = strtof(argv[2], NULL);
    agent->y = strtof(argv[3], NULL);
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
