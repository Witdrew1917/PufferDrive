// Dumps PufferDrive's zbin navigation features in the same format as tests/zdrive_navigation_oracle.c, for the
// parity test in tests/test_drive_zbin.py.
//
// Usage: zbin_navigation_harness <map.bin> <num_steps> <num_waypoints> <lookahead_distance>
#include "drive.h"

int main(int argc, char **argv) {
    if (argc != 5)
        return 2;
    Drive env = {0};
    env.map_name = argv[1];
    if (drive_map_format(env.map_name) != MAP_FORMAT_ZBIN)
        return 3;
    env.entities = load_map_binary(env.map_name, &env);
    if (env.entities == NULL)
        return 3;
    set_means(&env);
    int num_steps = atoi(argv[2]);
    env.navigation_enabled = 1;
    env.navigation_waypoints = atoi(argv[3]);
    env.navigation_lookahead_distance = strtof(argv[4], NULL);
    env.navigation_lookahead_norm = ZNAVI_DEFAULT_LOOKAHEAD_NORM;
    env.znavigation = load_znavigation(&env);
    if (env.znavigation == NULL)
        return 4;
    for (int i = 0; i < env.num_objects; i++)
        reset_navigation_state(&env, i);

    float ego[3];
    float waypoints[ZNAVI_LOOKAHEAD_WAYPOINTS_MAX * ZNAVI_WAYPOINT_FEATURES];
    for (int t = 0; t < num_steps; t++) {
        for (int i = 0; i < env.num_objects; i++) {
            Entity *agent = &env.entities[i];
            if (t >= agent->array_size || !agent->traj_valid[t]) {
                agent->x = INVALID_POSITION;
                agent->y = INVALID_POSITION;
                continue;
            }
            agent->x = agent->traj_x[t];
            agent->y = agent->traj_y[t];
            agent->heading = agent->traj_heading[t];
            agent->heading_x = cosf(agent->heading);
            agent->heading_y = sinf(agent->heading);
            update_navigation_state(&env, i);
            float distance_to_goal;
            znavi_localize(&env, agent, &distance_to_goal);
            znavi_write_observations(&env, agent, ego, waypoints);
            printf("%d %d %d %d %.9g %.9g %.9g %.9g %.9g", t, i, agent->id, agent->navi_goal_idx,
                   agent->navi_speed_limit, distance_to_goal, ego[0], ego[1], ego[2]);
            for (int k = 0; k < env.navigation_waypoints * ZNAVI_WAYPOINT_FEATURES; k++)
                printf(" %.9g", waypoints[k]);
            printf("\n");
        }
    }
    free_znavigation(env.znavigation);
    for (int i = 0; i < env.num_entities; i++)
        free_entity(&env.entities[i]);
    free(env.entities);
    free(env.tracks_to_predict_indices);
    return 0;
}
