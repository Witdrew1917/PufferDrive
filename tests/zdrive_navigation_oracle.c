// Reference dump of zpufferdrive's (zdrive) navigation features, used by tests/test_drive_zbin.py to check that
// PufferDrive's zbin navigation port (pufferlib/ocean/drive/zbin.h) matches it. Built as a unity build against a
// zpufferdrive checkout (see the test for the compile command).
//
// Usage: zdrive_navigation_oracle <map.bin> <num_steps> <num_waypoints> <lookahead_distance>
// Every logged object is moved to its logged pose at each step and localised through zdrive's policy-agent
// (NON_EXPERT) path in compute_agent_metrics; the lookahead walk mirrors compute_observations.
#include "zdrive_types.h"
void close_client(Client *client) { (void)client; }
void c_render(Zdrive *env) { (void)env; }

#include "log.c"
#include "object_history.c"
#include "zdrive_utils.c"
#include "zdrive_agent.c"
#include "zdrive_agent_obs.c"
#include "zdrive_geometry.c"
#include "zdrive_load.c"
#include "zdrive.c"
#include "zdrive_cosim.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <map.bin> <num_steps> <num_waypoints> <lookahead_distance>\n", argv[0]);
        return 2;
    }
    int num_steps = atoi(argv[2]);
    int n_waypoints = atoi(argv[3]);
    float lookahead_distance = (float)atof(argv[4]);

    Zdrive env;
    memset(&env, 0, sizeof(Zdrive));
    env.map_name = strdup(argv[1]);
    env.dt = 0.1f;
    env.episode_duration = 9.1f;
    env.max_episode_steps = 91;
    env.episode_length = 91;
    env.num_agents = MAX_AGENTS;
    env.max_controlled_agents = MAX_AGENTS;
    env.goal_radius = 2.0f;
    env.map_data.goal_target_distance = 50.0f;
    env.map_data.min_distance_to_goal = 2.0f;
    env.episodes_per_scenario = 1;
    env.fraction_rl_agents = 1.0f;
    env.lane_offset_offroad_threshold = 1000.0f;
    int available_map_ids[1] = {0};
    env.available_map_ids = available_map_ids;
    env.num_available_maps = 1;
    if (init(&env) != 0)
        return 1;

    for (int t = 0; t < num_steps; t++) {
        for (int i = 0; i < env.num_objects; i++) {
            Agent *agent = &env.agents[i];
            agent->mark_as_expert = NON_EXPERT;
            if (t >= agent->traj_length || !agent->traj_valid[t]) {
                agent->x = INVALID_POSITION;
                agent->y = INVALID_POSITION;
                continue;
            }
            agent->x = agent->traj_x[t];
            agent->y = agent->traj_y[t];
            agent->z = agent->traj_z[t];
            agent->vx = agent->traj_vx[t];
            agent->vy = agent->traj_vy[t];
            agent->speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
            agent->heading = agent->traj_heading[t];
            agent->heading_x = cosf(agent->heading);
            agent->heading_y = sinf(agent->heading);
        }
        for (int i = 0; i < env.num_objects; i++) {
            Agent *agent = &env.agents[i];
            if (agent->x == INVALID_POSITION)
                continue;
            agent->closest_navi_entity_idx = -1;
            agent->closest_navi_point_idx = -1;
            agent->left_lane_feasibility = 0.0f;
            agent->current_lane_feasibility = 0.0f;
            agent->right_lane_feasibility = 0.0f;
            agent->distance_to_goal = UNREACHABLE_DISTANCE_TO_GOAL;
            compute_agent_metrics(&env, i);

            float feasibility[3] = {0.0f, 0.0f, 0.0f};
            if (agent->current_speed_limit > 0.0f) {
                float raw[3] = {agent->left_lane_feasibility, agent->current_lane_feasibility,
                                agent->right_lane_feasibility};
                for (int k = 0; k < 3; k++)
                    feasibility[k] = CLAMP(raw[k] / agent->current_speed_limit / MAX_LANE_FEASIBILITY_TIME, 0.0f, 1.0f);
            }
            printf("%d %d %d %d %.9g %.9g %.9g %.9g %.9g", t, i, agent->id, agent->goal_index,
                   agent->current_speed_limit, agent->distance_to_goal, feasibility[0], feasibility[1], feasibility[2]);

            float spacing = lookahead_distance / (float)n_waypoints;
            NaviWalk walk = navi_walk_init(&env, agent->closest_navi_entity_idx, agent->closest_navi_point_idx,
                                           agent->navi_proj_x, agent->navi_proj_y);
            int w = 0;
            for (int step = 0; step < NAVI_LOOKAHEAD_MAX_WALK_STEPS && w < n_waypoints && navi_walk_step(&walk);
                 step++) {
                while (w < n_waypoints && walk.accum >= (float)(w + 1) * spacing) {
                    float wx, wy;
                    navi_walk_interpolate(&walk, (float)(w + 1) * spacing, &wx, &wy);
                    float dx = wx - agent->x;
                    float dy = wy - agent->y;
                    printf(" %.9g %.9g", (dx * agent->heading_x + dy * agent->heading_y) / NAVI_LOOKAHEAD_NORM_DIST,
                           (-dx * agent->heading_y + dy * agent->heading_x) / NAVI_LOOKAHEAD_NORM_DIST);
                    w++;
                }
            }
            for (; w < n_waypoints; w++)
                printf(" 0 0");
            printf("\n");
        }
    }
    return 0;
}
