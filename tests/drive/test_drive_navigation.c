#include "include/drive_fixture.h"
#include "include/test.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_NAVI_STEPS 30

typedef struct {
    unsigned char *bytes;
    size_t size;
} NaviBuffer;

static char g_navi_dir[64];

static void buffer_append(NaviBuffer *buffer, const void *data, size_t size) {
    buffer->bytes = (unsigned char *) realloc(buffer->bytes, buffer->size + size);
    memcpy(buffer->bytes + buffer->size, data, size);
    buffer->size += size;
}

// One Bellman-Ford sweep over longitudinal edges toward goal; returns 1 if any distance improved.
static int relax_toward_goal(const Drive *env, const struct NavigationData *nav, int goal, float *dist) {
    int improved = 0;
    for (int lane = 0; lane < nav->num_lanes; lane++) {
        const RoadMapElement *road = &env->road_elements[nav->lane_road_idx[lane]];
        for (int exit_idx = 0; exit_idx < road->num_exits && lane != goal; exit_idx++) {
            int exit_road = road->exit_lanes[exit_idx];
            int exit_lane = (exit_road >= 0) ? nav->road_to_navi_lane[exit_road] : -1;
            if (exit_lane == -1 || dist[exit_lane] >= NAVI_UNREACHABLE_M) {
                continue;
            }
            float candidate = road->length + dist[exit_lane];
            if (candidate < dist[lane]) {
                dist[lane] = candidate;
                improved = 1;
            }
        }
    }
    return improved;
}

// Serializes a valid v2 sidecar for the map in env: every lane is a goal, longitudinal edges only.
static NaviBuffer build_valid_sidecar(const Drive *env) {
    struct NavigationData nav = {0};
    nav.road_to_navi_lane = (int *) malloc(env->num_road_elements * sizeof(int));
    nav.lane_road_idx = (int *) malloc(env->num_road_elements * sizeof(int));
    for (int road_idx = 0; road_idx < env->num_road_elements; road_idx++) {
        nav.road_to_navi_lane[road_idx] = is_road_lane(env->road_elements[road_idx].type) ? nav.num_lanes : -1;
        if (nav.road_to_navi_lane[road_idx] != -1) {
            nav.lane_road_idx[nav.num_lanes++] = road_idx;
        }
    }
    size_t table_count = (size_t) nav.num_lanes * nav.num_lanes;
    float *dist = (float *) malloc(table_count * sizeof(float));
    float *center_feasibility = (float *) malloc(table_count * sizeof(float));
    for (size_t i = 0; i < table_count; i++) {
        dist[i] = NAVI_UNREACHABLE_M;
    }
    for (int goal = 0; goal < nav.num_lanes; goal++) {
        float *goal_dist = &dist[(size_t) goal * nav.num_lanes];
        goal_dist[goal] = env->road_elements[nav.lane_road_idx[goal]].length;
        for (int sweep = 0; sweep < nav.num_lanes && relax_toward_goal(env, &nav, goal, goal_dist); sweep++) {
        }
        for (int lane = 0; lane < nav.num_lanes; lane++) {
            float lane_dist = goal_dist[lane];
            center_feasibility[(size_t) goal * nav.num_lanes + lane] = (lane_dist >= NAVI_UNREACHABLE_M) ? 0.0f
                : (lane == goal) ? NAVI_FEASIBILITY_CAP_M
                                 : fminf(lane_dist, NAVI_FEASIBILITY_CAP_M);
        }
    }
    int *no_neighbor = (int *) malloc(nav.num_lanes * sizeof(int));
    for (int lane = 0; lane < nav.num_lanes; lane++) {
        no_neighbor[lane] = -1;
    }
    uint32_t map_crc32;
    uint64_t map_size_bytes;
    navi_file_fingerprint(env->map_name, &map_crc32, &map_size_bytes);
    int32_t version = NAVI_VERSION;
    int32_t num_roads = env->num_road_elements;
    NaviBuffer buffer = {0};
    buffer_append(&buffer, NAVI_MAGIC, NAVI_MAGIC_LENGTH);
    buffer_append(&buffer, &version, 4);
    buffer_append(&buffer, &map_crc32, 4);
    buffer_append(&buffer, &map_size_bytes, 8);
    buffer_append(&buffer, &num_roads, 4);
    buffer_append(&buffer, &nav.num_lanes, 4);
    buffer_append(&buffer, nav.lane_road_idx, nav.num_lanes * 4);
    buffer_append(&buffer, no_neighbor, nav.num_lanes * 4);
    buffer_append(&buffer, no_neighbor, nav.num_lanes * 4);
    buffer_append(&buffer, dist, table_count * 4);
    buffer_append(&buffer, center_feasibility, table_count * 4);
    free(nav.road_to_navi_lane);
    free(nav.lane_road_idx);
    free(dist);
    free(center_feasibility);
    free(no_neighbor);
    return buffer;
}

static void write_bytes(const char *path, const unsigned char *bytes, size_t size) {
    FILE *file = fopen(path, "wb");
    fwrite(bytes, 1, size, file);
    fclose(file);
}

static Drive load_map_only(const char *map_file) {
    Drive env = drive_test_env_config(map_file, SIMULATION_MODE_GIGAFLOW, 1, 0);
    load_map_binary(env.map_name, &env);
    return env;
}

static int load_bytes_status(const Drive *env, const unsigned char *bytes, size_t size) {
    char path[128];
    snprintf(path, sizeof(path), "%s/corrupt.navi", g_navi_dir);
    write_bytes(path, bytes, size);
    struct NavigationData *navigation = NULL;
    int status = load_navigation_binary(path, env, &navigation);
    free_navigation_data(navigation);
    unlink(path);
    return status;
}

static size_t lane_array_offset(int num_lanes, int array_idx) {
    return NAVI_HEADER_BYTES + 4 * (size_t) array_idx * num_lanes;
}

static size_t table_offset(int num_lanes, int table_idx) {
    return lane_array_offset(num_lanes, NAVI_LANE_INT_ARRAYS) + 4 * (size_t) table_idx * num_lanes * num_lanes;
}

static void write_town01_sidecar(char *path_out, size_t path_size) {
    Drive env = load_map_only(drive_carla_map());
    NaviBuffer buffer = build_valid_sidecar(&env);
    char *path = navigation_path_for_map(env.map_name, g_navi_dir);
    snprintf(path_out, path_size, "%s", path);
    write_bytes(path, buffer.bytes, buffer.size);
    free(path);
    free(buffer.bytes);
    free_loaded_map_data(&env);
}

static int test_sidecar_path_replaces_bin_suffix(void) {
    char *next_to_map = navigation_path_for_map("maps/carla/Town01.bin", "");
    char *in_dir = navigation_path_for_map("maps/carla/Town01.bin", "/data/navi");
    char *in_dir_slash = navigation_path_for_map("Town01.bin", "/data/navi/");
    EXPECT_EQ_INT(strcmp(next_to_map, "maps/carla/Town01.navi"), 0);
    EXPECT_EQ_INT(strcmp(in_dir, "/data/navi/Town01.navi"), 0);
    EXPECT_EQ_INT(strcmp(in_dir_slash, "/data/navi/Town01.navi"), 0);
    free(next_to_map);
    free(in_dir);
    free(in_dir_slash);
    return 0;
}

static int test_loader_accepts_valid_sidecar(void) {
    Drive env = load_map_only(drive_carla_map());
    NaviBuffer buffer = build_valid_sidecar(&env);
    char path[128];
    snprintf(path, sizeof(path), "%s/valid.navi", g_navi_dir);
    write_bytes(path, buffer.bytes, buffer.size);
    struct NavigationData *navigation = NULL;
    EXPECT_EQ_INT(load_navigation_binary(path, &env, &navigation), NAVI_LOAD_OK);
    for (int lane = 0; lane < navigation->num_lanes; lane++) {
        EXPECT_EQ_INT(navigation->road_to_navi_lane[navigation->lane_road_idx[lane]], lane);
    }
    free_navigation_data(navigation);
    unlink(path);
    free(buffer.bytes);
    free_loaded_map_data(&env);
    return 0;
}

static int test_loader_rejects_corrupt_sidecars(void) {
    Drive env = load_map_only(drive_carla_map());
    NaviBuffer valid = build_valid_sidecar(&env);
    int num_lanes;
    memcpy(&num_lanes, valid.bytes + 28, 4);
    unsigned char *bytes = (unsigned char *) malloc(valid.size + 4);

    struct NavigationData *navigation = NULL;
    EXPECT_EQ_INT(load_navigation_binary("/nonexistent/x.navi", &env, &navigation), NAVI_LOAD_MISSING);
    EXPECT_TRUE(navigation == NULL);

    EXPECT_EQ_INT(load_bytes_status(&env, valid.bytes, valid.size), NAVI_LOAD_OK);
    EXPECT_EQ_INT(load_bytes_status(&env, valid.bytes, valid.size - 4), NAVI_LOAD_INVALID);
    EXPECT_EQ_INT(load_bytes_status(&env, valid.bytes, 20), NAVI_LOAD_INVALID);
    memcpy(bytes, valid.bytes, valid.size);
    memset(bytes + valid.size, 0, 4);
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size + 4), NAVI_LOAD_INVALID);

    memcpy(bytes, valid.bytes, valid.size);
    bytes[0] = 'X';
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    memcpy(bytes, valid.bytes, valid.size);
    bytes[NAVI_MAGIC_LENGTH] = NAVI_VERSION + 1;
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    memcpy(bytes, valid.bytes, valid.size);
    bytes[12] ^= 0xFF; // map crc32
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    int32_t bad_index = num_lanes;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + lane_array_offset(num_lanes, 0), &bad_index, 4); // lane_road_idx[0]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    int32_t self_lane = 1;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + lane_array_offset(num_lanes, 1) + 4, &self_lane, 4); // left_lane[1] = 1
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    float not_finite = NAN;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + table_offset(num_lanes, 0), &not_finite, 4); // distance_to_goal_m[0][0]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    float over_cap = NAVI_FEASIBILITY_CAP_M + 1.0f;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + table_offset(num_lanes, 1), &over_cap, 4); // feasibility_center_m[0][0]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    free(bytes);
    free(valid.bytes);
    free_loaded_map_data(&env);
    return 0;
}

static Drive create_env_with_obs_navigation(
    const char *navigation_dir,
    int goal_source,
    int use_map_cache,
    int obs_navigation) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, use_map_cache);
    env.goal_source = goal_source;
    env.obs_navigation = obs_navigation;
    env.navigation_dir = drive_test_strdup(navigation_dir);
    allocate(&env);
    c_reset(&env);
    return env;
}

static Drive create_navigation_env(const char *navigation_dir, int goal_source, int use_map_cache) {
    return create_env_with_obs_navigation(navigation_dir, goal_source, use_map_cache, 1);
}

static float agent_lane_progress_m(Drive *env, const Agent *agent) {
    return compute_lane_progress(
        &env->road_elements[agent->current_lane_idx],
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        true,
        NULL);
}

// Navigation state is a pure function of (current lane, progress, final goal) and the sidecar tables.
static int expect_navigation_matches_tables(Drive *env, const Agent *agent) {
    const struct NavigationData *navigation = env->navigation;
    int goal_road_idx = agent->list_goal_lane[agent->goal_count - 1];
    EXPECT_EQ_INT(agent->navi_goal_lane_idx, goal_road_idx);
    if (agent->current_lane_idx == -1) {
        return 0;
    }
    size_t row = (size_t) navigation->road_to_navi_lane[goal_road_idx] * navigation->num_lanes;
    int navi_lane_idx = navigation->road_to_navi_lane[agent->current_lane_idx];
    float lane_start_distance_m = navigation->distance_to_goal_m[row + navi_lane_idx];
    float lane_start_center_m = navigation->feasibility_center_m[row + navi_lane_idx];
    float lane_progress_m = agent_lane_progress_m(env, agent);
    EXPECT_EQ_INT(agent->navi_off_route, lane_start_distance_m >= NAVI_UNREACHABLE_M);
    if (!agent->navi_off_route) {
        float expected_distance_m
            = fmaxf(lane_start_distance_m - lane_progress_m - agent->navi_goal_lane_remaining_m, 0.0f);
        EXPECT_NEAR(agent->navi_distance_to_goal_m, expected_distance_m, 1e-2f);
    }
    float expected_center_m = (lane_start_center_m >= NAVI_FEASIBILITY_CAP_M)
        ? lane_start_center_m
        : fmaxf(lane_start_center_m - lane_progress_m, 0.0f);
    EXPECT_NEAR(agent->navi_feasibility_center_m, expected_center_m, 1e-3f);
    return 0;
}

static int expect_navigation_follows_goals(int goal_source) {
    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    Drive env = create_navigation_env(g_navi_dir, goal_source, 0);
    EXPECT_TRUE(env.navigation != NULL);
    for (int step = 0; step < TEST_NAVI_STEPS; step++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    int checked_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        if (agent->removed || agent->goal_count == 0 || agent->list_goal_lane[agent->goal_count - 1] == -1) {
            continue;
        }
        EXPECT_EQ_INT(expect_navigation_matches_tables(&env, agent), 0);
        checked_agent_count += agent->current_lane_idx != -1;
    }
    EXPECT_TRUE(checked_agent_count > 0);
    free_allocated(&env);
    unlink(sidecar_path);
    return 0;
}

static int test_navigation_follows_route_goals(void) {
    return expect_navigation_follows_goals(GOAL_SOURCE_ROUTE);
}

static int test_navigation_follows_map_goals(void) {
    return expect_navigation_follows_goals(GOAL_SOURCE_MAP);
}

// Moving the final goal to another lane re-targets navigation without touching the route.
static int test_navigation_retargets_on_goal_change(void) {
    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    Drive env = create_navigation_env(g_navi_dir, GOAL_SOURCE_ROUTE, 0);
    Agent *agent = NULL;
    for (int i = 0; i < env.active_agent_count && agent == NULL; i++) {
        Agent *candidate = &env.agents[env.active_agent_indices[i]];
        if (!candidate->removed && candidate->current_lane_idx != -1 && candidate->goal_count > 0) {
            agent = candidate;
        }
    }
    EXPECT_TRUE(agent != NULL);
    int final_goal_idx = agent->goal_count - 1;
    int new_goal_road_idx = agent->current_lane_idx;
    const RoadMapElement *new_goal_lane = &env.road_elements[new_goal_road_idx];
    agent->list_goal_lane[final_goal_idx] = new_goal_road_idx;
    agent->list_goal_x[final_goal_idx] = new_goal_lane->x[new_goal_lane->segment_size - 1];
    agent->list_goal_y[final_goal_idx] = new_goal_lane->y[new_goal_lane->segment_size - 1];
    update_navigation_state(&env, (int) (agent - env.agents));
    EXPECT_EQ_INT(agent->navi_goal_lane_idx, new_goal_road_idx);
    EXPECT_NEAR(agent->navi_goal_lane_remaining_m, 0.0f, 1e-2f);
    EXPECT_EQ_INT(agent->navi_off_route, 0);
    EXPECT_NEAR(agent->navi_feasibility_center_m, NAVI_FEASIBILITY_CAP_M, 1e-3f);
    EXPECT_NEAR(
        agent->navi_distance_to_goal_m,
        fmaxf(new_goal_lane->length - agent_lane_progress_m(&env, agent), 0.0f),
        1e-2f);
    free_allocated(&env);
    unlink(sidecar_path);
    return 0;
}

static int test_missing_sidecar_aborts_env_creation(void) {
    fflush(stdout);
    pid_t child = fork();
    if (child == 0) {
        freopen("/dev/null", "w", stderr);
        Drive env = create_navigation_env(g_navi_dir, GOAL_SOURCE_ROUTE, 0);
        free_allocated(&env);
        _exit(0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    return 0;
}

// Sidecars drive navigation state and telemetry even when the policy does not observe them.
static int test_navigation_runs_without_obs_navigation(void) {
    Drive without_sidecar = create_env_with_obs_navigation(g_navi_dir, GOAL_SOURCE_ROUTE, 0, 0);
    EXPECT_TRUE(without_sidecar.navigation == NULL);
    free_allocated(&without_sidecar);

    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    Drive env = create_env_with_obs_navigation(g_navi_dir, GOAL_SOURCE_ROUTE, 0, 0);
    EXPECT_TRUE(env.navigation != NULL);
    EXPECT_EQ_INT(ego_feature_count(&env), EGO_FEATURES);
    int tracked_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        tracked_agent_count += env.agents[env.active_agent_indices[i]].navi_goal_lane_idx != -1;
    }
    EXPECT_TRUE(tracked_agent_count > 0);
    free_allocated(&env);
    unlink(sidecar_path);
    return 0;
}

static int test_cached_envs_share_navigation(void) {
    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    drive_map_cache_clear();
    Drive first = create_navigation_env(g_navi_dir, GOAL_SOURCE_ROUTE, 1);
    Drive second = create_navigation_env(g_navi_dir, GOAL_SOURCE_ROUTE, 1);
    EXPECT_TRUE(first.navigation != NULL);
    EXPECT_TRUE(first.navigation == second.navigation);
    EXPECT_TRUE(first.shared_map == second.shared_map);
    EXPECT_TRUE(first.shared_map->navigation == first.navigation);
    int num_lanes = first.navigation->num_lanes;
    free_allocated(&first);
    EXPECT_EQ_INT(second.navigation->num_lanes, num_lanes);
    free_allocated(&second);
    drive_map_cache_clear();
    unlink(sidecar_path);
    return 0;
}

int main(void) {
    snprintf(g_navi_dir, sizeof(g_navi_dir), "/tmp/pd_navi_test_XXXXXX");
    if (mkdtemp(g_navi_dir) == NULL) {
        printf("FAIL: mkdtemp\n");
        return 1;
    }
    int failures = 0;
    RUN_TEST(test_sidecar_path_replaces_bin_suffix);
    RUN_TEST(test_loader_accepts_valid_sidecar);
    RUN_TEST(test_loader_rejects_corrupt_sidecars);
    RUN_TEST(test_navigation_follows_route_goals);
    RUN_TEST(test_navigation_follows_map_goals);
    RUN_TEST(test_navigation_retargets_on_goal_change);
    RUN_TEST(test_missing_sidecar_aborts_env_creation);
    RUN_TEST(test_navigation_runs_without_obs_navigation);
    RUN_TEST(test_cached_envs_share_navigation);
    rmdir(g_navi_dir);
    return test_summary(failures);
}
