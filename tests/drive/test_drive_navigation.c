#include "include/drive_fixture.h"
#include "include/test.h"

#include <sys/stat.h>
#include <unistd.h>

#define TEST_NAVI_GOAL_COUNT 2
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

// One Bellman-Ford sweep over longitudinal edges; returns 1 if any distance improved.
static int relax_toward_goal(const Drive *env, const struct NavigationData *nav, int goal, float *dist, int *next) {
    int improved = 0;
    for (int lane = 0; lane < nav->num_lanes; lane++) {
        const RoadMapElement *road = &env->road_elements[nav->lane_road_idx[lane]];
        for (int exit_idx = 0; exit_idx < road->num_exits; exit_idx++) {
            int exit_road = road->exit_lanes[exit_idx];
            int exit_lane = (exit_road >= 0) ? nav->road_to_navi_lane[exit_road] : -1;
            if (exit_lane == -1 || lane == nav->goal_lane[goal] || dist[exit_lane] >= NAVI_UNREACHABLE_M) {
                continue;
            }
            float candidate = road->length + dist[exit_lane];
            if (candidate < dist[lane]) {
                dist[lane] = candidate;
                next[lane] = exit_lane;
                improved = 1;
            }
        }
    }
    return improved;
}

// Serializes a valid sidecar for the map in env: shortest longitudinal next hops, no lane changes.
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
    nav.num_goals = TEST_NAVI_GOAL_COUNT;
    int goal_lane[TEST_NAVI_GOAL_COUNT] = {0, nav.num_lanes / 2};
    nav.goal_lane = goal_lane;
    size_t table_count = (size_t) nav.num_goals * nav.num_lanes;
    int *next = (int *) malloc(table_count * sizeof(int));
    float *dist = (float *) malloc(table_count * sizeof(float));
    float *zeros = (float *) calloc(table_count, sizeof(float));
    for (size_t i = 0; i < table_count; i++) {
        next[i] = -1;
        dist[i] = NAVI_UNREACHABLE_M;
    }
    float goal_x[TEST_NAVI_GOAL_COUNT], goal_y[TEST_NAVI_GOAL_COUNT];
    for (int goal = 0; goal < nav.num_goals; goal++) {
        const RoadMapElement *road = &env->road_elements[nav.lane_road_idx[goal_lane[goal]]];
        goal_x[goal] = road->x[road->segment_size - 1];
        goal_y[goal] = road->y[road->segment_size - 1];
        dist[goal * nav.num_lanes + goal_lane[goal]] = road->length;
        for (int sweep = 0; sweep < nav.num_lanes; sweep++) {
            if (!relax_toward_goal(env, &nav, goal, &dist[goal * nav.num_lanes], &next[goal * nav.num_lanes])) {
                break;
            }
        }
    }
    float *center_feasibility = (float *) malloc(table_count * sizeof(float));
    for (size_t i = 0; i < table_count; i++) {
        center_feasibility[i] = (dist[i] >= NAVI_UNREACHABLE_M) ? 0.0f : fminf(dist[i], NAVI_FEASIBILITY_CAP_M);
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
    buffer_append(&buffer, &nav.num_goals, 4);
    buffer_append(&buffer, nav.lane_road_idx, nav.num_lanes * 4);
    buffer_append(&buffer, no_neighbor, nav.num_lanes * 4);
    buffer_append(&buffer, no_neighbor, nav.num_lanes * 4);
    buffer_append(&buffer, goal_lane, nav.num_goals * 4);
    buffer_append(&buffer, goal_x, nav.num_goals * 4);
    buffer_append(&buffer, goal_y, nav.num_goals * 4);
    buffer_append(&buffer, next, table_count * 4);
    buffer_append(&buffer, dist, table_count * 4);
    buffer_append(&buffer, zeros, table_count * 4);
    buffer_append(&buffer, center_feasibility, table_count * 4);
    buffer_append(&buffer, zeros, table_count * 4);
    free(nav.road_to_navi_lane);
    free(nav.lane_road_idx);
    free(next);
    free(dist);
    free(zeros);
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

static size_t table_offset(int num_lanes, int num_goals, int table_idx) {
    return NAVI_HEADER_BYTES + 4 * (3 * (size_t) num_lanes + 3 * (size_t) num_goals)
        + 4 * (size_t) table_idx * num_goals * num_lanes;
}

static int write_town01_sidecar(char *path_out, size_t path_size) {
    Drive env = load_map_only(drive_carla_map());
    NaviBuffer buffer = build_valid_sidecar(&env);
    char *path = navigation_path_for_map(env.map_name, g_navi_dir);
    snprintf(path_out, path_size, "%s", path);
    write_bytes(path, buffer.bytes, buffer.size);
    free(path);
    free(buffer.bytes);
    free_loaded_map_data(&env);
    return 0;
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
    EXPECT_EQ_INT(navigation->num_goals, TEST_NAVI_GOAL_COUNT);
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
    int num_lanes, num_goals;
    memcpy(&num_lanes, valid.bytes + 28, 4);
    memcpy(&num_goals, valid.bytes + 32, 4);
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
    memcpy(bytes + NAVI_HEADER_BYTES, &bad_index, 4); // lane_road_idx[0]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + table_offset(num_lanes, num_goals, 0) + 4, &bad_index, 4); // next_lane[0][1]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    // next_lane[0][1] pointing at itself is in range but never one of its own exits
    int32_t self_lane = 1;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + table_offset(num_lanes, num_goals, 0) + 4, &self_lane, 4);
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    float not_finite = NAN;
    memcpy(bytes, valid.bytes, valid.size);
    memcpy(bytes + table_offset(num_lanes, num_goals, 1), &not_finite, 4); // distance_to_goal_m[0][0]
    EXPECT_EQ_INT(load_bytes_status(&env, bytes, valid.size), NAVI_LOAD_INVALID);

    free(bytes);
    free(valid.bytes);
    free_loaded_map_data(&env);
    return 0;
}

static Drive create_navigation_env(const char *navigation_dir, int use_map_cache) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, use_map_cache);
    env.route_source = ROUTE_SOURCE_NAVIGATION;
    env.navigation_dir = drive_test_strdup(navigation_dir);
    allocate(&env);
    c_reset(&env);
    return env;
}

// Every navigation-tagged route entry is the next hop of its goal from the previous entry,
// and a leg only hands over to a new goal at the previous goal's lane.
static int expect_route_follows_next_hops(const Drive *env, const Agent *agent) {
    const struct NavigationData *navigation = env->navigation;
    for (int route_idx = 1; route_idx < agent->navi_route_length; route_idx++) {
        int goal = agent->navi_route_goal_idx[route_idx];
        int previous_goal = agent->navi_route_goal_idx[route_idx - 1];
        int previous_lane = navigation->road_to_navi_lane[agent->route[route_idx - 1]];
        int expected_next = navigation->next_lane[goal * navigation->num_lanes + previous_lane];
        EXPECT_TRUE(expected_next != -1);
        EXPECT_EQ_INT(agent->route[route_idx], navigation->lane_road_idx[expected_next]);
        EXPECT_TRUE(goal == previous_goal || previous_lane == navigation->goal_lane[previous_goal]);
    }
    return 0;
}

// Center feasibility counts from the agent's position, not from the start of its lane.
static int expect_center_feasibility_ahead_of_agent(Drive *env, Agent *agent) {
    if (agent->current_lane_idx == -1) {
        return 0;
    }
    const struct NavigationData *navigation = env->navigation;
    int navi_lane_idx = navigation->road_to_navi_lane[agent->current_lane_idx];
    float lane_start_center_m
        = navigation->feasibility_center_m[agent->navi_goal_idx * navigation->num_lanes + navi_lane_idx];
    if (lane_start_center_m >= NAVI_FEASIBILITY_CAP_M) {
        EXPECT_NEAR(agent->navi_feasibility_center_m, NAVI_FEASIBILITY_CAP_M, 1e-3f);
        return 0;
    }
    float lane_progress_m = compute_lane_progress(
        &env->road_elements[agent->current_lane_idx],
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        true,
        NULL);
    EXPECT_NEAR(agent->navi_feasibility_center_m, fmaxf(lane_start_center_m - lane_progress_m, 0.0f), 1e-3f);
    return 0;
}

static int test_navigation_routes_follow_next_hops(void) {
    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    Drive env = create_navigation_env(g_navi_dir, 0);
    EXPECT_TRUE(env.navigation != NULL);
    int navigated_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        if (agent->removed || agent->navi_route_length == 0) {
            continue;
        }
        navigated_agent_count++;
        EXPECT_TRUE(agent->navi_route_length <= agent->route_length);
        EXPECT_EQ_INT(expect_route_follows_next_hops(&env, agent), 0);
    }
    EXPECT_TRUE(navigated_agent_count > 0);
    int checked_agent_count = 0;
    for (int step = 0; step < TEST_NAVI_STEPS; step++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        if (agent->removed || agent->navi_goal_idx == -1) {
            continue;
        }
        EXPECT_TRUE(agent->navi_goal_idx < env.navigation->num_goals);
        EXPECT_TRUE(agent->navi_off_route || agent->navi_distance_to_goal_m < NAVI_UNREACHABLE_M);
        EXPECT_EQ_INT(expect_center_feasibility_ahead_of_agent(&env, agent), 0);
        checked_agent_count += agent->current_lane_idx != -1;
    }
    EXPECT_TRUE(checked_agent_count > 0);
    free_allocated(&env);
    unlink(sidecar_path);
    return 0;
}

static int test_missing_sidecar_falls_back_to_random_walk(void) {
    Drive env = create_navigation_env(g_navi_dir, 0);
    EXPECT_TRUE(env.navigation == NULL);
    int routed_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        EXPECT_EQ_INT(agent->navi_route_length, 0);
        routed_agent_count += !agent->removed && agent->route_length > 0;
    }
    EXPECT_TRUE(routed_agent_count > 0);
    free_allocated(&env);
    return 0;
}

static int test_cached_envs_share_navigation(void) {
    char sidecar_path[128];
    write_town01_sidecar(sidecar_path, sizeof(sidecar_path));
    drive_map_cache_clear();
    Drive first = create_navigation_env(g_navi_dir, 1);
    Drive second = create_navigation_env(g_navi_dir, 1);
    EXPECT_TRUE(first.navigation != NULL);
    EXPECT_TRUE(first.navigation == second.navigation);
    EXPECT_TRUE(first.shared_map == second.shared_map);
    EXPECT_TRUE(first.shared_map->navigation == first.navigation);
    free_allocated(&first);
    EXPECT_TRUE(second.navigation->num_goals == TEST_NAVI_GOAL_COUNT);
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
    RUN_TEST(test_navigation_routes_follow_next_hops);
    RUN_TEST(test_missing_sidecar_falls_back_to_random_walk);
    RUN_TEST(test_cached_envs_share_navigation);
    rmdir(g_navi_dir);
    return test_summary(failures);
}
