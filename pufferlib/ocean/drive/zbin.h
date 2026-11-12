// Loader and navigation runtime for zpufferdrive ("zdrive") v14 map binaries.
//
// Scene (Sections 1-2) is loaded into PufferDrive entities; the navigation data (Sections 3-4) is loaded into a
// ZNavigation that reproduces zdrive's non-expert navigation (lane localization, route lookup, NaviWalk lookahead)
// exactly. Ports cite the zpufferdrive source they mirror (paths relative to zpufferdrive/ocean/zdrive/).
// Included from drive.h right after struct Drive.

#define MAP_FORMAT_LEGACY 0
#define MAP_FORMAT_ZBIN 1

#define ZBIN_VERSION 14
#define ZBIN_NUM_SECTIONS 4
#define ZBIN_SCENARIO_ID_SIZE 512
#define ZBIN_JSON_SOURCE_FILE_SIZE 128
#define ZBIN_GIT_HASH_SIZE 48
// zdrive_data.h MAX_OBJECTS: objects beyond it are read and discarded, like zdrive.
#define ZBIN_MAX_OBJECTS 128
// zdrive_load.c:48-55, zdrive_constants.h:39,67-68
#define ZBIN_MAX_OBJECTS_ON_DISK 4096
#define ZBIN_MAX_TRACKS_TO_PREDICT 4096
#define ZBIN_MAX_ROADS 8192
#define ZBIN_MAX_WAYPOINTS_PER_ROAD 8192
#define ZBIN_MAX_TRAJECTORY_LENGTH 10000
#define ZBIN_MAX_NAVI_LINES 8192
#define ZBIN_MAX_GOALS 256
#define ZBIN_MAX_NAVI_ENTITIES 1000000
#define ZBIN_MAX_SUB_LINES 8192
#define ZBIN_MAX_LEFT_PERP_CONNECTIONS 64
#define ZBIN_LEFT_PERP_CONNECTION_DIM 3
// PufferDrive episodes index trajectories up to the standard WOMD length; shorter tracks are padded as invalid.
#define ZBIN_MIN_TRAJECTORY_LENGTH 91
#define ZBIN_AGENT_MOTORCYCLIST 4

// zdrive_constants.h
#define ZNAVI_GRID_CELL_SIZE 5.0f             // GRID_CELL_SIZE
#define ZNAVI_GRID_VISION_RANGE 30            // (int)MAX_ROAD_OBSERVATION_DISTANCE
#define ZNAVI_MAX_DETECTION_ROAD_ENTITIES 750 // MAX_ENTITIES_PER_CELL * 25
#define ZNAVI_SEGMENT_MAX_CELLS 256
#define ZNAVI_MAX_LANE_DISTANCE 10.0f       // MAX_LANE_DISTANCE
#define ZNAVI_MAX_NEARBY_LANE_DISTANCE 4.0f // MAX_NEARBY_LANE_DISTANCE
#define ZNAVI_LANE_TIE_DISTANCE 0.75f       // NAVI_LANE_TIE_DISTANCE
#define ZNAVI_MAX_LANE_FEASIBILITY_TIME 30.0F
#define ZNAVI_MAX_LANE_FEASIBILITY 500.0F
#define ZNAVI_UNREACHABLE_DISTANCE 2000.0f // UNREACHABLE_DISTANCE_TO_GOAL
#define ZNAVI_LOOKAHEAD_WAYPOINTS_MAX 50   // NAVI_LOOKAHEAD_WAYPOINTS_MAX
#define ZNAVI_LOOKAHEAD_MAX_WALK_STEPS 512 // NAVI_LOOKAHEAD_MAX_WALK_STEPS
#define ZNAVI_DEFAULT_LOOKAHEAD_DISTANCE 120.0f
#define ZNAVI_DEFAULT_LOOKAHEAD_NORM 150.0f // NAVI_LOOKAHEAD_NORM_DIST
#define ZNAVI_WAYPOINT_FEATURES 2           // ego-frame x, y

// zdrive_utils.h:15-17
#define ZB_MIN(a, b) ((a) < (b) ? (a) : (b))
#define ZB_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZB_CLAMP(v, m, M) (ZB_MIN(M, ZB_MAX(v, m)))

struct ZNavigation {
    int num_roads;
    int num_goals;
    int num_lines; // navi entities per goal; navi entity e = goal * num_lines + line
    int total;     // num_goals * num_lines
    int samples_per_goal;
    int *line_num_points;    // [num_lines] num_sub_lines + 1
    int *line_sample_offset; // [num_lines] offset of the line's samples within one goal
    float *line_x_start;     // [num_lines] navi segment endpoints, centred like the entities
    float *line_y_start;
    float *line_x_end;
    float *line_y_end;
    int *next_entity; // [num_goals * samples_per_goal]
    int *next_point;
    float *feasibility_left;
    float *feasibility_center;
    float *feasibility_right;
    float *distance_to_goal;
    int **road_navi_entity_idx; // [num_roads] NULL or [num_goals * (num_waypoints - 1)], zdrive layout

    // zdrive road grid (zdrive_geometry.c init_grid_map), in road-index space
    float grid_top_left_x;
    float grid_top_left_y;
    float grid_bottom_right_x;
    float grid_bottom_right_y;
    int grid_cols;
    int grid_rows;
    int *cell_offsets; // CSR [cells + 1]
    int *cell_road;
    int *cell_waypoint;
    int neighbor_offsets[ZNAVI_GRID_VISION_RANGE * ZNAVI_GRID_VISION_RANGE * 2];
    // Lazily built per-cell ROAD_LANE candidates: the first ZNAVI_MAX_DETECTION_ROAD_ENTITIES unique entries of
    // zdrive's neighbour cache for that cell, filtered to lanes. -1 count = not built yet.
    int *candidate_count;
    int **candidates; // pairs (entity_idx, waypoint_idx)
    int *road_waypoint_offset;
    int *dedup_stamp;
    int stamp;
};

static void zbin_fail(const char *filename, const char *message) {
    raise_error_with_message(ERROR_INITIALIZATION_FAILED,
                             "malformed zpufferdrive binary %s: %s (regenerate with process_maps.py)", filename,
                             message);
}

static void zbin_read(void *dst, size_t size, size_t count, FILE *file, const char *filename) {
    if (count > 0 && fread(dst, size, count, file) != count) {
        zbin_fail(filename, "truncated file");
    }
}

static void zbin_skip(size_t bytes, FILE *file, const char *filename) {
    if (bytes > 0 && fseek(file, (long)bytes, SEEK_CUR) != 0) {
        zbin_fail(filename, "truncated file");
    }
}

static void zbin_check_range(const char *filename, const char *field, int value, int min_value, int max_value) {
    if (value < min_value || value > max_value) {
        raise_error_with_message(ERROR_INITIALIZATION_FAILED,
                                 "malformed zpufferdrive binary %s: %s=%d out of range [%d, %d]", filename, field,
                                 value, min_value, max_value);
    }
}

static void *zbin_alloc(size_t count, size_t size) {
    void *ptr = calloc(count > 0 ? count : 1, size);
    if (ptr == NULL) {
        RAISE_MEMORY_ERROR_WITH_SIZE(count * size);
    }
    return ptr;
}

// Reads the v14 header; returns 1 and fills offsets when `file` is a zdrive binary.
static int zbin_read_header(FILE *file, uint32_t offsets[ZBIN_NUM_SECTIONS]) {
    uint32_t header[2];
    if (fseek(file, 0, SEEK_END) != 0)
        return 0;
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0)
        return 0;
    if (fread(header, sizeof(uint32_t), 2, file) != 2 || header[0] != ZBIN_VERSION || header[1] < ZBIN_NUM_SECTIONS ||
        header[1] > 64) {
        return 0;
    }
    if (fread(offsets, sizeof(uint32_t), ZBIN_NUM_SECTIONS, file) != ZBIN_NUM_SECTIONS)
        return 0;
    uint32_t header_bytes = 2 * sizeof(uint32_t) + header[1] * sizeof(uint32_t);
    if (offsets[0] != header_bytes)
        return 0;
    for (int i = 1; i < ZBIN_NUM_SECTIONS; i++) {
        if (offsets[i] < offsets[i - 1])
            return 0;
    }
    return offsets[ZBIN_NUM_SECTIONS - 1] <= (uint64_t)file_size;
}

// MAP_FORMAT_ZBIN for zdrive v14 binaries, MAP_FORMAT_LEGACY otherwise, -1 if the file cannot be opened.
int drive_map_format(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return -1;
    uint32_t offsets[ZBIN_NUM_SECTIONS];
    int is_zbin = zbin_read_header(file, offsets);
    fclose(file);
    return is_zbin ? MAP_FORMAT_ZBIN : MAP_FORMAT_LEGACY;
}

static int zbin_remap_road_type(int type) {
    switch (type) {
    case 4: // ROAD_LANE
        return ROAD_LANE;
    case 5: // ROAD_LINE
    case 6: // SOLID_ROAD_LINE
        return ROAD_LINE;
    case 7:
        return ROAD_EDGE;
    case 8:
        return STOP_SIGN;
    case 9:
        return CROSSWALK;
    case 10:
        return SPEED_BUMP;
    case 11:
        return DRIVEWAY;
    default:
        return -1;
    }
}

static void zbin_alloc_object_arrays(Entity *entity, int size) {
    entity->array_size = size;
    entity->traj_x = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_y = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_z = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_vx = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_vy = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_vz = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_heading = (float *)zbin_alloc(size, sizeof(float));
    entity->traj_valid = (int *)zbin_alloc(size, sizeof(int));
    entity->inferred_traj_x = (float *)zbin_alloc(size, sizeof(float));
    entity->inferred_traj_y = (float *)zbin_alloc(size, sizeof(float));
    float **expert_arrays[5] = {&entity->expert_accel, &entity->expert_steering, &entity->expert_delta_x,
                                &entity->expert_delta_y, &entity->expert_delta_yaw};
    for (int k = 0; k < 5; k++) {
        *expert_arrays[k] = (float *)zbin_alloc(size, sizeof(float));
        for (int t = 0; t < size; t++)
            (*expert_arrays[k])[t] = -1.0f; // zbins carry no expert actions
    }
}

// Sections 1-2 of zdrive_load.c (load_map_binaries) into PufferDrive entities. Entities are calloc'd; roads follow
// the (capped) objects, so road i is entity num_objects + i.
Entity *load_zbin_map(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return NULL;
    uint32_t offsets[ZBIN_NUM_SECTIONS];
    if (!zbin_read_header(file, offsets)) {
        fclose(file);
        zbin_fail(filename, "invalid header");
    }
    if (fseek(file, offsets[0], SEEK_SET) != 0)
        zbin_fail(filename, "truncated file");

    // --- Section 1: scenario metadata + agents ---
    char scenario_id[ZBIN_SCENARIO_ID_SIZE];
    zbin_read(scenario_id, 1, ZBIN_SCENARIO_ID_SIZE, file, filename);
    memcpy(env->scenario_id, scenario_id, sizeof(env->scenario_id));
    zbin_skip(ZBIN_JSON_SOURCE_FILE_SIZE + ZBIN_GIT_HASH_SIZE, file, filename);
    int goals_assigned_to_static_experts;
    zbin_read(&goals_assigned_to_static_experts, sizeof(int), 1, file, filename);
    int sdc_track_index;
    zbin_read(&sdc_track_index, sizeof(int), 1, file, filename);
    int num_tracks_to_predict;
    zbin_read(&num_tracks_to_predict, sizeof(int), 1, file, filename);
    zbin_check_range(filename, "num_tracks_to_predict", num_tracks_to_predict, 0, ZBIN_MAX_TRACKS_TO_PREDICT);
    int *tracks_to_predict = (int *)zbin_alloc(num_tracks_to_predict, sizeof(int));
    zbin_read(tracks_to_predict, sizeof(int), num_tracks_to_predict, file, filename);

    int num_objects_on_disk;
    zbin_read(&num_objects_on_disk, sizeof(int), 1, file, filename);
    zbin_check_range(filename, "num_objects", num_objects_on_disk, 0, ZBIN_MAX_OBJECTS_ON_DISK);
    int num_objects = ZB_MIN(num_objects_on_disk, ZBIN_MAX_OBJECTS);
    Entity *entities = (Entity *)zbin_alloc(num_objects, sizeof(Entity));

    for (int i = 0; i < num_objects_on_disk; i++) {
        int discard = i >= num_objects;
        int type, id, traj_length;
        float dims[3];
        zbin_read(&type, sizeof(int), 1, file, filename);
        zbin_check_range(filename, "agent type", type, VEHICLE, ZBIN_AGENT_MOTORCYCLIST);
        zbin_read(&id, sizeof(int), 1, file, filename);
        zbin_read(dims, sizeof(float), 3, file, filename);
        zbin_read(&traj_length, sizeof(int), 1, file, filename);
        zbin_check_range(filename, "trajectory_length", traj_length, 1, ZBIN_MAX_TRAJECTORY_LENGTH);
        Entity *entity = discard ? NULL : &entities[i];
        if (discard) {
            zbin_skip((size_t)traj_length * (7 * sizeof(float) + sizeof(int)), file, filename);
        } else {
            int size = ZB_MAX(traj_length, ZBIN_MIN_TRAJECTORY_LENGTH);
            zbin_alloc_object_arrays(entity, size);
            float *float_arrays[7] = {entity->traj_x,  entity->traj_y,  entity->traj_z,      entity->traj_vx,
                                      entity->traj_vy, entity->traj_vz, entity->traj_heading};
            for (int k = 0; k < 7; k++)
                zbin_read(float_arrays[k], sizeof(float), traj_length, file, filename);
            zbin_read(entity->traj_valid, sizeof(int), traj_length, file, filename);
            for (int t = traj_length; t < size; t++) {
                entity->traj_x[t] = INVALID_POSITION;
                entity->traj_y[t] = INVALID_POSITION;
                entity->traj_z[t] = INVALID_POSITION;
            }
        }
        int expert_type, goal_index, zombie;
        float speed_fields[2], goal[2];
        zbin_read(&expert_type, sizeof(int), 1, file, filename);
        zbin_check_range(filename, "expert_type", expert_type, 0, 3);
        zbin_read(speed_fields, sizeof(float), 2, file, filename); // speed, set_speed_fraction (unused)
        zbin_read(goal, sizeof(float), 2, file, filename);
        zbin_read(&goal_index, sizeof(int), 1, file, filename);
        zbin_read(&zombie, sizeof(int), 1, file, filename);
        // hybrid_a_long, hybrid_j_lat, turn_indicator (unused)
        zbin_skip((size_t)traj_length * (2 * sizeof(float) + sizeof(int)), file, filename);
        if (discard)
            continue;
        entity->type = type == ZBIN_AGENT_MOTORCYCLIST ? CYCLIST : type;
        entity->id = id;
        entity->width = dims[0];
        entity->length = dims[1];
        entity->height = dims[2];
        // zdrive snaps routed goals onto its navi graph. Undo that by using the last valid logged sample, which is
        // what legacy PufferDrive bins store (the scenario JSON goalPosition). The route (goal_index) is kept.
        entity->goal_position_x = goal[0];
        entity->goal_position_y = goal[1];
        entity->goal_position_z = 0.0f;
        for (int t = traj_length - 1; t >= 0; t--) {
            if (entity->traj_valid[t]) {
                entity->goal_position_x = entity->traj_x[t];
                entity->goal_position_y = entity->traj_y[t];
                entity->goal_position_z = entity->traj_z[t];
                break;
            }
        }
        entity->mark_as_expert = expert_type != 0 || zombie != 0;
        entity->zdrive_goal_index = goal_index;
    }
    if (ftell(file) != (long)offsets[1])
        zbin_fail(filename, "section 1 size does not match the header");

    // --- Section 2: roads ---
    int num_roads;
    zbin_read(&num_roads, sizeof(int), 1, file, filename);
    zbin_check_range(filename, "num_roads", num_roads, 0, ZBIN_MAX_ROADS);
    entities = (Entity *)realloc(entities, (size_t)(num_objects + num_roads) * sizeof(Entity));
    if (entities == NULL) {
        RAISE_MEMORY_ERROR_WITH_SIZE((size_t)(num_objects + num_roads) * sizeof(Entity));
    }
    memset(&entities[num_objects], 0, (size_t)num_roads * sizeof(Entity));
    for (int i = 0; i < num_roads; i++) {
        Entity *road = &entities[num_objects + i];
        int type, num_waypoints, priority;
        float width;
        zbin_read(&type, sizeof(int), 1, file, filename);
        zbin_read(&road->id, sizeof(int), 1, file, filename);
        zbin_read(&num_waypoints, sizeof(int), 1, file, filename);
        zbin_check_range(filename, "num_waypoints", num_waypoints, 1, ZBIN_MAX_WAYPOINTS_PER_ROAD);
        road->type = zbin_remap_road_type(type);
        if (road->type == -1) {
            zbin_check_range(filename, "road type", type, 4, 11);
        }
        road->array_size = num_waypoints;
        road->traj_x = (float *)zbin_alloc(num_waypoints, sizeof(float));
        road->traj_y = (float *)zbin_alloc(num_waypoints, sizeof(float));
        road->traj_z = (float *)zbin_alloc(num_waypoints, sizeof(float));
        zbin_read(road->traj_x, sizeof(float), num_waypoints, file, filename);
        zbin_read(road->traj_y, sizeof(float), num_waypoints, file, filename);
        zbin_read(road->traj_z, sizeof(float), num_waypoints, file, filename);
        zbin_read(&width, sizeof(float), 1, file, filename);
        zbin_read(&road->road_speed_limit, sizeof(float), 1, file, filename);
        zbin_read(&priority, sizeof(int), 1, file, filename);
        road->width = width;
        road->zdrive_goal_index = -1;
    }
    if (ftell(file) != (long)offsets[2])
        zbin_fail(filename, "section 2 size does not match the header");
    fclose(file);

    env->num_objects = num_objects;
    env->num_roads = num_roads;
    env->num_entities = num_objects + num_roads;
    env->sdc_track_index = sdc_track_index >= 0 && sdc_track_index < num_objects ? sdc_track_index : -1;
    // Tracks pointing at discarded objects are dropped.
    int kept_tracks = 0;
    for (int i = 0; i < num_tracks_to_predict; i++) {
        if (tracks_to_predict[i] >= 0 && tracks_to_predict[i] < num_objects)
            tracks_to_predict[kept_tracks++] = tracks_to_predict[i];
    }
    env->num_tracks_to_predict = kept_tracks;
    if (kept_tracks == 0) {
        free(tracks_to_predict);
        tracks_to_predict = NULL;
    }
    env->tracks_to_predict_indices = tracks_to_predict;
    return entities;
}

// zdrive.c set_means: running mean over objects valid at t=0 (first sample) and all road waypoints.
void zbin_set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;
    for (int i = 0; i < env->num_objects; i++) {
        if (env->entities[i].traj_valid[0]) {
            point_count++;
            mean_x += (env->entities[i].traj_x[0] - mean_x) / point_count;
            mean_y += (env->entities[i].traj_y[0] - mean_y) / point_count;
        }
    }
    for (int i = env->num_objects; i < env->num_entities; i++) {
        for (int j = 0; j < env->entities[i].array_size; j++) {
            point_count++;
            mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
            mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_objects; i++) {
        Entity *entity = &env->entities[i];
        for (int j = 0; j < entity->array_size; j++) {
            if (entity->traj_x[j] == INVALID_POSITION)
                continue;
            entity->traj_x[j] -= mean_x;
            entity->traj_y[j] -= mean_y;
        }
        entity->goal_position_x -= mean_x;
        entity->goal_position_y -= mean_y;
    }
    for (int i = env->num_objects; i < env->num_entities; i++) {
        for (int j = 0; j < env->entities[i].array_size; j++) {
            env->entities[i].traj_x[j] -= mean_x;
            env->entities[i].traj_y[j] -= mean_y;
        }
    }
}

void free_znavigation(ZNavigation *zn) {
    if (zn == NULL)
        return;
    free(zn->line_num_points);
    free(zn->line_sample_offset);
    free(zn->line_x_start);
    free(zn->line_y_start);
    free(zn->line_x_end);
    free(zn->line_y_end);
    free(zn->next_entity);
    free(zn->next_point);
    free(zn->feasibility_left);
    free(zn->feasibility_center);
    free(zn->feasibility_right);
    free(zn->distance_to_goal);
    if (zn->road_navi_entity_idx != NULL) {
        for (int i = 0; i < zn->num_roads; i++)
            free(zn->road_navi_entity_idx[i]);
    }
    free(zn->road_navi_entity_idx);
    free(zn->cell_offsets);
    free(zn->cell_road);
    free(zn->cell_waypoint);
    int cell_count = zn->grid_cols * zn->grid_rows;
    if (zn->candidates != NULL) {
        for (int i = 0; i < cell_count; i++)
            free(zn->candidates[i]);
    }
    free(zn->candidates);
    free(zn->candidate_count);
    free(zn->road_waypoint_offset);
    free(zn->dedup_stamp);
    free(zn);
}

typedef struct ZbinCursor ZbinCursor;
struct ZbinCursor {
    const unsigned char *data;
    size_t size;
    size_t pos;
    const char *filename;
};

static void zbin_take(ZbinCursor *cursor, void *dst, size_t size, size_t count) {
    size_t bytes = size * count;
    if (bytes > cursor->size - cursor->pos) {
        zbin_fail(cursor->filename, "truncated navigation sections");
    }
    if (dst != NULL && bytes > 0) {
        memcpy(dst, cursor->data + cursor->pos, bytes);
    }
    cursor->pos += bytes;
}

static int zbin_take_int(ZbinCursor *cursor) {
    int value;
    zbin_take(cursor, &value, sizeof(int), 1);
    return value;
}

// zdrive find_lane_by_id: first ROAD_LANE with the given id, as a road index.
static int zbin_find_lane_by_id(const Drive *env, int id) {
    for (int i = 0; i < env->num_roads; i++) {
        const Entity *road = &env->entities[env->num_objects + i];
        if (road->type == ROAD_LANE && road->id == id)
            return i;
    }
    return -1;
}

static int znavi_grid_index(const ZNavigation *zn, float x, float y) {
    // zdrive_geometry.h get_grid_index
    if (zn->grid_top_left_x >= zn->grid_bottom_right_x || zn->grid_bottom_right_y >= zn->grid_top_left_y) {
        return -1;
    }
    float relative_x = x - zn->grid_top_left_x;
    float relative_y = y - zn->grid_bottom_right_y;
    int grid_x = (int)(relative_x / ZNAVI_GRID_CELL_SIZE);
    int grid_y = (int)(relative_y / ZNAVI_GRID_CELL_SIZE);
    if (grid_x < 0 || grid_x >= zn->grid_cols || grid_y < 0 || grid_y >= zn->grid_rows) {
        return -1;
    }
    return (grid_y * zn->grid_cols) + grid_x;
}

// zdrive_geometry.c walk_segment_grid_cells
static int znavi_walk_segment_cells(const ZNavigation *zn, float x0, float y0, float x1, float y1, int *out_indices,
                                    int max_out) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float seg_len = sqrtf(dx * dx + dy * dy);
    float step = ZNAVI_GRID_CELL_SIZE * 0.5f;
    int n_steps = ZB_MAX(2, (int)(seg_len / step) + 1);
    int count = 0;
    int prev_index = -1;
    for (int s = 0; s <= n_steps; s++) {
        float t = (float)s / (float)n_steps;
        int grid_index = znavi_grid_index(zn, x0 + t * dx, y0 + t * dy);
        if (grid_index == -1 || grid_index == prev_index)
            continue;
        int duplicate = 0;
        for (int k = 0; k < ZB_MIN(count, max_out); k++) {
            if (out_indices[k] == grid_index) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            prev_index = grid_index;
            continue;
        }
        if (count < max_out)
            out_indices[count] = grid_index;
        count++;
        prev_index = grid_index;
    }
    return count;
}

// zdrive_geometry.c init_grid_map + init_neighbor_offsets, on the centred road geometry.
static void znavi_build_grid(const Drive *env, ZNavigation *zn) {
    float top_left_x = 0.0f, top_left_y = 0.0f, bottom_right_x = 0.0f, bottom_right_y = 0.0f;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_roads; i++) {
        const Entity *road = &env->entities[env->num_objects + i];
        for (int j = 0; j < road->array_size; j++) {
            if (road->traj_x[j] == INVALID_POSITION || road->traj_y[j] == INVALID_POSITION)
                continue;
            if (!first_valid_point) {
                top_left_x = bottom_right_x = road->traj_x[j];
                top_left_y = bottom_right_y = road->traj_y[j];
                first_valid_point = 1;
                continue;
            }
            if (road->traj_x[j] < top_left_x)
                top_left_x = road->traj_x[j];
            if (road->traj_x[j] > bottom_right_x)
                bottom_right_x = road->traj_x[j];
            if (road->traj_y[j] > top_left_y)
                top_left_y = road->traj_y[j];
            if (road->traj_y[j] < bottom_right_y)
                bottom_right_y = road->traj_y[j];
        }
    }
    zn->grid_top_left_x = top_left_x;
    zn->grid_top_left_y = top_left_y;
    zn->grid_bottom_right_x = bottom_right_x;
    zn->grid_bottom_right_y = bottom_right_y;
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    zn->grid_cols = ceil(grid_width / ZNAVI_GRID_CELL_SIZE);
    zn->grid_rows = ceil(grid_height / ZNAVI_GRID_CELL_SIZE);
    int cell_count = zn->grid_cols * zn->grid_rows;

    zn->cell_offsets = (int *)zbin_alloc((size_t)cell_count + 1, sizeof(int));
    int tmp_indices[ZNAVI_SEGMENT_MAX_CELLS];
    for (int i = 0; i < env->num_roads; i++) {
        const Entity *road = &env->entities[env->num_objects + i];
        for (int j = 0; j < road->array_size - 1; j++) {
            int n_cells = znavi_walk_segment_cells(zn, road->traj_x[j], road->traj_y[j], road->traj_x[j + 1],
                                                   road->traj_y[j + 1], tmp_indices, ZNAVI_SEGMENT_MAX_CELLS);
            for (int c = 0; c < n_cells && c < ZNAVI_SEGMENT_MAX_CELLS; c++)
                zn->cell_offsets[tmp_indices[c] + 1]++;
        }
    }
    for (int c = 0; c < cell_count; c++)
        zn->cell_offsets[c + 1] += zn->cell_offsets[c];
    int total_entries = zn->cell_offsets[cell_count];
    zn->cell_road = (int *)zbin_alloc(total_entries, sizeof(int));
    zn->cell_waypoint = (int *)zbin_alloc(total_entries, sizeof(int));
    int *insert_index = (int *)zbin_alloc((size_t)cell_count + 1, sizeof(int));
    memcpy(insert_index, zn->cell_offsets, ((size_t)cell_count + 1) * sizeof(int));
    for (int i = 0; i < env->num_roads; i++) {
        const Entity *road = &env->entities[env->num_objects + i];
        for (int j = 0; j < road->array_size - 1; j++) {
            int n_cells = znavi_walk_segment_cells(zn, road->traj_x[j], road->traj_y[j], road->traj_x[j + 1],
                                                   road->traj_y[j + 1], tmp_indices, ZNAVI_SEGMENT_MAX_CELLS);
            for (int c = 0; c < n_cells && c < ZNAVI_SEGMENT_MAX_CELLS; c++) {
                int slot = insert_index[tmp_indices[c]]++;
                zn->cell_road[slot] = i;
                zn->cell_waypoint[slot] = j;
            }
        }
    }
    free(insert_index);

    // Spiral neighbour offsets
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0, steps_to_take = 1, steps_taken = 0, segments_completed = 0, total = 0;
    int max_offsets = ZNAVI_GRID_VISION_RANGE * ZNAVI_GRID_VISION_RANGE;
    int curr_idx = 0;
    zn->neighbor_offsets[curr_idx++] = 0;
    zn->neighbor_offsets[curr_idx++] = 0;
    total++;
    while (total < max_offsets) {
        x += dx[dir];
        y += dy[dir];
        if (abs(x) <= ZNAVI_GRID_VISION_RANGE / 2 && abs(y) <= ZNAVI_GRID_VISION_RANGE / 2) {
            zn->neighbor_offsets[curr_idx++] = x;
            zn->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;
        dir = (dir + 1) % 4;
        segments_completed++;
        if (segments_completed % 2 == 0)
            steps_to_take++;
    }

    zn->candidate_count = (int *)zbin_alloc(cell_count, sizeof(int));
    for (int c = 0; c < cell_count; c++)
        zn->candidate_count[c] = -1;
    zn->candidates = (int **)zbin_alloc(cell_count, sizeof(int *));
    zn->road_waypoint_offset = (int *)zbin_alloc((size_t)env->num_roads + 1, sizeof(int));
    for (int i = 0; i < env->num_roads; i++)
        zn->road_waypoint_offset[i + 1] = zn->road_waypoint_offset[i] + env->entities[env->num_objects + i].array_size;
    zn->dedup_stamp = (int *)zbin_alloc(zn->road_waypoint_offset[env->num_roads], sizeof(int));
    zn->stamp = 0;
}

// ROAD_LANE subset of zdrive's neighbour cache for `cell` (zdrive_geometry.c cache_neighbor_offsets, deduplicated
// by (road, waypoint) keeping the first, then truncated by get_neighbor_cache_entities to
// MAX_DETECTION_ROAD_ENTITIES). Built on first use.
static int znavi_cell_candidates(const Drive *env, ZNavigation *zn, int cell, const int **out) {
    if (cell < 0 || cell >= zn->grid_cols * zn->grid_rows)
        return 0;
    if (zn->candidate_count[cell] < 0) {
        if (++zn->stamp == INT32_MAX) {
            memset(zn->dedup_stamp, 0, (size_t)zn->road_waypoint_offset[env->num_roads] * sizeof(int));
            zn->stamp = 1;
        }
        int pairs[ZNAVI_MAX_DETECTION_ROAD_ENTITIES * 2];
        int unique = 0, lanes = 0;
        int cell_x = cell % zn->grid_cols;
        int cell_y = cell / zn->grid_cols;
        int num_offsets = ZNAVI_GRID_VISION_RANGE * ZNAVI_GRID_VISION_RANGE;
        for (int j = 0; j < num_offsets && unique < ZNAVI_MAX_DETECTION_ROAD_ENTITIES; j++) {
            int x = cell_x + zn->neighbor_offsets[j * 2];
            int y = cell_y + zn->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= zn->grid_cols || y < 0 || y >= zn->grid_rows)
                continue;
            int grid_index = zn->grid_cols * y + x;
            for (int k = zn->cell_offsets[grid_index];
                 k < zn->cell_offsets[grid_index + 1] && unique < ZNAVI_MAX_DETECTION_ROAD_ENTITIES; k++) {
                int road = zn->cell_road[k];
                int waypoint = zn->cell_waypoint[k];
                int key = zn->road_waypoint_offset[road] + waypoint;
                if (zn->dedup_stamp[key] == zn->stamp)
                    continue;
                zn->dedup_stamp[key] = zn->stamp;
                unique++;
                if (env->entities[env->num_objects + road].type == ROAD_LANE) {
                    pairs[lanes * 2] = env->num_objects + road;
                    pairs[lanes * 2 + 1] = waypoint;
                    lanes++;
                }
            }
        }
        zn->candidates[cell] = (int *)zbin_alloc((size_t)lanes * 2, sizeof(int));
        memcpy(zn->candidates[cell], pairs, (size_t)lanes * 2 * sizeof(int));
        zn->candidate_count[cell] = lanes;
    }
    *out = zn->candidates[cell];
    return zn->candidate_count[cell];
}

// Sections 3-4 of zdrive_load.c (navi topology and routes) for the already loaded and centred zbin scene, plus
// zdrive's road grid. Aborts on malformed data like zdrive.
ZNavigation *load_znavigation(Drive *env) {
    const char *filename = env->map_name;
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        RAISE_FILE_ERROR(filename);
    }
    uint32_t offsets[ZBIN_NUM_SECTIONS];
    if (!zbin_read_header(file, offsets)) {
        fclose(file);
        zbin_fail(filename, "invalid header");
    }
    fseek(file, 0, SEEK_END);
    size_t bytes = (size_t)ftell(file) - offsets[2];
    unsigned char *data = (unsigned char *)zbin_alloc(bytes, 1);
    if (fseek(file, offsets[2], SEEK_SET) != 0 || fread(data, 1, bytes, file) != bytes) {
        fclose(file);
        zbin_fail(filename, "truncated file");
    }
    fclose(file);
    ZbinCursor cursor = {data, bytes, 0, filename};

    ZNavigation *zn = (ZNavigation *)zbin_alloc(1, sizeof(ZNavigation));
    zn->num_roads = env->num_roads;

    // --- Section 3: navi topology ---
    int num_lines = zbin_take_int(&cursor);
    int max_lpc = zbin_take_int(&cursor);
    int max_sub_lines = zbin_take_int(&cursor);
    zbin_check_range(filename, "num_lines", num_lines, 0, ZBIN_MAX_NAVI_LINES);
    zbin_check_range(filename, "max_lpc", max_lpc, 0, ZBIN_MAX_LEFT_PERP_CONNECTIONS);
    zbin_check_range(filename, "max_sub_lines_global", max_sub_lines, 0, ZBIN_MAX_SUB_LINES);
    size_t lpc_stride_point = (size_t)max_lpc * ZBIN_LEFT_PERP_CONNECTION_DIM;
    zn->num_lines = num_lines;
    zn->line_num_points = (int *)zbin_alloc(num_lines, sizeof(int));
    zn->line_sample_offset = (int *)zbin_alloc(num_lines, sizeof(int));
    zn->line_x_start = (float *)zbin_alloc(num_lines, sizeof(float));
    zn->line_y_start = (float *)zbin_alloc(num_lines, sizeof(float));
    zn->line_x_end = (float *)zbin_alloc(num_lines, sizeof(float));
    zn->line_y_end = (float *)zbin_alloc(num_lines, sizeof(float));
    int *line_road = (int *)zbin_alloc(num_lines, sizeof(int));
    int *line_index = (int *)zbin_alloc(num_lines, sizeof(int));
    int samples_per_goal = 0;
    for (int j = 0; j < num_lines; j++) {
        int line_id = zbin_take_int(&cursor);
        line_index[j] = zbin_take_int(&cursor);
        int num_sub_lines = zbin_take_int(&cursor);
        zbin_check_range(filename, "num_sub_lines", num_sub_lines, 0, max_sub_lines);
        int road = zbin_find_lane_by_id(env, line_id);
        // A lane with <= 1 waypoint carries no navi table, like zdrive.
        if (road >= 0 && env->entities[env->num_objects + road].array_size < 2)
            road = -1;
        line_road[j] = road;
        if (road >= 0) {
            zbin_check_range(filename, "line_index", line_index[j], 0,
                             env->entities[env->num_objects + road].array_size - 2);
            const Entity *lane = &env->entities[env->num_objects + road];
            zn->line_x_start[j] = lane->traj_x[line_index[j]];
            zn->line_y_start[j] = lane->traj_y[line_index[j]];
            zn->line_x_end[j] = lane->traj_x[line_index[j] + 1];
            zn->line_y_end[j] = lane->traj_y[line_index[j] + 1];
        } else {
            // zdrive stores 0 and centres it afterwards.
            zn->line_x_start[j] = zn->line_x_end[j] = 0.0f - env->world_mean_x;
            zn->line_y_start[j] = zn->line_y_end[j] = 0.0f - env->world_mean_y;
        }
        zbin_take(&cursor, NULL, sizeof(int), (size_t)num_sub_lines * lpc_stride_point); // left-perp connections
        zn->line_num_points[j] = num_sub_lines + 1;
        zn->line_sample_offset[j] = samples_per_goal;
        samples_per_goal += num_sub_lines;
    }
    if (cursor.pos != (size_t)(offsets[3] - offsets[2]))
        zbin_fail(filename, "section 3 size does not match the header");

    // --- Section 4: navi routes ---
    int num_goals = zbin_take_int(&cursor);
    zbin_check_range(filename, "num_goals", num_goals, 0, ZBIN_MAX_GOALS);
    if (num_lines > 0 && num_goals > ZBIN_MAX_NAVI_ENTITIES / num_lines)
        zbin_fail(filename, "too many navi entities");
    zn->num_goals = num_goals;
    zn->total = num_goals * num_lines;
    zn->samples_per_goal = samples_per_goal;
    size_t sample_count = (size_t)num_goals * samples_per_goal;
    zn->next_entity = (int *)zbin_alloc(sample_count, sizeof(int));
    zn->next_point = (int *)zbin_alloc(sample_count, sizeof(int));
    zn->feasibility_left = (float *)zbin_alloc(sample_count, sizeof(float));
    zn->feasibility_center = (float *)zbin_alloc(sample_count, sizeof(float));
    zn->feasibility_right = (float *)zbin_alloc(sample_count, sizeof(float));
    zn->distance_to_goal = (float *)zbin_alloc(sample_count, sizeof(float));
    zn->road_navi_entity_idx = (int **)zbin_alloc(env->num_roads, sizeof(int *));
    for (int i = 0; i < env->num_roads; i++) {
        const Entity *road = &env->entities[env->num_objects + i];
        if (road->type != ROAD_LANE || road->array_size < 2)
            continue;
        size_t table_size = (size_t)num_goals * (road->array_size - 1);
        zn->road_navi_entity_idx[i] = (int *)zbin_alloc(table_size, sizeof(int));
        memset(zn->road_navi_entity_idx[i], -1, table_size * sizeof(int));
    }
    for (int goal = 0; goal < num_goals; goal++) {
        for (int j = 0; j < num_lines; j++) {
            int navi_idx = goal * num_lines + j;
            int n_sub = zn->line_num_points[j] - 1;
            size_t base = (size_t)goal * samples_per_goal + zn->line_sample_offset[j];
            zbin_take(&cursor, &zn->next_entity[base], sizeof(int), n_sub);
            zbin_take(&cursor, &zn->next_point[base], sizeof(int), n_sub);
            zbin_take(&cursor, &zn->feasibility_left[base], sizeof(float), n_sub);
            zbin_take(&cursor, &zn->feasibility_center[base], sizeof(float), n_sub);
            zbin_take(&cursor, &zn->feasibility_right[base], sizeof(float), n_sub);
            zbin_take(&cursor, &zn->distance_to_goal[base], sizeof(float), n_sub);
            for (int s = 0; s < n_sub; s++) {
                zn->feasibility_left[base + s] = fmin(ZNAVI_MAX_LANE_FEASIBILITY, zn->feasibility_left[base + s]);
                zn->feasibility_center[base + s] = fmin(ZNAVI_MAX_LANE_FEASIBILITY, zn->feasibility_center[base + s]);
                zn->feasibility_right[base + s] = fmin(ZNAVI_MAX_LANE_FEASIBILITY, zn->feasibility_right[base + s]);
            }
            int road = line_road[j];
            if (road >= 0) {
                int num_segments = env->entities[env->num_objects + road].array_size - 1;
                zn->road_navi_entity_idx[road][(size_t)line_index[j] + (size_t)goal * num_segments] = navi_idx;
            }
        }
    }
    free(line_road);
    free(line_index);
    free(data);
    znavi_build_grid(env, zn);
    return zn;
}

static inline int znavi_num_points(const ZNavigation *zn, int navi_idx) {
    return zn->line_num_points[navi_idx % zn->num_lines];
}

static inline size_t znavi_sample_base(const ZNavigation *zn, int navi_idx) {
    int goal = navi_idx / zn->num_lines;
    return (size_t)goal * zn->samples_per_goal + zn->line_sample_offset[navi_idx - goal * zn->num_lines];
}

// zdrive_geometry/utils: navi point k of entity navi_idx
static inline void znavi_point(const ZNavigation *zn, int navi_idx, int point, float *out_x, float *out_y) {
    int line = navi_idx % zn->num_lines;
    int denom = ZB_MAX(zn->line_num_points[line] - 1, 1);
    *out_x = zn->line_x_start[line] + (zn->line_x_end[line] - zn->line_x_start[line]) / (float)denom * point;
    *out_y = zn->line_y_start[line] + (zn->line_y_end[line] - zn->line_y_start[line]) / (float)denom * point;
}

// zdrive_utils.h point_to_segment_projection_scalar_2d
static inline float znavi_projection_scalar(float px, float py, float x1, float y1, float x2, float y2, int clamp) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    if (dx == 0 && dy == 0) {
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
    if (clamp) {
        if (t < 0)
            t = 0;
        else if (t > 1)
            t = 1;
    }
    return t;
}

// zdrive_utils.h point_to_segment_distance_2d
static inline float znavi_point_segment_distance(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float hypot2 = (dx * dx + dy * dy);
    if (hypot2 < 0.001F) {
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }
    float t = ((px - x1) * dx + (py - y1) * dy) / hypot2;
    if (t < 0)
        t = 0;
    else if (t > 1)
        t = 1;
    float closest_x = x1 + t * dx;
    float closest_y = y1 + t * dy;
    return sqrtf((px - closest_x) * (px - closest_x) + (py - closest_y) * (py - closest_y));
}

// zdrive_agent_obs.c lane_navi_entity_index
static int znavi_lane_entity_index(const Drive *env, int entity_idx, int waypoint_idx, int goal_index) {
    const ZNavigation *zn = env->znavigation;
    const int *table = zn->road_navi_entity_idx[entity_idx - env->num_objects];
    if (table == NULL)
        return -1;
    int num_segments = env->entities[entity_idx].array_size - 1;
    if (waypoint_idx >= num_segments || goal_index < 0 || goal_index >= zn->num_goals)
        return -1;
    int navi_idx = table[waypoint_idx + num_segments * goal_index];
    if (navi_idx < 0 || navi_idx >= zn->total)
        return -1;
    return navi_idx;
}

// zdrive_agent_obs.c lane_has_navi_route + lane_navi_route_distance_to_goal: 1 and the first reachable sample's
// distance when the lane segment carries a route to the goal.
static int znavi_lane_route(const Drive *env, int entity_idx, int waypoint_idx, int goal_index, float *out_distance) {
    const ZNavigation *zn = env->znavigation;
    int navi_idx = znavi_lane_entity_index(env, entity_idx, waypoint_idx, goal_index);
    if (navi_idx < 0)
        return 0;
    int num_points = znavi_num_points(zn, navi_idx);
    if (num_points < 2)
        return 0;
    const float *distance = &zn->distance_to_goal[znavi_sample_base(zn, navi_idx)];
    for (int s = 0; s < num_points - 1; s++) {
        if (distance[s] < (ZNAVI_UNREACHABLE_DISTANCE - 0.01F)) {
            *out_distance = distance[s];
            return 1;
        }
    }
    return 0;
}

// Non-expert localization of zdrive_agent_obs.c compute_agent_metrics (closest/routed lane selection,
// find_navi_edge_from_lane and the feasibility/distance lookup). Writes the observation state (route projection,
// speed limit, feasibility); returns the selected lane entity (-1 when none within MAX_LANE_DISTANCE) and the
// route distance to goal (ZNAVI_UNREACHABLE_DISTANCE when navigation fails).
static int znavi_localize(Drive *env, Entity *agent, float *out_distance_to_goal) {
    ZNavigation *zn = env->znavigation;
    int goal_index = agent->navi_goal_idx;
    float min_distance = ZNAVI_MAX_LANE_DISTANCE;
    float closest_speed_limit = 0.0F;
    int closest_lane_entity_idx = -1;
    int closest_lane_waypoint_idx = -1;
    float min_navi_lane_distance = ZNAVI_MAX_NEARBY_LANE_DISTANCE;
    float navi_lane_best_route_distance = ZNAVI_UNREACHABLE_DISTANCE;
    int closest_navi_lane_entity_idx = -1;
    int closest_navi_lane_waypoint_idx = -1;
    float closest_navi_lane_speed_limit = 0.0F;

    const int *candidates = NULL;
    int list_size = znavi_cell_candidates(env, zn, znavi_grid_index(zn, agent->x, agent->y), &candidates);
    for (int i = 0; i < list_size; i++) {
        int entity_idx = candidates[i * 2];
        int waypoint_idx = candidates[i * 2 + 1];
        const Entity *road_entity = &env->entities[entity_idx];
        float start[2] = {road_entity->traj_x[waypoint_idx], road_entity->traj_y[waypoint_idx]};
        float end[2] = {road_entity->traj_x[waypoint_idx + 1], road_entity->traj_y[waypoint_idx + 1]};
        float dist = znavi_point_segment_distance(agent->x, agent->y, start[0], start[1], end[0], end[1]);
        float heading_diff = fabsf(atan2f(end[1] - start[1], end[0] - start[0]) - agent->heading);
        if (heading_diff > M_PI)
            heading_diff = 2.0f * M_PI - heading_diff;
        if (heading_diff > (M_PI / 3.0f))
            dist += 1000.0f;
        if (dist < min_distance) {
            min_distance = dist;
            closest_lane_entity_idx = entity_idx;
            closest_lane_waypoint_idx = waypoint_idx;
            closest_speed_limit = road_entity->road_speed_limit;
        }
        float route_distance;
        if (dist < ZNAVI_MAX_NEARBY_LANE_DISTANCE &&
            znavi_lane_route(env, entity_idx, waypoint_idx, goal_index, &route_distance)) {
            int tied_with_best =
                closest_navi_lane_entity_idx >= 0 && fabsf(dist - min_navi_lane_distance) <= ZNAVI_LANE_TIE_DISTANCE;
            int wins =
                closest_navi_lane_entity_idx < 0 ||
                (tied_with_best ? route_distance < navi_lane_best_route_distance : dist < min_navi_lane_distance);
            if (wins) {
                min_navi_lane_distance = dist;
                navi_lane_best_route_distance = route_distance;
                closest_navi_lane_entity_idx = entity_idx;
                closest_navi_lane_waypoint_idx = waypoint_idx;
                closest_navi_lane_speed_limit = road_entity->road_speed_limit;
            }
        }
    }
    if (closest_navi_lane_entity_idx >= 0) {
        closest_lane_entity_idx = closest_navi_lane_entity_idx;
        closest_lane_waypoint_idx = closest_navi_lane_waypoint_idx;
        closest_speed_limit = closest_navi_lane_speed_limit;
    }

    // find_navi_edge_from_lane
    int navi_entity_idx = -1;
    int navi_point_idx = -1;
    float proj_x = agent->x;
    float proj_y = agent->y;
    if (closest_lane_entity_idx >= 0 && closest_lane_waypoint_idx >= 0) {
        int navi_idx = znavi_lane_entity_index(env, closest_lane_entity_idx, closest_lane_waypoint_idx, goal_index);
        if (navi_idx >= 0) {
            const Entity *lane = &env->entities[closest_lane_entity_idx];
            navi_entity_idx = navi_idx;
            float t = znavi_projection_scalar(
                agent->x, agent->y, lane->traj_x[closest_lane_waypoint_idx], lane->traj_y[closest_lane_waypoint_idx],
                lane->traj_x[closest_lane_waypoint_idx + 1], lane->traj_y[closest_lane_waypoint_idx + 1], 0);
            int navi_line_size = znavi_num_points(zn, navi_idx) - 1;
            float scaled = t * navi_line_size;
            int point = (scaled > -2.0e9f && scaled < 2.0e9f) ? (int)scaled : INT32_MIN; // x86 cvttss2si overflow
            navi_point_idx = ZB_CLAMP(point, 0, navi_line_size - 1);
            float x_a, y_a, x_b, y_b;
            znavi_point(zn, navi_idx, navi_point_idx, &x_a, &y_a);
            znavi_point(zn, navi_idx, navi_point_idx + 1, &x_b, &y_b);
            float s = znavi_projection_scalar(agent->x, agent->y, x_a, y_a, x_b, y_b, 1);
            proj_x = x_a + s * (x_b - x_a);
            proj_y = y_a + s * (y_b - y_a);
        }
    }

    float feasibility[3] = {0.0F, 0.0F, 0.0F};
    float distance_to_goal = ZNAVI_UNREACHABLE_DISTANCE;
    if (navi_entity_idx != -1) {
        int lane_feasibility_index = ZB_MAX(navi_point_idx - 1, 0);
        if (lane_feasibility_index < znavi_num_points(zn, navi_entity_idx) - 1) {
            size_t sample = znavi_sample_base(zn, navi_entity_idx) + lane_feasibility_index;
            feasibility[0] = zn->feasibility_left[sample];
            feasibility[1] = zn->feasibility_center[sample];
            feasibility[2] = zn->feasibility_right[sample];
            distance_to_goal = zn->distance_to_goal[sample];
        }
    }
    agent->znavi_entity_idx = navi_entity_idx;
    agent->znavi_point_idx = navi_point_idx;
    agent->znavi_proj_x = proj_x;
    agent->znavi_proj_y = proj_y;
    agent->navi_speed_limit = closest_speed_limit;
    agent->navi_feasibility_left_m = feasibility[0];
    agent->navi_feasibility_center_m = feasibility[1];
    agent->navi_feasibility_right_m = feasibility[2];
    *out_distance_to_goal = distance_to_goal;
    return closest_lane_entity_idx;
}

// zdrive_agent.h NaviWalk: cursor over the chained navi route, one edge per step.
typedef struct ZNaviWalk ZNaviWalk;
struct ZNaviWalk {
    int navi_e;
    int navi_p;
    float x, y;
    float prev_x, prev_y;
    float edge_len;
    float accum;
};

static inline ZNaviWalk znavi_walk_init(int navi_e, int navi_p, float start_x, float start_y) {
    ZNaviWalk walk = {0};
    walk.navi_e = navi_e;
    walk.navi_p = navi_p;
    walk.x = walk.prev_x = start_x;
    walk.y = walk.prev_y = start_y;
    return walk;
}

static inline int znavi_walk_step(const ZNavigation *zn, ZNaviWalk *walk) {
    if (walk->navi_e < 0 || walk->navi_e >= zn->total)
        return 0;
    if (walk->navi_p < 0 || walk->navi_p >= znavi_num_points(zn, walk->navi_e) - 1)
        return 0;
    size_t sample = znavi_sample_base(zn, walk->navi_e) + walk->navi_p;
    int next_e = zn->next_entity[sample];
    int next_p = zn->next_point[sample];
    if (next_e < 0 || next_e >= zn->total)
        return 0;
    float nx, ny;
    znavi_point(zn, next_e, next_p, &nx, &ny);
    walk->prev_x = walk->x;
    walk->prev_y = walk->y;
    walk->edge_len = sqrtf((nx - walk->x) * (nx - walk->x) + (ny - walk->y) * (ny - walk->y));
    walk->accum += walk->edge_len;
    walk->x = nx;
    walk->y = ny;
    walk->navi_e = next_e;
    walk->navi_p = next_p;
    return 1;
}

static inline void znavi_walk_interpolate(const ZNaviWalk *walk, float dist, float *out_x, float *out_y) {
    float t = (walk->edge_len > 1e-6f) ? 1.0f - (walk->accum - dist) / walk->edge_len : 1.0f;
    *out_x = walk->prev_x + t * (walk->x - walk->prev_x);
    *out_y = walk->prev_y + t * (walk->y - walk->prev_y);
}

// World-frame route lookahead points at arclength (w + 1) * distance / n_waypoints from the agent's route
// projection (zdrive_agent_obs.c navi lookahead). Returns the number written; the route may end earlier.
int znavi_lookahead_points(const Drive *env, const Entity *agent, int n_waypoints, float distance, float *xs,
                           float *ys) {
    const ZNavigation *zn = env->znavigation;
    if (zn == NULL || n_waypoints <= 0)
        return 0;
    float spacing = distance / (float)n_waypoints;
    ZNaviWalk walk =
        znavi_walk_init(agent->znavi_entity_idx, agent->znavi_point_idx, agent->znavi_proj_x, agent->znavi_proj_y);
    int w = 0;
    for (int step = 0; step < ZNAVI_LOOKAHEAD_MAX_WALK_STEPS && w < n_waypoints && znavi_walk_step(zn, &walk); step++) {
        while (w < n_waypoints && walk.accum >= (float)(w + 1) * spacing) {
            znavi_walk_interpolate(&walk, (float)(w + 1) * spacing, &xs[w], &ys[w]);
            w++;
        }
    }
    return w;
}

// zdrive's lane-feasibility-time ego features and navi lookahead observation block.
void znavi_write_observations(const Drive *env, const Entity *agent, float *ego_navi_obs, float *waypoint_obs) {
    float feasibility_norm[3] = {0.0f, 0.0f, 0.0f};
    if (agent->navi_speed_limit > 0.0F) {
        float feasibility_m[3] = {agent->navi_feasibility_left_m, agent->navi_feasibility_center_m,
                                  agent->navi_feasibility_right_m};
        for (int k = 0; k < 3; k++) {
            float feasibility_time = feasibility_m[k] / agent->navi_speed_limit;
            feasibility_norm[k] = ZB_CLAMP(feasibility_time / ZNAVI_MAX_LANE_FEASIBILITY_TIME, 0.0f, 1.0f);
        }
    }
    for (int k = 0; k < 3; k++)
        ego_navi_obs[k] = feasibility_norm[k];

    int n_waypoints = env->navigation_waypoints;
    float xs[ZNAVI_LOOKAHEAD_WAYPOINTS_MAX];
    float ys[ZNAVI_LOOKAHEAD_WAYPOINTS_MAX];
    int count = znavi_lookahead_points(env, agent, n_waypoints, env->navigation_lookahead_distance, xs, ys);
    for (int w = 0; w < n_waypoints; w++) {
        float forward = 0.0f, left = 0.0f;
        if (w < count) {
            float dx = xs[w] - agent->x;
            float dy = ys[w] - agent->y;
            forward = (dx * agent->heading_x + dy * agent->heading_y) / env->navigation_lookahead_norm;
            left = (-dx * agent->heading_y + dy * agent->heading_x) / env->navigation_lookahead_norm;
        }
        waypoint_obs[w * ZNAVI_WAYPOINT_FEATURES] = forward;
        waypoint_obs[w * ZNAVI_WAYPOINT_FEATURES + 1] = left;
    }
}

void znavi_reset_navigation_state(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int goal = agent->zdrive_goal_index;
    agent->navi_goal_idx = goal >= 0 && goal < env->znavigation->num_goals ? goal : -1;
    agent->znavi_entity_idx = -1;
    agent->znavi_point_idx = -1;
    agent->znavi_proj_x = agent->x;
    agent->znavi_proj_y = agent->y;
    agent->navi_speed_limit = 0.0f;
}

// Refreshes the zdrive navigation state. The observation state follows zdrive every step; the reward state
// (distance to goal, off-route) keeps its last value while no lane is within reach, like the legacy backend.
void znavi_update_navigation_state(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->x == INVALID_POSITION)
        return;
    float distance_to_goal;
    int lane = znavi_localize(env, agent, &distance_to_goal);
    agent->current_lane_idx = lane;
    if (lane == -1)
        return;
    int previous_lane = agent->navi_lane_idx;
    agent->navi_lane_idx = lane;
    if (agent->navi_goal_idx == -1)
        return;
    int off_route = distance_to_goal >= (ZNAVI_UNREACHABLE_DISTANCE - 0.01F);
    agent->navi_off_route = off_route;
    if (off_route)
        agent->navi_off_route_sticky = 1;
    float distance_m = off_route ? NAVI_UNREACHABLE_M : distance_to_goal;
    agent->prev_navi_distance_to_goal_m = previous_lane == -1 ? distance_m : agent->navi_distance_to_goal_m;
    if (previous_lane == -1)
        agent->closest_navi_distance_to_goal_m = distance_m;
    agent->navi_distance_to_goal_m = distance_m;
    agent->navi_position_x = agent->x;
    agent->navi_position_y = agent->y;
}
