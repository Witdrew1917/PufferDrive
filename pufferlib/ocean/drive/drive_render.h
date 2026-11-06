// Raylib rendering for the Drive environment: headless video recording
// (ffmpeg / EGL) and the interactive pop-up eval viewer. Included from drive.h.
#pragma once

typedef struct Client Client;

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
    int recorder_pipefd[2];
    pid_t recorder_pid;
    pid_t xvfb_pid;
    int xvfb_display_num;
    int render_mode;
    int egl_mode;
    Mesh road_tri_mesh;
    Material road_material;
    float *road_line_verts;
    unsigned char *road_line_colors;
    int road_line_count;
    int road_cache_valid;
#ifdef DRIVE_HAS_EGL
    unsigned int pbo[2];
#endif
    int pbo_index;
    int pbo_frame_count;
};

static int g_glfw_ready = 0;
#ifdef DRIVE_HAS_EGL
static int g_egl_available = -1;
#endif
static pid_t g_xvfb_pid = 0;
static int g_xvfb_display_num = 0;

void stop_recorder(Client *client) {
    if (client->recorder_pid > 0) {
        close(client->recorder_pipefd[1]);
        waitpid(client->recorder_pid, NULL, 0);
        client->recorder_pid = 0;
    }
}

void start_recorder(Client *client, const char *scenario_id, int width, int height) {
    stop_recorder(client);

    if (pipe(client->recorder_pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe for recorder\n");
        return;
    }

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dx%d", width, height);

    char filename[256];
    snprintf(filename, sizeof(filename), "%s.mp4", scenario_id);

    client->recorder_pid = fork();
    if (client->recorder_pid == -1) {
        fprintf(stderr, "Failed to fork recorder\n");
        close(client->recorder_pipefd[0]);
        close(client->recorder_pipefd[1]);
        return;
    }

    if (client->recorder_pid == 0) {
        close(client->recorder_pipefd[1]);
        dup2(client->recorder_pipefd[0], STDIN_FILENO);
        close(client->recorder_pipefd[0]);
        for (int fd = 3; fd < 256; fd++)
            close(fd);
        execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size_str, "-r", "30", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-loglevel", "error",
               filename, NULL);
        fprintf(stderr, "execlp ffmpeg failed\n");
        _exit(1);
    }
    close(client->recorder_pipefd[0]);

#ifdef DRIVE_HAS_EGL
    int pipe_sz = fcntl(client->recorder_pipefd[1], F_SETPIPE_SZ, 4 * 1024 * 1024);
    if (pipe_sz > 0) {
        fprintf(stderr, "[drive] Pipe buffer set to %d bytes\n", pipe_sz);
    }
#endif
}

Client *make_client(Drive *env) {

    Client *client = (Client *)calloc(1, sizeof(Client));
    client->render_mode = env->render_mode;

    if (env->render_mode == RENDER_WINDOW) {
        // The pop-up window outlives individual clients (map resampling closes and recreates the client),
        // so GUI state such as camera, toggles and window size persists between scenarios.
        if (!IsWindowReady()) {
            SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
            SetTraceLogLevel(LOG_WARNING);
            InitWindow(1280, 720, "PufferDrive");
            if (IsWindowReady()) {
                SetWindowMinSize(640, 360);
                SetExitKey(KEY_NULL);
                SetTargetFPS(30);
            }
        }
        client->width = (float)GetScreenWidth();
        client->height = (float)GetScreenHeight();
    } else { // Headless rendering
        // Clean-eval videos use a fixed 16:9 frame. This also matches the
        // persistent EGL pbuffer model: every env records into the same size.
        client->width = 1920;
        client->height = 1080;

        if (!g_glfw_ready) {
            if (getenv("DISPLAY") == NULL) {
                g_xvfb_display_num = 100 + (getpid() % 900);
                char display_str[16];
                char lock_file[32];
                char socket_file[32];
                snprintf(display_str, sizeof(display_str), ":%d", g_xvfb_display_num);
                snprintf(lock_file, sizeof(lock_file), "/tmp/.X%d-lock", g_xvfb_display_num);
                snprintf(socket_file, sizeof(socket_file), "/tmp/.X11-unix/X%d", g_xvfb_display_num);

                FILE *f = fopen(lock_file, "r");
                if (f) {
                    pid_t pid = -1;
                    fscanf(f, "%d", &pid);
                    fclose(f);
                    if (pid > 0 && kill(pid, 0) != 0) {
                        unlink(lock_file);
                        unlink(socket_file);
                    }
                }

                g_xvfb_pid = fork();
                if (g_xvfb_pid == 0) {
                    close(STDOUT_FILENO);
                    close(STDERR_FILENO);
                    execlp("Xvfb", "Xvfb", display_str, "-screen", "0", "1280x720x24", "+extension", "GLX", "-ac",
                           "-noreset", NULL);
                    _exit(1);
                }
                setenv("DISPLAY", display_str, 1);
                for (int i = 0; i < 20 && access(lock_file, F_OK) != 0; i++)
                    usleep(100000);
                usleep(200000);

                client->xvfb_pid = g_xvfb_pid;
                client->xvfb_display_num = g_xvfb_display_num;
            }

            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            SetTargetFPS(6000);
            SetTraceLogLevel(LOG_WARNING);
            InitWindow(client->width, client->height, "PufferDrive");
            g_glfw_ready = 1;
        }
    }

    if (!IsWindowReady()) {
        fprintf(stderr, "WARNING: Failed to initialize render window. Rendering disabled.\n");
        free(client);
        return NULL;
    }

#ifdef DRIVE_HAS_EGL
    if (env->render_mode == RENDER_HEADLESS && g_egl_available != 0) {
        if (!g_egl_ctx.active) {
            if (egl_headless_init((int)client->width, (int)client->height) && egl_switch_to_gpu()) {
                rlglInit((int)client->width, (int)client->height);
                rlViewport(0, 0, (int)client->width, (int)client->height);
                rlEnableDepthTest();
                g_egl_available = 1;
            }
            if (g_egl_available < 0) {
                g_egl_available = 0;
                fprintf(stderr, "[drive] EGL GPU unavailable, using Xvfb/Mesa rendering\n");
            }
        } else {
            egl_headless_resize((int)client->width, (int)client->height);
            rlViewport(0, 0, (int)client->width, (int)client->height);
        }
        if (g_egl_available == 1)
            client->egl_mode = 1;
    }
#endif

    if (env->render_mode == RENDER_WINDOW) {
        return client; // Pop-up draws agents as boxes, no models needed
    }

    // Load assets
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }

    return client;
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (obs_only == 0) {
        return;
    }

    int ego_dim = ego_feature_count(env);
    int max_obs = observation_size(env);
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // Navigation waypoints
    const float *waypoint_obs = &agent_obs[max_obs - waypoint_feature_count(env)];
    for (int k = 0; env->navigation_enabled && k < env->navigation_waypoints; k++) {
        const float *waypoint = &waypoint_obs[k * NAVIGATION_WAYPOINT_FEATURES];
        if (waypoint[2] == 0.0f) {
            break;
        }
        float waypoint_x = waypoint[0] * 50;
        float waypoint_y = waypoint[1] * 50;
        if (mode == 1) {
            float local_x = waypoint_x;
            waypoint_x = px + (local_x * heading_self_x - waypoint_y * heading_self_y);
            waypoint_y = py + (local_x * heading_self_y + waypoint_y * heading_self_x);
        }
        DrawSphere((Vector3){waypoint_x, waypoint_y, Z_AGENT_DETAILS}, NAVI_RENDER_WAYPOINT_RADIUS_M, PUFF_CYAN);
    }
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;

    int agent_type = env->entities[active_idx].type;
    Color goal_color = LIGHTBLUE;
    if (agent_type == PEDESTRIAN)
        goal_color = LIGHT_ORANGE;
    else if (agent_type == CYCLIST)
        goal_color = LIGHT_PURPLE;

    if (mode == 0) { // agent-relative coordinates
        DrawSphere((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(goal_color, 0.3f));
    }

    if (mode == 1) { // world coordinates

        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1},
                     90.0f, Fade(goal_color, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, Z_AGENT_DETAILS}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, Z_AGENT_DETAILS}, (Vector3){partner_x, partner_y, Z_AGENT_DETAILS}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, Z_AGENT_DETAILS}, (Vector3){arrow_x, arrow_y, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, Z_AGENT_DETAILS},
                       (Vector3){arrow_x_world, arrow_y_world, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end1, arrow_y_end1, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end2, arrow_y_end2, 0.0}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 0.0}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1); // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.0f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void build_road_cache(Drive *env, Client *client) {
    int tri_count = 0;
    int line_count = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE)
            continue;
        int segs = road->array_size - 1;
        if (segs <= 0)
            continue;
        if (road->type == ROAD_EDGE)
            tri_count += segs * 12;
        else if (road->type == ROAD_LANE || road->type == ROAD_LINE)
            line_count += segs;
    }

    int num_tri_verts = tri_count * 3;
    float *tri_verts = num_tri_verts > 0 ? (float *)RL_CALLOC(num_tri_verts * 3, sizeof(float)) : NULL;
    unsigned char *tri_colors =
        num_tri_verts > 0 ? (unsigned char *)RL_CALLOC(num_tri_verts * 4, sizeof(unsigned char)) : NULL;

    free(client->road_line_verts);
    free(client->road_line_colors);
    client->road_line_verts = line_count > 0 ? (float *)malloc(line_count * 2 * 3 * sizeof(float)) : NULL;
    client->road_line_colors = line_count > 0 ? (unsigned char *)malloc(line_count * 2 * 4) : NULL;
    client->road_line_count = 0;
    int actual_tri_count = 0;

#define PUSH_TRI(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, cr, cg, cb, ca)                                          \
    do {                                                                                                               \
        int _ti = actual_tri_count * 9;                                                                                \
        int _ci = actual_tri_count * 12;                                                                               \
        tri_verts[_ti + 0] = vx1;                                                                                      \
        tri_verts[_ti + 1] = vy1;                                                                                      \
        tri_verts[_ti + 2] = vz1;                                                                                      \
        tri_verts[_ti + 3] = vx2;                                                                                      \
        tri_verts[_ti + 4] = vy2;                                                                                      \
        tri_verts[_ti + 5] = vz2;                                                                                      \
        tri_verts[_ti + 6] = vx3;                                                                                      \
        tri_verts[_ti + 7] = vy3;                                                                                      \
        tri_verts[_ti + 8] = vz3;                                                                                      \
        for (int _v = 0; _v < 3; _v++) {                                                                               \
            tri_colors[_ci + _v * 4 + 0] = cr;                                                                         \
            tri_colors[_ci + _v * 4 + 1] = cg;                                                                         \
            tri_colors[_ci + _v * 4 + 2] = cb;                                                                         \
            tri_colors[_ci + _v * 4 + 3] = ca;                                                                         \
        }                                                                                                              \
        actual_tri_count++;                                                                                            \
    } while (0)

#define PUSH_LINE(vx1, vy1, vz1, vx2, vy2, vz2, cr, cg, cb, ca)                                                        \
    do {                                                                                                               \
        int _li = client->road_line_count * 6;                                                                         \
        int _ci = client->road_line_count * 8;                                                                         \
        client->road_line_verts[_li + 0] = vx1;                                                                        \
        client->road_line_verts[_li + 1] = vy1;                                                                        \
        client->road_line_verts[_li + 2] = vz1;                                                                        \
        client->road_line_verts[_li + 3] = vx2;                                                                        \
        client->road_line_verts[_li + 4] = vy2;                                                                        \
        client->road_line_verts[_li + 5] = vz2;                                                                        \
        for (int _v = 0; _v < 2; _v++) {                                                                               \
            client->road_line_colors[_ci + _v * 4 + 0] = cr;                                                           \
            client->road_line_colors[_ci + _v * 4 + 1] = cg;                                                           \
            client->road_line_colors[_ci + _v * 4 + 2] = cb;                                                           \
            client->road_line_colors[_ci + _v * 4 + 3] = ca;                                                           \
        }                                                                                                              \
        client->road_line_count++;                                                                                     \
    } while (0)

    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE)
            continue;
        for (int j = 0; j < road->array_size - 1; j++) {
            float sx = road->traj_x[j];
            float sy = road->traj_y[j];
            float ex = road->traj_x[j + 1];
            float ey = road->traj_y[j + 1];
            if (sx == INVALID_POSITION || sy == INVALID_POSITION || ex == INVALID_POSITION || ey == INVALID_POSITION)
                continue;
            float sz = Z_ROAD_MARKINGS;
            float ez = Z_ROAD_MARKINGS;

            if (road->type == ROAD_EDGE) {
                sz = Z_ROAD_SURFACE;
                ez = Z_ROAD_SURFACE;
                float curb_height = 0.5f;
                float curb_width = 0.3f;
                float dx = ex - sx;
                float dy = ey - sy;
                float len = sqrtf(dx * dx + dy * dy);
                if (len < 1e-6f)
                    continue;
                float nx = -dy / len;
                float ny = dx / len;
                float hw = curb_width * 0.5f;

                float b1x = sx - nx * hw, b1y = sy - ny * hw, b2x = sx + nx * hw, b2y = sy + ny * hw;
                float b3x = ex + nx * hw, b3y = ey + ny * hw, b4x = ex - nx * hw, b4y = ey - ny * hw;
                float t1z = sz + curb_height, t2z = sz + curb_height;
                float t3z = ez + curb_height, t4z = ez + curb_height;

                PUSH_TRI(b1x, b1y, sz, b2x, b2y, sz, b3x, b3y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, sz, b3x, b3y, ez, b4x, b4y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, t1z, b3x, b3y, t3z, b2x, b2y, t2z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, t1z, b4x, b4y, t4z, b3x, b3y, t3z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, sz, b1x, b1y, t1z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b1x, b1y, t1z, b2x, b2y, t2z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, sz, b2x, b2y, t2z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, t2z, b3x, b3y, t3z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, ez, b3x, b3y, t3z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, t3z, b4x, b4y, t4z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, ez, b4x, b4y, t4z, b1x, b1y, sz, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, t4z, b1x, b1y, t1z, b1x, b1y, sz, 180, 180, 180, 255);
            } else if (road->type == ROAD_LANE && client->road_line_verts) {
                Color c = Fade(SOFT_YELLOW, 0.25f);
                PUSH_LINE(sx, sy, sz, ex, ey, ez, c.r, c.g, c.b, c.a);
            } else if (road->type == ROAD_LINE && client->road_line_verts) {
                PUSH_LINE(sx, sy, sz, ex, ey, ez, 255, 255, 255, 255);
            }
        }
    }

#undef PUSH_TRI
#undef PUSH_LINE

    if (actual_tri_count > 0) {
        Mesh mesh = {0};
        mesh.vertexCount = actual_tri_count * 3;
        mesh.triangleCount = actual_tri_count;
        mesh.vertices = tri_verts;
        mesh.colors = tri_colors;
        UploadMesh(&mesh, false);
        client->road_tri_mesh = mesh;
        client->road_material = LoadMaterialDefault();
    } else {
        RL_FREE(tri_verts);
        RL_FREE(tri_colors);
    }

    client->road_cache_valid = 1;
    fprintf(stderr, "[drive] Road cache: %d triangles, %d lines\n", actual_tri_count, client->road_line_count);
}

void draw_road_cached(Client *client) {
    if (client->road_tri_mesh.vertexCount > 0)
        DrawMesh(client->road_tri_mesh, client->road_material, MatrixIdentity());

    if (client->road_line_count > 0) {
        rlSetLineWidth(2.0f);
        rlBegin(RL_LINES);
        int nv = client->road_line_count * 2;
        for (int i = 0; i < nv; i++) {
            rlColor4ub(client->road_line_colors[i * 4], client->road_line_colors[i * 4 + 1],
                       client->road_line_colors[i * 4 + 2], client->road_line_colors[i * 4 + 3]);
            rlVertex3f(client->road_line_verts[i * 3], client->road_line_verts[i * 3 + 1],
                       client->road_line_verts[i * 3 + 2]);
        }
        rlEnd();
    }
}

void draw_inferred_trajectory(Drive *env, int agent_array_idx) {
    int idx = env->active_agent_indices[agent_array_idx];
    Entity *e = &env->entities[idx];
    if (!e->inferred_traj_x || !e->traj_valid)
        return;

    float zg = Z_ROAD_MARKINGS, zs = Z_ROAD_MARKINGS + 0.01f;

    for (int t = 0; t < e->array_size - 1; t++) {
        if (!e->traj_valid[t])
            continue;

        // Ground truth (green)
        DrawSphere((Vector3){e->traj_x[t], e->traj_y[t], zg}, 0.15f, LIGHTGREEN);
        if (e->traj_valid[t + 1])
            DrawLine3D((Vector3){e->traj_x[t], e->traj_y[t], zg}, (Vector3){e->traj_x[t + 1], e->traj_y[t + 1], zg},
                       Fade(LIGHTGREEN, 0.6f));

        // Inferred (purple)
        DrawSphere((Vector3){e->inferred_traj_x[t], e->inferred_traj_y[t], zs}, 0.2f, LIGHT_PURPLE);
        DrawLine3D((Vector3){e->inferred_traj_x[t], e->inferred_traj_y[t], zs},
                   (Vector3){e->inferred_traj_x[t + 1], e->inferred_traj_y[t + 1], zs}, Fade(LIGHT_PURPLE, 0.6f));

        // Error shading (red)
        if (e->traj_valid[t + 1]) {
            Vector3 a = {e->traj_x[t], e->traj_y[t], zg};
            Vector3 b = {e->inferred_traj_x[t], e->inferred_traj_y[t], zg};
            Vector3 c = {e->traj_x[t + 1], e->traj_y[t + 1], zg};
            Vector3 d = {e->inferred_traj_x[t + 1], e->inferred_traj_y[t + 1], zg};
            Color err = Fade(PUFF_RED, 0.15f);
            DrawTriangle3D(a, b, c, err);
            DrawTriangle3D(c, b, d, err);
            DrawTriangle3D(c, b, a, err);
            DrawTriangle3D(d, b, c, err);
        }
    }
}

void draw_lane_polyline(const Entity *lane, float z, Color color) {
    for (int point_idx = 0; point_idx + 1 < lane->array_size; point_idx++) {
        DrawLine3D((Vector3){lane->traj_x[point_idx], lane->traj_y[point_idx], z},
                   (Vector3){lane->traj_x[point_idx + 1], lane->traj_y[point_idx + 1], z}, color);
    }
}

// Lanes coloured by centre feasibility toward the selected agent's goal (red→green), grey if unreachable.
void draw_navigation_overlay(Drive *env) {
    const Navigation *navigation = env->navigation;
    const Entity *agent = &env->entities[env->active_agent_indices[env->human_agent_idx]];
    if (agent->navi_goal_idx == -1) {
        return;
    }
    int goal_table_offset = agent->navi_goal_idx * navigation->num_lanes;
    rlSetLineWidth(NAVI_RENDER_LANE_WIDTH_PX);
    for (int navi_lane_idx = 0; navi_lane_idx < navigation->num_lanes; navi_lane_idx++) {
        const Entity *lane = &env->entities[navigation->lane_entity_idx[navi_lane_idx]];
        Color lane_color = Fade(GRAY, NAVI_RENDER_UNREACHABLE_ALPHA);
        if (navigation->distance_to_goal_m[goal_table_offset + navi_lane_idx] < NAVI_UNREACHABLE_M) {
            float feasibility_ratio = fminf(navigation->feasibility_center_m[goal_table_offset + navi_lane_idx] /
                                                NAVI_OBS_FEASIBILITY_HORIZON_M,
                                            1.0f);
            lane_color = (Color){(unsigned char)(255.0f * (1.0f - feasibility_ratio)),
                                 (unsigned char)(255.0f * feasibility_ratio), 0, 255};
        }
        draw_lane_polyline(lane, Z_ROAD_MARKINGS + NAVI_RENDER_Z_OFFSET, lane_color);
    }
    if (agent->current_lane_idx != -1) {
        rlSetLineWidth(NAVI_RENDER_ROUTE_WIDTH_PX);
        int navi_lane_idx = navigation->entity_to_navi_lane[agent->current_lane_idx];
        for (int hop_count = 0; hop_count < navigation->num_lanes && navi_lane_idx != -1; hop_count++) {
            draw_lane_polyline(&env->entities[navigation->lane_entity_idx[navi_lane_idx]],
                               Z_ROAD_MARKINGS + 2.0f * NAVI_RENDER_Z_OFFSET, PUFF_CYAN);
            navi_lane_idx = navigation->next_lane[goal_table_offset + navi_lane_idx];
        }
    }
    rlSetLineWidth(1.0f);
    float waypoint_x[NAVI_MAX_WAYPOINTS];
    float waypoint_y[NAVI_MAX_WAYPOINTS];
    int waypoint_count = navigation_route_waypoints(env, agent, env->navigation_waypoints,
                                                    env->navigation_waypoint_spacing, waypoint_x, waypoint_y);
    for (int k = 0; k < waypoint_count; k++) {
        DrawSphere((Vector3){waypoint_x[k], waypoint_y[k], Z_AGENT_DETAILS}, NAVI_RENDER_WAYPOINT_RADIUS_M, WHITE);
    }
    DrawSphere(
        (Vector3){navigation->goal_x[agent->navi_goal_idx], navigation->goal_y[agent->navi_goal_idx], Z_AGENT_DETAILS},
        NAVI_RENDER_GOAL_RADIUS_M, PUFF_CYAN);
}

// Logged (and inferred) expert trajectories for the replay / inferred-expert verification modes.
// Returns 1 when inferred trajectories were drawn (so ADE labels should be shown).
int draw_verification_trajectories(Drive *env) {
    if (env->control_mode != CONTROL_REPLAY_LOGS && env->control_mode != CONTROL_INFERRED_EXPERT_ACTIONS) {
        return 0;
    }
    Color traj_color = (env->control_mode == CONTROL_REPLAY_LOGS) ? EXPERT_REPLAY : LIGHT_PURPLE;
    for (int i = 0; i < env->active_agent_count; i++) {
        int idx = env->active_agent_indices[i];
        Entity *e = &env->entities[idx];
        for (int t = env->init_steps; t < e->array_size - 1; t++) {
            if (e->traj_valid[t] && e->traj_valid[t + 1]) {
                DrawLine3D((Vector3){e->traj_x[t], e->traj_y[t], Z_ROAD_MARKINGS},
                           (Vector3){e->traj_x[t + 1], e->traj_y[t + 1], Z_ROAD_MARKINGS}, Fade(traj_color, 0.4f));
            }
            if (e->traj_valid[t]) {
                DrawSphere((Vector3){e->traj_x[t], e->traj_y[t], Z_ROAD_MARKINGS}, 0.15f, traj_color);
            }
        }
    }

    if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
        for (int i = 0; i < env->active_agent_count; i++) {
            draw_inferred_trajectory(env, i);
        }
        return 1;
    }
    return 0;
}

// Per-agent ADE labels projected to screen (2D, call outside BeginMode3D).
void draw_ade_labels(Drive *env, Camera3D camera, int width, int height) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int idx = env->active_agent_indices[i];
        Entity *e = &env->entities[idx];
        if (e->inferred_ade < 0)
            continue;
        int mid = e->array_size / 2;
        Vector2 sp = GetWorldToScreenEx((Vector3){e->traj_x[mid], e->traj_y[mid], Z_AGENT_DETAILS}, camera, width,
                                        height);
        int sx = (int)sp.x, sy = (int)sp.y;
        if (sx < 0 || sx > width || sy < 0 || sy > height)
            continue;
        char buf[32];
        // Check if unfit (same criteria as c_collect_expert_data)
        int valid_count = 0;
        int is_static = 1;
        if (e->traj_valid) {
            for (int t = 0; t < e->array_size; t++) {
                if (!e->traj_valid[t])
                    continue;
                valid_count++;
                if (valid_count > 1 && (e->traj_x[t] != e->traj_x[0] || e->traj_y[t] != e->traj_y[0]))
                    is_static = 0;
            }
        }
        int unfit = (e->inferred_ade < 0.0f || e->inferred_ade > ADE_THRESHOLD || valid_count < 3 || is_static);
        Color label_color = unfit ? PUFF_RED : PUFF_WHITE;
        snprintf(buf, sizeof(buf), "ADE:%.2fm", e->inferred_ade);
        DrawText(buf, sx - MeasureText(buf, 16) / 2, sy, 16, label_color);
    }
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

    int has_inferred = 0;
    if (show_grid) {
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 0.0f}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, Fade(PUFF_BACKGROUND2, 0.3f));
            }
        }
    }

    // Draw expert trajectories for verification modes
    has_inferred = draw_verification_trajectories(env);

    if (obs_only == 0 && client->road_cache_valid) {
        draw_road_cached(client);
    }

    // Draw entities
    for (int i = 0; i < env->num_entities; i++) {
        // Draw objects
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {

            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    agent_index = j;
                    if (!env->entities[agent_index].removed) {
                        is_active_agent = true;
                        break;
                    }
                }
            }

            for (int j = 0; j < env->expert_static_agent_count; j++) {
                if (env->expert_static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if (!is_active_agent && !is_static_agent) {
                continue;
            }

            if (env->entities[i].removed) {
                continue;
            }

            Vector3 position;
            float heading;
            position = (Vector3){env->entities[i].x, env->entities[i].y, Z_AGENTS};
            heading = env->entities[i].heading;

            // Create size vector
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

            // Save current transform
            if (mode == 1) {
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                              position.y + (half_len * sin_heading + half_width * cos_heading), position.z},
                    (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                              position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                              position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                              position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

                };

                if (agent_index == env->human_agent_idx &&
                    !env->entities[env->active_agent_indices[agent_index]].current_goal_reached) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if (obs_only && agent_index != env->human_agent_idx) {
                    continue;
                }

                // Pick agent color (reference scheme)
                Color agent_color = GRAY;
                if (is_expert) {
                    if (env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST)
                        agent_color = EXPERT_REPLAY_SMALL;
                    else
                        agent_color = EXPERT_REPLAY;
                }
                if (is_active_agent) {
                    if (env->control_mode == CONTROL_REPLAY_LOGS)
                        agent_color = EXPERT_REPLAY;
                    else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS)
                        agent_color = LIGHT_PURPLE;
                    else if (env->entities[i].type == PEDESTRIAN)
                        agent_color = LIGHT_ORANGE;
                    else if (env->entities[i].type == CYCLIST)
                        agent_color = LIGHT_PURPLE;
                    else
                        agent_color = BLUE;
                }
                if (is_active_agent && env->entities[i].collision_state > 0)
                    agent_color = RED;
                if (is_active_agent && env->entities[i].offroad_state > 0)
                    agent_color = LIGHTYELLOW;

                // Filled cube + wireframe outline at the same color
                rlPushMatrix();
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
                DrawCube((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, Fade(agent_color, 0.5f));
                DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, agent_color);
                rlPopMatrix();

                // Heading arrow scaled by speed
                float agent_speed =
                    sqrtf(env->entities[i].vx * env->entities[i].vx + env->entities[i].vy * env->entities[i].vy);
                float speed_frac = fminf(agent_speed / MAX_SPEED, 1.0f);
                float arrow_len = half_len * 1.5f + speed_frac * 15.0f;
                float shaft_w = 0.4f;
                float head_len = 1.0f;
                float head_w = 1.4f;
                float az = position.z + 0.51f;

                float shaft_end = arrow_len - head_len;
                Vector3 s_start_l = {position.x - sin_heading * shaft_w * 0.5f,
                                     position.y + cos_heading * shaft_w * 0.5f, az};
                Vector3 s_start_r = {position.x + sin_heading * shaft_w * 0.5f,
                                     position.y - cos_heading * shaft_w * 0.5f, az};
                Vector3 s_end_l = {position.x + cos_heading * shaft_end - sin_heading * shaft_w * 0.5f,
                                   position.y + sin_heading * shaft_end + cos_heading * shaft_w * 0.5f, az};
                Vector3 s_end_r = {position.x + cos_heading * shaft_end + sin_heading * shaft_w * 0.5f,
                                   position.y + sin_heading * shaft_end - cos_heading * shaft_w * 0.5f, az};
                DrawTriangle3D(s_start_l, s_start_r, s_end_r, agent_color);
                DrawTriangle3D(s_start_l, s_end_r, s_end_l, agent_color);
                DrawTriangle3D(s_end_r, s_start_r, s_start_l, agent_color);
                DrawTriangle3D(s_end_l, s_end_r, s_start_l, agent_color);

                Vector3 tip = {position.x + cos_heading * arrow_len, position.y + sin_heading * arrow_len, az};
                Vector3 head_l = {position.x + cos_heading * shaft_end - sin_heading * head_w * 0.5f,
                                  position.y + sin_heading * shaft_end + cos_heading * head_w * 0.5f, az};
                Vector3 head_r = {position.x + cos_heading * shaft_end + sin_heading * head_w * 0.5f,
                                  position.y + sin_heading * shaft_end - cos_heading * head_w * 0.5f, az};
                DrawTriangle3D(head_l, head_r, tip, agent_color);
                DrawTriangle3D(tip, head_r, head_l, agent_color);

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model (skip index 0)
                Model car_model = client->cars[(i % 5) + 1]; // Cycles through indices 1-5

                if (env->control_mode == CONTROL_REPLAY_LOGS || env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
                    car_model = client->cars[1]; // White car for verification modes
                } else if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0];
                } else if (is_active_agent) {
                    car_model = client->cars[(i % 5) + 1];
                    if (env->entities[i].collision_state > 0) {
                        car_model = client->cars[0];
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->entities[env->active_agent_indices[agent_index]].current_goal_reached ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};

                if (env->entities[i].type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (env->entities[i].type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY;
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1)
                        wire_color = EXPERT_REPLAY;
                    if (is_active_agent) {
                        if (env->control_mode == CONTROL_REPLAY_LOGS)
                            wire_color = EXPERT_REPLAY;
                        else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS)
                            wire_color = LIGHT_PURPLE;
                        else
                            wire_color = BLUE; // Policy-controlled
                    }
                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = RED;

                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = LIGHTYELLOW;

                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // Draw goal position for active agents
            if (!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if (obs_only == 0 && env->control_mode != CONTROL_REPLAY_LOGS &&
                env->control_mode != CONTROL_INFERRED_EXPERT_ACTIONS) {
                // Match clean-eval: goal markers are bright lime, with a faint
                // radius circle on the road surface.
                Color goal_color = LIME;
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_AGENTS},
                           1.5f, goal_color);
                DrawCircle3D(
                    (Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_ROAD_MARKINGS},
                    env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(goal_color, 0.3f));
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 || env->entities[i].type >= 7) {
            continue;
        }
        if (client->road_cache_valid) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], Z_ROAD_MARKINGS};
            Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], Z_ROAD_MARKINGS};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = WHITE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = Fade(WHITE, 0.7f);
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;

            if (obs_only == 0) {
                if (env->entities[i].type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y);
                } else if (env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    if (env->navigation != NULL && obs_only == 0) {
        draw_navigation_overlay(env);
    }

    EndMode3D();

    // Draw per-agent ADE labels projected to screen for interactive verify mode
    if (has_inferred) {
        draw_ade_labels(env, client->camera, (int)client->width, (int)client->height);
    }
}

void draw_thick_line_3d(Vector3 a, Vector3 b, float thickness, Color color) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-6f)
        return;
    float nx = -dy / len * thickness * 0.5f;
    float ny = dx / len * thickness * 0.5f;

    Vector3 p0 = {a.x + nx, a.y + ny, a.z};
    Vector3 p1 = {a.x - nx, a.y - ny, a.z};
    Vector3 p2 = {b.x - nx, b.y - ny, b.z};
    Vector3 p3 = {b.x + nx, b.y + ny, b.z};

    DrawTriangle3D(p0, p1, p2, color);
    DrawTriangle3D(p0, p2, p3, color);
    DrawTriangle3D(p2, p1, p0, color);
    DrawTriangle3D(p3, p2, p0, color);
}

void draw_sensor_noise_view(Drive *env, Client *client) {
    int sdc_array_idx = 0;
    if (env->sdc_track_index >= 0) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (env->active_agent_indices[i] == env->sdc_track_index) {
                sdc_array_idx = i;
                break;
            }
        }
    }

    int ego_idx = env->active_agent_indices[sdc_array_idx];
    Entity *ego = &env->entities[ego_idx];
    if (ego->removed == 1)
        return;

    float cos_h = ego->heading_x;
    float sin_h = ego->heading_y;
    int num_samples = 5;

    for (int j = 0; j < MAX_AGENTS; j++) {
        int index = -1;
        if (j < env->active_agent_count)
            index = env->active_agent_indices[j];
        else if (j < env->num_actors && env->static_agent_count > 0)
            index = env->static_agent_indices[j - env->active_agent_count];
        if (index == -1 || index == ego_idx)
            continue;

        Entity *other = &env->entities[index];
        if (other->type > CYCLIST)
            continue;
        if (other->x == INVALID_POSITION)
            continue;
        if (other->removed == 1)
            continue;

        float dx = other->x - ego->x;
        float dy = other->y - ego->y;
        if ((dx * dx + dy * dy) > 2500.0f)
            continue;

        float gt_cos = other->heading_x;
        float gt_sin = other->heading_y;
        float gt_hl = other->length * 0.5f;
        float gt_hw = other->width * 0.5f;

        // Ground truth box (green)
        Vector3 gt[4] = {
            {other->x + gt_hl * gt_cos - gt_hw * gt_sin, other->y + gt_hl * gt_sin + gt_hw * gt_cos, Z_AGENTS},
            {other->x + gt_hl * gt_cos + gt_hw * gt_sin, other->y + gt_hl * gt_sin - gt_hw * gt_cos, Z_AGENTS},
            {other->x - gt_hl * gt_cos + gt_hw * gt_sin, other->y - gt_hl * gt_sin - gt_hw * gt_cos, Z_AGENTS},
            {other->x - gt_hl * gt_cos - gt_hw * gt_sin, other->y - gt_hl * gt_sin + gt_hw * gt_cos, Z_AGENTS},
        };
        for (int k = 0; k < 4; k++)
            draw_thick_line_3d(gt[k], gt[(k + 1) % 4], 0.45f, LIGHTGREEN);

        // Noisy perception samples (purple)
        for (int s = 0; s < num_samples; s++) {
            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;
            float noisy_rx = rel_x + gaussian_noise(env->obs_partner_noise_pos) / 0.02f;
            float noisy_ry = rel_y + gaussian_noise(env->obs_partner_noise_pos) / 0.02f;
            float px = ego->x + noisy_rx * cos_h - noisy_ry * sin_h;
            float py = ego->y + noisy_rx * sin_h + noisy_ry * cos_h;

            float p_hw =
                (other->width / MAX_VEH_WIDTH + gaussian_noise(env->obs_partner_noise_pos)) * MAX_VEH_WIDTH * 0.5f;
            float p_hl =
                (other->length / MAX_VEH_LEN + gaussian_noise(env->obs_partner_noise_pos)) * MAX_VEH_LEN * 0.5f;

            Vector3 pc[4] = {
                {px + p_hl * gt_cos - p_hw * gt_sin, py + p_hl * gt_sin + p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px + p_hl * gt_cos + p_hw * gt_sin, py + p_hl * gt_sin - p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px - p_hl * gt_cos + p_hw * gt_sin, py - p_hl * gt_sin - p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px - p_hl * gt_cos - p_hw * gt_sin, py - p_hl * gt_sin + p_hw * gt_cos, Z_AGENTS + 0.01f},
            };
            Color sc = Fade(LIGHT_PURPLE, 0.6f);
            for (int k = 0; k < 4; k++)
                draw_thick_line_3d(pc[k], pc[(k + 1) % 4], 0.4f, sc);
        }
    }

    // Ego box (cyan)
    float hl = ego->length * 0.5f;
    float hw = ego->width * 0.5f;
    Vector3 ec[4] = {
        {ego->x + hl * cos_h - hw * sin_h, ego->y + hl * sin_h + hw * cos_h, Z_AGENTS},
        {ego->x + hl * cos_h + hw * sin_h, ego->y + hl * sin_h - hw * cos_h, Z_AGENTS},
        {ego->x - hl * cos_h + hw * sin_h, ego->y - hl * sin_h - hw * cos_h, Z_AGENTS},
        {ego->x - hl * cos_h - hw * sin_h, ego->y - hl * sin_h + hw * cos_h, Z_AGENTS},
    };
    for (int k = 0; k < 4; k++)
        draw_thick_line_3d(ec[k], ec[(k + 1) % 4], 0.5f, DEEPBLUE);

    // Ego heading arrow
    Vector3 as = {ego->x, ego->y, Z_AGENTS};
    Vector3 ae = {ego->x + cos_h * hl * 2.0f, ego->y + sin_h * hl * 2.0f, Z_AGENTS};
    draw_thick_line_3d(as, ae, 0.25f, DEEPBLUE);
}

static void write_all_bytes(int fd, const unsigned char *data, size_t size) {
    size_t remaining = size;
    const unsigned char *p = data;
    while (remaining > 0) {
        ssize_t written = write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "[drive] write failed errno=%d(%s)\n", errno, strerror(errno));
            break;
        }
        p += written;
        remaining -= (size_t)written;
    }
}

#ifdef DRIVE_HAS_EGL
static void write_gl_frame_flipped(int fd, unsigned char *ptr, int width, int height) {
    int row_bytes = width * 4;
    int rows_remaining = height;
    int row_top = 0;
    int iov_max = 1024;

    while (rows_remaining > 0) {
        int chunk = rows_remaining < iov_max ? rows_remaining : iov_max;
        struct iovec iov[1024];
        size_t chunk_bytes = 0;
        for (int i = 0; i < chunk; i++) {
            int src_row = height - 1 - (row_top + i);
            iov[i].iov_base = ptr + (size_t)src_row * row_bytes;
            iov[i].iov_len = row_bytes;
            chunk_bytes += row_bytes;
        }

        struct iovec *cur = iov;
        int cur_cnt = chunk;
        size_t cur_remaining = chunk_bytes;
        while (cur_remaining > 0) {
            ssize_t written = writev(fd, cur, cur_cnt);
            if (written < 0) {
                if (errno == EINTR)
                    continue;
                fprintf(stderr, "[drive-pbo] writev failed errno=%d(%s)\n", errno, strerror(errno));
                return;
            }
            cur_remaining -= (size_t)written;
            size_t consumed = (size_t)written;
            while (cur_cnt > 0 && consumed >= cur[0].iov_len) {
                consumed -= cur[0].iov_len;
                cur++;
                cur_cnt--;
            }
            if (cur_cnt > 0 && consumed > 0) {
                cur[0].iov_base = (unsigned char *)cur[0].iov_base + consumed;
                cur[0].iov_len -= consumed;
            }
        }

        row_top += chunk;
        rows_remaining -= chunk;
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Interactive pop-up viewer (render_mode = RENDER_WINDOW)
//
// The simulator keeps driving the loop: every c_render call draws at least one frame. While paused, c_render keeps
// drawing and handling input until the user resumes or single-steps, so Python eval / verify loops stay unchanged.
// GUI state lives in a process-wide struct so it survives client re-creation on map resampling.
// ---------------------------------------------------------------------------------------------------------------------

#define GUI_CAM_BEV 0
#define GUI_CAM_ORBIT 1
#define GUI_CAM_FPV 2
#define GUI_NAV_OFF 0
#define GUI_NAV_ROUTE 1
#define GUI_NAV_ROUTE_LANES 2
#define GUI_NAV_ALL 3
#define GUI_GOALS_ALL 0
#define GUI_GOALS_SELECTED 1
#define GUI_GOALS_OFF 2
#define GUI_FEAS_OFF 0
#define GUI_FEAS_ARROWS 1
#define GUI_FEAS_TEXT 2
#define GUI_FEAS_LANE_OFFSET_M 3.5f // lateral offset of the left/right indicators (about one lane width)
#define GUI_FEAS_Z 2.5f             // above the agent boxes so neighbours don't hide the indicators
#define GUI_FEAS_MIN_LEN_M 0.8f
#define GUI_FEAS_MAX_LEN_M 5.0f
#define GUI_FEAS_TEXT_GAP_PX 55.0f

#define GUI_ORBIT_FOVY 45.0f
#define GUI_FPV_FOVY 60.0f
#define GUI_BEV_CAMERA_HEIGHT 500.0f
#define GUI_MIN_VIEW_M 5.0f
#define GUI_MAX_VIEW_M 3000.0f
#define GUI_MAX_ORBIT_VIEW_M 700.0f // keeps the orbit camera inside the far clip plane
#define GUI_DRAG_THRESHOLD_PX 4.0f
#define GUI_DOUBLE_CLICK_S 0.3
#define GUI_INERTIA_DECAY 0.85f
#define GUI_TRAIL_LEN 60
#define GUI_PICK_MARGIN_M 0.5f
#define GUI_PICK_FALLBACK_M 3.0f
#define GUI_WAYPOINT_RADIUS_M 0.6f
#define GUI_FONT 20
#define GUI_LINE 22

// NULL-terminated so the help panel can measure every state label
static const char *const GUI_CAM_NAMES[] = {"BEV", "3D orbit", "FPV", NULL};
static const char *const GUI_NAV_NAMES[] = {"off", "route", "route + lanes", "all routes", NULL};
static const char *const GUI_GOAL_NAMES[] = {"all", "selected", "off", NULL};
static const char *const GUI_FEAS_NAMES[] = {"off", "arrows", "text", NULL};
static const char *const GUI_ONOFF_NAMES[] = {"off", "on", NULL};

typedef struct PopupGui PopupGui;
struct PopupGui {
    int initialized;
    Client *client_seen;

    // Camera: orbit/BEV share target, yaw and view height (metres visible vertically)
    int cam_mode;
    Vector3 target;
    float yaw;
    float pitch;
    float view_h;
    int follow;
    Vector2 orbit_vel;
    Vector3 pan_vel;
    float zoom_vel;
    Vector2 zoom_anchor;
    Vector3 fpv_pos;
    Vector3 fpv_target;
    int fpv_valid;
    float fpv_azimuth;
    float fpv_elevation;
    float fpv_dist;
    Camera3D camera;

    // Mouse drag / click tracking
    int drag_button; // -1 when no button is held
    int drag_active;
    Vector2 press_pos;
    Vector3 pan_anchor;
    int pan_anchor_valid;
    double last_click_time;
    Vector2 last_click_pos;

    // Toggles
    int obs_view;
    int lasers;
    int nav_mode;
    int goal_mode;
    int feas_mode;
    int trails;
    int help;

    // Flow control
    int paused;
    int step_requested;
    int quit_confirm;
    int map_request; // -1 / +1: previous / next map, consumed by Python via gui_take_map_request()

    int sel; // index into env->active_agent_indices

    // Trails: ring buffer of GUI_TRAIL_LEN positions per active agent
    Vector2 *trail;
    int *trail_head;
    int *trail_len;
    int trail_agents;
    int trail_timestep;
};

static PopupGui g_gui = {0};
static volatile sig_atomic_t g_gui_sigint = 0;

typedef struct GuiBinding GuiBinding;
struct GuiBinding {
    const char *keys;
    const char *desc;
    const int *state;         // NULL for plain commands
    const char *const *names; // state labels, indexed by *state
};

static const GuiBinding GUI_BINDINGS[] = {
    {"SPACE / P", "Pause / resume", &g_gui.paused, GUI_ONOFF_NAMES},
    {"ENTER / ->", "Step one frame (paused)", NULL, NULL},
    {"V", "Cycle camera BEV / 3D / FPV", &g_gui.cam_mode, GUI_CAM_NAMES},
    {"B", "Toggle BEV", NULL, NULL},
    {"F", "Follow selected agent", &g_gui.follow, GUI_ONOFF_NAMES},
    {"R", "Reset camera to map", NULL, NULL},
    {"TAB / S-TAB", "Next / previous agent", NULL, NULL},
    {"S", "Select agent nearest cursor", NULL, NULL},
    {"CTRL", "Observation view", &g_gui.obs_view, GUI_ONOFF_NAMES},
    {"O", "Observation lasers", &g_gui.lasers, GUI_ONOFF_NAMES},
    {"N", "Navigation overlay", &g_gui.nav_mode, GUI_NAV_NAMES},
    {"L", "Lane feasibility (selected)", &g_gui.feas_mode, GUI_FEAS_NAMES},
    {"G", "Goal markers", &g_gui.goal_mode, GUI_GOAL_NAMES},
    {"T", "Motion trails", &g_gui.trails, GUI_ONOFF_NAMES},
    {"J / K", "Previous / next map", NULL, NULL},
    {"H", "This help", &g_gui.help, GUI_ONOFF_NAMES},
    {"Q / ESC", "Quit viewer", NULL, NULL},
    {"LMB click", "Select agent", NULL, NULL},
    {"LMB drag", "Orbit (3D) / pan (BEV) / look (FPV)", NULL, NULL},
    {"RMB/MMB drag", "Pan", NULL, NULL},
    {"Wheel", "Zoom to cursor", NULL, NULL},
    {"Double-click", "Centre view on point", NULL, NULL},
};
#define GUI_NUM_BINDINGS ((int)(sizeof(GUI_BINDINGS) / sizeof(GUI_BINDINGS[0])))

static void gui_sigint_handler(int sig) {
    (void)sig;
    g_gui_sigint = 1;
}

static int gui_agent_selectable(Drive *env, int active_idx) {
    if (active_idx < 0 || active_idx >= env->active_agent_count)
        return 0;
    Entity *e = &env->entities[env->active_agent_indices[active_idx]];
    return !e->removed && e->x != INVALID_POSITION;
}

static Entity *gui_selected_entity(Drive *env) {
    if (!gui_agent_selectable(env, g_gui.sel))
        return NULL;
    return &env->entities[env->active_agent_indices[g_gui.sel]];
}

static void gui_set_selection(Drive *env, int active_idx) {
    g_gui.sel = active_idx;
    env->human_agent_idx = active_idx;
    g_gui.fpv_valid = 0;
}

static void gui_cycle_selection(Drive *env, int dir) {
    int n = env->active_agent_count;
    for (int k = 1; k <= n; k++) {
        int idx = ((g_gui.sel + dir * k) % n + n) % n;
        if (gui_agent_selectable(env, idx)) {
            gui_set_selection(env, idx);
            return;
        }
    }
}

// Golden-angle hue spacing: consecutive indices get well-separated colours however many goals the map has
// (agents typically use only a few low goal indices). Agents sharing a navigation goal share a colour.
static Color gui_goal_color(Drive *env, int active_idx) {
    Entity *e = &env->entities[env->active_agent_indices[active_idx]];
    int idx = active_idx;
    if (env->navigation != NULL && e->navi_goal_idx >= 0)
        idx = e->navi_goal_idx;
    return ColorFromHSV(fmodf((float)idx * 137.508f, 360.0f), 0.85f, 0.90f);
}

// ------------------------------------------------- camera -----------------------------------------------------------

static Camera3D gui_build_camera(void) {
    Camera3D camera = {0};
    if (g_gui.cam_mode == GUI_CAM_FPV && g_gui.fpv_valid) {
        camera.position = g_gui.fpv_pos;
        camera.target = g_gui.fpv_target;
        camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        camera.fovy = GUI_FPV_FOVY;
        camera.projection = CAMERA_PERSPECTIVE;
        return camera;
    }
    int bev = g_gui.cam_mode != GUI_CAM_ORBIT;
    float phi = bev ? PI * 0.5f : g_gui.pitch;
    float psi = g_gui.yaw;
    float dist = bev ? GUI_BEV_CAMERA_HEIGHT : g_gui.view_h / (2.0f * tanf(GUI_ORBIT_FOVY * 0.5f * DEG2RAD));
    Vector3 dir = {cosf(phi) * cosf(psi), cosf(phi) * sinf(psi), sinf(phi)};
    camera.target = g_gui.target;
    camera.position = Vector3Add(g_gui.target, Vector3Scale(dir, dist));
    camera.up = (Vector3){-sinf(phi) * cosf(psi), -sinf(phi) * sinf(psi), cosf(phi)};
    camera.fovy = bev ? g_gui.view_h : GUI_ORBIT_FOVY;
    camera.projection = bev ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
    return camera;
}

static int gui_ray_plane(Vector2 screen, Camera3D camera, float z, Vector3 *out) {
    Ray ray = GetScreenToWorldRayEx(screen, camera, GetScreenWidth(), GetScreenHeight());
    if (fabsf(ray.direction.z) < 1e-6f)
        return 0;
    float t = (z - ray.position.z) / ray.direction.z;
    if (t < 0.0f)
        return 0;
    *out = Vector3Add(ray.position, Vector3Scale(ray.direction, t));
    return 1;
}

static void gui_clamp_view(void) {
    float max_view = g_gui.cam_mode == GUI_CAM_ORBIT ? GUI_MAX_ORBIT_VIEW_M : GUI_MAX_VIEW_M;
    g_gui.view_h = Clamp(g_gui.view_h, GUI_MIN_VIEW_M, max_view);
}

static void gui_autoframe(Drive *env) {
    float map_w = fabsf(env->grid_map->bottom_right_x - env->grid_map->top_left_x);
    float map_h = fabsf(env->grid_map->top_left_y - env->grid_map->bottom_right_y);
    float aspect = (float)GetScreenWidth() / fmaxf((float)GetScreenHeight(), 1.0f);
    g_gui.target = (Vector3){(env->grid_map->top_left_x + env->grid_map->bottom_right_x) * 0.5f,
                             (env->grid_map->top_left_y + env->grid_map->bottom_right_y) * 0.5f, 0.0f};
    g_gui.view_h = fmaxf(map_h, map_w / aspect) * 1.05f;
    g_gui.yaw = PI * 0.5f; // matches the headless top-down orientation
    g_gui.pitch = 60.0f * DEG2RAD;
    g_gui.orbit_vel = (Vector2){0};
    g_gui.pan_vel = (Vector3){0};
    g_gui.zoom_vel = 0.0f;
    g_gui.follow = 0;
    gui_clamp_view();
}

static void gui_zoom(float wheel, Vector2 anchor_screen) {
    if (g_gui.cam_mode == GUI_CAM_FPV) {
        g_gui.fpv_dist = Clamp(g_gui.fpv_dist * powf(0.9f, wheel), 5.0f, 150.0f);
        return;
    }
    Vector3 before, after;
    int anchored = !g_gui.follow && gui_ray_plane(anchor_screen, gui_build_camera(), 0.0f, &before);
    g_gui.view_h *= powf(0.9f, wheel);
    gui_clamp_view();
    if (anchored && gui_ray_plane(anchor_screen, gui_build_camera(), 0.0f, &after)) {
        g_gui.target.x += before.x - after.x;
        g_gui.target.y += before.y - after.y;
    }
}

static void gui_update_fpv(Drive *env) {
    Entity *agent = gui_selected_entity(env);
    if (agent == NULL)
        return;
    float h = agent->heading + g_gui.fpv_azimuth;
    float ce = cosf(g_gui.fpv_elevation), se = sinf(g_gui.fpv_elevation);
    Vector3 pos = {agent->x - g_gui.fpv_dist * ce * cosf(h), agent->y - g_gui.fpv_dist * ce * sinf(h),
                   2.0f + g_gui.fpv_dist * se};
    Vector3 target = {agent->x + 8.0f * cosf(agent->heading), agent->y + 8.0f * sinf(agent->heading), 1.0f};
    if (!g_gui.fpv_valid) {
        g_gui.fpv_pos = pos;
        g_gui.fpv_target = target;
        g_gui.fpv_valid = 1;
    } else {
        g_gui.fpv_pos = Vector3Lerp(g_gui.fpv_pos, pos, 0.25f);
        g_gui.fpv_target = Vector3Lerp(g_gui.fpv_target, target, 0.25f);
    }
}

static void gui_set_cam_mode(int mode) {
    g_gui.cam_mode = mode;
    g_gui.fpv_valid = 0;
    g_gui.orbit_vel = (Vector2){0};
    g_gui.pan_vel = (Vector3){0};
    gui_clamp_view();
}

// ------------------------------------------------- selection --------------------------------------------------------

// Agent under the cursor: ray vs. oriented box (with margin) at half box height, else nearest centre within
// GUI_PICK_FALLBACK_M of the ground hit. Returns an active-agent index or -1.
static int gui_pick_agent(Drive *env, Vector2 screen, Camera3D camera) {
    int best_box = -1;
    int best_near = -1;
    float best_box_d2 = 1e30f;
    float best_near_d2 = GUI_PICK_FALLBACK_M * GUI_PICK_FALLBACK_M;
    for (int i = 0; i < env->active_agent_count; i++) {
        if (!gui_agent_selectable(env, i))
            continue;
        Entity *e = &env->entities[env->active_agent_indices[i]];
        float box_h = e->height > 0.0f ? e->height : 1.5f;
        Vector3 hit;
        if (!gui_ray_plane(screen, camera, box_h * 0.5f, &hit))
            continue;
        float dx = hit.x - e->x, dy = hit.y - e->y;
        float lx = dx * e->heading_x + dy * e->heading_y;
        float ly = -dx * e->heading_y + dy * e->heading_x;
        float d2 = dx * dx + dy * dy;
        if (fabsf(lx) <= e->length * 0.5f + GUI_PICK_MARGIN_M && fabsf(ly) <= e->width * 0.5f + GUI_PICK_MARGIN_M) {
            if (d2 < best_box_d2) {
                best_box_d2 = d2;
                best_box = i;
            }
        } else if (d2 < best_near_d2) {
            best_near_d2 = d2;
            best_near = i;
        }
    }
    return best_box >= 0 ? best_box : best_near;
}

static int gui_nearest_agent(Drive *env, Vector2 screen, Camera3D camera) {
    Vector3 hit;
    if (!gui_ray_plane(screen, camera, 0.0f, &hit))
        return -1;
    int best = -1;
    float best_d2 = 1e30f;
    for (int i = 0; i < env->active_agent_count; i++) {
        if (!gui_agent_selectable(env, i))
            continue;
        Entity *e = &env->entities[env->active_agent_indices[i]];
        float d2 = (hit.x - e->x) * (hit.x - e->x) + (hit.y - e->y) * (hit.y - e->y);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

// ------------------------------------------------- input ------------------------------------------------------------

static void gui_quit(void) {
    CloseWindow();
    exit(0);
}

static void gui_handle_keys(Drive *env, Camera3D camera) {
    if (g_gui.quit_confirm) {
        if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER) || IsKeyPressed(KEY_Y))
            gui_quit();
        if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_N))
            g_gui.quit_confirm = 0;
        return;
    }
    int shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

    if (IsKeyPressed(KEY_Q))
        gui_quit();
    if (IsKeyPressed(KEY_ESCAPE))
        g_gui.quit_confirm = 1;
    if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P))
        g_gui.paused = !g_gui.paused;
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER) || IsKeyPressed(KEY_RIGHT) ||
        IsKeyPressedRepeat(KEY_RIGHT))
        g_gui.step_requested = 1;
    if (IsKeyPressed(KEY_H))
        g_gui.help = !g_gui.help;

    if (IsKeyPressed(KEY_V))
        gui_set_cam_mode((g_gui.cam_mode + 1) % 3);
    if (IsKeyPressed(KEY_B))
        gui_set_cam_mode(g_gui.cam_mode == GUI_CAM_BEV ? GUI_CAM_ORBIT : GUI_CAM_BEV);
    if (IsKeyPressed(KEY_F))
        g_gui.follow = !g_gui.follow;
    if (IsKeyPressed(KEY_R))
        gui_autoframe(env);

    if (IsKeyPressed(KEY_TAB) && env->active_agent_count > 0)
        gui_cycle_selection(env, shift ? -1 : 1);
    if (IsKeyPressed(KEY_S)) {
        int idx = gui_nearest_agent(env, GetMousePosition(), camera);
        if (idx >= 0)
            gui_set_selection(env, idx);
    }

    if (IsKeyPressed(KEY_LEFT_CONTROL) || IsKeyPressed(KEY_RIGHT_CONTROL))
        g_gui.obs_view = !g_gui.obs_view;
    if (IsKeyPressed(KEY_O))
        g_gui.lasers = !g_gui.lasers;
    if (IsKeyPressed(KEY_N) && env->navigation != NULL)
        g_gui.nav_mode = (g_gui.nav_mode + 1) % 4;
    if (IsKeyPressed(KEY_L) && env->navigation != NULL)
        g_gui.feas_mode = (g_gui.feas_mode + 1) % 3;
    if (IsKeyPressed(KEY_J))
        g_gui.map_request = -1;
    if (IsKeyPressed(KEY_K))
        g_gui.map_request = 1;
    if (IsKeyPressed(KEY_G))
        g_gui.goal_mode = (g_gui.goal_mode + 1) % 3;
    if (IsKeyPressed(KEY_T))
        g_gui.trails = !g_gui.trails;
}

static void gui_handle_mouse(Drive *env, Camera3D camera) {
    Vector2 mouse = GetMousePosition();
    Vector2 delta = GetMouseDelta();

    if (g_gui.drag_button < 0) {
        int buttons[3] = {MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE};
        for (int b = 0; b < 3; b++) {
            if (IsMouseButtonPressed(buttons[b])) {
                g_gui.drag_button = buttons[b];
                g_gui.drag_active = 0;
                g_gui.press_pos = mouse;
                g_gui.orbit_vel = (Vector2){0};
                g_gui.pan_vel = (Vector3){0};
                break;
            }
        }
    }

    if (g_gui.drag_button >= 0) {
        int button = g_gui.drag_button;
        if (!g_gui.drag_active && Vector2Distance(mouse, g_gui.press_pos) > GUI_DRAG_THRESHOLD_PX) {
            g_gui.drag_active = 1;
            g_gui.pan_anchor_valid = gui_ray_plane(mouse, camera, 0.0f, &g_gui.pan_anchor);
        }
        if (g_gui.drag_active) {
            int pan = button != MOUSE_BUTTON_LEFT || g_gui.cam_mode == GUI_CAM_BEV;
            if (g_gui.cam_mode == GUI_CAM_FPV) {
                if (button == MOUSE_BUTTON_LEFT) {
                    g_gui.fpv_azimuth -= delta.x * 0.005f;
                    g_gui.fpv_elevation = Clamp(g_gui.fpv_elevation + delta.y * 0.005f, 0.05f, 1.4f);
                }
            } else if (pan) {
                Vector3 hit;
                if (g_gui.pan_anchor_valid && gui_ray_plane(mouse, camera, 0.0f, &hit)) {
                    Vector3 shift = {g_gui.pan_anchor.x - hit.x, g_gui.pan_anchor.y - hit.y, 0.0f};
                    g_gui.target = Vector3Add(g_gui.target, shift);
                    g_gui.pan_vel = Vector3Lerp(g_gui.pan_vel, shift, 0.5f);
                    g_gui.follow = 0;
                }
            } else {
                Vector2 v = {-delta.x * 0.005f, delta.y * 0.005f};
                g_gui.yaw += v.x;
                g_gui.pitch = Clamp(g_gui.pitch + v.y, 5.0f * DEG2RAD, 89.0f * DEG2RAD);
                g_gui.orbit_vel = Vector2Lerp(g_gui.orbit_vel, v, 0.5f);
            }
        }
        if (IsMouseButtonReleased(button)) {
            if (!g_gui.drag_active && button == MOUSE_BUTTON_LEFT) {
                double now = GetTime();
                int double_click = now - g_gui.last_click_time < GUI_DOUBLE_CLICK_S &&
                                   Vector2Distance(mouse, g_gui.last_click_pos) < 6.0f;
                Vector3 hit;
                if (double_click && g_gui.cam_mode != GUI_CAM_FPV && gui_ray_plane(mouse, camera, 0.0f, &hit)) {
                    g_gui.target = (Vector3){hit.x, hit.y, 0.0f};
                    g_gui.follow = 0;
                    g_gui.last_click_time = 0.0;
                } else {
                    int idx = gui_pick_agent(env, mouse, camera);
                    if (idx >= 0)
                        gui_set_selection(env, idx);
                    g_gui.last_click_time = now;
                    g_gui.last_click_pos = mouse;
                }
            }
            if (!g_gui.drag_active || button != MOUSE_BUTTON_LEFT || g_gui.cam_mode == GUI_CAM_BEV) {
                g_gui.orbit_vel = (Vector2){0};
            }
            if (!g_gui.drag_active) {
                g_gui.pan_vel = (Vector3){0};
            }
            g_gui.drag_button = -1;
            g_gui.drag_active = 0;
        }
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        g_gui.zoom_vel += wheel;
        g_gui.zoom_anchor = mouse;
    }
}

static void gui_apply_inertia(Drive *env) {
    if (g_gui.drag_button < 0) {
        if (Vector2Length(g_gui.orbit_vel) > 1e-4f && g_gui.cam_mode == GUI_CAM_ORBIT) {
            g_gui.yaw += g_gui.orbit_vel.x;
            g_gui.pitch = Clamp(g_gui.pitch + g_gui.orbit_vel.y, 5.0f * DEG2RAD, 89.0f * DEG2RAD);
            g_gui.orbit_vel = Vector2Scale(g_gui.orbit_vel, GUI_INERTIA_DECAY);
        } else {
            g_gui.orbit_vel = (Vector2){0};
        }
        if (Vector3Length(g_gui.pan_vel) > 1e-3f * g_gui.view_h && g_gui.cam_mode != GUI_CAM_FPV) {
            g_gui.target = Vector3Add(g_gui.target, g_gui.pan_vel);
            g_gui.pan_vel = Vector3Scale(g_gui.pan_vel, GUI_INERTIA_DECAY);
        } else {
            g_gui.pan_vel = (Vector3){0};
        }
    }
    if (fabsf(g_gui.zoom_vel) > 0.01f) {
        float step = g_gui.zoom_vel * 0.35f;
        gui_zoom(step, g_gui.zoom_anchor);
        g_gui.zoom_vel -= step;
    } else {
        g_gui.zoom_vel = 0.0f;
    }

    Entity *agent = gui_selected_entity(env);
    if (g_gui.follow && agent != NULL && g_gui.cam_mode != GUI_CAM_FPV) {
        g_gui.target = Vector3Lerp(g_gui.target, (Vector3){agent->x, agent->y, 0.0f}, 0.2f);
    }
    if (g_gui.cam_mode == GUI_CAM_FPV) {
        gui_update_fpv(env);
    }
}

// ------------------------------------------------- trails -----------------------------------------------------------

static void gui_reset_trails(Drive *env) {
    free(g_gui.trail);
    free(g_gui.trail_head);
    free(g_gui.trail_len);
    int n = env->active_agent_count;
    g_gui.trail_agents = n;
    g_gui.trail = n > 0 ? (Vector2 *)calloc((size_t)n * GUI_TRAIL_LEN, sizeof(Vector2)) : NULL;
    g_gui.trail_head = n > 0 ? (int *)calloc(n, sizeof(int)) : NULL;
    g_gui.trail_len = n > 0 ? (int *)calloc(n, sizeof(int)) : NULL;
    g_gui.trail_timestep = -1;
}

static void gui_update_trails(Drive *env) {
    if (env->active_agent_count != g_gui.trail_agents || env->timestep < g_gui.trail_timestep) {
        gui_reset_trails(env);
    }
    if (env->timestep == g_gui.trail_timestep)
        return;
    g_gui.trail_timestep = env->timestep;
    for (int i = 0; i < g_gui.trail_agents; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];
        if (e->removed || e->x == INVALID_POSITION)
            continue;
        int head = (g_gui.trail_head[i] + 1) % GUI_TRAIL_LEN;
        g_gui.trail[i * GUI_TRAIL_LEN + head] = (Vector2){e->x, e->y};
        g_gui.trail_head[i] = head;
        if (g_gui.trail_len[i] < GUI_TRAIL_LEN)
            g_gui.trail_len[i]++;
    }
}

static void gui_draw_trails(Drive *env) {
    for (int i = 0; i < g_gui.trail_agents; i++) {
        int len = g_gui.trail_len[i];
        if (len < 2 || !gui_agent_selectable(env, i))
            continue;
        Color color = gui_goal_color(env, i);
        for (int k = 0; k + 1 < len; k++) {
            int a = (g_gui.trail_head[i] - k + GUI_TRAIL_LEN) % GUI_TRAIL_LEN;
            int b = (a - 1 + GUI_TRAIL_LEN) % GUI_TRAIL_LEN;
            Vector2 pa = g_gui.trail[i * GUI_TRAIL_LEN + a];
            Vector2 pb = g_gui.trail[i * GUI_TRAIL_LEN + b];
            float alpha = 0.9f * (1.0f - (float)k / (float)GUI_TRAIL_LEN);
            DrawLine3D((Vector3){pa.x, pa.y, 0.15f}, (Vector3){pb.x, pb.y, 0.15f}, Fade(color, alpha));
        }
    }
}

// ------------------------------------------------- scene ------------------------------------------------------------

static void gui_set_line_width(float width) {
    rlDrawRenderBatchActive(); // line width applies to the whole pending batch
    rlSetLineWidth(width);
}

static void gui_draw_feasibility_lanes(Drive *env, const Entity *agent) {
    const Navigation *navigation = env->navigation;
    int goal_table_offset = agent->navi_goal_idx * navigation->num_lanes;
    gui_set_line_width(NAVI_RENDER_LANE_WIDTH_PX);
    for (int navi_lane_idx = 0; navi_lane_idx < navigation->num_lanes; navi_lane_idx++) {
        const Entity *lane = &env->entities[navigation->lane_entity_idx[navi_lane_idx]];
        Color lane_color = Fade(GRAY, NAVI_RENDER_UNREACHABLE_ALPHA);
        if (navigation->distance_to_goal_m[goal_table_offset + navi_lane_idx] < NAVI_UNREACHABLE_M) {
            float feasibility_ratio = fminf(navigation->feasibility_center_m[goal_table_offset + navi_lane_idx] /
                                                NAVI_OBS_FEASIBILITY_HORIZON_M,
                                            1.0f);
            lane_color = (Color){(unsigned char)(255.0f * (1.0f - feasibility_ratio)),
                                 (unsigned char)(255.0f * feasibility_ratio), 0, 255};
        }
        draw_lane_polyline(lane, Z_ROAD_MARKINGS + NAVI_RENDER_Z_OFFSET, lane_color);
    }
    gui_set_line_width(1.0f);
}

static void gui_draw_route(Drive *env, int active_idx, Color color, int is_selected) {
    const Navigation *navigation = env->navigation;
    const Entity *agent = &env->entities[env->active_agent_indices[active_idx]];
    if (agent->navi_goal_idx < 0)
        return;
    int goal_table_offset = agent->navi_goal_idx * navigation->num_lanes;
    int lane_idx = agent->current_lane_idx != -1 ? navigation->entity_to_navi_lane[agent->current_lane_idx]
                                                 : agent->navi_lane_idx;
    gui_set_line_width(is_selected ? NAVI_RENDER_ROUTE_WIDTH_PX : NAVI_RENDER_LANE_WIDTH_PX);
    for (int hop = 0; hop < navigation->num_lanes && lane_idx != -1; hop++) {
        draw_lane_polyline(&env->entities[navigation->lane_entity_idx[lane_idx]],
                           Z_ROAD_MARKINGS + 2.0f * NAVI_RENDER_Z_OFFSET, is_selected ? color : Fade(color, 0.7f));
        lane_idx = navigation->next_lane[goal_table_offset + lane_idx];
    }
    gui_set_line_width(1.0f);

    float xs[NAVI_MAX_WAYPOINTS];
    float ys[NAVI_MAX_WAYPOINTS];
    int max_waypoints = env->navigation_waypoints > 0 ? env->navigation_waypoints : NAVI_DEFAULT_WAYPOINTS;
    float spacing = env->navigation_waypoint_spacing > 0.0f ? env->navigation_waypoint_spacing
                                                            : NAVI_DEFAULT_WAYPOINT_SPACING_M;
    int count = navigation_route_waypoints(env, agent, max_waypoints, spacing, xs, ys);
    float radius = is_selected ? GUI_WAYPOINT_RADIUS_M : GUI_WAYPOINT_RADIUS_M * 0.6f;
    for (int k = 0; k < count; k++) {
        Vector3 base = {xs[k], ys[k], Z_AGENT_DETAILS};
        DrawCylinderEx(base, (Vector3){xs[k], ys[k], Z_AGENT_DETAILS + 0.05f}, radius, radius, 16, ORANGE);
        DrawCircle3D((Vector3){xs[k], ys[k], Z_AGENT_DETAILS + 0.06f}, radius, (Vector3){0, 0, 1}, 0.0f,
                     Fade(BLACK, 0.6f));
    }
    Vector3 goal = {navigation->goal_x[agent->navi_goal_idx], navigation->goal_y[agent->navi_goal_idx],
                    Z_AGENT_DETAILS};
    DrawSphere(goal, is_selected ? NAVI_RENDER_GOAL_RADIUS_M : NAVI_RENDER_GOAL_RADIUS_M * 0.6f, color);
}

static void gui_draw_nav(Drive *env) {
    if (env->navigation == NULL || g_gui.nav_mode == GUI_NAV_OFF)
        return;
    if (g_gui.nav_mode == GUI_NAV_ALL) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (i != g_gui.sel && gui_agent_selectable(env, i))
                gui_draw_route(env, i, gui_goal_color(env, i), 0);
        }
    }
    Entity *agent = gui_selected_entity(env);
    if (agent == NULL || agent->navi_goal_idx < 0)
        return;
    if (g_gui.nav_mode == GUI_NAV_ROUTE_LANES)
        gui_draw_feasibility_lanes(env, agent);
    gui_draw_route(env, g_gui.sel, gui_goal_color(env, g_gui.sel), 1);
}

// World-size multiplier so the indicators stay readable when zoomed out (BEV / orbit); 1 in FPV.
static float gui_feasibility_scale(void) {
    if (g_gui.cam_mode == GUI_CAM_FPV)
        return 1.0f;
    return fmaxf(1.0f, g_gui.view_h / 100.0f);
}

// Selected agent's left / current / right lane feasibility (metres the goal stays reachable when driving in that
// lane). Indicators sit ahead of the agent, one per lane; has_lane[i] is 0 when there is no such neighbour lane.
static int gui_feasibility_state(Drive *env, Vector3 base[3], Vector2 *dir, float vals[3], int has_lane[3]) {
    Entity *agent = gui_selected_entity(env);
    if (env->navigation == NULL || agent == NULL || agent->navi_goal_idx < 0 || agent->navi_lane_idx < 0)
        return 0;
    float scale = gui_feasibility_scale();
    float hx = cosf(agent->heading);
    float hy = sinf(agent->heading);
    float ahead = agent->length * 0.5f + 1.0f * scale;
    float offset = GUI_FEAS_LANE_OFFSET_M * scale;
    float lateral[3] = {offset, 0.0f, -offset};
    vals[0] = agent->navi_feasibility_left_m;
    vals[1] = agent->navi_feasibility_center_m;
    vals[2] = agent->navi_feasibility_right_m;
    has_lane[0] = env->navigation->left_lane[agent->navi_lane_idx] != -1;
    has_lane[1] = 1;
    has_lane[2] = env->navigation->right_lane[agent->navi_lane_idx] != -1;
    for (int i = 0; i < 3; i++) {
        base[i] = (Vector3){agent->x + hx * ahead - hy * lateral[i], agent->y + hy * ahead + hx * lateral[i],
                            GUI_FEAS_Z};
    }
    *dir = (Vector2){hx, hy};
    return 1;
}

static Color gui_feasibility_color(float value_m) {
    if (value_m <= 0.0f)
        return RED;
    float ratio = fminf(value_m / NAVI_OBS_FEASIBILITY_HORIZON_M, 1.0f);
    return (Color){(unsigned char)(255.0f * (1.0f - ratio)), (unsigned char)(255.0f * ratio), 0, 255};
}

static void gui_draw_feasibility_arrows(Drive *env) {
    Vector3 base[3];
    Vector2 dir;
    float vals[3];
    int has_lane[3];
    if (g_gui.feas_mode != GUI_FEAS_ARROWS || !gui_feasibility_state(env, base, &dir, vals, has_lane))
        return;
    Vector2 side = {-dir.y, dir.x};
    float k = gui_feasibility_scale();
    for (int i = 0; i < 3; i++) {
        if (!has_lane[i])
            continue;
        Color color = gui_feasibility_color(vals[i]);
        Vector3 p = base[i];
        if (vals[i] <= 0.0f) {
            float s = 0.5f * k;
            Vector3 a1 = {p.x + (dir.x + side.x) * s, p.y + (dir.y + side.y) * s, p.z};
            Vector3 a2 = {p.x - (dir.x + side.x) * s, p.y - (dir.y + side.y) * s, p.z};
            Vector3 b1 = {p.x + (dir.x - side.x) * s, p.y + (dir.y - side.y) * s, p.z};
            Vector3 b2 = {p.x - (dir.x - side.x) * s, p.y - (dir.y - side.y) * s, p.z};
            DrawCylinderEx(a1, a2, 0.1f * k, 0.1f * k, 8, color);
            DrawCylinderEx(b1, b2, 0.1f * k, 0.1f * k, 8, color);
            continue;
        }
        float ratio = fminf(vals[i] / NAVI_OBS_FEASIBILITY_HORIZON_M, 1.0f);
        float len = (GUI_FEAS_MIN_LEN_M + (GUI_FEAS_MAX_LEN_M - GUI_FEAS_MIN_LEN_M) * ratio) * k;
        float head = fminf(0.6f * k, len * 0.5f);
        Vector3 neck = {p.x + dir.x * (len - head), p.y + dir.y * (len - head), p.z};
        Vector3 tip = {p.x + dir.x * len, p.y + dir.y * len, p.z};
        DrawCylinderEx(p, neck, 0.1f * k, 0.1f * k, 8, color);
        DrawCylinderEx(neck, tip, 0.35f * k, 0.0f, 12, color);
    }
}

static const char *gui_feasibility_label(float value_m, int has_lane) {
    if (!has_lane)
        return "-";
    if (value_m <= 0.0f)
        return "X";
    if (value_m >= NAVI_FEASIBILITY_CAP_M)
        return TextFormat("%.0f+m", NAVI_FEASIBILITY_CAP_M);
    return TextFormat("%.0fm", value_m);
}

static void gui_draw_feasibility_text(Drive *env, Camera3D camera) {
    Vector3 base[3];
    Vector2 dir;
    float vals[3];
    int has_lane[3];
    if (g_gui.feas_mode != GUI_FEAS_TEXT || !gui_feasibility_state(env, base, &dir, vals, has_lane))
        return;
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    // Skip when the centre point is behind the camera (FPV / orbit)
    if (camera.projection == CAMERA_PERSPECTIVE &&
        Vector3DotProduct(forward, Vector3Subtract(base[1], camera.position)) <= 0.0f)
        return;
    // Keep left/right labels at least GUI_FEAS_TEXT_GAP_PX from the centre one on screen
    Vector2 center = GetWorldToScreen(base[1], camera);
    Vector2 left = GetWorldToScreen(base[0], camera);
    Vector2 lat = Vector2Subtract(left, center);
    float lat_len = Vector2Length(lat);
    lat = lat_len > 1e-3f ? Vector2Scale(lat, 1.0f / lat_len) : (Vector2){-1.0f, 0.0f};
    float gap = fmaxf(lat_len, GUI_FEAS_TEXT_GAP_PX);
    float sides[3] = {1.0f, 0.0f, -1.0f};
    for (int i = 0; i < 3; i++) {
        if (!has_lane[i])
            continue;
        Vector2 sp = Vector2Add(center, Vector2Scale(lat, sides[i] * gap));
        const char *label = gui_feasibility_label(vals[i], 1);
        int w = MeasureText(label, GUI_FONT);
        DrawRectangle((int)sp.x - w / 2 - 4, (int)sp.y - GUI_FONT / 2 - 3, w + 8, GUI_FONT + 6, Fade(BLACK, 0.65f));
        DrawText(label, (int)sp.x - w / 2, (int)sp.y - GUI_FONT / 2, GUI_FONT, gui_feasibility_color(vals[i]));
    }
}

static void gui_draw_goals(Drive *env) {
    if (g_gui.goal_mode == GUI_GOALS_OFF || env->control_mode == CONTROL_REPLAY_LOGS ||
        env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS)
        return;
    for (int i = 0; i < env->active_agent_count; i++) {
        if (g_gui.goal_mode == GUI_GOALS_SELECTED && i != g_gui.sel)
            continue;
        Entity *e = &env->entities[env->active_agent_indices[i]];
        if (e->removed || e->valid == 0)
            continue;
        Color color = gui_goal_color(env, i);
        DrawSphere((Vector3){e->goal_position_x, e->goal_position_y, 1.0f}, i == g_gui.sel ? 1.2f : 0.8f, color);
        DrawCircle3D((Vector3){e->goal_position_x, e->goal_position_y, Z_ROAD_MARKINGS}, env->goal_radius,
                     (Vector3){0, 0, 1}, 0.0f, Fade(color, 0.5f));
    }
}

static void gui_draw_box(const Entity *e, Color fill, Color wire, Color marker) {
    float box_h = e->height > 0.0f ? e->height : 1.5f;
    rlPushMatrix();
    rlTranslatef(e->x, e->y, box_h * 0.5f);
    rlRotatef(e->heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
    DrawCube((Vector3){0}, e->length, e->width, box_h, fill);
    DrawCubeWires((Vector3){0}, e->length, e->width, box_h, wire);
    if (marker.a > 0) {
        float top = box_h * 0.5f + 0.02f;
        float half_len = e->length * 0.5f;
        float tri_len = fminf(e->length * 0.45f, 2.0f);
        Vector3 tip = {half_len, 0.0f, top};
        Vector3 left = {half_len - tri_len, e->width * 0.4f, top};
        Vector3 right = {half_len - tri_len, -e->width * 0.4f, top};
        DrawTriangle3D(tip, left, right, marker);
        DrawTriangle3D(tip, right, left, marker);
    }
    rlPopMatrix();
}

static void gui_draw_agents(Drive *env) {
    int ghost = g_gui.obs_view;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type != VEHICLE && e->type != PEDESTRIAN && e->type != CYCLIST)
            continue;
        if (e->removed || e->x == INVALID_POSITION)
            continue;
        int active_idx = -1;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                active_idx = j;
                break;
            }
        }
        int is_static = 0;
        for (int j = 0; active_idx < 0 && !is_static && j < env->expert_static_agent_count; j++)
            is_static = env->expert_static_agent_indices[j] == i;
        for (int j = 0; active_idx < 0 && !is_static && j < env->static_agent_count; j++)
            is_static = env->static_agent_indices[j] == i;
        if (active_idx < 0 && !is_static)
            continue;

        int selected = active_idx >= 0 && active_idx == g_gui.sel;
        if (ghost && !selected) {
            float box_h = e->height > 0.0f ? e->height : 1.5f;
            rlPushMatrix();
            rlTranslatef(e->x, e->y, box_h * 0.5f);
            rlRotatef(e->heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
            DrawCubeWires((Vector3){0}, e->length, e->width, box_h, Fade(GRAY, 0.35f));
            rlPopMatrix();
            continue;
        }

        Color base = WHITE;
        if (active_idx < 0) {
            base = e->mark_as_expert == 1 ? EXPERT_REPLAY : GRAY;
        } else if (env->control_mode == CONTROL_REPLAY_LOGS) {
            base = EXPERT_REPLAY;
        } else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
            base = LIGHT_PURPLE;
        } else if (e->collision_state > 0) {
            base = RED;
        } else if (e->offroad_state > 0) {
            base = YELLOW;
        }
        Color marker = active_idx >= 0 ? gui_goal_color(env, active_idx) : BLANK;
        Color wire = selected ? PUFF_CYAN : Fade(DARKGRAY, 0.9f);
        gui_draw_box(e, Fade(base, active_idx >= 0 ? 0.85f : 0.6f), wire, marker);

        if (selected) {
            float r = fmaxf(e->length, e->width) * 0.75f;
            for (int k = 0; k < 3; k++) {
                DrawCircle3D((Vector3){e->x, e->y, 0.1f}, r + 0.15f * k, (Vector3){0, 0, 1}, 0.0f, PUFF_CYAN);
            }
        }
    }
}

static void gui_draw_faint_roads(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE)
            continue;
        Color color = road->type == ROAD_EDGE ? Fade(GRAY, 0.45f) : Fade(GRAY, 0.15f);
        for (int j = 0; j + 1 < road->array_size; j++) {
            if (road->traj_x[j] == INVALID_POSITION || road->traj_x[j + 1] == INVALID_POSITION)
                continue;
            DrawLine3D((Vector3){road->traj_x[j], road->traj_y[j], Z_ROAD_MARKINGS},
                       (Vector3){road->traj_x[j + 1], road->traj_y[j + 1], Z_ROAD_MARKINGS}, color);
        }
    }
}

static int gui_draw_world(Drive *env, Client *client) {
    int has_inferred = draw_verification_trajectories(env);
    if (g_gui.obs_view) {
        gui_draw_faint_roads(env);
    } else if (client->road_cache_valid) {
        draw_road_cached(client);
    }
    if (!g_gui.obs_view) {
        gui_draw_nav(env);
        gui_draw_goals(env);
    }
    if (g_gui.trails) {
        gui_draw_trails(env);
    }
    gui_draw_agents(env);
    gui_draw_feasibility_arrows(env);
    if (g_gui.obs_view && gui_selected_entity(env) != NULL) {
        draw_agent_obs(env, g_gui.sel, 1, 1, g_gui.lasers);
    }
    return has_inferred;
}

// ------------------------------------------------- HUD --------------------------------------------------------------

static const char *gui_control_mode_name(int control_mode) {
    switch (control_mode) {
    case CONTROL_VEHICLES:
        return "vehicles";
    case CONTROL_AGENTS:
        return "agents";
    case CONTROL_WOSAC:
        return "WOSAC";
    case CONTROL_SDC_ONLY:
        return "SDC only";
    case CONTROL_MIXED_PLAY:
        return "mixed play";
    case CONTROL_INFERRED_EXPERT_ACTIONS:
        return "inferred expert actions";
    case CONTROL_REPLAY_LOGS:
        return "replay logs";
    default:
        return "?";
    }
}

static const char *gui_type_name(int type) {
    return type == PEDESTRIAN ? "pedestrian" : type == CYCLIST ? "cyclist" : "vehicle";
}

static void gui_text(const char *text, int x, int *y, Color color) {
    DrawText(text, x, *y, GUI_FONT, color);
    *y += GUI_LINE;
}

static void gui_draw_action(Drive *env, int x, int *y) {
    if (env->actions == NULL || g_gui.sel < 0 || g_gui.sel >= env->active_agent_count)
        return;
    if (env->control_mode == CONTROL_REPLAY_LOGS)
        return;
    if (env->action_type == 0) {
        int action = ((int *)env->actions)[g_gui.sel];
        if (env->dynamics_model == CLASSIC) {
            int accel_idx = action / NUM_STEER_BINS;
            int steer_idx = action % NUM_STEER_BINS;
            if (accel_idx >= 0 && accel_idx < NUM_ACCEL_BINS && steer_idx >= 0) {
                gui_text(TextFormat("accel %.2f m/s^2  steer %.3f", ACCELERATION_VALUES[accel_idx],
                                    STEERING_VALUES[steer_idx]),
                         x, y, PUFF_WHITE);
                return;
            }
        } else if (env->dynamics_model == JERK) {
            int long_idx = action / 3;
            int lat_idx = action % 3;
            if (long_idx >= 0 && long_idx < 4 && lat_idx >= 0) {
                gui_text(TextFormat("jerk long %.1f  lat %.1f m/s^3", JERK_LONG[long_idx], JERK_LAT[lat_idx]), x, y,
                         PUFF_WHITE);
                return;
            }
        }
        gui_text(TextFormat("action %d", action), x, y, PUFF_WHITE);
    } else {
        int dim = env->dynamics_model == DELTA_LOCAL ? 3 : 2;
        float *a = &((float *)env->actions)[g_gui.sel * dim];
        if (dim == 3)
            gui_text(TextFormat("action %.2f  %.2f  %.2f", a[0], a[1], a[2]), x, y, PUFF_WHITE);
        else
            gui_text(TextFormat("accel %.2f  steer %.2f", a[0], a[1]), x, y, PUFF_WHITE);
    }
}

static void gui_draw_hud(Drive *env) {
    int sw = GetScreenWidth();
    int sh = GetScreenHeight();
    int x = 12;
    int y = 10;
    DrawRectangle(0, 0, 400, 10 * GUI_LINE + 16, Fade(BLACK, 0.45f));

    gui_text(TextFormat("t = %d / %d", env->timestep, env->episode_length), x, &y,
             g_gui.paused ? YELLOW : PUFF_WHITE);
    Color mode_color = env->control_mode == CONTROL_REPLAY_LOGS              ? EXPERT_REPLAY
                       : env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS ? LIGHT_PURPLE
                                                                              : PUFF_WHITE;
    gui_text(TextFormat("control: %s", gui_control_mode_name(env->control_mode)), x, &y, mode_color);
    const char *map_base = env->map_name != NULL ? strrchr(env->map_name, '/') : NULL;
    gui_text(TextFormat("map: %s  (J / K: prev / next)",
                        map_base != NULL ? map_base + 1 : (env->map_name != NULL ? env->map_name : "?")),
             x, &y, PUFF_WHITE);
    gui_text(TextFormat("camera: %s%s", GUI_CAM_NAMES[g_gui.cam_mode], g_gui.follow ? " (follow)" : ""), x, &y,
             PUFF_WHITE);

    Entity *agent = gui_selected_entity(env);
    if (agent != NULL) {
        Color color = gui_goal_color(env, g_gui.sel);
        DrawRectangle(x, y + 4, 12, 12, color);
        gui_text(TextFormat("   agent %d / %d  (entity %d, %s)", g_gui.sel, env->active_agent_count,
                            env->active_agent_indices[g_gui.sel], gui_type_name(agent->type)),
                 x, &y, PUFF_WHITE);
        const char *status = agent->collision_state > 0  ? "collision"
                             : agent->offroad_state > 0  ? "offroad"
                             : agent->current_goal_reached ? "goal reached"
                                                           : "ok";
        Color status_color = agent->collision_state > 0  ? RED
                             : agent->offroad_state > 0  ? YELLOW
                             : agent->current_goal_reached ? LIME
                                                           : PUFF_WHITE;
        gui_text(TextFormat("speed %.1f m/s   %s", sqrtf(agent->vx * agent->vx + agent->vy * agent->vy), status), x,
                 &y, status_color);
        gui_draw_action(env, x, &y);
        if (env->navigation != NULL && agent->navi_goal_idx >= 0 && agent->navi_lane_idx >= 0) {
            int has_left = env->navigation->left_lane[agent->navi_lane_idx] != -1;
            int has_right = env->navigation->right_lane[agent->navi_lane_idx] != -1;
            const char *left = gui_feasibility_label(agent->navi_feasibility_left_m, has_left);
            char left_buf[32];
            snprintf(left_buf, sizeof(left_buf), "%s", left); // TextFormat reuses a small ring of buffers
            const char *center = gui_feasibility_label(agent->navi_feasibility_center_m, 1);
            char center_buf[32];
            snprintf(center_buf, sizeof(center_buf), "%s", center);
            const char *right = gui_feasibility_label(agent->navi_feasibility_right_m, has_right);
            gui_text(TextFormat("feasible  L %s  C %s  R %s%s", left_buf, center_buf, right,
                                agent->navi_off_route ? "  (off route)" : ""),
                     x, &y, agent->navi_off_route ? ORANGE : PUFF_WHITE);
        }
    } else {
        gui_text("no agent selected", x, &y, GRAY);
    }
    gui_text(TextFormat("obs %s  nav %s  goals %s  trails %s", g_gui.obs_view ? "on" : "off",
                        env->navigation != NULL ? GUI_NAV_NAMES[g_gui.nav_mode] : "n/a",
                        GUI_GOAL_NAMES[g_gui.goal_mode], g_gui.trails ? "on" : "off"),
             x, &y, LIGHTGRAY);

    const char *hint = "H: help   SPACE: pause   ENTER: step   Q: quit";
    DrawText(hint, 12, sh - 28, GUI_FONT, Fade(PUFF_WHITE, 0.8f));
    DrawText(TextFormat("%d FPS", GetFPS()), sw - 80, sh - 28, GUI_FONT, Fade(PUFF_WHITE, 0.6f));

    if (g_gui.paused) {
        const char *msg = "PAUSED  -  SPACE resume, ENTER step";
        int w = MeasureText(msg, 24);
        DrawRectangle(sw / 2 - w / 2 - 12, sh - 80, w + 24, 36, Fade(BLACK, 0.55f));
        DrawText(msg, sw / 2 - w / 2, sh - 74, 24, YELLOW);
    }

    if (g_gui.help) {
        int key_w = 0;
        int desc_w = 0;
        int value_w = 0;
        for (int i = 0; i < GUI_NUM_BINDINGS; i++) {
            key_w = fmaxf(key_w, MeasureText(GUI_BINDINGS[i].keys, 18));
            desc_w = fmaxf(desc_w, MeasureText(GUI_BINDINGS[i].desc, 18));
            for (int k = 0; GUI_BINDINGS[i].state != NULL && GUI_BINDINGS[i].names[k] != NULL; k++) {
                value_w = fmaxf(value_w, MeasureText(GUI_BINDINGS[i].names[k], 18));
            }
        }
        int panel_w = 12 + key_w + 16 + desc_w + 16 + value_w + 12;
        int panel_h = GUI_NUM_BINDINGS * GUI_LINE + 44;
        int px = sw - panel_w - 12;
        int py = 12;
        DrawRectangle(px, py, panel_w, panel_h, Fade(BLACK, 0.7f));
        DrawRectangleLines(px, py, panel_w, panel_h, Fade(PUFF_CYAN, 0.8f));
        DrawText("Key bindings", px + 12, py + 10, GUI_FONT, PUFF_CYAN);
        int ly = py + 36;
        for (int i = 0; i < GUI_NUM_BINDINGS; i++) {
            const GuiBinding *b = &GUI_BINDINGS[i];
            DrawText(b->keys, px + 12, ly, 18, YELLOW);
            DrawText(b->desc, px + 12 + key_w + 16, ly, 18, PUFF_WHITE);
            if (b->state != NULL) {
                const char *value = b->names[*b->state];
                DrawText(value, px + panel_w - 12 - MeasureText(value, 18), ly, 18, LIME);
            }
            ly += GUI_LINE;
        }
    }

    if (g_gui.quit_confirm) {
        const char *msg = "Quit viewer?   ENTER / Y: yes    ESC / N: no";
        int w = MeasureText(msg, 24);
        DrawRectangle(0, 0, sw, sh, Fade(BLACK, 0.4f));
        DrawRectangle(sw / 2 - w / 2 - 20, sh / 2 - 30, w + 40, 60, Fade(BLACK, 0.85f));
        DrawRectangleLines(sw / 2 - w / 2 - 20, sh / 2 - 30, w + 40, 60, PUFF_RED);
        DrawText(msg, sw / 2 - w / 2, sh / 2 - 12, 24, PUFF_WHITE);
    }
}

// ------------------------------------------------- frame loop -------------------------------------------------------

static void gui_sync(Drive *env, Client *client) {
    if (!g_gui.initialized) {
        g_gui.initialized = 1;
        g_gui.cam_mode = GUI_CAM_BEV;
        g_gui.nav_mode = env->navigation != NULL ? GUI_NAV_ROUTE : GUI_NAV_OFF;
        g_gui.goal_mode = GUI_GOALS_ALL;
        g_gui.feas_mode = env->navigation != NULL ? GUI_FEAS_ARROWS : GUI_FEAS_OFF;
        g_gui.paused = 1;
        g_gui.drag_button = -1;
        g_gui.fpv_dist = 25.0f;
        g_gui.fpv_elevation = 0.45f;
        g_gui.sel = env->human_agent_idx;
    }
    if (g_gui.client_seen != client) {
        g_gui.client_seen = client;
        int follow = g_gui.follow;
        gui_autoframe(env);
        g_gui.follow = follow;
        gui_reset_trails(env);
        g_gui.fpv_valid = 0;
    }
    if (env->navigation == NULL) {
        g_gui.nav_mode = GUI_NAV_OFF;
        g_gui.feas_mode = GUI_FEAS_OFF;
    }
    if (!gui_agent_selectable(env, g_gui.sel)) {
        if (g_gui.sel < 0 || g_gui.sel >= env->active_agent_count)
            g_gui.sel = 0;
        if (env->active_agent_count > 0 && !gui_agent_selectable(env, g_gui.sel))
            gui_cycle_selection(env, 1);
    }
    env->human_agent_idx = g_gui.sel;
    gui_update_trails(env);
}

static void gui_frame(Drive *env, Client *client) {
    if (WindowShouldClose())
        gui_quit();
    client->width = (float)GetScreenWidth();
    client->height = (float)GetScreenHeight();

    Camera3D camera = gui_build_camera();
    gui_handle_keys(env, camera);
    if (!g_gui.quit_confirm)
        gui_handle_mouse(env, camera);
    gui_apply_inertia(env);
    env->human_agent_idx = g_gui.sel;
    camera = gui_build_camera();
    g_gui.camera = camera;
    client->camera = camera;

    BeginDrawing();
    ClearBackground(ROAD_COLOR);
    BeginMode3D(camera);
    int has_inferred = gui_draw_world(env, client);
    EndMode3D();
    if (has_inferred)
        draw_ade_labels(env, camera, GetScreenWidth(), GetScreenHeight());
    gui_draw_feasibility_text(env, camera);
    gui_draw_hud(env);
    EndDrawing();
}

// Returns and clears the pending map change (-1 previous, +1 next, 0 none). Python rebuilds the envs on request.
int gui_take_map_request(void) {
    int request = g_gui.map_request;
    g_gui.map_request = 0;
    return request;
}

void render_popup(Drive *env, Client *client) {
    gui_sync(env, client);
    g_gui.step_requested = 0;

    // While paused we block here; let Ctrl+C through to Python as a KeyboardInterrupt.
    struct sigaction sa = {0};
    struct sigaction old_sa;
    sa.sa_handler = gui_sigint_handler;
    sigemptyset(&sa.sa_mask);
    g_gui_sigint = 0;
    int installed = sigaction(SIGINT, &sa, &old_sa) == 0;

    do {
        gui_frame(env, client);
    } while (g_gui.paused && !g_gui.step_requested && !g_gui.map_request && !g_gui_sigint);

    if (installed)
        sigaction(SIGINT, &old_sa, NULL);
    if (g_gui_sigint) {
        g_gui_sigint = 0;
        raise(SIGINT);
    }
}

void c_render(Drive *env, int view_mode, int draw_traces) {

    // Create client on first render call
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (env->client == NULL) {
        return; // Silently skip rendering
    }

    Client *client = env->client;

    if (!client->road_cache_valid) {
        build_road_cache(env, client);
    }

    if (env->render_mode == RENDER_HEADLESS && client->recorder_pid <= 0) {
        start_recorder(client, env->scenario_id, (int)client->width, (int)client->height);
    }

    if (env->render_mode == RENDER_HEADLESS) { // Headless rendering via ffmpeg
        Camera3D camera = {0};

        if (view_mode == VIEW_MODE_SIM_STATE) {
            if (env->sdc_track_index >= 0 && env->control_mode == CONTROL_SDC_ONLY) {
                // Follow the SDC agent (zoomed-in tracking shot)
                Entity *sdc = &env->entities[env->sdc_track_index];
                camera.position = (Vector3){sdc->x, sdc->y, 400.0f};
                camera.target = (Vector3){sdc->x, sdc->y, 0.0f};
                camera.up = (Vector3){0.0f, -1.0f, 0.0f};
                camera.projection = CAMERA_ORTHOGRAPHIC;
                camera.fovy = 75.0f;
            } else {
                // Orthographic bird's-eye view over the entire map.
                float map_width = fabsf(env->grid_map->bottom_right_x - env->grid_map->top_left_x);
                float map_height = fabsf(env->grid_map->top_left_y - env->grid_map->bottom_right_y);
                float map_center_x = (env->grid_map->top_left_x + env->grid_map->bottom_right_x) * 0.5f;
                float map_center_y = (env->grid_map->top_left_y + env->grid_map->bottom_right_y) * 0.5f;
                float aspect = client->width / client->height;
                float fovy = fmaxf(map_height, map_width / aspect) * 1.05f;
                camera.position = (Vector3){map_center_x, map_center_y, 400.0f};
                camera.target = (Vector3){map_center_x, map_center_y, 0.0f};
                camera.up = (Vector3){0.0f, -1.0f, 0.0f};
                camera.projection = CAMERA_ORTHOGRAPHIC;
                camera.fovy = fovy;
            }
            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);

            if (draw_traces) { // Show logged trajectories of active agents and expert static agents
                for (int i = 0; i < env->active_agent_count; i++) {
                    int idx = env->active_agent_indices[i];
                    int t_end = env->episode_length;
                    if (t_end > env->entities[idx].array_size)
                        t_end = env->entities[idx].array_size;
                    for (int t = env->init_steps; t < t_end; t++) {
                        if (env->entities[idx].traj_valid && !env->entities[idx].traj_valid[t])
                            continue;
                        Color agent_color = LIGHTBLUE;
                        if (env->entities[idx].type == PEDESTRIAN) {
                            agent_color = LIGHT_ORANGE;
                        } else if (env->entities[idx].type == CYCLIST) {
                            agent_color = LIGHT_PURPLE;
                        }
                        DrawPoint3D(
                            (Vector3){env->entities[idx].traj_x[t], env->entities[idx].traj_y[t], Z_AGENT_DETAILS},
                            agent_color);
                    }
                }

                for (int i = 0; i < env->expert_static_agent_count; i++) {
                    int idx = env->expert_static_agent_indices[i];
                    int t_end = env->episode_length;
                    if (t_end > env->entities[idx].array_size)
                        t_end = env->entities[idx].array_size;
                    for (int t = env->init_steps; t < t_end; t++) {
                        if (env->entities[idx].traj_valid && !env->entities[idx].traj_valid[t])
                            continue;
                        DrawPoint3D(
                            (Vector3){env->entities[idx].traj_x[t], env->entities[idx].traj_y[t], Z_AGENT_DETAILS},
                            EXPERT_REPLAY);
                    }
                }
            }

            draw_scene(env, client, 1, 0, 0, 0);

            // Draw timestep
            DrawText(TextFormat("t=%d", env->timestep), 10, 10, 26, PUFF_WHITE);

        } else if (view_mode == VIEW_MODE_BEV_AGENT_OBS) {
            // Orthographic bird's-eye view centered on the selected agent,
            // showing only that agent's observations
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            camera.position = (Vector3){agent->x, agent->y, 400.0f};
            camera.target = (Vector3){agent->x, agent->y, 0.0f};
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = env->grid_map->vision_range * GRID_CELL_SIZE * 2.0f;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);
            draw_scene(env, client, 1, 1, 0, 0);

        } else if (view_mode == VIEW_MODE_SENSOR_NOISE) {
            // Sensor noise visualization: BEV centered on SDC showing
            // ground truth (green) vs noisy perception (purple)
            int sdc_array_idx = 0;
            if (env->sdc_track_index >= 0) {
                for (int i = 0; i < env->active_agent_count; i++) {
                    if (env->active_agent_indices[i] == env->sdc_track_index) {
                        sdc_array_idx = i;
                        break;
                    }
                }
            }
            int ego_idx = env->active_agent_indices[sdc_array_idx];
            Entity *sdc = &env->entities[ego_idx];

            camera.position = (Vector3){sdc->x, sdc->y, 400.0f};
            camera.target = (Vector3){sdc->x, sdc->y, 0.0f};
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = 100.0f;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);

            // Draw roads first
            for (int i = 0; i < env->num_entities; i++) {
                if (env->entities[i].type < ROAD_LANE || env->entities[i].type > ROAD_EDGE)
                    continue;
                for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                    Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], Z_ROAD_MARKINGS};
                    Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], Z_ROAD_MARKINGS};
                    if (env->entities[i].type == ROAD_EDGE)
                        draw_road_edge(env, start.x, start.y, end.x, end.y);
                    else {
                        Color c = (env->entities[i].type == ROAD_LANE) ? Fade(SOFT_YELLOW, 0.25f) : WHITE;
                        rlSetLineWidth(2.0f);
                        DrawLine3D(start, end, c);
                    }
                }
            }

            draw_sensor_noise_view(env, client);
            EndMode3D();

        } else { // First-person perspective from a selected agent
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            // Position camera behind and above the agent
            camera.position =
                (Vector3){agent->x - (25.0f * cosf(agent->heading)), agent->y - (25.0f * sinf(agent->heading)), 15.0f};
            camera.target =
                (Vector3){agent->x + 40.0f * cosf(agent->heading), agent->y + 40.0f * sinf(agent->heading), 1.0f};
            camera.up = (Vector3){0.0f, 0.0f, 1.0f};
            camera.fovy = 60.0f;
            camera.projection = CAMERA_PERSPECTIVE;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);
            draw_scene(env, client, 0, 0, 0, 1);
        }

#ifdef DRIVE_HAS_EGL
        if (client->egl_mode) {
            rlDrawRenderBatchActive();
        } else
#endif
        {
            EndDrawing();
        }

        int w = (int)client->width;
        int h = (int)client->height;
        int frame_bytes = w * h * 4;

#ifdef DRIVE_HAS_EGL
        if (client->egl_mode) {
            if (client->pbo[0] == 0) {
                glGenBuffers(2, client->pbo);
                for (int i = 0; i < 2; i++) {
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[i]);
                    glBufferData(GL_PIXEL_PACK_BUFFER, frame_bytes, NULL, GL_STREAM_READ);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            int curr = client->pbo_index;
            int prev = 1 - curr;

            glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[curr]);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

            if (client->pbo_frame_count > 0) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
                unsigned char *ptr = (unsigned char *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                if (ptr) {
                    write_gl_frame_flipped(client->recorder_pipefd[1], ptr, w, h);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                } else {
                    fprintf(stderr, "[drive-pbo] frame=%d glMapBuffer returned NULL, GL error=0x%x\n",
                            client->pbo_frame_count, glGetError());
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            client->pbo_index = prev;
            client->pbo_frame_count++;
        } else
#endif
        {
            unsigned char *screen_data = rlReadScreenPixels(w, h);
            if (screen_data) {
                write_all_bytes(client->recorder_pipefd[1], screen_data, (size_t)frame_bytes);
                RL_FREE(screen_data);
            }
        }
    } else { // Pop-up window
        render_popup(env, client);
    }
}

void close_client(Client *client) {
    if (g_gui.client_seen == client)
        g_gui.client_seen = NULL; // a new client may reuse this address; force re-framing
#ifdef DRIVE_HAS_EGL
    if (client->egl_mode && client->pbo_frame_count > 0 && client->pbo[0] != 0) {
        int prev = 1 - client->pbo_index;
        int w = (int)client->width;
        int h = (int)client->height;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
        unsigned char *ptr = (unsigned char *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (ptr) {
            write_gl_frame_flipped(client->recorder_pipefd[1], ptr, w, h);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        } else {
            fprintf(stderr, "[drive-pbo] final glMapBuffer returned NULL, GL error=0x%x\n", glGetError());
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glDeleteBuffers(2, client->pbo);
    }
#endif

    if (client->road_cache_valid && client->road_tri_mesh.vertexCount > 0) {
        UnloadMesh(client->road_tri_mesh);
    }
    free(client->road_line_verts);
    free(client->road_line_colors);

    stop_recorder(client);

    if (client->render_mode == RENDER_HEADLESS) {
        for (int i = 0; i < 6; i++)
            UnloadModel(client->cars[i]);
        UnloadModel(client->cyclist);
        UnloadModel(client->pedestrian);
    }

    free(client);
}
