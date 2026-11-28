// znavi_build.h - build zpufferdrive navigation tables (routes, distances, lane feasibilities, goals) from a
// map's polylines at load time, instead of reading them from a pre-processed zbin (Sections 3-4).
//
// C port of zpufferdrive/scripts/python/goal.py (Goals) as driven by load_map.py. Parity is behavioural: float
// rounding and tie-breaking can differ, and the random draws (feasibility bonus, random goals) use a different
// generator than numpy.
//
// The header is simulator-agnostic: it depends on libc/libm only and takes plain arrays, so it can be dropped into
// other PufferDrive forks. Include it once (all functions are static).
//
// Usage:
//   ZNaviMapInput map = {...};           // polylines (CSR points), type ids, speed limits, type role table
//   ZNaviAgentInput agents = {...};      // optional: anchors and specified goals of the agents to route
//   ZNaviBuildConfig cfg = znavi_default_config();
//   ZNaviTables t;
//   znavi_build(&map, &agents, &cfg, &t);
//   ... t.next_sample[g * t.num_samples + s], t.distance_to_goal[...], t.agent_goal[a] ...
//   znavi_tables_free(&t);
//
// Nodes ("samples") are sub-lines: each segment k of a lane polyline p is a navi line, split uniformly into
// line_num_sub[line] = ceil(length / (discretization_time * max(speed_limit, 1))) sub-lines. Samples are ordered by
// line, lines by (polyline, segment). Boundary polylines (markers / edges) are obstacles for lateral connections:
// a lane change may cross a boundary only if its type is crossable; lane markers (crossable or not) also mark
// forward connections that are really lane changes.
//
// Differences to the Python pipeline: no 5000-point road graph truncation; a lane speed limit <= 0 uses
// default_speed_limit (Python: 0, i.e. 0.75 m sub-lines and no lane changes); dead-end goals beyond max_goals are
// dropped; Python's robust left-perp connections (Section 3, zdrive undertake only) are not produced.

#ifndef ZNAVI_BUILD_H
#define ZNAVI_BUILD_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum {
    ZNB_ROLE_IGNORE = 0,
    ZNB_ROLE_LANE = 1,   // lane center line: graph nodes
    ZNB_ROLE_MARKER = 2, // lane marker (solid/dashed line between lanes)
    ZNB_ROLE_EDGE = 3,   // road edge / curb
} ZNaviRole;

typedef struct {
    int num_polylines;
    const int *point_offset;             // [num_polylines + 1]: points of polyline i are [offset[i], offset[i + 1])
    const float *x, *y;                  // [point_offset[num_polylines]] any common origin
    const int *type;                     // [num_polylines] caller type id in [0, num_types), others are ignored
    const float *speed_limit;            // [num_polylines] m/s, NULL or <= 0 -> default_speed_limit
    int num_types;                       // caller (fork) specific type table:
    const ZNaviRole *type_role;          // [num_types]
    const unsigned char *type_crossable; // [num_types] 1: lane changes may cross it (NULL: none crossable)
} ZNaviMapInput;

// Agents to assign goals to (all are treated as controlled). Any pointer except x/y/heading may be NULL.
typedef struct {
    int num_agents;
    const float *x, *y, *heading;                            // primary anchor (initial pose)
    const float *fallback_x, *fallback_y, *fallback_heading; // used when the primary anchor has no start line
    const float *length;                                     // > long_vehicle_length: no lane-change goals
    const float *goal_x, *goal_y;                            // specified goal, NaN = none
} ZNaviAgentInput;

typedef struct {
    // Features (1 = on)
    int perp_connections;          // F1 perpendicular/adjacent connections (side feasibilities, lane-change prune)
    int lane_changes;              // F2 lane-change connections in the navigation graph
    int gap_probe;                 // F3 lane-change gap/median probes
    int feasibility;               // F4 lane feasibilities (else 0)
    int dead_end_goals;            // F5 dead-end goals besides the specified ones
    int assign_goals;              // F6 agent goal assignment (agent_goal), else -1
    int assign_random_goals;       //    random reachable goal when the specified one is missing/unreachable
    int long_vehicle_rule;         // F6a long vehicles only get goals reachable without lane change
    int fallback_anchor;           // F6b retry from the fallback anchor
    int retain_unreachable_goals;  // F6c keep specified goals even when unreachable
    int virtual_goal_reachability; // F7 lane-change margins for goal reachability
    int marker_crossings;          // F8 drop forward connections crossing a lane marker from the no-lane-change graph
    int snap_goals;                // specified goal = closest sub-line only (else all within goal_set_radius)
    int goal_set_along_lane_cost;  // non-snapped goal sets: seed each member with its along-lane distance to the
                                   // goal (relative to the closest member) instead of 0 (Python), so a lane change
                                   // into a neighbouring lane's earlier-starting goal sub-line does not win
    int fixed_step_sublines;       // sub-lines of length step plus a shorter remainder (Python), else equal
                                   // lengths (matches the uniform sample positions of znavi_point)
    int max_goals;
    int verbose;
    uint64_t seed;
    // Constants (defaults = Python values)
    double discretization_time;   // s, sub-line length = time * max(speed, 1)
    double default_speed_limit;   // m/s, used for speed limits <= 0
    double cell_size;             // m, obstruction grid
    double connection_threshold;  // m, forward connection end-start gap
    double connection_max_angle;  // rad
    double unreachable_distance;  // m, distance_to_goal of unreachable samples
    double lane_change_cost;      // m, routing cost added to lane changes
    double goal_dedup_distance;   // m, dead-end goals closer than this to an accepted one are dropped
    double goal_set_radius;       // m, non-snapped goal set radius
    double lane_change_margin;    // m, F7 first/last lane-change margin, F4 last lane-change margin
    double max_route_deviation;   // m, F4 cap above the distance to goal
    double feasibility_bonus_min; // m, F4 random bonus on the goal's own lane
    double feasibility_bonus_max;
    double start_max_distance;  // m, agent start line search
    double start_max_angle;     // rad
    double long_vehicle_length; // m
} ZNaviBuildConfig;

typedef struct {
    int num_lines;           // navi lines = lane segments
    int *line_polyline;      // [num_lines]
    int *line_segment;       // [num_lines] segment index within the polyline
    int *line_num_sub;       // [num_lines] sub-lines (samples) of the line, may be 0
    int *line_sample_offset; // [num_lines]
    int num_samples;
    int *sample_line;  // [num_samples]
    int *sample_index; // [num_samples] index within its line
    int num_goals;
    float *goal_x, *goal_y; // [num_goals]
    int *goal_sample;       // [num_goals] primary goal sample
    // [num_goals * num_samples], index g * num_samples + s
    int *next_sample;        // next sample on the route, -1 at the goal or when unreachable
    float *distance_to_goal; // m, unreachable_distance when unreachable
    float *feasibility_left, *feasibility_center, *feasibility_right; // m, unclamped (may be inf)
    int num_agents;
    int *agent_goal; // [num_agents] goal index or -1
} ZNaviTables;

static ZNaviBuildConfig znavi_default_config(void) {
    ZNaviBuildConfig c;
    memset(&c, 0, sizeof(c));
    c.perp_connections = c.lane_changes = c.gap_probe = c.feasibility = c.dead_end_goals = 1;
    c.assign_goals = c.assign_random_goals = c.long_vehicle_rule = c.fallback_anchor = 1;
    c.retain_unreachable_goals = 0;
    c.virtual_goal_reachability = c.marker_crossings = c.snap_goals = 1;
    c.fixed_step_sublines = 1;
    c.max_goals = 256;
    c.seed = 42;
    c.discretization_time = 0.75;
    c.default_speed_limit = 13.89;
    c.cell_size = 4.0;
    c.connection_threshold = 0.2;
    c.connection_max_angle = M_PI / 4;
    c.unreachable_distance = 2000.0;
    c.lane_change_cost = 10.0;
    c.goal_dedup_distance = 1.0;
    c.goal_set_radius = 1.5;
    c.goal_set_along_lane_cost = 1;
    c.lane_change_margin = 30.0;
    c.max_route_deviation = 30.0;
    c.feasibility_bonus_min = 100.0;
    c.feasibility_bonus_max = 500.0;
    c.start_max_distance = 5.0;
    c.start_max_angle = M_PI / 2;
    c.long_vehicle_length = 8.0;
    return c;
}

// ------------------------------------------------------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------------------------------------------------------

#define ZNB_INF HUGE_VAL
#define ZNB_MIN(a, b) ((a) < (b) ? (a) : (b))
#define ZNB_MAX(a, b) ((a) > (b) ? (a) : (b))

static void *znb_alloc(size_t n, size_t size) {
    void *p = calloc(n > 0 ? n : 1, size);
    if (p == NULL) {
        fprintf(stderr, "znavi_build: out of memory (%zu bytes)\n", n * size);
        abort();
    }
    return p;
}

static void *znb_realloc(void *p, size_t n, size_t size) {
    p = realloc(p, (n > 0 ? n : 1) * size);
    if (p == NULL) {
        fprintf(stderr, "znavi_build: out of memory (%zu bytes)\n", n * size);
        abort();
    }
    return p;
}

typedef struct {
    int *a;
    int n, cap;
} ZnbIVec;

static void znb_ipush(ZnbIVec *v, int x) {
    if (v->n == v->cap) {
        v->cap = v->cap ? 2 * v->cap : 16;
        v->a = (int *)znb_realloc(v->a, v->cap, sizeof(int));
    }
    v->a[v->n++] = x;
}

static int znb_cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static uint64_t znb_rng_next(uint64_t *s) { // splitmix64
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double znb_rng_uniform(uint64_t *s) { return (double)(znb_rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }

static double znb_wrap(double a) {
    double r = fmod(a + M_PI, 2 * M_PI);
    if (r < 0)
        r += 2 * M_PI;
    return r - M_PI;
}

// Segment a0-a1 vs b0-b1; *ua = parameter along a. Parallel segments never intersect.
static int znb_intersect(double a0x, double a0y, double a1x, double a1y, double b0x, double b0y, double b1x, double b1y,
                         double *ua) {
    double adx = a1x - a0x, ady = a1y - a0y, bdx = b1x - b0x, bdy = b1y - b0y;
    double denom = bdy * adx - bdx * ady;
    if (fabs(denom) <= 1e-6)
        return 0;
    double dx = a0x - b0x, dy = a0y - b0y;
    double u = (bdx * dy - bdy * dx) / denom;
    double v = (adx * dy - ady * dx) / denom;
    if (u < 0 || u > 1 || v < 0 || v > 1)
        return 0;
    *ua = u;
    return 1;
}

static double znb_point_segment_distance(double px, double py, double x0, double y0, double x1, double y1) {
    double dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
    double t = l2 > 0 ? ((px - x0) * dx + (py - y0) * dy) / l2 : 0;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return hypot(px - (x0 + t * dx), py - (y0 + t * dy));
}

// Binary min-heap with lazy deletion.
typedef struct {
    double *k;
    int *v;
    int n, cap;
} ZnbHeap;

static void znb_hpush(ZnbHeap *h, double k, int v) {
    if (h->n == h->cap) {
        h->cap = h->cap ? 2 * h->cap : 64;
        h->k = (double *)znb_realloc(h->k, h->cap, sizeof(double));
        h->v = (int *)znb_realloc(h->v, h->cap, sizeof(int));
    }
    int i = h->n++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->k[p] <= k)
            break;
        h->k[i] = h->k[p];
        h->v[i] = h->v[p];
        i = p;
    }
    h->k[i] = k;
    h->v[i] = v;
}

static void znb_hpop(ZnbHeap *h, double *k, int *v) {
    *k = h->k[0];
    *v = h->v[0];
    if (--h->n == 0)
        return;
    double lk = h->k[h->n];
    int lv = h->v[h->n], i = 0;
    for (;;) {
        int l = 2 * i + 1;
        if (l >= h->n)
            break;
        int m = (l + 1 < h->n && h->k[l + 1] < h->k[l]) ? l + 1 : l;
        if (h->k[m] >= lk)
            break;
        h->k[i] = h->k[m];
        h->v[i] = h->v[m];
        i = m;
    }
    h->k[i] = lk;
    h->v[i] = lv;
}

typedef struct {
    int n;
    int *off; // [n + 1]
    int *dst;
    double *w;
} ZnbCsr;

// Edges keep their input order within a source row.
static ZnbCsr znb_csr(int n, int m, const int *src, const int *dst, const double *w) {
    ZnbCsr g;
    g.n = n;
    g.off = (int *)znb_alloc(n + 1, sizeof(int));
    g.dst = (int *)znb_alloc(m, sizeof(int));
    g.w = (double *)znb_alloc(m, sizeof(double));
    for (int e = 0; e < m; e++)
        g.off[src[e] + 1]++;
    for (int i = 0; i < n; i++)
        g.off[i + 1] += g.off[i];
    int *pos = (int *)znb_alloc(n + 1, sizeof(int));
    memcpy(pos, g.off, (n + 1) * sizeof(int));
    for (int e = 0; e < m; e++) {
        int k = pos[src[e]]++;
        g.dst[k] = dst[e];
        g.w[k] = w ? w[e] : 1.0;
    }
    free(pos);
    return g;
}

static ZnbCsr znb_csr_transpose(const ZnbCsr *g) {
    int m = g->off[g->n];
    int *src = (int *)znb_alloc(m, sizeof(int));
    for (int u = 0; u < g->n; u++)
        for (int e = g->off[u]; e < g->off[u + 1]; e++)
            src[e] = u;
    ZnbCsr t = znb_csr(g->n, m, g->dst, src, g->w);
    free(src);
    return t;
}

static void znb_csr_free(ZnbCsr *g) {
    free(g->off);
    free(g->dst);
    free(g->w);
    memset(g, 0, sizeof(*g));
}

static int znb_csr_has(const ZnbCsr *g, int u, int v) {
    for (int e = g->off[u]; e < g->off[u + 1]; e++)
        if (g->dst[e] == v)
            return 1;
    return 0;
}

// Multi-source Dijkstra; next[v] = the node v was relaxed from (on a reversed graph: v's next hop to the sources).
static void znb_dijkstra_seeded(const ZnbCsr *g, const int *src, const double *seed, int ns, double *dist, int *next,
                                ZnbHeap *h) {
    for (int i = 0; i < g->n; i++) {
        dist[i] = ZNB_INF;
        if (next)
            next[i] = -1;
    }
    h->n = 0;
    for (int s = 0; s < ns; s++) {
        double d0 = seed ? seed[s] : 0.0;
        if (d0 < dist[src[s]]) {
            dist[src[s]] = d0;
            znb_hpush(h, d0, src[s]);
        }
    }
    while (h->n > 0) {
        double d;
        int u;
        znb_hpop(h, &d, &u);
        if (d > dist[u])
            continue;
        for (int e = g->off[u]; e < g->off[u + 1]; e++) {
            int v = g->dst[e];
            double nd = d + g->w[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                if (next)
                    next[v] = u;
                znb_hpush(h, nd, v);
            }
        }
    }
}

static void znb_dijkstra(const ZnbCsr *g, const int *src, int ns, double *dist, int *next, ZnbHeap *h) {
    znb_dijkstra_seeded(g, src, NULL, ns, dist, next, h);
}

// ------------------------------------------------------------------------------------------------------------------
// Spatial grids
// ------------------------------------------------------------------------------------------------------------------

#define ZNB_MAX_GRID_CELLS 4000000

typedef struct {
    double ox, oy, cs;
    int cols, rows;
    int *off; // [cells + 1]
    int *idx;
} ZnbPtGrid;

static void znb_grid_frame(double minx, double miny, double maxx, double maxy, double cs, double *ox, double *oy,
                           double *out_cs, int *cols, int *rows) {
    if (!(maxx >= minx)) {
        minx = maxx = miny = maxy = 0;
    }
    for (;;) {
        double c = floor((maxx - minx) / cs) + 1, r = floor((maxy - miny) / cs) + 1;
        if (c * r <= ZNB_MAX_GRID_CELLS) {
            *cols = (int)c;
            *rows = (int)r;
            break;
        }
        cs *= 2;
    }
    *ox = minx;
    *oy = miny;
    *out_cs = cs;
}

static int znb_cell_coord(double v, double o, double cs, int count) {
    int c = (int)floor((v - o) / cs);
    return c < 0 ? 0 : (c >= count ? count - 1 : c);
}

static void znb_ptgrid_build(ZnbPtGrid *g, int n, const double *x, const double *y, double cs) {
    double minx = ZNB_INF, miny = ZNB_INF, maxx = -ZNB_INF, maxy = -ZNB_INF;
    for (int i = 0; i < n; i++) {
        minx = ZNB_MIN(minx, x[i]);
        maxx = ZNB_MAX(maxx, x[i]);
        miny = ZNB_MIN(miny, y[i]);
        maxy = ZNB_MAX(maxy, y[i]);
    }
    znb_grid_frame(minx, miny, maxx, maxy, cs, &g->ox, &g->oy, &g->cs, &g->cols, &g->rows);
    int cells = g->cols * g->rows;
    g->off = (int *)znb_alloc(cells + 1, sizeof(int));
    g->idx = (int *)znb_alloc(n, sizeof(int));
    int *cell = (int *)znb_alloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        cell[i] = znb_cell_coord(y[i], g->oy, g->cs, g->rows) * g->cols + znb_cell_coord(x[i], g->ox, g->cs, g->cols);
        g->off[cell[i] + 1]++;
    }
    for (int c = 0; c < cells; c++)
        g->off[c + 1] += g->off[c];
    int *pos = (int *)znb_alloc(cells + 1, sizeof(int));
    memcpy(pos, g->off, (cells + 1) * sizeof(int));
    for (int i = 0; i < n; i++)
        g->idx[pos[cell[i]]++] = i;
    free(pos);
    free(cell);
}

// All points in cells overlapping the square [x - r, x + r] x [y - r, y + r] (superset of the disc).
static void znb_ptgrid_query(const ZnbPtGrid *g, double x, double y, double r, ZnbIVec *out) {
    out->n = 0;
    int cx0 = znb_cell_coord(x - r, g->ox, g->cs, g->cols), cx1 = znb_cell_coord(x + r, g->ox, g->cs, g->cols);
    int cy0 = znb_cell_coord(y - r, g->oy, g->cs, g->rows), cy1 = znb_cell_coord(y + r, g->oy, g->cs, g->rows);
    for (int cy = cy0; cy <= cy1; cy++)
        for (int cx = cx0; cx <= cx1; cx++) {
            int c = cy * g->cols + cx;
            for (int k = g->off[c]; k < g->off[c + 1]; k++)
                znb_ipush(out, g->idx[k]);
        }
}

static void znb_ptgrid_free(ZnbPtGrid *g) {
    free(g->off);
    free(g->idx);
}

// Segment grid: each segment is stored in every cell it touches (supercover, small margin), so ray queries only need
// the cells of the ray.
typedef struct {
    double ox, oy, cs;
    int cols, rows;
    int *off, *item;
    int n;
    double *x0, *y0, *x1, *y1;
    int *stamp;
    int cur;
} ZnbSegGrid;

typedef struct {
    int item;
    double ua;
} ZnbHit;

typedef struct {
    ZnbHit *a;
    int n, cap;
} ZnbHitVec;

static int znb_clip(double x0, double y0, double x1, double y1, double xmin, double ymin, double xmax, double ymax) {
    double t0 = 0, t1 = 1, dx = x1 - x0, dy = y1 - y0;
    double p[4] = {-dx, dx, -dy, dy}, q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};
    for (int k = 0; k < 4; k++) {
        if (p[k] == 0) {
            if (q[k] < 0)
                return 0;
        } else {
            double r = q[k] / p[k];
            if (p[k] < 0) {
                if (r > t1)
                    return 0;
                if (r > t0)
                    t0 = r;
            } else {
                if (r < t0)
                    return 0;
                if (r < t1)
                    t1 = r;
            }
        }
    }
    return 1;
}

static void znb_seg_cells(const ZnbSegGrid *g, double x0, double y0, double x1, double y1, ZnbIVec *out) {
    const double eps = 1e-4;
    out->n = 0;
    double u0 = (x0 - g->ox) / g->cs, v0 = (y0 - g->oy) / g->cs, u1 = (x1 - g->ox) / g->cs, v1 = (y1 - g->oy) / g->cs;
    double umin = ZNB_MIN(u0, u1) - eps, umax = ZNB_MAX(u0, u1) + eps;
    double vmin = ZNB_MIN(v0, v1) - eps, vmax = ZNB_MAX(v0, v1) + eps;
    if (umax < 0 || vmax < 0 || umin >= g->cols || vmin >= g->rows)
        return;
    int cx0 = ZNB_MAX(0, (int)floor(umin)), cx1 = ZNB_MIN(g->cols - 1, (int)floor(umax));
    int cy0 = ZNB_MAX(0, (int)floor(vmin)), cy1 = ZNB_MIN(g->rows - 1, (int)floor(vmax));
    for (int cy = cy0; cy <= cy1; cy++)
        for (int cx = cx0; cx <= cx1; cx++)
            if (znb_clip(u0, v0, u1, v1, cx - eps, cy - eps, cx + 1 + eps, cy + 1 + eps))
                znb_ipush(out, cy * g->cols + cx);
}

static void znb_seggrid_build(ZnbSegGrid *g, int n, const double *x0, const double *y0, const double *x1,
                              const double *y1, double cs, double minx, double miny, double maxx, double maxy) {
    znb_grid_frame(minx, miny, maxx, maxy, cs, &g->ox, &g->oy, &g->cs, &g->cols, &g->rows);
    g->n = n;
    size_t bytes = (size_t)n * sizeof(double);
    g->x0 = (double *)znb_alloc(n, sizeof(double));
    g->y0 = (double *)znb_alloc(n, sizeof(double));
    g->x1 = (double *)znb_alloc(n, sizeof(double));
    g->y1 = (double *)znb_alloc(n, sizeof(double));
    if (n > 0) {
        memcpy(g->x0, x0, bytes);
        memcpy(g->y0, y0, bytes);
        memcpy(g->x1, x1, bytes);
        memcpy(g->y1, y1, bytes);
    }
    g->stamp = (int *)znb_alloc(n, sizeof(int));
    g->cur = 0;
    int cells = g->cols * g->rows;
    g->off = (int *)znb_alloc(cells + 1, sizeof(int));
    ZnbIVec buf = {0};
    for (int i = 0; i < n; i++) {
        znb_seg_cells(g, x0[i], y0[i], x1[i], y1[i], &buf);
        for (int k = 0; k < buf.n; k++)
            g->off[buf.a[k] + 1]++;
    }
    for (int c = 0; c < cells; c++)
        g->off[c + 1] += g->off[c];
    g->item = (int *)znb_alloc(g->off[cells], sizeof(int));
    int *pos = (int *)znb_alloc(cells + 1, sizeof(int));
    memcpy(pos, g->off, (cells + 1) * sizeof(int));
    for (int i = 0; i < n; i++) {
        znb_seg_cells(g, x0[i], y0[i], x1[i], y1[i], &buf);
        for (int k = 0; k < buf.n; k++)
            g->item[pos[buf.a[k]]++] = i;
    }
    free(pos);
    free(buf.a);
}

static void znb_seggrid_query(ZnbSegGrid *g, double ax, double ay, double bx, double by, ZnbHitVec *out,
                              ZnbIVec *cells) {
    out->n = 0;
    znb_seg_cells(g, ax, ay, bx, by, cells);
    if (++g->cur == 0x7fffffff) {
        memset(g->stamp, 0, (size_t)g->n * sizeof(int));
        g->cur = 1;
    }
    for (int k = 0; k < cells->n; k++) {
        int c = cells->a[k];
        for (int e = g->off[c]; e < g->off[c + 1]; e++) {
            int it = g->item[e];
            if (g->stamp[it] == g->cur)
                continue;
            g->stamp[it] = g->cur;
            double ua;
            if (znb_intersect(ax, ay, bx, by, g->x0[it], g->y0[it], g->x1[it], g->y1[it], &ua)) {
                if (out->n == out->cap) {
                    out->cap = out->cap ? 2 * out->cap : 32;
                    out->a = (ZnbHit *)znb_realloc(out->a, out->cap, sizeof(ZnbHit));
                }
                out->a[out->n].item = it;
                out->a[out->n].ua = ua;
                out->n++;
            }
        }
    }
}

static void znb_seggrid_free(ZnbSegGrid *g) {
    free(g->off);
    free(g->item);
    free(g->x0);
    free(g->y0);
    free(g->x1);
    free(g->y1);
    free(g->stamp);
}

// ------------------------------------------------------------------------------------------------------------------
// Build context
// ------------------------------------------------------------------------------------------------------------------

typedef struct {
    int *node;
    double *d;
    int n;
    double bound; // < 0: not computed, INF: complete
} ZnbReach;

typedef struct {
    const ZNaviMapInput *in;
    const ZNaviBuildConfig *cfg;
    int N;
    double *x0, *y0, *x1, *y1, *cx, *cy, *fx, *fy, *len, *speed;
    int *poly, *chain;
    double max_len;
    double minx, miny, maxx, maxy;
    // Boundary segments
    int num_static; // obstacle (non-crossable) segments
    double *sx0, *sy0, *sx1, *sy1;
    int num_markers;
    double *mx0, *my0, *mx1, *my1;
    ZnbSegGrid obs_grid;   // [static obstacles, sub-lines]
    ZnbPtGrid center_grid; // sub-line centers
    // Longitudinal graph, same-lane weights (len_i + len_j) / 2
    ZnbCsr lon, lon_rev;
    ZnbReach *reach;
    double *ws_dist;
    int *ws_stamp, ws_cur;
    ZnbHeap heap;
    ZnbIVec ivec, cells;
    ZnbHitVec hits;
} ZnbCtx;

static ZNaviRole znb_role(const ZNaviMapInput *in, int poly) {
    int t = in->type[poly];
    return (t >= 0 && t < in->num_types) ? in->type_role[t] : ZNB_ROLE_IGNORE;
}

static int znb_crossable(const ZNaviMapInput *in, int poly) {
    int t = in->type[poly];
    return in->type_crossable != NULL && t >= 0 && t < in->num_types && in->type_crossable[t];
}

// Forward same-lane-weight distances from a, complete up to at least `need`.
static const ZnbReach *znb_reach(ZnbCtx *c, int a, double need) {
    ZnbReach *r = &c->reach[a];
    if (r->bound >= need)
        return r;
    double bound = r->bound < 0 ? ZNB_MAX(need, 64.0) : ZNB_MAX(need, 2 * r->bound);
    if (++c->ws_cur == 0x7fffffff) {
        memset(c->ws_stamp, 0, (size_t)c->N * sizeof(int));
        c->ws_cur = 1;
    }
    int cur = c->ws_cur, n = 0, cap = 16, exhausted = 1;
    int *node = (int *)znb_alloc(cap, sizeof(int));
    double *dd = (double *)znb_alloc(cap, sizeof(double));
    ZnbHeap *h = &c->heap;
    h->n = 0;
    c->ws_stamp[a] = cur;
    c->ws_dist[a] = 0;
    znb_hpush(h, 0, a);
    while (h->n > 0) {
        double d;
        int u;
        znb_hpop(h, &d, &u);
        if (d > c->ws_dist[u])
            continue;
        if (d > bound) {
            exhausted = 0;
            break;
        }
        if (n == cap) {
            cap *= 2;
            node = (int *)znb_realloc(node, cap, sizeof(int));
            dd = (double *)znb_realloc(dd, cap, sizeof(double));
        }
        node[n] = u;
        dd[n++] = d;
        c->ws_dist[u] = -1; // settled
        for (int e = c->lon.off[u]; e < c->lon.off[u + 1]; e++) {
            int v = c->lon.dst[e];
            double nd = d + c->lon.w[e];
            if (c->ws_stamp[v] != cur) {
                c->ws_stamp[v] = cur;
                c->ws_dist[v] = nd;
                znb_hpush(h, nd, v);
            } else if (c->ws_dist[v] >= 0 && nd < c->ws_dist[v]) {
                c->ws_dist[v] = nd;
                znb_hpush(h, nd, v);
            }
        }
    }
    h->n = 0;
    // Sort by node (insertion sort on small lists, else qsort of pairs).
    int *order = (int *)znb_alloc(n, sizeof(int));
    for (int i = 0; i < n; i++)
        order[i] = i;
    for (int i = 1; i < n; i++) { // lists are short; nodes along a chain are nearly sorted
        int k = order[i], j = i - 1;
        while (j >= 0 && node[order[j]] > node[k]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = k;
    }
    free(r->node);
    free(r->d);
    r->node = (int *)znb_alloc(n, sizeof(int));
    r->d = (double *)znb_alloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        r->node[i] = node[order[i]];
        r->d[i] = dd[order[i]];
    }
    r->n = n;
    r->bound = exhausted ? ZNB_INF : bound;
    free(order);
    free(node);
    free(dd);
    return r;
}

static double znb_reach_lookup(const ZnbReach *r, int b) {
    int lo = 0, hi = r->n - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (r->node[m] == b)
            return r->d[m];
        if (r->node[m] < b)
            lo = m + 1;
        else
            hi = m - 1;
    }
    return ZNB_INF;
}

// goal.py get_longitudinal_set_mask: forward path (either way) at most twice the center distance.
static int znb_same_lane(ZnbCtx *c, int a, int b) {
    if (a == b)
        return 1;
    double limit = 2.0 * hypot(c->cx[b] - c->cx[a], c->cy[b] - c->cy[a]);
    if (znb_reach_lookup(znb_reach(c, a, limit), b) <= limit)
        return 1;
    return znb_reach_lookup(znb_reach(c, b, limit), a) <= limit;
}

static double znb_signed_angle(const ZnbCtx *c, int s, int d, double px, double py) {
    double rx = c->fx[s] + c->fx[d], ry = c->fy[s] + c->fy[d];
    double vx = px - c->cx[s], vy = py - c->cy[s];
    return atan2(rx * vy - ry * vx, rx * vx + ry * vy);
}

// ------------------------------------------------------------------------------------------------------------------
// Lateral connections (goal.py get_laterally_reachable_lanes)
// ------------------------------------------------------------------------------------------------------------------

typedef struct {
    const double *radius;          // [N] search radius per source, <= 0: none
    const double *ang_lo, *ang_hi; // [N] angle window per source (both sides)
    int num_steps;
    int steps[2];
    double same_dir, same_pos;         // angle tolerances
    const int *lane_ids;               // [N]
    const double *along_lo, *along_hi; // [N] or NULL
    double max_lateral;                // < 0: none
    double gap;                        // < 0: none (max lane center gap probes)
    int one_target_lane;               // only with symmetric = 0 and num_steps = 1
    int symmetric;
    double target_angle; // < 0: none (one target per lane, closest to this angle)
    // PufferDrive refinement: in step `junction_step` (< 0: none), for pairs at most `junction_max_distance` apart,
    // only lanes within `junction_parallel_angle` of (anti)parallel to the source can lie in between; lanes crossing
    // the probe more steeply (junction connectors) are ignored.
    int junction_step;
    double junction_parallel_angle, junction_max_distance;
} ZnbLatParams;

typedef struct {
    int src, dst;
    unsigned char side[2];
    unsigned char evaluated;
    unsigned char blocked[2];
    double dist;
} ZnbPair;

static int znb_gap_blocked(ZnbCtx *c, int i, int j, double gap) {
    const double spacing = 2.5, probe = 20.0;
    double rx = c->fx[i] + c->fx[j], ry = c->fy[i] + c->fy[j];
    double rn = ZNB_MAX(hypot(rx, ry), 1e-9);
    rx /= rn;
    ry /= rn;
    double nx = -ry, ny = rx, half = gap / 2;
    double dx = c->cx[j] - c->cx[i], dy = c->cy[j] - c->cy[i];
    int samples = (int)ceil(hypot(dx, dy) / spacing);
    if (samples < 1)
        samples = 1;
    for (int k = 0; k < samples; k++) {
        double t = (k + 0.5) / samples, px = c->cx[i] + t * dx, py = c->cy[i] + t * dy;
        znb_seggrid_query(&c->obs_grid, px - nx * half, py - ny * half, px + nx * half, py + ny * half, &c->hits,
                          &c->cells);
        int covered = 0;
        for (int h = 0; h < c->hits.n && !covered; h++)
            covered = c->hits.a[h].item >= c->num_static;
        if (!covered) {
            return 1;
        }
    }
    for (int o = 0; o < 2; o++) {
        int origin = o == 0 ? i : j, other = o == 0 ? j : i;
        double side = ((c->cx[other] - c->cx[origin]) * nx + (c->cy[other] - c->cy[origin]) * ny) >= 0 ? 1.0 : -1.0;
        double ex = c->cx[origin] + nx * side * probe, ey = c->cy[origin] + ny * side * probe;
        znb_seggrid_query(&c->obs_grid, c->cx[origin], c->cy[origin], ex, ey, &c->hits, &c->cells);
        // Copy: same_lane below may not touch hits, but keep the query buffer independent.
        double first_static = ZNB_INF, first_lane = ZNB_INF, first_other = ZNB_INF;
        for (int h = 0; h < c->hits.n; h++) {
            double dist = c->hits.a[h].ua * probe;
            int it = c->hits.a[h].item;
            if (it < c->num_static) {
                first_static = ZNB_MIN(first_static, dist);
                continue;
            }
            int k = it - c->num_static;
            if (dist < first_lane && !znb_same_lane(c, origin, k))
                first_lane = dist;
            if (dist < first_other && znb_same_lane(c, other, k))
                first_other = dist;
        }
        if (first_static < ZNB_MIN(first_lane, gap))
            return 1;
        if (first_other > gap && first_other < ZNB_INF)
            return 1;
    }
    return 0;
}

// Lanes crossed by the centre-to-centre probe i -> j (`items`: probe hits). Sets *blocked on a static obstacle, an
// opposing or same-lane crossing, or any crossing when `any_crossing_blocks`. With `parallel_cos` > 0, lanes not
// within acos(parallel_cos) of (anti)parallel to i are ignored.
static int znb_count_crossings(ZnbCtx *c, const ZnbLatParams *P, int i, int j, const int *items, int nh,
                               int any_crossing_blocks, double parallel_cos, int *blocked) {
    const double cos_opposing = cos(M_PI - M_PI / 4);
    int crossed_cap = 8, crossed = 0;
    int *lanes = (int *)znb_alloc(crossed_cap, sizeof(int));
    *blocked = 0;
    for (int h = 0; h < nh && !*blocked; h++) {
        int it = items[h];
        if (it < c->num_static) {
            *blocked = 1;
            break;
        }
        int k = it - c->num_static;
        if (k == i || k == j)
            continue;
        double dot = c->fx[k] * c->fx[i] + c->fy[k] * c->fy[i];
        if (parallel_cos > 0 && fabs(dot) < parallel_cos)
            continue;
        if (any_crossing_blocks || dot < cos_opposing || znb_same_lane(c, i, k) || znb_same_lane(c, j, k)) {
            *blocked = 1;
            break;
        }
        int lane = P->lane_ids[k], seen = 0;
        for (int q = 0; q < crossed && !seen; q++)
            seen = lanes[q] == lane;
        if (!seen) {
            if (crossed == crossed_cap) {
                crossed_cap *= 2;
                lanes = (int *)znb_realloc(lanes, crossed_cap, sizeof(int));
            }
            lanes[crossed++] = lane;
        }
    }
    free(lanes);
    return crossed;
}

static void znb_eval_pair(ZnbCtx *c, const ZnbLatParams *P, ZnbPair *p) {
    if (p->evaluated)
        return;
    p->evaluated = 1;
    int any_crossing_blocks = 1;
    for (int s = 0; s < P->num_steps; s++)
        any_crossing_blocks &= P->steps[s] == 1;
    int i = p->src, j = p->dst, blocked = 0;
    znb_seggrid_query(&c->obs_grid, c->cx[i], c->cy[i], c->cx[j], c->cy[j], &c->hits, &c->cells);
    int nh = c->hits.n;
    int *items = (int *)znb_alloc(nh, sizeof(int));
    for (int h = 0; h < nh; h++)
        items[h] = c->hits.a[h].item;
    int crossed = znb_count_crossings(c, P, i, j, items, nh, any_crossing_blocks, -1.0, &blocked);
    int junction_blocked = blocked, junction_crossed = crossed;
    if (P->junction_step >= 0 && P->junction_step < P->num_steps && p->dist <= P->junction_max_distance)
        junction_crossed = znb_count_crossings(c, P, i, j, items, nh, any_crossing_blocks,
                                               cos(P->junction_parallel_angle), &junction_blocked);
    free(items);
    if (!blocked && P->gap > 0)
        blocked = znb_gap_blocked(c, i, j, P->gap);
    if (!junction_blocked && P->gap > 0)
        junction_blocked = znb_gap_blocked(c, i, j, P->gap);
    for (int s = 0; s < P->num_steps; s++) {
        int b = s == P->junction_step ? junction_blocked : blocked;
        int n = s == P->junction_step ? junction_crossed : crossed;
        p->blocked[s] = b || (P->steps[s] > 0 && n >= P->steps[s]);
    }
}

static int znb_cmp_pair_key(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

typedef struct {
    double key;
    int idx;
} ZnbKeyIdx;

static int znb_cmp_keyidx(const void *a, const void *b) {
    const ZnbKeyIdx *x = (const ZnbKeyIdx *)a, *y = (const ZnbKeyIdx *)b;
    if (x->key != y->key)
        return x->key < y->key ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

// Sorted unique (src, dst) keys -> CSR; one target per lane if P->target_angle >= 0.
static ZnbCsr znb_lateral_output(ZnbCtx *c, const ZnbLatParams *P, int64_t *keys, int nk, int side) {
    int N = c->N;
    qsort(keys, nk, sizeof(int64_t), znb_cmp_pair_key);
    int m = 0;
    for (int k = 0; k < nk; k++)
        if (m == 0 || keys[k] != keys[m - 1])
            keys[m++] = keys[k];
    unsigned char *keep = (unsigned char *)znb_alloc(m, 1);
    memset(keep, 1, m);
    if (P->target_angle >= 0) {
        double *score = (double *)znb_alloc(m, sizeof(double));
        for (int k = 0; k < m; k++) {
            int s = (int)(keys[k] / N), d = (int)(keys[k] % N);
            double a = znb_signed_angle(c, s, d, c->cx[d], c->cy[d]) * (side == 0 ? 1 : -1);
            score[k] = fabs(a - P->target_angle);
        }
        for (int g0 = 0; g0 < m;) {
            int s = (int)(keys[g0] / N), g1 = g0;
            while (g1 < m && keys[g1] / N == s)
                g1++;
            for (int b = g0; b < g1; b++)
                for (int a = g0; a < g1 && keep[b]; a++) {
                    if (a == b || !(score[a] < score[b] || (score[a] == score[b] && a < b)))
                        continue;
                    int da = (int)(keys[a] % N), db = (int)(keys[b] % N);
                    if (P->lane_ids[da] == P->lane_ids[db] || znb_same_lane(c, da, db))
                        keep[b] = 0;
                }
            g0 = g1;
        }
        free(score);
    }
    int *src = (int *)znb_alloc(m, sizeof(int)), *dst = (int *)znb_alloc(m, sizeof(int));
    int e = 0;
    for (int k = 0; k < m; k++)
        if (keep[k]) {
            src[e] = (int)(keys[k] / N);
            dst[e++] = (int)(keys[k] % N);
        }
    ZnbCsr g = znb_csr(N, e, src, dst, NULL);
    free(src);
    free(dst);
    free(keep);
    return g;
}

// out[steps][side] (side 0 = left), diagonal not included.
static void znb_lateral(ZnbCtx *c, const ZnbLatParams *P, ZnbCsr out[2][2]) {
    int N = c->N;
    double *radius = (double *)znb_alloc(N, sizeof(double));
    double max_radius = 0;
    for (int i = 0; i < N; i++)
        radius[i] = P->radius[i];
    if (P->along_hi != NULL) {
        int all_ok = 1;
        for (int i = 0; i < N && all_ok; i++)
            all_ok = ZNB_MAX(fabs(P->ang_lo[i]), fabs(P->ang_hi[i])) < M_PI / 2 - 1e-3;
        if (all_ok)
            for (int i = 0; i < N; i++) {
                double maxabs = ZNB_MAX(fabs(P->ang_lo[i]), fabs(P->ang_hi[i]));
                double bound = (P->along_hi[i] + c->max_len) / cos(maxabs);
                radius[i] = ZNB_MIN(radius[i], bound + c->max_len / 2 + 1.0);
            }
    }
    for (int i = 0; i < N; i++)
        max_radius = ZNB_MAX(max_radius, radius[i]);
    (void)max_radius;
    const double cos_dir = cos(P->same_dir), cos_pos = cos(P->same_pos);

    // Step 1-2: candidate pairs and eligibility, in (src, dst) order.
    ZnbPair *pairs = NULL;
    int np = 0, pcap = 0;
    int *pair_start = (int *)znb_alloc(N + 1, sizeof(int));
    ZnbIVec cand = {0};
    for (int i = 0; i < N; i++) {
        pair_start[i] = np;
        if (radius[i] <= 0)
            continue;
        znb_ptgrid_query(&c->center_grid, c->cx[i], c->cy[i], radius[i], &cand);
        qsort(cand.a, cand.n, sizeof(int), znb_cmp_int);
        for (int q = 0; q < cand.n; q++) {
            int j = cand.a[q];
            if (j == i)
                continue;
            double rx = c->cx[j] - c->cx[i], ry = c->cy[j] - c->cy[i], dist = hypot(rx, ry);
            if (dist > radius[i])
                continue;
            if (c->fx[i] * c->fx[j] + c->fy[i] * c->fy[j] < cos_dir)
                continue;
            if (rx * c->fx[j] + ry * c->fy[j] < cos_pos * dist)
                continue;
            double ac = znb_signed_angle(c, i, j, c->cx[j], c->cy[j]);
            double ds = znb_wrap(znb_signed_angle(c, i, j, c->x0[j], c->y0[j]) - ac);
            double de = znb_wrap(znb_signed_angle(c, i, j, c->x1[j], c->y1[j]) - ac);
            double lo_e = ac + ZNB_MIN(ds, de), hi_e = ac + ZNB_MAX(ds, de);
            int in0 = hi_e >= P->ang_lo[i] && lo_e <= P->ang_hi[i];
            int in1 = -lo_e >= P->ang_lo[i] && -hi_e <= P->ang_hi[i];
            if (!in0 && !in1)
                continue;
            if (P->along_hi != NULL || P->max_lateral >= 0) {
                double ux = c->fx[i] + c->fx[j], uy = c->fy[i] + c->fy[j], un = ZNB_MAX(hypot(ux, uy), 1e-9);
                ux /= un;
                uy /= un;
                double sx = c->x0[j] - c->cx[i], sy = c->y0[j] - c->cy[i];
                double ex = c->x1[j] - c->cx[i], ey = c->y1[j] - c->cy[i];
                if (P->along_hi != NULL) {
                    double as = sx * ux + sy * uy, ae = ex * ux + ey * uy;
                    if (!(ZNB_MAX(as, ae) >= P->along_lo[i] && ZNB_MIN(as, ae) <= P->along_hi[i]))
                        continue;
                }
                if (P->max_lateral >= 0) {
                    double cs = -sx * uy + sy * ux, ce = -ex * uy + ey * ux;
                    double min_across = cs * ce <= 0 ? 0 : ZNB_MIN(fabs(cs), fabs(ce));
                    if (min_across > P->max_lateral)
                        continue;
                }
            }
            if (znb_same_lane(c, i, j))
                continue;
            if (np == pcap) {
                pcap = pcap ? 2 * pcap : 256;
                pairs = (ZnbPair *)znb_realloc(pairs, pcap, sizeof(ZnbPair));
            }
            ZnbPair *p = &pairs[np++];
            memset(p, 0, sizeof(*p));
            p->src = i;
            p->dst = j;
            p->side[0] = (unsigned char)in0;
            p->side[1] = (unsigned char)in1;
            p->dist = dist;
        }
    }
    pair_start[N] = np;
    free(cand.a);

    // Step 3: obstruction and target selection.
    int64_t *keys[2][2] = {{NULL, NULL}, {NULL, NULL}};
    int nk[2][2] = {{0, 0}, {0, 0}}, kcap[2][2] = {{0, 0}, {0, 0}};
#define ZNB_KEY_PUSH(st, sd, s, d)                                                                                     \
    do {                                                                                                               \
        if (nk[st][sd] == kcap[st][sd]) {                                                                              \
            kcap[st][sd] = kcap[st][sd] ? 2 * kcap[st][sd] : 256;                                                      \
            keys[st][sd] = (int64_t *)znb_realloc(keys[st][sd], kcap[st][sd], sizeof(int64_t));                        \
        }                                                                                                              \
        keys[st][sd][nk[st][sd]++] = (int64_t)(s) * N + (d);                                                           \
    } while (0)

    if (P->one_target_lane && !P->symmetric && P->num_steps == 1) {
        // Nearest unobstructed target per source and side, then the unobstructed targets on its lane.
        ZnbKeyIdx *order = (ZnbKeyIdx *)znb_alloc(np, sizeof(ZnbKeyIdx));
        for (int i = 0; i < N; i++) {
            for (int side = 0; side < 2; side++) {
                int n = 0;
                for (int k = pair_start[i]; k < pair_start[i + 1]; k++)
                    if (pairs[k].side[side]) {
                        order[n].key = pairs[k].dist;
                        order[n++].idx = k;
                    }
                if (n == 0)
                    continue;
                qsort(order, n, sizeof(ZnbKeyIdx), znb_cmp_keyidx);
                int nearest = -1;
                for (int q = 0; q < n && nearest < 0; q++) {
                    ZnbPair *p = &pairs[order[q].idx];
                    znb_eval_pair(c, P, p);
                    if (!p->blocked[0])
                        nearest = p->dst;
                }
                if (nearest < 0)
                    continue;
                for (int k = pair_start[i]; k < pair_start[i + 1]; k++) {
                    ZnbPair *p = &pairs[k];
                    if (!p->side[side] || !(p->dst == nearest || znb_same_lane(c, nearest, p->dst)))
                        continue;
                    znb_eval_pair(c, P, p);
                    if (!p->blocked[0])
                        ZNB_KEY_PUSH(0, side, i, p->dst);
                }
            }
        }
        free(order);
    } else {
        for (int k = 0; k < np; k++) {
            ZnbPair *p = &pairs[k];
            znb_eval_pair(c, P, p);
            for (int st = 0; st < P->num_steps; st++)
                for (int side = 0; side < 2; side++) {
                    if (!p->side[side] || p->blocked[st])
                        continue;
                    ZNB_KEY_PUSH(st, side, p->src, p->dst);
                    if (P->symmetric) // j left of i <=> i right of j
                        ZNB_KEY_PUSH(st, 1 - side, p->dst, p->src);
                }
        }
    }
#undef ZNB_KEY_PUSH
    for (int st = 0; st < 2; st++)
        for (int side = 0; side < 2; side++) {
            if (st < P->num_steps)
                out[st][side] = znb_lateral_output(c, P, keys[st][side], nk[st][side], side);
            else
                memset(&out[st][side], 0, sizeof(ZnbCsr));
            free(keys[st][side]);
        }
    free(pairs);
    free(pair_start);
    free(radius);
}

// goal.py _restrict_to_perp_reachable: keep lane change i -> j if j is forward-reachable from a perp target of the
// source chain (forward from i, and backward up to max_back[i]).
static void znb_restrict_lane_changes(ZnbCtx *c, ZnbCsr lc[2], ZnbCsr perp[2], const double *max_back) {
    int N = c->N;
    int *mark[2] = {(int *)znb_alloc(N, sizeof(int)), (int *)znb_alloc(N, sizeof(int))};
    int *seen = (int *)znb_alloc(N, sizeof(int));
    int *stack = (int *)znb_alloc(N, sizeof(int));
    double *bd = (double *)znb_alloc(N, sizeof(double));
    int stamp = 0;
    unsigned char *keep[2];
    for (int side = 0; side < 2; side++) {
        keep[side] = (unsigned char *)znb_alloc(lc[side].off[N], 1);
    }
    for (int i = 0; i < N; i++) {
        if (lc[0].off[i] == lc[0].off[i + 1] && lc[1].off[i] == lc[1].off[i + 1])
            continue;
        int id = i + 1;
        // Forward chain (unbounded)
        stamp++;
        int sp = 0;
        stack[sp++] = i;
        seen[i] = stamp;
        while (sp > 0) {
            int u = stack[--sp];
            for (int side = 0; side < 2; side++)
                for (int e = perp[side].off[u]; e < perp[side].off[u + 1]; e++)
                    if (perp[side].dst[e] != u)
                        mark[side][perp[side].dst[e]] = id;
            for (int e = c->lon.off[u]; e < c->lon.off[u + 1]; e++) {
                int v = c->lon.dst[e];
                if (seen[v] != stamp) {
                    seen[v] = stamp;
                    stack[sp++] = v;
                }
            }
        }
        // Backward chain (bounded Dijkstra on the reversed same-lane-weight graph)
        {
            ZnbHeap *h = &c->heap;
            h->n = 0;
            stamp++;
            seen[i] = stamp;
            bd[i] = 0;
            znb_hpush(h, 0, i);
            while (h->n > 0) {
                double d;
                int u;
                znb_hpop(h, &d, &u);
                if (d > bd[u])
                    continue;
                for (int side = 0; side < 2; side++)
                    for (int e = perp[side].off[u]; e < perp[side].off[u + 1]; e++)
                        if (perp[side].dst[e] != u)
                            mark[side][perp[side].dst[e]] = id;
                for (int e = c->lon_rev.off[u]; e < c->lon_rev.off[u + 1]; e++) {
                    int v = c->lon_rev.dst[e];
                    double nd = d + c->lon_rev.w[e];
                    if (nd > max_back[i])
                        continue;
                    if (seen[v] != stamp || nd < bd[v]) {
                        seen[v] = stamp;
                        bd[v] = nd;
                        znb_hpush(h, nd, v);
                    }
                }
            }
        }
        // Targets: any backward-reachable (unbounded) node of j marked
        for (int side = 0; side < 2; side++)
            for (int e = lc[side].off[i]; e < lc[side].off[i + 1]; e++) {
                int j = lc[side].dst[e], found = mark[side][j] == id;
                stamp++;
                sp = 0;
                stack[sp++] = j;
                seen[j] = stamp;
                while (sp > 0 && !found) {
                    int u = stack[--sp];
                    for (int q = c->lon_rev.off[u]; q < c->lon_rev.off[u + 1] && !found; q++) {
                        int v = c->lon_rev.dst[q];
                        if (seen[v] == stamp)
                            continue;
                        seen[v] = stamp;
                        found = mark[side][v] == id;
                        stack[sp++] = v;
                    }
                }
                keep[side][e] = (unsigned char)found;
            }
    }
    for (int side = 0; side < 2; side++) {
        int m = lc[side].off[N], e2 = 0;
        int *src = (int *)znb_alloc(m, sizeof(int)), *dst = (int *)znb_alloc(m, sizeof(int));
        for (int u = 0; u < N; u++)
            for (int e = lc[side].off[u]; e < lc[side].off[u + 1]; e++)
                if (keep[side][e]) {
                    src[e2] = u;
                    dst[e2++] = lc[side].dst[e];
                }
        znb_csr_free(&lc[side]);
        lc[side] = znb_csr(N, e2, src, dst, NULL);
        free(src);
        free(dst);
        free(keep[side]);
        free(mark[side]);
    }
    free(seen);
    free(stack);
    free(bd);
}

// ------------------------------------------------------------------------------------------------------------------
// Lane feasibility (goal.py _max_longitudinal_distance_all_starts)
// ------------------------------------------------------------------------------------------------------------------

static void znb_feasibility_row(const ZnbCsr *g, const double *dist, const double *fwd_from_goal,
                                const double *rev_to_goal, const ZNaviBuildConfig *cfg, double *out, double *f,
                                unsigned char *eligible, int *outdeg, int *queue, const ZnbCsr *grev) {
    int N = g->n;
    const double close_limit = cfg->unreachable_distance - 0.01;
    for (int u = 0; u < N; u++) {
        double d = dist[u] < ZNB_INF ? dist[u] : cfg->unreachable_distance;
        int past = fwd_from_goal[u] < ZNB_INF && fwd_from_goal[u] > 0;
        int close = d < close_limit;
        int lane_ok = cfg->lane_change_margin == 0 || rev_to_goal[u] < ZNB_INF || d >= cfg->lane_change_margin;
        eligible[u] = (unsigned char)(!past && close && lane_ok);
        f[u] = eligible[u] ? 0.0 : -1.0;
    }
    // Peel eligible nodes without remaining eligible successors (reverse topological DP); the rest reach a cycle.
    int qh = 0, qt = 0;
    for (int u = 0; u < N; u++) {
        if (!eligible[u])
            continue;
        outdeg[u] = 0;
        for (int e = g->off[u]; e < g->off[u + 1]; e++)
            outdeg[u] += eligible[g->dst[e]];
        if (outdeg[u] == 0)
            queue[qt++] = u;
    }
    unsigned char *done = (unsigned char *)znb_alloc(N, 1);
    while (qh < qt) {
        int u = queue[qh++];
        done[u] = 1;
        double best = 0;
        for (int e = g->off[u]; e < g->off[u + 1]; e++) {
            int v = g->dst[e];
            if (eligible[v] && g->w[e] + f[v] > best)
                best = g->w[e] + f[v];
        }
        f[u] = best;
        for (int e = grev->off[u]; e < grev->off[u + 1]; e++) {
            int p = grev->dst[e];
            if (eligible[p] && --outdeg[p] == 0)
                queue[qt++] = p;
        }
    }
    for (int u = 0; u < N; u++)
        if (eligible[u] && !done[u])
            f[u] = ZNB_INF;
    free(done);
    for (int u = 0; u < N; u++) {
        double d = dist[u] < ZNB_INF ? dist[u] : cfg->unreachable_distance;
        int past = fwd_from_goal[u] < ZNB_INF && fwd_from_goal[u] > 0;
        int reachable = !past && d < close_limit;
        double r = 0;
        if (eligible[u]) {
            r = f[u] >= 0 ? f[u] : 0;
        } else if (reachable) {
            for (int e = g->off[u]; e < g->off[u + 1]; e++) {
                int v = g->dst[e];
                if (eligible[v] && g->w[e] + ZNB_MAX(f[v], 0.0) > r)
                    r = g->w[e] + ZNB_MAX(f[v], 0.0);
            }
        }
        out[u] = reachable ? ZNB_MIN(r, d + cfg->max_route_deviation) : 0.0;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Main entry
// ------------------------------------------------------------------------------------------------------------------

static void znavi_tables_free(ZNaviTables *t) {
    free(t->line_polyline);
    free(t->line_segment);
    free(t->line_num_sub);
    free(t->line_sample_offset);
    free(t->sample_line);
    free(t->sample_index);
    free(t->goal_x);
    free(t->goal_y);
    free(t->goal_sample);
    free(t->next_sample);
    free(t->distance_to_goal);
    free(t->feasibility_left);
    free(t->feasibility_center);
    free(t->feasibility_right);
    free(t->agent_goal);
    memset(t, 0, sizeof(*t));
}

typedef struct {
    int *members;
    int n;
    double *seed; // [n] initial route cost per member, NULL = all 0
} ZnbGoalSet;

static int znb_cmp_goalset(const void *a, const void *b) {
    const ZnbGoalSet *x = (const ZnbGoalSet *)a, *y = (const ZnbGoalSet *)b;
    for (int k = 0; k < x->n && k < y->n; k++)
        if (x->members[k] != y->members[k])
            return x->members[k] < y->members[k] ? -1 : 1;
    return (x->n > y->n) - (x->n < y->n);
}

// Candidate start sub-lines (goal.py get_candidate_start_lines): closest per lane within distance and angle.
static int znb_start_candidates(ZnbCtx *c, double x, double y, double heading, ZnbIVec *out) {
    out->n = 0;
    if (!isfinite(x) || !isfinite(y))
        return 0;
    const ZNaviBuildConfig *cfg = c->cfg;
    double hx = cos(heading), hy = sin(heading);
    ZnbIVec q = {0};
    znb_ptgrid_query(&c->center_grid, x, y, cfg->start_max_distance + c->max_len / 2 + 1e-3, &q);
    ZnbKeyIdx *cand = (ZnbKeyIdx *)znb_alloc(q.n, sizeof(ZnbKeyIdx));
    int n = 0;
    double cos_max = cos(cfg->start_max_angle);
    for (int k = 0; k < q.n; k++) {
        int s = q.a[k];
        double d = znb_point_segment_distance(x, y, c->x0[s], c->y0[s], c->x1[s], c->y1[s]);
        if (d > cfg->start_max_distance)
            continue;
        double dot = c->fx[s] * hx + c->fy[s] * hy;
        if (isfinite(heading) && dot < cos_max - 1e-12)
            continue;
        cand[n].key = d;
        cand[n++].idx = s;
    }
    qsort(cand, n, sizeof(ZnbKeyIdx), znb_cmp_keyidx);
    for (int k = 0; k < n; k++) {
        int s = cand[k].idx, dup = 0;
        for (int q2 = 0; q2 < out->n && !dup; q2++)
            dup = c->poly[out->a[q2]] == c->poly[s];
        if (!dup)
            znb_ipush(out, s);
    }
    free(cand);
    free(q.a);
    return out->n;
}

// Builds the tables; returns 0 on success. `agents` may be NULL.
static int znavi_build(const ZNaviMapInput *in, const ZNaviAgentInput *agents, const ZNaviBuildConfig *cfg,
                       ZNaviTables *t) {
    memset(t, 0, sizeof(*t));
    ZnbCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ZnbCtx *c = &ctx;
    c->in = in;
    c->cfg = cfg;
    const int P = in->num_polylines;

    // ---- C2: navi lines and sub-lines ----
    int num_lines = 0, N = 0;
    for (int p = 0; p < P; p++) {
        int np = in->point_offset[p + 1] - in->point_offset[p];
        if (znb_role(in, p) == ZNB_ROLE_LANE && np >= 2)
            num_lines += np - 1;
    }
    t->num_lines = num_lines;
    t->line_polyline = (int *)znb_alloc(num_lines, sizeof(int));
    t->line_segment = (int *)znb_alloc(num_lines, sizeof(int));
    t->line_num_sub = (int *)znb_alloc(num_lines, sizeof(int));
    t->line_sample_offset = (int *)znb_alloc(num_lines, sizeof(int));
    double *line_speed = (double *)znb_alloc(num_lines, sizeof(double));
    int l = 0;
    for (int p = 0; p < P; p++) {
        int o = in->point_offset[p], np = in->point_offset[p + 1] - o;
        if (znb_role(in, p) != ZNB_ROLE_LANE || np < 2)
            continue;
        double v = in->speed_limit ? in->speed_limit[p] : 0;
        if (!(v > 0))
            v = cfg->default_speed_limit;
        double step = cfg->discretization_time * ZNB_MAX(v, 1.0);
        for (int k = 0; k + 1 < np; k++, l++) {
            double L = hypot((double)in->x[o + k + 1] - in->x[o + k], (double)in->y[o + k + 1] - in->y[o + k]);
            t->line_polyline[l] = p;
            t->line_segment[l] = k;
            t->line_num_sub[l] = L > 1e-6 ? (int)ceil((L - 1e-6) / step) : 0;
            t->line_sample_offset[l] = N;
            line_speed[l] = v;
            N += t->line_num_sub[l];
        }
    }
    c->N = t->num_samples = N;
    t->sample_line = (int *)znb_alloc(N, sizeof(int));
    t->sample_index = (int *)znb_alloc(N, sizeof(int));
    double **arrs[10] = {&c->x0, &c->y0, &c->x1, &c->y1, &c->cx, &c->cy, &c->fx, &c->fy, &c->len, &c->speed};
    for (int k = 0; k < 10; k++)
        *arrs[k] = (double *)znb_alloc(N, sizeof(double));
    c->poly = (int *)znb_alloc(N, sizeof(int));
    c->chain = (int *)znb_alloc(N, sizeof(int));
    c->minx = c->miny = ZNB_INF;
    c->maxx = c->maxy = -ZNB_INF;
    for (l = 0; l < num_lines; l++) {
        int p = t->line_polyline[l], o = in->point_offset[p] + t->line_segment[l], n = t->line_num_sub[l];
        double ax = in->x[o], ay = in->y[o], bx = in->x[o + 1], by = in->y[o + 1];
        double L = hypot(bx - ax, by - ay);
        for (int s = 0; s < n; s++) {
            int i = t->line_sample_offset[l] + s;
            t->sample_line[i] = l;
            t->sample_index[i] = s;
            double u0 = (double)s / n, u1 = (double)(s + 1) / n;
            if (cfg->fixed_step_sublines) {
                double step = cfg->discretization_time * ZNB_MAX(line_speed[l], 1.0);
                u0 = ZNB_MIN(s * step, L) / L;
                u1 = ZNB_MIN((s + 1) * step, L) / L;
            }
            c->x0[i] = ax + (bx - ax) * u0;
            c->y0[i] = ay + (by - ay) * u0;
            c->x1[i] = ax + (bx - ax) * u1;
            c->y1[i] = ay + (by - ay) * u1;
            c->cx[i] = (c->x0[i] + c->x1[i]) / 2;
            c->cy[i] = (c->y0[i] + c->y1[i]) / 2;
            c->fx[i] = (bx - ax) / L;
            c->fy[i] = (by - ay) / L;
            c->len[i] = L * (u1 - u0);
            c->speed[i] = line_speed[l];
            c->poly[i] = p;
            c->max_len = ZNB_MAX(c->max_len, c->len[i]);
        }
    }
    free(line_speed);
    for (int i = 0; i < in->point_offset[P]; i++) {
        c->minx = ZNB_MIN(c->minx, in->x[i]);
        c->maxx = ZNB_MAX(c->maxx, in->x[i]);
        c->miny = ZNB_MIN(c->miny, in->y[i]);
        c->maxy = ZNB_MAX(c->maxy, in->y[i]);
    }

    // ---- Boundary segments ----
    int num_static = 0, num_markers = 0;
    for (int p = 0; p < P; p++) {
        ZNaviRole role = znb_role(in, p);
        int ns = in->point_offset[p + 1] - in->point_offset[p] - 1;
        if (ns < 1 || (role != ZNB_ROLE_MARKER && role != ZNB_ROLE_EDGE))
            continue;
        if (!znb_crossable(in, p))
            num_static += ns;
        if (role == ZNB_ROLE_MARKER)
            num_markers += ns;
    }
    c->num_static = num_static;
    c->num_markers = num_markers;
    int num_obs = num_static + N;
    double *ox0 = (double *)znb_alloc(num_obs, sizeof(double)), *oy0 = (double *)znb_alloc(num_obs, sizeof(double));
    double *ox1 = (double *)znb_alloc(num_obs, sizeof(double)), *oy1 = (double *)znb_alloc(num_obs, sizeof(double));
    double *mx0 = (double *)znb_alloc(num_markers, sizeof(double)),
           *my0 = (double *)znb_alloc(num_markers, sizeof(double));
    double *mx1 = (double *)znb_alloc(num_markers, sizeof(double)),
           *my1 = (double *)znb_alloc(num_markers, sizeof(double));
    int so = 0, mo = 0;
    for (int p = 0; p < P; p++) {
        ZNaviRole role = znb_role(in, p);
        int o = in->point_offset[p], ns = in->point_offset[p + 1] - o - 1;
        if (ns < 1 || (role != ZNB_ROLE_MARKER && role != ZNB_ROLE_EDGE))
            continue;
        for (int k = 0; k < ns; k++) {
            if (!znb_crossable(in, p)) {
                ox0[so] = in->x[o + k];
                oy0[so] = in->y[o + k];
                ox1[so] = in->x[o + k + 1];
                oy1[so++] = in->y[o + k + 1];
            }
            if (role == ZNB_ROLE_MARKER) {
                mx0[mo] = in->x[o + k];
                my0[mo] = in->y[o + k];
                mx1[mo] = in->x[o + k + 1];
                my1[mo++] = in->y[o + k + 1];
            }
        }
    }
    for (int i = 0; i < N; i++) {
        ox0[num_static + i] = c->x0[i];
        oy0[num_static + i] = c->y0[i];
        ox1[num_static + i] = c->x1[i];
        oy1[num_static + i] = c->y1[i];
    }
    const double probe_margin = 20.0;
    znb_seggrid_build(&c->obs_grid, num_obs, ox0, oy0, ox1, oy1, cfg->cell_size, c->minx - probe_margin,
                      c->miny - probe_margin, c->maxx + probe_margin, c->maxy + probe_margin);
    free(ox0);
    free(oy0);
    free(ox1);
    free(oy1);
    znb_ptgrid_build(&c->center_grid, N, c->cx, c->cy, 16.0);
    c->ws_dist = (double *)znb_alloc(N, sizeof(double));
    c->ws_stamp = (int *)znb_alloc(N, sizeof(int));
    c->reach = (ZnbReach *)znb_alloc(N, sizeof(ZnbReach));
    for (int i = 0; i < N; i++)
        c->reach[i].bound = -1;

    // ---- C3: longitudinal connections ----
    ZnbIVec esrc = {0}, edst = {0};
    {
        ZnbPtGrid start_grid;
        znb_ptgrid_build(&start_grid, N, c->x0, c->y0, 4.0);
        const double thr = cfg->connection_threshold, cos_max = cos(cfg->connection_max_angle);
        ZnbIVec q = {0};
        for (int i = 0; i < N; i++) {
            znb_ptgrid_query(&start_grid, c->x1[i], c->y1[i], thr, &q);
            int next_in_line = t->sample_index[i] + 1 < t->line_num_sub[t->sample_line[i]] ? i + 1 : -1;
            const int nq = q.n;
            for (int k = 0; k < nq; k++) {
                int j = q.a[k];
                if (j == i)
                    continue;
                int ok = j == next_in_line || (hypot(c->x0[j] - c->x1[i], c->y0[j] - c->y1[i]) < thr &&
                                               c->fx[i] * c->fx[j] + c->fy[i] * c->fy[j] > cos_max);
                if (ok)
                    znb_ipush(&q, -1 - j); // marked, collected below
            }
            int m0 = esrc.n;
            for (int k = nq; k < q.n; k++)
                if (q.a[k] < 0) {
                    znb_ipush(&esrc, i);
                    znb_ipush(&edst, -1 - q.a[k]);
                }
            if (next_in_line >= 0) {
                int has = 0;
                for (int k = m0; k < esrc.n && !has; k++)
                    has = edst.a[k] == next_in_line;
                if (!has) {
                    znb_ipush(&esrc, i);
                    znb_ipush(&edst, next_in_line);
                }
            }
            qsort(edst.a + m0, esrc.n - m0, sizeof(int), znb_cmp_int);
        }
        free(q.a);
        znb_ptgrid_free(&start_grid);
    }
    int num_lon = esrc.n;
    {
        double *w = (double *)znb_alloc(num_lon, sizeof(double));
        for (int e = 0; e < num_lon; e++)
            w[e] = (c->len[esrc.a[e]] + c->len[edst.a[e]]) / 2;
        c->lon = znb_csr(N, num_lon, esrc.a, edst.a, w);
        c->lon_rev = znb_csr_transpose(&c->lon);
        free(w);
    }

    // Lane chain ids (goal.py get_lane_chain_ids): polylines linked one-to-one are one lane.
    {
        int *succ = (int *)znb_alloc(P, sizeof(int)), *pred = (int *)znb_alloc(P, sizeof(int));
        int *parent = (int *)znb_alloc(P, sizeof(int));
        for (int p = 0; p < P; p++) {
            succ[p] = pred[p] = -1; // -1 none, -2 several
            parent[p] = p;
        }
        for (int e = 0; e < num_lon; e++) {
            int a = c->poly[esrc.a[e]], b = c->poly[edst.a[e]];
            if (a == b)
                continue;
            succ[a] = succ[a] == -1 || succ[a] == b ? b : -2;
            pred[b] = pred[b] == -1 || pred[b] == a ? a : -2;
        }
        for (int a = 0; a < P; a++) {
            int b = succ[a];
            if (b < 0 || pred[b] != a)
                continue;
            int ra = a, rb = b;
            while (parent[ra] != ra)
                ra = parent[ra];
            while (parent[rb] != rb)
                rb = parent[rb];
            parent[ZNB_MAX(ra, rb)] = ZNB_MIN(ra, rb);
        }
        for (int i = 0; i < N; i++) {
            int r = c->poly[i];
            while (parent[r] != r)
                r = parent[r];
            c->chain[i] = r;
        }
        free(succ);
        free(pred);
        free(parent);
    }

    // F8: forward connections crossing a lane marker (between different polylines).
    unsigned char *crosses = (unsigned char *)znb_alloc(num_lon, 1);
    if (cfg->marker_crossings && num_markers > 0) {
        ZnbSegGrid mg;
        memset(&mg, 0, sizeof(mg));
        znb_seggrid_build(&mg, num_markers, mx0, my0, mx1, my1, cfg->cell_size, c->minx, c->miny, c->maxx, c->maxy);
        for (int e = 0; e < num_lon; e++) {
            int i = esrc.a[e], j = edst.a[e];
            if (c->poly[i] == c->poly[j])
                continue;
            znb_seggrid_query(&mg, c->x1[i], c->y1[i], c->x0[j], c->y0[j], &c->hits, &c->cells);
            crosses[e] = c->hits.n > 0;
        }
        znb_seggrid_free(&mg);
    }
    free(mx0);
    free(my0);
    free(mx1);
    free(my1);

    // ---- F1: perpendicular / adjacent connections ----
    ZnbCsr perp[2], adjacent[2];
    memset(perp, 0, sizeof(perp));
    memset(adjacent, 0, sizeof(adjacent));
    double *lo = (double *)znb_alloc(N, sizeof(double)), *hi = (double *)znb_alloc(N, sizeof(double));
    double *rad = (double *)znb_alloc(N, sizeof(double));
    if (cfg->perp_connections) {
        for (int i = 0; i < N; i++) {
            lo[i] = M_PI / 2 - M_PI / 3;
            hi[i] = M_PI / 2 + M_PI / 3;
            rad[i] = 20.0;
        }
        ZnbLatParams lp;
        memset(&lp, 0, sizeof(lp));
        lp.radius = rad;
        lp.ang_lo = lo;
        lp.ang_hi = hi;
        lp.num_steps = 2;
        lp.steps[0] = 5;
        lp.steps[1] = 1;
        lp.same_dir = M_PI / 5;
        lp.same_pos = M_PI;
        lp.lane_ids = c->poly;
        lp.max_lateral = -1;
        lp.gap = -1;
        lp.symmetric = 1;
        lp.target_angle = M_PI / 2;
        lp.junction_step = 1; // adjacent lanes (side feasibilities only)
        lp.junction_parallel_angle = M_PI / 6;
        lp.junction_max_distance = 6.0;
        ZnbCsr out[2][2];
        znb_lateral(c, &lp, out);
        perp[0] = out[0][0];
        perp[1] = out[0][1];
        adjacent[0] = out[1][0];
        adjacent[1] = out[1][1];
    } else {
        for (int s = 0; s < 2; s++) {
            perp[s] = znb_csr(N, 0, NULL, NULL, NULL);
            adjacent[s] = znb_csr(N, 0, NULL, NULL, NULL);
        }
    }

    // ---- F2/F3: lane changes ----
    ZnbCsr lc[2];
    if (cfg->lane_changes) {
        double *along_lo = (double *)znb_alloc(N, sizeof(double)), *along_hi = (double *)znb_alloc(N, sizeof(double));
        double *max_back = (double *)znb_alloc(N, sizeof(double));
        for (int i = 0; i < N; i++) {
            double v = ZNB_MIN(c->speed[i], 36.11);
            double angle_5s = asin(ZNB_MIN(3.5 / (ZNB_MAX(v, 3.5 / 5.0) * 5.0), 1.0));
            lo[i] = angle_5s * 0.95;
            hi[i] = M_PI / 4;
            double vv = ZNB_MAX(v, 1.0);
            along_lo[i] = vv * 2.0 / 1.2;
            along_hi[i] = vv * 5.0 / 0.95;
            rad[i] = max_back[i] = v * 5.0 * 2.5;
        }
        ZnbLatParams lp;
        memset(&lp, 0, sizeof(lp));
        lp.radius = rad;
        lp.ang_lo = lo;
        lp.ang_hi = hi;
        lp.num_steps = 1;
        lp.steps[0] = 1;
        lp.same_dir = M_PI / 4;
        lp.same_pos = M_PI / 4;
        lp.lane_ids = c->chain;
        lp.along_lo = along_lo;
        lp.along_hi = along_hi;
        lp.max_lateral = 15.0;
        lp.gap = cfg->gap_probe ? 5.0 : -1;
        lp.one_target_lane = 1;
        lp.target_angle = -1;
        lp.junction_step = -1;
        ZnbCsr out[2][2];
        znb_lateral(c, &lp, out);
        lc[0] = out[0][0];
        lc[1] = out[0][1];
        if (cfg->perp_connections)
            znb_restrict_lane_changes(c, lc, perp, max_back);
        free(along_lo);
        free(along_hi);
        free(max_back);
    } else {
        lc[0] = znb_csr(N, 0, NULL, NULL, NULL);
        lc[1] = znb_csr(N, 0, NULL, NULL, NULL);
    }
    free(lo);
    free(hi);
    free(rad);

    // ---- Graphs: navigation (weight / distance), plain longitudinal, no-lane-change (F8) ----
    ZnbCsr nav_w_rev, nav_d_rev, nav_rev_struct, lg, lg_rev, f8, f8_rev;
    {
        ZnbIVec ns = {0}, nd = {0};
        int cap_w = num_lon + lc[0].off[N] + lc[1].off[N];
        double *ww = (double *)znb_alloc(cap_w, sizeof(double)), *wd = (double *)znb_alloc(cap_w, sizeof(double));
        int *ls = (int *)znb_alloc(num_lon, sizeof(int)), *ld = (int *)znb_alloc(num_lon, sizeof(int));
        double *lw = (double *)znb_alloc(num_lon, sizeof(double));
        int *fs = (int *)znb_alloc(num_lon, sizeof(int)), *fd = (int *)znb_alloc(num_lon, sizeof(int));
        double *fw = (double *)znb_alloc(num_lon, sizeof(double));
        int nl = 0, nf = 0;
        for (int i = 0; i < N; i++) {
            // longitudinal edges of i
            for (int e = c->lon.off[i]; e < c->lon.off[i + 1]; e++) {
                int j = c->lon.dst[e];
                double d = hypot(c->cx[j] - c->cx[i], c->cy[j] - c->cy[i]);
                if (!(d > 0))
                    continue;
                ww[ns.n] = d;
                wd[ns.n] = d;
                znb_ipush(&ns, i);
                znb_ipush(&nd, j);
                ls[nl] = i;
                ld[nl] = j;
                lw[nl++] = d;
                if (!crosses[e]) {
                    fs[nf] = i;
                    fd[nf] = j;
                    fw[nf++] = d;
                }
            }
            for (int side = 0; side < 2; side++)
                for (int e = lc[side].off[i]; e < lc[side].off[i + 1]; e++) {
                    int j = lc[side].dst[e];
                    if (j == i || znb_csr_has(&c->lon, i, j))
                        continue;
                    int dup = 0;
                    for (int k = ns.n - 1; k >= 0 && ns.a[k] == i && !dup; k--)
                        dup = nd.a[k] == j;
                    double d = hypot(c->cx[j] - c->cx[i], c->cy[j] - c->cy[i]);
                    if (dup || !(d > 0))
                        continue;
                    ww[ns.n] = d + cfg->lane_change_cost;
                    wd[ns.n] = d;
                    znb_ipush(&ns, i);
                    znb_ipush(&nd, j);
                }
        }
        ZnbCsr g = znb_csr(N, ns.n, ns.a, nd.a, ww);
        nav_w_rev = znb_csr_transpose(&g);
        znb_csr_free(&g);
        g = znb_csr(N, ns.n, ns.a, nd.a, wd);
        nav_d_rev = znb_csr_transpose(&g);
        znb_csr_free(&g);
        nav_rev_struct = nav_d_rev;
        lg = znb_csr(N, nl, ls, ld, lw);
        lg_rev = znb_csr_transpose(&lg);
        f8 = znb_csr(N, nf, fs, fd, fw);
        f8_rev = znb_csr_transpose(&f8);
        free(ns.a);
        free(nd.a);
        free(ww);
        free(wd);
        free(ls);
        free(ld);
        free(lw);
        free(fs);
        free(fd);
        free(fw);
    }
    free(crosses);

    // ---- C5: specified goals ----
    int A = agents ? agents->num_agents : 0;
    int *agent_set = (int *)znb_alloc(A, sizeof(int)); // index into unique sets, -1 none
    ZnbGoalSet *sets = (ZnbGoalSet *)znb_alloc(A + 1, sizeof(ZnbGoalSet));
    int *set_primary = (int *)znb_alloc(A + 1, sizeof(int));
    int num_sets = 0;
    {
        ZnbGoalSet *agent_sets = (ZnbGoalSet *)znb_alloc(A, sizeof(ZnbGoalSet));
        int *agent_closest = (int *)znb_alloc(A, sizeof(int));
        for (int a = 0; a < A; a++) {
            agent_set[a] = -1;
            agent_closest[a] = -1;
            if (N == 0 || agents->goal_x == NULL || agents->goal_y == NULL || !isfinite(agents->goal_x[a]) ||
                !isfinite(agents->goal_y[a]))
                continue;
            double gx = agents->goal_x[a], gy = agents->goal_y[a], best = ZNB_INF;
            for (int s = 0; s < N; s++) {
                double d = znb_point_segment_distance(gx, gy, c->x0[s], c->y0[s], c->x1[s], c->y1[s]);
                if (d < best) {
                    best = d;
                    agent_closest[a] = s;
                }
            }
            ZnbIVec mem = {0};
            if (!cfg->snap_goals)
                for (int s = 0; s < N; s++)
                    if (znb_point_segment_distance(gx, gy, c->x0[s], c->y0[s], c->x1[s], c->y1[s]) <=
                        cfg->goal_set_radius)
                        znb_ipush(&mem, s);
            if (mem.n == 0)
                znb_ipush(&mem, agent_closest[a]);
            agent_sets[a].members = mem.a;
            agent_sets[a].n = mem.n;
        }
        // Unique sets, sorted; primary = closest sub-line of the first agent with that set.
        for (int a = 0; a < A; a++) {
            if (agent_closest[a] < 0)
                continue;
            int found = 0;
            for (int k = 0; k < num_sets && !found; k++)
                found = znb_cmp_goalset(&sets[k], &agent_sets[a]) == 0;
            if (!found) {
                sets[num_sets].n = agent_sets[a].n;
                sets[num_sets].members = (int *)znb_alloc(agent_sets[a].n, sizeof(int));
                memcpy(sets[num_sets].members, agent_sets[a].members, agent_sets[a].n * sizeof(int));
                sets[num_sets].seed = NULL;
                if (cfg->goal_set_along_lane_cost && agent_sets[a].n > 1) {
                    double *seed = (double *)znb_alloc(agent_sets[a].n, sizeof(double)), lo = ZNB_INF;
                    for (int k = 0; k < agent_sets[a].n; k++) {
                        int m = agent_sets[a].members[k];
                        double dx = c->x1[m] - c->x0[m], dy = c->y1[m] - c->y0[m], len = hypot(dx, dy);
                        double along =
                            len > 0 ? ((agents->goal_x[a] - c->x0[m]) * dx + (agents->goal_y[a] - c->y0[m]) * dy) / len
                                    : 0.0;
                        seed[k] = ZNB_MAX(along, 0.0);
                        lo = ZNB_MIN(lo, seed[k]);
                    }
                    for (int k = 0; k < agent_sets[a].n; k++)
                        seed[k] -= lo;
                    sets[num_sets].seed = seed;
                }
                set_primary[num_sets++] = agent_closest[a];
            }
        }
        // Sort sets together with primaries (insertion sort, few sets).
        for (int k = 1; k < num_sets; k++) {
            ZnbGoalSet s = sets[k];
            int pr = set_primary[k], j = k - 1;
            while (j >= 0 && znb_cmp_goalset(&sets[j], &s) > 0) {
                sets[j + 1] = sets[j];
                set_primary[j + 1] = set_primary[j];
                j--;
            }
            sets[j + 1] = s;
            set_primary[j + 1] = pr;
        }
        for (int a = 0; a < A; a++) {
            if (agent_closest[a] < 0)
                continue;
            for (int k = 0; k < num_sets; k++)
                if (znb_cmp_goalset(&sets[k], &agent_sets[a]) == 0)
                    agent_set[a] = k;
            free(agent_sets[a].members);
        }
        free(agent_sets);
        free(agent_closest);
    }
    if (num_sets > cfg->max_goals) {
        fprintf(stderr, "znavi_build: %d specified goals exceed max_goals=%d; dropping the rest\n", num_sets,
                cfg->max_goals);
        for (int k = cfg->max_goals; k < num_sets; k++)
            free(sets[k].members);
        for (int a = 0; a < A; a++)
            if (agent_set[a] >= cfg->max_goals)
                agent_set[a] = -1;
        num_sets = cfg->max_goals;
    }

    // ---- F5: dead-end goals ----
    ZnbIVec gen = {0};
    if (cfg->dead_end_goals) {
        int dropped = 0;
        for (int s = 0; s < N; s++) {
            if (c->lon.off[s] != c->lon.off[s + 1])
                continue;
            int dup = 0;
            for (int k = 0; k < gen.n && !dup; k++)
                dup = hypot(c->x1[gen.a[k]] - c->x1[s], c->y1[gen.a[k]] - c->y1[s]) < cfg->goal_dedup_distance;
            if (dup)
                continue;
            znb_ipush(&gen, s);
        }
        int m = 0;
        for (int k = 0; k < gen.n; k++) {
            int s = gen.a[k], dup = 0;
            for (int q = 0; q < num_sets && !dup; q++)
                dup = sets[q].n == 1 && sets[q].members[0] == s;
            if (dup)
                continue;
            if (num_sets + m >= cfg->max_goals) {
                dropped++;
                continue;
            }
            gen.a[m++] = s;
        }
        gen.n = m;
        if (dropped > 0 && cfg->verbose)
            fprintf(stderr, "znavi_build: dropped %d dead-end goals beyond max_goals=%d\n", dropped, cfg->max_goals);
    }
    int G = num_sets + gen.n;
    t->num_goals = G;
    t->goal_x = (float *)znb_alloc(G, sizeof(float));
    t->goal_y = (float *)znb_alloc(G, sizeof(float));
    t->goal_sample = (int *)znb_alloc(G, sizeof(int));
    ZnbGoalSet *goal_sets = (ZnbGoalSet *)znb_alloc(G, sizeof(ZnbGoalSet));
    for (int g = 0; g < G; g++) {
        if (g < num_sets) {
            goal_sets[g] = sets[g];
            t->goal_sample[g] = set_primary[g];
            t->goal_x[g] = (float)c->cx[set_primary[g]];
            t->goal_y[g] = (float)c->cy[set_primary[g]];
        } else {
            int s = gen.a[g - num_sets];
            goal_sets[g].members = (int *)znb_alloc(1, sizeof(int));
            goal_sets[g].members[0] = s;
            goal_sets[g].n = 1;
            goal_sets[g].seed = NULL;
            t->goal_sample[g] = s;
            t->goal_x[g] = (float)c->x1[s];
            t->goal_y[g] = (float)c->y1[s];
        }
    }
    free(sets);
    free(set_primary);
    free(gen.a);

    // ---- F6 preparation: agent start candidates and their longitudinal frontiers (F7) ----
    int do_agents = cfg->assign_goals && A > 0;
    ZnbIVec *cands = (ZnbIVec *)znb_alloc(A, sizeof(ZnbIVec));
    int *cand_off = (int *)znb_alloc(A + 1, sizeof(int));
    int num_cand = 0;
    unsigned char *is_long = (unsigned char *)znb_alloc(A, 1);
    if (do_agents) {
        for (int a = 0; a < A; a++) {
            znb_start_candidates(c, agents->x[a], agents->y[a], agents->heading[a], &cands[a]);
            if (cands[a].n == 0 && cfg->fallback_anchor && agents->fallback_x && agents->fallback_y &&
                agents->fallback_heading)
                znb_start_candidates(c, agents->fallback_x[a], agents->fallback_y[a], agents->fallback_heading[a],
                                     &cands[a]);
            is_long[a] = cfg->long_vehicle_rule && agents->length && agents->length[a] > cfg->long_vehicle_length;
            if (is_long[a] && cands[a].n > 1)
                cands[a].n = 1;
            cand_off[a] = num_cand;
            num_cand += cands[a].n;
        }
    }
    cand_off[A] = num_cand;
    int *cand_node = (int *)znb_alloc(num_cand, sizeof(int));
    for (int a = 0; a < A; a++)
        for (int k = 0; k < cands[a].n; k++)
            cand_node[cand_off[a] + k] = cands[a].a[k];
    // Frontier: nodes first at >= margin along the longitudinal graph from the candidate.
    ZnbIVec *frontier = (ZnbIVec *)znb_alloc(num_cand, sizeof(ZnbIVec));
    double *dist = (double *)znb_alloc(N, sizeof(double));
    if (cfg->virtual_goal_reachability) {
        for (int k = 0; k < num_cand; k++) {
            int s = cand_node[k];
            for (int i = 0; i < N; i++)
                dist[i] = ZNB_INF;
            ZnbHeap *h = &c->heap;
            h->n = 0;
            dist[s] = 0;
            znb_hpush(h, 0, s);
            while (h->n > 0) {
                double d;
                int u;
                znb_hpop(h, &d, &u);
                if (d > dist[u])
                    continue;
                if (d >= cfg->lane_change_margin) {
                    znb_ipush(&frontier[k], u);
                    dist[u] = -1;
                    continue;
                }
                for (int e = lg.off[u]; e < lg.off[u + 1]; e++) {
                    int v = lg.dst[e];
                    double nd = d + lg.w[e];
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        znb_hpush(h, nd, v);
                    }
                }
            }
        }
    }
    // reach bits per candidate and goal: 1 reachable, 2 reachable without lane change
    unsigned char *reach = (unsigned char *)znb_alloc((size_t)num_cand * (G > 0 ? G : 1), 1);

    // ---- Per goal: distances, next hops, feasibilities, reachability ----
    size_t total = (size_t)G * N;
    t->next_sample = (int *)znb_alloc(total, sizeof(int));
    t->distance_to_goal = (float *)znb_alloc(total, sizeof(float));
    t->feasibility_left = (float *)znb_alloc(total, sizeof(float));
    t->feasibility_center = (float *)znb_alloc(total, sizeof(float));
    t->feasibility_right = (float *)znb_alloc(total, sizeof(float));
    double *wdist = (double *)znb_alloc(N, sizeof(double));
    double *long_rev = (double *)znb_alloc(N, sizeof(double));
    double *f8_fwd = (double *)znb_alloc(N, sizeof(double));
    double *f8_rev_d = (double *)znb_alloc(N, sizeof(double));
    double *row = (double *)znb_alloc(N, sizeof(double));
    double *fbuf = (double *)znb_alloc(N, sizeof(double));
    unsigned char *elig = (unsigned char *)znb_alloc(N, 1);
    unsigned char *vg_reach = (unsigned char *)znb_alloc(N, 1);
    int *outdeg = (int *)znb_alloc(N, sizeof(int)), *queue = (int *)znb_alloc(N, sizeof(int));
    uint64_t rng = cfg->seed;
    double *bonus = (double *)znb_alloc(G, sizeof(double));
    for (int g = 0; g < G; g++)
        bonus[g] = cfg->feasibility_bonus_min +
                   (cfg->feasibility_bonus_max - cfg->feasibility_bonus_min) * znb_rng_uniform(&rng);
    for (int g = 0; g < G; g++) {
        const int *mem = goal_sets[g].members;
        int nm = goal_sets[g].n;
        size_t base = (size_t)g * N;
        znb_dijkstra_seeded(&nav_d_rev, mem, goal_sets[g].seed, nm, dist, NULL, &c->heap);
        znb_dijkstra_seeded(&nav_w_rev, mem, goal_sets[g].seed, nm, wdist, t->next_sample + base, &c->heap);
        for (int i = 0; i < N; i++) {
            t->distance_to_goal[base + i] = (float)(dist[i] < ZNB_INF ? dist[i] : cfg->unreachable_distance);
            if (wdist[i] == ZNB_INF)
                t->next_sample[base + i] = -1;
        }
        for (int k = 0; k < nm; k++)
            t->next_sample[base + mem[k]] = -1;
        int need_f8 = cfg->feasibility || (do_agents && cfg->long_vehicle_rule);
        if (need_f8) {
            znb_dijkstra(&f8_rev, mem, nm, f8_rev_d, NULL, &c->heap);
        }
        if (cfg->feasibility) {
            znb_dijkstra(&f8, mem, nm, f8_fwd, NULL, &c->heap);
            znb_feasibility_row(&f8, dist, f8_fwd, f8_rev_d, cfg, row, fbuf, elig, outdeg, queue, &f8_rev);
            for (int i = 0; i < N; i++)
                if (f8_rev_d[i] < ZNB_INF)
                    row[i] += bonus[g];
            for (int i = 0; i < N; i++) {
                double sides[2] = {0, 0};
                for (int side = 0; side < 2; side++) {
                    const ZnbCsr *gs[2] = {&lc[side], &adjacent[side]};
                    for (int q = 0; q < 2; q++)
                        for (int e = gs[q]->off[i]; e < gs[q]->off[i + 1]; e++) {
                            int j = gs[q]->dst[e];
                            if (j != i && row[j] > sides[side])
                                sides[side] = row[j];
                        }
                }
                t->feasibility_left[base + i] = (float)sides[0];
                t->feasibility_center[base + i] = (float)ZNB_MAX(row[i], 0.0);
                t->feasibility_right[base + i] = (float)sides[1];
            }
        }
        if (do_agents && num_cand > 0) {
            int use_vg = cfg->virtual_goal_reachability;
            if (use_vg) {
                znb_dijkstra(&lg_rev, mem, nm, long_rev, NULL, &c->heap);
                // Virtual goals: at/past the last lane-change margin with a successor inside it.
                int sp = 0, any_vg = 0;
                memset(vg_reach, 0, N);
                for (int u = 0; u < N; u++) {
                    if (!(long_rev[u] < ZNB_INF && long_rev[u] >= cfg->lane_change_margin))
                        continue;
                    int vg = 0;
                    for (int e = lg.off[u]; e < lg.off[u + 1] && !vg; e++)
                        vg = long_rev[lg.dst[e]] < cfg->lane_change_margin;
                    if (vg) {
                        vg_reach[u] = 1;
                        queue[sp++] = u;
                        any_vg = 1;
                    }
                }
                if (any_vg) {
                    while (sp > 0) {
                        int u = queue[--sp];
                        for (int e = nav_rev_struct.off[u]; e < nav_rev_struct.off[u + 1]; e++) {
                            int v = nav_rev_struct.dst[e];
                            if (!vg_reach[v]) {
                                vg_reach[v] = 1;
                                queue[sp++] = v;
                            }
                        }
                    }
                } else {
                    for (int u = 0; u < N; u++)
                        vg_reach[u] = dist[u] < ZNB_INF;
                }
            }
            for (int k = 0; k < num_cand; k++) {
                int s = cand_node[k];
                int ok = dist[s] < ZNB_INF;
                if (ok && use_vg && !(long_rev[s] < ZNB_INF)) {
                    ok = 0;
                    for (int q = 0; q < frontier[k].n && !ok; q++)
                        ok = vg_reach[frontier[k].a[q]];
                }
                int nolc = need_f8 ? f8_rev_d[s] < ZNB_INF : 0;
                reach[(size_t)k * G + g] = (unsigned char)(ok | (nolc << 1));
            }
        }
    }

    // ---- F6: agent goals ----
    t->num_agents = A;
    t->agent_goal = (int *)znb_alloc(A, sizeof(int));
    for (int a = 0; a < A; a++)
        t->agent_goal[a] = -1;
    if (do_agents) {
        int *allowed = (int *)znb_alloc(G, sizeof(int));
        for (int a = 0; a < A; a++) {
            int spec = agent_set[a], goal = -1;
            int nc = cands[a].n;
#define ZNB_OK(k, g) ((reach[(size_t)(k) * G + (g)] & 1) && (!is_long[a] || (reach[(size_t)(k) * G + (g)] & 2)))
            if (spec >= 0 && (cfg->retain_unreachable_goals || nc > 0)) {
                if (cfg->retain_unreachable_goals)
                    goal = spec;
                else
                    for (int k = cand_off[a]; k < cand_off[a + 1] && goal < 0; k++)
                        if (ZNB_OK(k, spec))
                            goal = spec;
            }
            if (goal < 0 && cfg->assign_random_goals && G > 0) {
                int na = 0;
                for (int g = 0; g < G; g++) {
                    int ok = 0;
                    for (int k = cand_off[a]; k < cand_off[a + 1] && !ok; k++)
                        ok = ZNB_OK(k, g);
                    if (ok)
                        allowed[na++] = g;
                }
                if (na > 0) {
                    int pick = (int)(znb_rng_uniform(&rng) * na);
                    goal = allowed[pick < na ? pick : na - 1];
                }
            }
#undef ZNB_OK
            t->agent_goal[a] = goal;
        }
        free(allowed);
    }

    // ---- Cleanup ----
    for (int g = 0; g < G; g++) {
        free(goal_sets[g].members);
        free(goal_sets[g].seed);
    }
    free(goal_sets);
    for (int a = 0; a < A; a++)
        free(cands[a].a);
    free(cands);
    for (int k = 0; k < num_cand; k++)
        free(frontier[k].a);
    free(frontier);
    free(cand_off);
    free(cand_node);
    free(is_long);
    free(agent_set);
    free(reach);
    free(dist);
    free(wdist);
    free(long_rev);
    free(f8_fwd);
    free(f8_rev_d);
    free(row);
    free(fbuf);
    free(elig);
    free(vg_reach);
    free(outdeg);
    free(queue);
    free(bonus);
    znb_csr_free(&nav_w_rev);
    znb_csr_free(&nav_d_rev);
    znb_csr_free(&lg);
    znb_csr_free(&lg_rev);
    znb_csr_free(&f8);
    znb_csr_free(&f8_rev);
    for (int s = 0; s < 2; s++) {
        znb_csr_free(&perp[s]);
        znb_csr_free(&adjacent[s]);
        znb_csr_free(&lc[s]);
    }
    free(esrc.a);
    free(edst.a);
    znb_csr_free(&c->lon);
    znb_csr_free(&c->lon_rev);
    for (int i = 0; i < N; i++) {
        free(c->reach[i].node);
        free(c->reach[i].d);
    }
    free(c->reach);
    free(c->ws_dist);
    free(c->ws_stamp);
    free(c->heap.k);
    free(c->heap.v);
    free(c->ivec.a);
    free(c->cells.a);
    free(c->hits.a);
    znb_seggrid_free(&c->obs_grid);
    znb_ptgrid_free(&c->center_grid);
    for (int k = 0; k < 10; k++)
        free(*arrs[k]);
    free(c->poly);
    free(c->chain);
    return 0;
}

#endif // ZNAVI_BUILD_H
