# Navigation sidecars

Goal-directed lane routing for gigaflow mode. It replaces the random-walk route
(`compute_new_route`) with routes toward goals sampled from a per-map navigation
table. This is the 3.0 port of zpufferdrive's navigation sections.

Navigation lives in a separate `<map_stem>.navi` file, so map binaries and the
map loader are unchanged. The suffix is not `.bin` because every `*.bin` in
`map_dir` is loaded as a map.

## Generating

```bash
python data_utils/process_navigation.py --input data/nuplan/mini/train            # .navi next to each map
python data_utils/process_navigation.py --input maps/ --output-dir navi/ --workers 16 --overwrite
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | required | map `.bin` or a directory of them |
| `--output-dir` | next to each map | where to write `.navi` files |
| `--max-goals` | 64 | goals per map (dead ends + logged route ends, then farthest-point sampling) |
| `--seed` | 42 | farthest-point sampling seed |
| `--workers`, `--timeout-s` | half the CPUs, 600 s | worker processes, per-map timeout |
| `--overwrite` | off | rebuild existing sidecars |
| `--plot-dir` | off | also write HTML/SVG plots (see Inspecting) |

Sidecars are bound to the exact map bytes (CRC32 + size). After regenerating a
map, regenerate its sidecar.

## Inspecting

`data_utils/plot_navigation.py` replaces zpufferdrive's verbose SVG/HTML map
overview. It needs `plotly` and `matplotlib`. Each map gets
`<map_stem>.html` and `<map_stem>.svg`:

```bash
python data_utils/plot_navigation.py --input <map.bin|map_dir> [--navigation-dir navi/] \
    [--output-dir navi_plots] [--goal 0] [--format html svg] [--max-maps 20]
python data_utils/process_navigation.py --input maps/ --output-dir navi/ --plot-dir navi_plots/   # plot while generating
```

- **HTML** (open in a browser; plotly is loaded from its CDN): pan and zoom,
  with legend toggles for edges, road lines, lanes, left/right lane changes and
  goals. A dropdown selects the goal. For the selected goal, lanes that reach it
  by following `next_lane` are blue and lanes that reach it only through a lane
  change are violet. Per-lane arrows show driving direction, coloured by
  `distance_to_goal_m`. Hovering an arrow shows the lane and road index, next
  hop, distance and left/center/right feasibility.
- **SVG**: the same layers for the single goal given by `--goal`.

## Using

```bash
puffer train puffer_drive env.simulation_mode=gigaflow env.goal_source=route \
    env.route_source=navigation env.navigation_dir=navi/
```

- `env.route_source`: `random_walk` (default) or `navigation`. Navigation
  requires `simulation_mode=gigaflow` with `goal_source=route`.
- `env.navigation_dir`: the sidecar directory. An empty value means the map's
  own directory.
- **Missing sidecar**: a warning is printed and that map falls back to random walk.
- **Corrupt or stale sidecar** (bad magic or version, size, fingerprint, an index
  out of range, a `next_lane` hop that is not a lane exit, a non-finite value):
  env creation aborts.

At spawn and on route exhaustion, the route chains legs. Each leg picks a goal
uniformly (env RNG) among those longitudinally reachable from the current lane,
then follows `next_lane` to it. Chaining continues up to the route target
length. Lane changes do not appear in the route; they only affect
`distance_to_goal_m` and the feasibility values. Per-agent state is
`navi_goal_idx`, `navi_distance_to_goal_m`, `navi_feasibility_{left,center,right}_m`
and `navi_off_route`. At runtime, feasibility is measured from the agent's position: the progress along its own lane (center) or the neighbour lane (left/right) is subtracted from the sidecar's lane-start value; capped values stay at the cap. The logged metrics are `navi_goal_active_rate` and
`navi_off_route_rate`. Navigation adds no rewards.

### Navigation observations (opt-in)

Set `env.obs_navigation: true` (requires `route_source=navigation`) to append 4
features to the ego state (`ego_features` grows from 11 to 15):

| Feature | Encoding |
|---|---|
| navi active | 1 when the agent follows a navigation goal, 0 on random-walk fallback |
| feas left / lane / right | feasibility in seconds at the current lane speed limit over a 30 s horizon, clamped to [0, 1]; capped (unbounded) feasibility gives 1 |

The feasibilities are 0 when navigation is inactive, when the agent is off-lane, or
when the lane has no speed limit. The flag changes the observation size, so checkpoints
trained with it must be evaluated with it. Eval and fine-tuning restore it from the
checkpoint's `config.yaml`. The default (`false`) keeps old checkpoints compatible.

To inspect navigation in rollouts, see the navigation overlays in [evaluation.md](evaluation.md).

## Format (version 1, little-endian)

```
char[8]  magic "PDNAVI01"
int32    version
uint32   map_crc32           # zlib CRC32 of the map .bin
uint64   map_size_bytes
int32    num_roads, num_lanes (L), num_goals (G)
int32    lane_road_idx[L]    # ascending road indices of lane-type roads (0-9)
int32    left_lane[L], right_lane[L]          # navi lane idx of legal lane change, or -1
int32    goal_lane[G]
float32  goal_x[G], goal_y[G]                 # goal lane end point
int32    next_lane[G][L]                      # longitudinal next hop, -1 at goal / unreachable
float32  distance_to_goal_m[G][L]             # from lane start, 1e9 = unreachable
float32  feasibility_{left,center,right}_m[G][L]   # capped at 1000 m
```

The file size must equal `36 + 4 * (3L + 3G + 5GL)` exactly. Python IO and
validation are in `data_utils/navigation_bin.py`; the C loader is
`load_navigation_binary` in `pufferlib/ocean/drive/map_data.h`.
