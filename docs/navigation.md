# Navigation sidecars

Goal-conditioned lane navigation. Given the agent's goal, the env tells the
policy how to reach it: how far it can keep its current lane, and whether
changing left or right keeps the goal reachable. Navigation does not produce
goals. It conditions on whatever goal the agent already has (`goal_source`
`route`, `map`, `gt_map` or `external`), so it can run as an optional
observation in any mode where goals have lanes.

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
| `--workers`, `--timeout-s` | half the CPUs, 600 s | worker processes, per-map timeout |
| `--overwrite` | off | rebuild existing sidecars |
| `--plot-dir` | off | also write HTML/SVG plots (see Inspecting) |

Every lane is a goal, so the tables are `L x L` (CARLA and nuPlan maps have
roughly 100 to 700 lanes, giving sidecars of 0.1 to 4 MB built in a few seconds).
Sidecars are bound to the exact map bytes (CRC32 + size). After regenerating a
map, regenerate its sidecar.

## Inspecting

`data_utils/plot_navigation.py` replaces zpufferdrive's verbose SVG/HTML map
overview. It needs `plotly` and `matplotlib`. Each map gets
`<map_stem>.html` and `<map_stem>.svg`:

```bash
python data_utils/plot_navigation.py --input <map.bin|map_dir> [--navigation-dir navi/] \
    [--output-dir navi_plots] [--goals 12 40] [--format html svg] [--max-maps 20]
python data_utils/process_navigation.py --input maps/ --output-dir navi/ --plot-dir navi_plots/   # plot while generating
```

- **HTML** (open in a browser; plotly is loaded from its CDN): pan and zoom,
  with legend toggles for edges, road lines, lanes, left/right lane changes and
  goals. A dropdown selects the goal lane (`--goals`, navi lane indices; default
  8 evenly spaced lanes). For the selected goal, lanes that reach it by
  following exits only are blue and lanes that reach it only through a lane
  change are violet. Per-lane arrows show driving direction, coloured by
  `distance_to_goal_m`. Hovering an arrow shows the lane and road index,
  distance and left/center/right feasibility.
- **SVG**: the same layers for the first goal in `--goals`.

## Using

```bash
puffer train puffer_drive env.goal_source=map env.obs_navigation=true env.navigation_dir=navi/
```

- `env.navigation_dir`: the sidecar directory. An empty value means the map's
  own directory. A map's `<map_stem>.navi` is loaded whenever it exists, so
  navigation state, telemetry and replay overlays work even when the policy does
  not observe it.
- `env.obs_navigation`: appends the navigation features below to the policy
  observation and requires a sidecar for every map. Works with any `goal_source`.
- **Corrupt or stale sidecar** (bad magic or version, size, fingerprint, an index
  out of range, a non-finite or out-of-range value) always aborts env creation; a
  **missing** sidecar aborts only with `env.obs_navigation=true`.

The navigation goal is the **last goal of the agent's current goal set**
(`list_goal_lane[goal_count - 1]`). It is re-read every step, so regenerated or
externally pushed goals retarget navigation immediately. Each step the runtime
looks up the goal's table row (O(1)) and updates `navi_goal_lane_idx`,
`navi_distance_to_goal_m`, `navi_feasibility_{left,center,right}_m` and
`navi_off_route`:

- **distance**: the table value for the agent's lane, minus its progress along
  that lane and the part of the goal lane beyond the goal point.
- **feasibility**: the sidecar's lane-start value minus the agent's progress
  along its own lane (center) or the neighbour lane (left/right). Capped values
  stay at the cap. Feasibility is capped once following exits reaches the goal
  lane. On loop maps, any lane that can still reach the goal is feasible, even
  if that means going around the loop again.
- **off route**: the goal is unreachable from the agent's lane.

Agents whose goal has no lane (`goal_source=gt`) or who have no goals get
inactive (all-zero) navigation. Off-lane agents keep their last lane-relative
state. The logged metrics are `navi_goal_active_rate` and
`navi_off_route_rate`. Navigation adds no rewards.

### Navigation observations

`env.obs_navigation: true` appends 4 features to the ego state (`ego_features`
grows from 11 to 15):

| Feature | Encoding |
|---|---|
| navi reachable | 1 when the agent's final goal is reachable from its lane, else 0 |
| feas left / lane / right | feasibility in seconds at the current lane speed limit over a 30 s horizon, clamped to [0, 1]; capped (unbounded) feasibility gives 1 |

The feasibilities are 0 when the goal is unreachable, when the agent is off-lane, or
when the lane has no speed limit. The flag changes the observation size, so checkpoints
trained with it must be evaluated with it. Eval and fine-tuning restore it from the
checkpoint's `config.yaml`. The default (`false`) keeps old checkpoints compatible.
Checkpoints trained with version 1 sidecars (`route_source=navigation`, removed)
keep the same observation size, but the features now mean goal-conditioned
feasibility.

To inspect navigation in rollouts, see the navigation overlays in [evaluation.md](evaluation.md).

## Format (version 2, little-endian)

```
char[8]  magic "PDNAVI02"
int32    version
uint32   map_crc32           # zlib CRC32 of the map .bin
uint64   map_size_bytes
int32    num_roads, num_lanes (L)
int32    lane_road_idx[L]    # ascending road indices of lane-type roads (0-9)
int32    left_lane[L], right_lane[L]          # navi lane idx of legal lane change, or -1
float32  distance_to_goal_m[L][L]             # [goal lane][lane], lane start -> goal lane end, 1e9 = unreachable
float32  feasibility_center_m[L][L]           # [goal lane][lane], metres along exits with goal reachable, cap 1000
```

Shortest paths use lane length per exit edge and `NAVI_LANE_CHANGE_COST_M`
(10 m) per legal lane change. Left/right feasibility is
`feasibility_center_m[goal][left_lane[lane]]` (or `right_lane`). The file size
must equal `32 + 4 * (3L + 2L^2)` exactly. Python IO and validation are in
`data_utils/navigation_bin.py`; the C loader is `load_navigation_binary` in
`pufferlib/ocean/drive/map_data.h`.
