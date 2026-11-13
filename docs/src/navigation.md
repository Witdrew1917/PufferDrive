# Lane navigation

Lane navigation gives agents a lane-level route to their goal. There are two backends, chosen by the map format:

- **Legacy PufferDrive maps** use a precomputed `map_XXX.navi` sidecar next to every `map_XXX.bin` (see [Sidecars](#sidecars)).
- **zpufferdrive (zdrive) v14 maps** carry their own routes; PufferDrive follows zdrive's navigation with a
  projected-distance lane-selection refinement (see [zpufferdrive maps](#zpufferdrive-maps)).

Navigation data is **loaded whenever it is present** (sidecar or built-in sections). It then drives the
`navi_*` metrics and the render overlay. Agents only **use** it when `navigation = 1`. With `navigation = 0`
(default), observations, rewards, goals and the policy are identical to a run without navigation data unless
`snap_goal_to_navi_endpoint` is enabled.

With `navigation = 1` the environment:

- tracks each agent's current lane, distance to goal and lane feasibility,
- extends the **ego observation by 3** (left / center / right lane feasibility),
- appends a **route-waypoint stream** to the observation (`navigation_waypoints` × 3 floats for legacy maps,
  × 2 for zpufferdrive maps),
- enables the `reward_progress_navi` reward,
- for legacy maps only, snaps newly sampled goals to sidecar goal lanes.

## Quick start

```bash
# 1. Maps must be in the current .bin format (see data.md)
python -m pufferlib.ocean.drive.drive maps --data-folder data/processed/training

# 2. Build sidecars next to the bins (bins are not modified)
python -m pufferlib.ocean.drive.drive navigation --map-dir resources/drive/binaries/training --workers 16

# 3. Train with navigation enabled
puffer train puffer_drive --env.navigation 1 --env.map-dir resources/drive/binaries/training

# 4. Render a rollout with the navigation overlay (window: 0, mp4: 1)
scripts/render_navigation.sh resources/drive/binaries/training latest 1
```

The C extension needs no rebuild when switching `navigation` on or off; all settings are read at runtime.

## Configuration

`[env]` section of `pufferlib/config/ocean/drive.ini` (override with `--env.<name>`):

| Key | Default | Description |
|-----|---------|-------------|
| `navigation` | `0` | `1` lets agents use navigation. Legacy maps then need a `.navi` sidecar for every map in `map_dir`. |
| `snap_goal_to_navi_endpoint` | `0` | Snap initial logged goals: project onto the goal lane for legacy maps, or use the nearest route terminal for zpufferdrive maps. Independent of `navigation`; legacy maps require sidecars, and generated goals are unchanged. |
| `navigation_waypoints` | `16` | Route waypoints observed per agent: `[1, 64]` (`NAVI_MAX_WAYPOINTS`) for legacy maps, `[1, 50]` (`NAVI_LOOKAHEAD_WAYPOINTS_MAX`) for zpufferdrive maps. |
| `navigation_waypoint_spacing` | `5.0` | Legacy maps: metres between consecutive waypoints along the route (> 0). |
| `navigation_lookahead_distance` | `120.0` | zpufferdrive maps: route arclength covered by the waypoints, in metres. Must satisfy `0 < distance <= navigation_lookahead_norm`. |
| `navigation_lookahead_norm` | `150.0` | zpufferdrive maps: metres that waypoint coordinates are divided by. |

> [!IMPORTANT]
> `navigation`, `navigation_waypoints` and the map format change the observation size. A policy trained with one
> setting cannot be loaded with another. `navigation_waypoint_spacing`, `navigation_lookahead_distance` and
> `navigation_lookahead_norm` keep the shape but change the meaning of the inputs.

## Sidecars

### Generating

```bash
python -m pufferlib.ocean.drive.drive navigation --map-dir <map_dir> [--workers N] [--max-goals 64] [--seed 42] [--overwrite]
```

- `--max-goals`: goal lanes per map (default 64, max 1024). Lanes holding a logged agent goal are always included.
  The rest are chosen by farthest-point sampling, dead-end lanes first.
- `--seed`: seed for goal selection.
- `--overwrite`: rebuild sidecars that already exist. Without it existing sidecars are skipped.

Each sidecar stores a CRC-32 and the byte size of its `.bin`. If a map is regenerated, its sidecar becomes stale and
environment creation aborts with a fingerprint mismatch. Rebuild with `--overwrite`. Creating an env with
`navigation = 1` or `snap_goal_to_navi_endpoint = 1` also fails early, with the command to run, if any map lacks a sidecar. Sidecars that exist are
also loaded (and validated) with `navigation = 0`, for metrics and rendering.

Split-aware navigation and overlapping goal terminals require version 4 sidecars. Regenerate older sidecars with
`--overwrite` and rebuild the native extension; map binaries and policy observation shapes do not change.
Existing weights remain loadable,
but their guidance and evaluation results can change at ambiguous intersections.

### What is computed (`pufferlib/ocean/drive/navigation.py`)

Map bins carry no lane connectivity, so it is inferred from geometry:

- **Successors**: lane B follows lane A if B starts within 0.5 m of A's end with a compatible heading.
- **Split branches**: nearby starts or common predecessors identify candidate siblings. A pair must share at least
  2 m of aligned approach, with centerlines no more than 1 m apart and tangent cosine at least 0.9. The shared
  prefix ends when either condition fails. Relations are pairwise, not transitive, and do not permit jumping
  between separate turn pockets, crossing lanes, or ordinary adjacent lanes.
- **Lateral neighbours**: probed at 25/50/75 % of the lane, one lane width (3.7 m) to each side. All probes must hit
  the same aligned lane, and no road edge may lie between the lanes (a road edge forbids the lane change).
- **Goal lanes**: the visible goal is the end point of a representative goal lane. Each object's logged goal is
  matched to the lane
  containing it (aligned with the object's final heading). That gives `object_goal_idx`, the goal each   agent drives
  to at reset. Other same-direction lanes sharing the goal's final 2 m of approach (within 1 m, tangent cosine
  at least 0.9) receive hidden routing copies at the goal's exact projection. This includes a straight lane
  overlapping the end of a turning lane, but excludes adjacent, crossing and opposing lanes.
  Copies share the original goal ID and marker and do not consume additional goal slots or move the goal.
- **Per goal × lane tables** (Dijkstra toward the goal lane's end, lane change cost 10 m):
  - `next_lane`: longitudinal next hop toward any copy of the goal without lane changes, or -1.
  - `longitudinal_distance_to_goal_m`: distance from the lane start without lane changes, used to rank ambiguous
    split branches. A lane with `next_lane = -1` and a distance below `1e9` is a terminal: the distance is the
    stopping arc length on that lane, which may be before its end. `next_lane = -1` with an unreachable
    longitudinal distance still means no longitudinal route. This reuses the Dijkstra pass already needed for
    `next_lane`, seeded with all copies of the same goal.
  - `distance_to_goal_m`: distance from the lane start to the goal with lane changes allowed, or `1e9` (unreachable).
  - `feasibility_left_m / center_m / right_m`: metres the agent can keep following the left / own / right lane
    (taking the best successor) while the goal stays reachable, capped at 1000 m. 0 means that lane is not
    available or does not lead to the goal.

Split detection compares local candidate prefixes rather than all pairs of lane polylines. It adds one
`float32[num_goals, num_lanes]` distance table (20% more storage for the per-goal/lane tables) and sparse split
metadata, but no additional Dijkstra pass. Runtime selection projects onto local sibling candidates; it does not
run a shortest-path search per agent.

### File format

Little-endian. 44-byte header: magic `"PDNAVI01"`, `version` (int32 = 4), `map_crc32` (uint32), `map_size_bytes`
(uint64), then `num_objects`, `num_roads`, `num_lanes`, `num_goals`, `num_branches` (int32). Followed by:

| Array | Type | Size |
|-------|------|------|
| `lane_entity_idx`, `left_lane`, `right_lane` | int32 | `num_lanes` |
| `goal_lane` | int32 | `num_goals` |
| `goal_x`, `goal_y` | float32 | `num_goals` |
| `object_goal_idx` | int32 | `num_objects` |
| `next_lane` | int32 | `num_goals × num_lanes` |
| `distance_to_goal_m`, `feasibility_left_m`, `feasibility_center_m`, `feasibility_right_m` | float32 | `num_goals × num_lanes` |
| `longitudinal_distance_to_goal_m` | float32 | `num_goals × num_lanes` |
| `branch_offsets` | int32 | `num_lanes + 1` |
| `branch_lane` | int32 | `num_branches` |
| `branch_end_m`, `branch_other_end_m` | float32 | `num_branches` |

`branch_offsets[lane]:branch_offsets[lane + 1]` selects its sibling entries. Each entry holds the sibling index
and shared-prefix arc-length limits on the source and sibling lanes. Entries in each row are strictly ascending.

Version 4 retains version 3's array layout; the version bump identifies the new multi-terminal routing semantics.
Hidden goal copies require no additional arrays: a finite longitudinal distance with no next hop encodes a
terminal's stopping position.

Lane indices in the sidecar are *navi lane* indices (position in `lane_entity_idx`), not entity indices. The C loader
(`load_navigation_binary` in `drive.h`) rejects missing, truncated, trailing-byte, out-of-range or stale files.

## Runtime state (C, `drive.h`)

Every step, for each controlled agent:

1. **Current lane** (`update_current_lane`): the closest `ROAD_LANE` within 4 m. Lanes whose heading differs by more
   than 30° get a 3 m distance penalty. This remains the physical lane match.
2. **Navigation anchor** (`navigation_branch_lane`): while projections onto sibling branches remain within their
   shared prefixes, within 1 m of the agent and one another, and aligned with its heading, choose the shorter
   reachable longitudinal route from the projected position. Near ties (within 0.1 m) prefer the previous
   navigation anchor, then geometric fit and lane index. If none has a longitudinal route, keep the physical
   match. After divergence the physical branch wins, even if it is longer. Off-lane preserves the last known
   anchor and state. No goal means normal geometric matching.
3. **Navigation state** (`update_navigation_state`): from the selected anchor's lane-start table values minus the
   agent's projected progress, compute remaining distance, off-route status, and left / center / right feasibility.
   These retain their original lane-change-aware semantics; only split ranking excludes lane changes. When sibling
   anchors change, progress is measured on the new anchor for both positions, so movement still counts but
   selection alone cannot earn progress reward.
   Side feasibility uses projection onto the selected anchor's neighbouring lane.
   The progress reward requires a new closest navigation distance for the current goal; retracing a route does not
   earn the same progress again. The closest distance is initialized again when a new goal is sampled.
4. **Goals**: each agent retains its logged goal and associated `object_goal_idx` by default. With
   `snap_goal_to_navi_endpoint = 1`, the initial goal is projected onto its associated goal lane after agent
   selection; agents without a navigation goal retain their logged goal. Sidecar routes and their distance tables
   still lead to the lane's original terminal, which may be beyond the projected goal. When a new goal is sampled
   (`sample_new_goal`, e.g. goal behavior 1), it is snapped to the reachable sidecar goal whose route distance is
   closest to `goal_target_distance`, resolving ambiguous branches for each prospective goal without changing
   agent state during comparison. If no sidecar goal is reachable, the normal lane-point sampling is used and
   the agent has no navigation goal.

## zpufferdrive maps

zpufferdrive v14 binaries (`map_XXX.bin` written by zpufferdrive's `scripts/python/process_maps.py`) are detected
automatically: the file starts with version 14, at least 4 section offsets, and the offsets are increasing and
within the file. The format of a map directory is taken from `map_000.bin`; all maps in it must share that format.
`pufferlib/ocean/drive/zbin.h` holds the loader and the navigation runtime. No sidecar or conversion is needed.

### Scene

Sections 1 and 2 populate the agents and roads:

- Object types: vehicle, pedestrian and cyclist keep their type. Motorcyclist becomes cyclist.
- Road types: solid road line becomes road line, and road edge, stop sign, crosswalk, speed bump and driveway are
  shifted down by one to PufferDrive's numbering. Each road keeps its `speed_limit`, which only navigation uses.
- Each object's goal is its last valid logged position, as in legacy PufferDrive maps (the scenario JSON
  `goalPosition`). zdrive's route-snapped `goal_x`, `goal_y` is ignored, so a route (`goal_index`, which is kept) can
  end a few metres from the goal. With `snap_goal_to_navi_endpoint = 1`, initial controlled-agent goals instead
  use the nearest usable terminal of that route; agents without one keep their logged goal (with a warning when a
  route exists). Snapping occurs after agent selection and initialization, so eligibility is determined by the
  logged goals.
  Objects with a non-zero `expert_type`, and zombies, are marked as experts and replay their logs.
  Expert-action arrays are filled with -1.
- Up to 128 objects are loaded. `scenario_id` is truncated to 16 characters. Hybrid-action and turn-indicator data
  are ignored.

### Navigation runtime

Sections 3 and 4 hold, per goal, a route graph over sub-sampled lane segments ("navi entities"). Each sample has a
next hop (which may move to another lane or mid-lane), the remaining distance to the goal, and left / center /
right feasibility (capped at 500 m). The loader validates every index and rejects truncated or trailing-byte files
with `malformed zpufferdrive binary`.

Each step, for each agent with a route, PufferDrive follows zdrive's non-expert path (`compute_agent_metrics` in
`zdrive_agent_obs.c`), except for the projected-distance refinement below.

1. **Localize**: find the closest lane, with a 1000 m penalty for lanes whose heading differs by more than 60°.
   Among lanes within 4 m, prefer those routed to the agent's goal. Near-ties (within 0.75 m) go to the shorter
   distance to goal at the projected sample. Unlike zdrive's first-reachable-sample ranking, a candidate is
   reachable only if that projected sample is reachable. If none qualifies, use the geometric match.
   Candidates come from the same spatial grid and in the same order as zdrive, since the tie-break depends on it.
2. **Look up** the navi entity and point on that lane. `distance_to_goal` and feasibility are read at point
   `max(p - 1, 0)`. Values of 1999.99 m or more, or a failed lookup, mean unreachable: 0 feasibility and
   2000 m distance (`navi_off_route`).
3. **Goals**: the agent keeps its original route, including after `goal_behavior = 1` resamples the visible goal
   with the normal (non-snapping) lane-point sampler, as zdrive does. `reward_progress_navi` is driven by
   `distance_to_goal`.

## Observations

Layout with `navigation = 1`:

```
[ ego (dynamics ego + 3) | partners ((MAX_AGENTS-1) × 7) | roads (128 × 7) | waypoints (navigation_waypoints × F) ]
```

`F` (`env.waypoint_features`) is 3 for legacy maps and 2 for zpufferdrive maps. The partner and road blocks keep
their offsets. Waypoints come last.

### Ego: lane feasibility (+3)

The last 3 ego features are `left, center, right` lane feasibility: how far the agent can stay in that lane and
still reach its goal. 0 means the lane doesn't exist, isn't allowed or doesn't lead to the goal. All three are 0
when the agent has no navigation goal.

- Legacy maps: `min(feasibility_m / 200, 1)`, a 200 m horizon.
- zpufferdrive maps: `clamp(feasibility_m / speed_limit / 30, 0, 1)`, time at the current lane's speed limit over a
  30 s horizon. All three are 0 when the lane has no speed limit.

### Waypoints, zpufferdrive maps (`navigation_waypoints` × 2)

Per waypoint: `x_forward, y_left`, in the ego frame and divided by `navigation_lookahead_norm`. Waypoint `w`
(0-based) lies `(w + 1) × navigation_lookahead_distance / navigation_waypoints` metres along the route from the
agent's localized navi point, following the route's next hops (zdrive's `NaviWalk`). Slots beyond the route end,
and all slots for agents without a route, are `(0, 0)`. There is no valid flag.

### Waypoints, legacy maps (`navigation_waypoints` × 3)

Per waypoint: `rel_x, rel_y, valid`. Positions are in the ego frame and scaled by 0.02, like partner and road
features.

Route walk (`navigation_route_waypoints`):

- Start at the agent's projection onto its resolved navigation anchor (or last known anchor when off-lane).
- Every `navigation_waypoint_spacing` metres along the lane polyline, emit a waypoint.
- At the lane end, continue on `next_lane`.
- If the current lane has no `next_lane` toward the goal (a lane change is needed), hop to the left or right
  neighbour with the smaller `distance_to_goal_m`, then keep walking from the projected position. Distance strictly
  decreases, so the walk always terminates.
- Stop at the sidecar goal's projected position on any terminal lane, even if the lane continues beyond it,
  or when no hop is possible.

Unused slots, and all slots for agents without a goal or off route, are zero (`valid = 0`).

## Policy (`pufferlib/ocean/torch.py`)

The `Drive` policy reads `env.max_waypoints` and `env.waypoint_features`. When navigation is on it adds a fourth
branch next to the ego, partner and road encoders:

- `waypoint_encoder`: a shared per-waypoint `Linear(F, input_size) → LayerNorm → Linear(input_size, input_size)`,
  max-pooled over waypoints.
- `shared_embedding` input grows from `3 × input_size` to `4 × input_size`.

The ego encoder picks up the 3 extra features automatically through `env.ego_features`. With navigation off, the
model and its checkpoints are unchanged.

The C demo policy (`drivenet.h`, used by `drive.c`) has no waypoint branch and refuses to run with `navigation = 1`.

## Metrics

Logged whenever navigation data is loaded:

- `navi_goal_active_rate`: fraction of agent-steps with a navigation goal.
- `navi_off_route_rate`: fraction of agent-steps on a lane from which the goal is unreachable.

## Rendering

When navigation data is loaded the renderer draws, for the selected agent:

- lanes coloured by feasibility toward its goal (red → green, grey = unreachable),
- its route in cyan: legacy `next_lane` from the resolved navigation anchor, or the zdrive route lookahead,
- its observed route waypoints (small spheres),
- its goal.

Only the selected goal marker is drawn. Routes and observed waypoints can reach any hidden copy and stop there;
they do not continue to the end of a longer overlapping lane.

The observation-only view also decodes the waypoint block. `scripts/render_navigation.sh [MAP_DIR] [MODEL|latest|none]
[0|1]` generates missing sidecars and runs `puffer eval` with the overlay.

## Tests

```bash
python -m unittest tests.test_drive_navigation tests.test_drive_zbin
```

`test_drive_navigation` covers sidecar topology and round trip, `next_lane` termination, road-edge-blocked lane
changes, reader validation, observation sizes, waypoint count and spacing, lane-change waypoints and stale-sidecar
rejection.

`test_drive_zbin` covers zpufferdrive maps on a synthetic binary: format detection, observation sizes and values,
route-end padding, `navigation = 0` ignoring loaded routes, unmoved goals, config validation and truncated-file
rejection. Its parity test compiles `tests/zdrive_navigation_oracle.c` against a zpufferdrive checkout and
`tests/zbin_navigation_harness.c` against PufferDrive. Both teleport every object to each logged pose and print
`distance_to_goal`, feasibility and the lookahead with `%.9g`. The existing real-map fixture outputs match byte for
byte; other maps may differ where projected-sample lane selection differs from zdrive. Configure it with
`ZPUFFERDRIVE_ROOT` (default `~/Repositories/zpufferdrive`), `ZBIN_MAP_DIR` (default
`~/Data/zpufferdrive/waymo_binaries/interactive_1k`) and `ZBIN_PARITY_MAPS` (default 20). It is skipped when these
are missing.
