# Lane navigation

Lane navigation gives agents a lane-level route to their goal. Every map format uses zpufferdrive (zdrive)
navigation (`ZNavigation`, `pufferlib/ocean/drive/zbin.h`), with a projected-distance lane-selection refinement
(see [zpufferdrive maps](#zpufferdrive-maps)):

- **zpufferdrive v14 maps** carry their own routes (Sections 3-4), which are read at load time.
- **Legacy PufferDrive maps** have no routes; they are built from the road entities
  (see [Built navigation](#built-navigation)) and cached in a `map_XXX.znavi` sidecar next to the map
  (see [Navigation sidecars](#navigation-sidecars)). A missing sidecar is written on first use.
- With `zbin_navigation_source = 1`, routes are built in memory at load time for every map format, for the loaded
  agents only, and no sidecar is read or written.

Navigation data is always loaded. It drives the `navi_*` metrics and the render overlay. Agents only **use** it
when `navigation = 1`. With `navigation = 0` (default), observations, rewards, goals and the policy are identical
to a run without navigation data unless `snap_goal_to_navi_endpoint` is enabled.

With `navigation = 1` the environment:

- tracks each agent's current lane, distance to goal and lane feasibility,
- extends the **ego observation by 3** (left / center / right lane feasibility),
- appends a **route-waypoint stream** to the observation (`navigation_waypoints` × 2 floats),
- enables the `reward_progress_navi` reward.

## Quick start

```bash
# 1. Maps must be in the current .bin format (see data.md); conversion also writes the map_XXX.znavi sidecars
python -m pufferlib.ocean.drive.drive maps --data-folder data/processed/training
# (existing legacy maps) write the missing sidecars up front instead of on first use
python -m pufferlib.ocean.drive.drive navigation --map-dir resources/drive/binaries/training

# 2. Train with navigation enabled
puffer train puffer_drive --env.navigation 1 --env.map-dir resources/drive/binaries/training

# 3. Render a rollout with the navigation overlay (window: 0, mp4: 1)
scripts/render_navigation.sh resources/drive/binaries/training latest 1
```

The C extension needs no rebuild when switching `navigation` on or off; all settings are read at runtime.

## Configuration

`[env]` section of `pufferlib/config/ocean/drive.ini` (override with `--env.<name>`):

| Key | Default | Description |
|-----|---------|-------------|
| `navigation` | `0` | `1` lets agents use navigation. |
| `snap_goal_to_navi_endpoint` | `0` | Snap initial logged goals to the nearest usable terminal of their route. Independent of `navigation`; generated goals are unchanged. |
| `navigation_waypoints` | `16` | Route waypoints observed per agent, `[1, 50]` (`NAVI_LOOKAHEAD_WAYPOINTS_MAX`). |
| `navigation_lookahead_distance` | `120.0` | Route arclength covered by the waypoints, in metres. Must satisfy `0 < distance <= navigation_lookahead_norm`. |
| `navigation_lookahead_norm` | `150.0` | Metres that waypoint coordinates are divided by. |
| `append_goal_to_navi_lookahead` | `0` | `1` extends the route lookahead from its reachable terminal to the mission goal. |
| `zbin_navigation_source` | `0` | `0`: stored navigation; zpufferdrive maps read their Sections 3-4, legacy maps read their `map_XXX.znavi` sidecar (written when missing). `1`: always build navigation in memory at load time for the loaded agents. |

## Navigation PID controller

Select `navi_pid` with `--env.sdc-controller navi_pid` or `--env.non-sdc-controller navi_pid`. It follows
navigation routes even when policy navigation observations are disabled. It uses the localized lane speed limit
multiplied by `navi_pid_speed_limit_fraction`, or `navi_pid_default_speed` when no usable limit is available
(legacy maps have no lane speed limits; built navigation then assumes 50 km/h). For speed control, the controller
previews the route by arclength and can lower the target for upcoming lane limits or for the tightest sampled bend
(`sqrt(4.0 / |curvature|)` in m/s). Preview never raises the current target. Steering still uses local curvature.
`append_goal_to_navi_lookahead = 1` also extends the PID route from its reachable terminal to
the mission goal. The PID uses that final segment for steering and braking even when policy navigation is disabled.

The PID produces actions for the selected `dynamics_model`, just like a policy: acceleration and steering for
`classic`, longitudinal and lateral jerk for `jerk`, or ego-frame `(dx, dy, dyaw)` for `delta_local`. Discrete
actions use the model's bins. The native dynamics update and constrain the motion; for controlled agents, the
generated actions are visible in `env.actions` after `step()`.

For PPO with discrete `delta_local` actions, set `reg_mode = kl_pid_anchor` to use the native PID as a
checkpoint-free teacher. Three target bins travel with each rollout observation but are hidden from the policy.
PPO still controls the car. When `fix_lambdas = True`, set `lambda_value` above zero to give the auxiliary
loss weight; the default is zero.

| Key | Default | Description |
|-----|---------|-------------|
| `navi_pid_speed_gain` | `1.0` | Gain on desired-speed error. |
| `navi_pid_gap_position_gain` | `1.0` | Gain on following-distance error. |
| `navi_pid_gap_velocity_gain` | `1.0` | Gain on lead-versus-ego speed error. |
| `navi_pid_lateral_offset_gain` | `0.01` | Gain on lateral route-offset error (m). |
| `navi_pid_heading_gain` | `0.25` | Gain on heading error (rad). |
| `navi_pid_speed_limit_fraction` | `1.0` | Fraction of the localized lane speed limit used as the target, in `[0, 1]`. |
| `navi_pid_default_speed` | `9.0` | Fallback target speed in m/s when no usable lane limit is available. |
| `navi_pid_speed_lookahead_distance` | `20.0` | Route preview distance in metres, in `[0, 80]`. `0` keeps local-only speed control. |

All gains must be finite and non-negative, and the fallback speed must be positive.

> [!IMPORTANT]
> `navigation` and `navigation_waypoints` change the observation size. A policy trained with one setting cannot
> be loaded with another. `navigation_lookahead_distance` and `navigation_lookahead_norm` keep the shape but change
> the meaning of the inputs. Policies trained with the removed legacy `.navi` navigation (3 features per waypoint)
> cannot be loaded with `navigation = 1`.

## zpufferdrive maps

zpufferdrive v14 binaries (`map_XXX.bin` written by zpufferdrive's `scripts/python/process_maps.py`) are detected
automatically: the file starts with version 14, at least 4 section offsets, and the offsets are increasing and
within the file. The format of a map directory is taken from `map_000.bin`; all maps in it must share that format.
`pufferlib/ocean/drive/zbin.h` holds the loader and the navigation runtime. No conversion is needed.

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
`zdrive_agent_obs.c`), with the lane-selection differences below.

1. **Localize**: find the closest lane, with a 1000 m penalty for lanes whose heading differs by more than 60°.
   Among segments less than 4 m away, find the closest one with a reachable route at the projected sample.
   Then choose the shortest route to the goal from among reachable segments within 1.5 m of that closest
   distance. Unlike zdrive's ranking, the tie window does not move when a farther segment wins on route length.
   The route lengths compared are the remaining distances: the stored sample distance minus the agent's progress
   past the sample start. Otherwise long samples (sparse legacy polylines) would lose to candidates that merely
   start further ahead, such as a merging turn lane.
   If no candidate qualifies, use the geometric match.
2. **Look up** the navi entity and point on that lane. `distance_to_goal` and feasibility are read at point
   `max(p - 1, 0)`. Values of 1999.99 m or more, or a failed lookup, mean unreachable: 0 feasibility and
   2000 m distance (`navi_off_route`).
3. **Goals**: the agent keeps its original route, including after `goal_behavior = 1` resamples the visible goal
   with the normal (non-snapping) lane-point sampler, as zdrive does. `reward_progress_navi` is driven by
   `distance_to_goal`.

### Built navigation

Legacy maps (and zpufferdrive maps with `zbin_navigation_source = 1`) build the Sections 3-4 tables at load time, with
`pufferlib/ocean/drive/znavi_build.h`, a C port of zpufferdrive's `goal.py` that depends only on libc/libm. The
adapter `build_znavigation` in `zbin.h` passes it the road polylines and agents and wraps the result in the same
`ZNavigation`, so the runtime above is unchanged for both map formats.

- **Lanes and obstacles**: lanes are graph nodes, sub-sampled every `0.75 s × max(speed_limit, 1 m/s)`. Road edges
  and zbin solid road lines block lane changes, while other road lines can be crossed. Legacy maps have no solid-line
  information and no lane speed limits, so missing lane speed limits default to 50 km/h, as in zpufferdrive's
  `load_map.py`.
- **Goals**: each agent's logged goal is localized to every lane sub-line within 4 m (`ZBIN_NAVI_GOAL_SET_RADIUS`;
  zpufferdrive `snap_goals=False`). The route therefore ends on whichever nearby lane is best reachable, not on the
  geometrically closest one. Unlike zpufferdrive, each of these goal sub-lines starts the route search with its
  along-lane distance to the goal, measured relative to the closest one. So staying in lane beats a lane change into
  a neighbouring lane whose goal sub-line merely starts earlier. Start candidates are all lanes within 5 m and 90° of the agent.
  - Agents whose goal is unreachable get a random reachable dead-end goal, seeded from the map's CRC-32.
  - Agents longer than 8 m are only routed from their closest start lane.
- **Parity**: on maps regenerated with current zpufferdrive, routes match the stored tables for about 99.5% of
  samples. The remaining differences come from line types that zbin files do not keep: yellow solid lines are
  stored as dashed. Building takes tens of milliseconds per map, versus a few milliseconds to load the sections.

### Navigation sidecars

With `zbin_navigation_source = 0`, legacy maps cache their built navigation in `map_XXX.znavi` next to
`map_XXX.bin`. The sidecar belongs to the map binary, not to a run:

- it routes **every logged object** with at least one valid sample, using the trajectories as loaded (before
  `remove_bad_trajectories` or any other run-dependent preprocessing). Every run then uses the goals of the objects
  it controls, whatever `init_mode`, controllers or agent budget it uses;
- it is bound to the map by the map file's CRC-32 and byte size. If the map changes, or the sidecar is truncated or
  invalid, environment creation fails with an error; regenerate with
  `python -m pufferlib.ocean.drive.drive navigation --map-dir <dir> --overwrite`;
- it is written atomically (temporary file + rename) by `drive.py maps` after each map binary, by the
  `navigation` subcommand (`--map-dir`, `--workers`, `--overwrite`; zpufferdrive maps are skipped), or by the
  environment when the sidecar is missing. If the directory is read-only, the environment only prints a warning
  and uses the built navigation.

Because the sidecar routes every object, its goal set (and so its random fallback goals) can differ from
`zbin_navigation_source = 1`, which only routes the agents a run loads.

## Observations

Layout with `navigation = 1`:

```
[ ego (dynamics ego + 3) | partners ((MAX_AGENTS-1) × 7) | roads (128 × 7) | waypoints (navigation_waypoints × F) ]
```

`F` (`env.waypoint_features`) is 2. The partner and road blocks keep their offsets. Waypoints come last.

### Ego: lane feasibility (+3)

The last 3 ego features are `left, center, right` lane feasibility: how far the agent can stay in that lane and
still reach its goal. 0 means the lane doesn't exist, isn't allowed or doesn't lead to the goal. All three are 0
when the agent has no navigation goal. Each is `clamp(feasibility_m / speed_limit / 30, 0, 1)`, time at the
current lane's speed limit over a 30 s horizon; all three are 0 when the lane has no speed limit.

### Waypoints (`navigation_waypoints` × 2)

Per waypoint: `x_forward, y_left`, in the ego frame and divided by `navigation_lookahead_norm`. Waypoint `w`
(0-based) lies `(w + 1) × navigation_lookahead_distance / navigation_waypoints` metres along the route from the
agent's localized navi point, following the route's next hops (zdrive's `NaviWalk`). Slots beyond the route end,
and all slots for agents without a route, are `(0, 0)`. There is no valid flag.

## Policy (`pufferlib/ocean/torch.py`)

The `Drive` policy reads `env.max_waypoints` and `env.waypoint_features`. When navigation is on it adds a fourth
branch next to the ego, partner and road encoders:

- `waypoint_encoder`: a shared per-waypoint `Linear(F, input_size) → LayerNorm → Linear(input_size, input_size)`,
  max-pooled over waypoints.
- `shared_embedding` input grows from `3 × input_size` to `4 × input_size`.

The ego encoder picks up the 3 extra features automatically through `env.ego_features`. With navigation off, the
model and its checkpoints are unchanged.

The C demo policy (`drivenet.h`, used by `drive.c`) has no waypoint branch and refuses to run with `navigation = 1`.

## Navigation controller

Set `sdc_controller`, `non_sdc_controller`, or `non_vehicle_controller` to `navi_pid` to follow the navigation
route with a kinematic bicycle controller. It uses proportional speed and route-tracking feedback and a
proportional/derivative following-gap controller; it does not use a learned policy or the IDM extension. It works
with either map format and also works with `navigation = 0`, without changing policy observation sizes. With `non_sdc_controller = "navi_pid"`, the default
`non_vehicle_controller = "auto"` replays non-vehicles. Explicitly selected non-expert background agents also use
`navi_pid` when the active-agent limit is reached.

The controller uses lane speed limits, or `navi_pid_default_speed` (9 m/s) when no usable limit is available. It brakes for vehicles
ahead and near the end of the available route. If localization fails or the agent has no reachable navigation
goal, it warns and brakes rather than continuing without guidance. The controller follows the navigation
terminal, which can differ from the visible logged goal, and retains its original route after a goal resample.
Lateral tracking uses 2D route waypoints, so it does not model road elevation changes.

## Metrics

Always logged:

- `navi_goal_active_rate`: fraction of agent-steps with a navigation goal.
- `navi_off_route_rate`: fraction of agent-steps on a lane from which the goal is unreachable.

## Rendering

The renderer draws, for the selected agent:

- lanes coloured by feasibility toward its goal (red → green, grey = unreachable),
- its route in cyan (the zdrive route lookahead),
- its observed route waypoints (small spheres),
- its goal.

The observation-only view also decodes the waypoint block. `scripts/render_navigation.sh [MAP_DIR] [MODEL|latest|none]
[0|1]` runs `puffer eval` with the overlay.

## Tests

```bash
python -m unittest tests.test_drive_navigation tests.test_drive_zbin
```

`test_drive_navigation` covers navigation built at load time on synthetic legacy maps: observation sizes, waypoint
count, goal snapping, goal observation dropout, the `navi_pid` controller and the `map_XXX.znavi` sidecar (round
trip, CLI, stale-map error).

`test_drive_zbin` covers zpufferdrive maps on a synthetic binary: format detection, observation sizes and values,
route-end padding, `navigation = 0` ignoring loaded routes, unmoved goals, config validation and truncated-file
rejection. Its parity test compiles `tests/zdrive_navigation_oracle.c` against a zpufferdrive checkout and
`tests/zbin_navigation_harness.c` against PufferDrive. Both teleport every object to each logged pose and print
the selected route sample, `distance_to_goal`, feasibility and the lookahead. Feature outputs must match byte for
byte when both select the same route sample. Synthetic tests cover PufferDrive's lane-selection rules. Configure it with
`ZPUFFERDRIVE_ROOT` (default `~/Repositories/zpufferdrive`), `ZBIN_MAP_DIR` (default
`~/Data/zpufferdrive/waymo_binaries/interactive_1k`) and `ZBIN_PARITY_MAPS` (default 20). It is skipped when these
are missing.
