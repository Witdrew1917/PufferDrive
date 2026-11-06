## Interactive viewer

Rendering with `render_mode` set to the pop-up window (e.g. `puffer eval` or the standalone `./drive` demo) opens a
resizable interactive viewer. The simulator and eval loop run as usual; the viewer only adds camera, selection and
overlay controls. It starts **paused**: press **SPACE** to run or **ENTER** to advance one step.

### Local rendering

To launch an interactive renderer, first build:

```bash
bash scripts/build_ocean.sh drive local
```

then launch:

```bash
./drive
```

This will run `demo()` with an existing model checkpoint.

## Arguments & Configuration

The `drive` tool supports similar CLI arguments as the visualizer to control the environment and rendering. It also reads the `pufferlib/config/ocean/drive.ini` file for default environment settings.

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--map-name <path>` | Path to the map binary file (e.g., `resources/drive/binaries/training/map_000.bin`). If omitted, picks a random map out of `num_maps` from `map_dir` in `drive.ini`. | Random |
| `--policy-name <path>` | `Path to the policy weights file (.bin).` | `resources/drive/puffer_drive_weights.bin` |
| `--view <mode>` | Selects which views to render: `agent`, `topdown`, or `both`. | `both` |
| `--frame-skip <n>` | Renders every Nth frame to speed up simulation (framerate remains 30fps). | `1` |
| `--num-maps <n>` | Overrides the number of maps to sample from if `--map-name` is not set. | `drive.ini` value |

### Visualization Flags

| Flag | Description |
| :--- | :--- |
| `--show-grid` | Draws the underlying nav-graph/grid on the map. |
| `--obs-only` | Hides objects not currently visible to the agent's sensors (fog of war). |
| `--lasers` | Visualizes the raycast sensor lines from the agent. |
| `--log-trajectories` | Draws the ground-truth "human" expert trajectories as green lines. |
| `--zoom-in` | Zooms the camera mainly on the active region rather than the full map bounds. |

### Controls

Press **H** in the viewer to show the key-binding panel with the current state of each toggle.

**Simulation**

- **SPACE / P** - Pause / resume
- **ENTER / →** - Step one frame (while paused)
- **Q** - Quit viewer (exits the process); **ESC** asks for confirmation. **Ctrl+C** in the terminal also works while paused.

**Camera**

- **V** - Cycle camera: BEV (2D top-down) / 3D orbit / FPV (behind the selected agent)
- **B** - Toggle BEV
- **F** - Follow the selected agent
- **R** - Reset camera to the map
- **LMB drag** - Orbit (3D), pan (BEV) or look around (FPV)
- **RMB / MMB drag** - Pan
- **Mouse wheel** - Zoom towards the cursor
- **Double-click** - Centre view on the clicked point

**Selection**

- **LMB click** - Select the agent under the cursor
- **TAB / SHIFT+TAB** - Next / previous agent
- **S** - Select the agent nearest to the cursor

**Overlays**

- **CTRL** - Toggle observation view (what the selected agent sees)
- **O** - Toggle observation lasers
- **N** - Navigation overlay: off / selected route / route + feasible lanes / all routes (waypoints in orange)
- **L** - Lane feasibility of the selected agent: off / arrows / text. One indicator per lane (left, current, right)
  shows how many metres the goal stays reachable in that lane; red **X** means the goal is unreachable from it
  (arrow length and colour scale red → green up to 200 m). The HUD always shows the values as `feasible L / C / R`.
- **G** - Goal markers: all / selected agent / off
- **T** - Motion trails

**Maps**

- **J / K** - Previous / next map (loads a new map batch and pauses at t = 0; the HUD shows the current map)
