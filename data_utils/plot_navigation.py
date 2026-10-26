#!/usr/bin/env python3
"""plot_navigation.py: visualize navigation sidecars (.navi) on top of their map binaries.

Port of zpufferdrive's verbose map overview (SVG + interactive HTML) for the 3.0 sidecar format.

Every lane is a potential goal in the v2 sidecar; --goals picks which goal lanes (navi lane idx) to show.

HTML (plotly; pan/zoom, legend toggles, goal dropdown):
  * map            road edges, solid / broken road lines, lanes
  * lateral        legal left (green) / right (orange) lane changes, drawn from each lane to its neighbour
  * goals          the end point of every shown goal lane, labelled g<lane>
  * selected goal  lanes reaching it by following exits only (blue), lanes reaching it only via a lane change
                   (violet), per-lane direction arrows coloured by distance_to_goal_m; hover shows lane/road idx,
                   distance and left/center/right feasibility
SVG (matplotlib): the same layers for the first goal.

Usage:
    python data_utils/plot_navigation.py --input pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin
    python data_utils/plot_navigation.py --input ~/Data/nuplan_mini_val --navigation-dir ~/Data/nuplan_mini_val_navi \\
        --output-dir navi_plots --max-maps 20
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirror_map_bin import _is_lane, read_bin  # noqa: E402
from navigation_bin import NAVI_UNREACHABLE_M, map_fingerprint, navigation_path_for, read_navigation  # noqa: E402
from process_navigation import ROAD_EDGE_TYPE_RANGE, ROAD_LINE_TYPE_RANGE, SOLID_ROAD_LINE_TYPES  # noqa: E402

NAV_BLUE = "#0066FF"
LANE_CHANGE_VIOLET = "#9933CC"
LEFT_LINK_GREEN = "#2CA02C"
RIGHT_LINK_ORANGE = "#FF7F0E"
ROAD_EDGE_COLOR = "#333333"
ROAD_LINE_COLOR = "#999999"
LANE_COLOR = "#D0D0D0"
GOAL_RED = "#D62728"
UNREACHABLE_GRAY = "#B0B0B0"
# left links sit at 1/3 of the lane, right links at 2/3, so a symmetric left/right pair does not overlap
LEFT_LINK_FRACTION = 1.0 / 3.0
RIGHT_LINK_FRACTION = 2.0 / 3.0
MAX_VERTICES_PER_LINE = 12
COORD_DECIMALS = 2
SVG_WIDTH_IN = 14.0
SVG_HEIGHT_RANGE_IN = (4.0, 28.0)
HTML_HEIGHT_PX = 900
DEFAULT_OUTPUT_DIR = "navi_plots"
FORMATS = ("html", "svg")
DEFAULT_GOAL_COUNT = 8


def _subsampled_xy(road):
    xs, ys = np.asarray(road["x"]), np.asarray(road["y"])
    if len(xs) > MAX_VERTICES_PER_LINE:
        keep = np.unique(np.linspace(0, len(xs) - 1, MAX_VERTICES_PER_LINE).round().astype(int))
        xs, ys = xs[keep], ys[keep]
    return np.round(xs, COORD_DECIMALS), np.round(ys, COORD_DECIMALS)


def _point_at(road, fraction):
    vertex_idx = int(round(fraction * (len(road["x"]) - 1)))
    return road["x"][vertex_idx], road["y"][vertex_idx]


def _mid_heading_rad(road):
    vertex_idx = max(1, len(road["x"]) // 2)
    return math.atan2(
        road["y"][vertex_idx] - road["y"][vertex_idx - 1], road["x"][vertex_idx] - road["x"][vertex_idx - 1]
    )


def _lateral_links(roads, navi, neighbours, fraction):
    links = []
    for lane, neighbour in enumerate(neighbours):
        if neighbour < 0:
            continue
        start = _point_at(roads[navi["lane_road_idx"][lane]], fraction)
        end = _point_at(roads[navi["lane_road_idx"][neighbour]], fraction)
        links.append((start, end))
    return links


def map_layers(data, navi):
    """Static geometry: lists of road indices per category, lateral link segments."""
    roads = data["roads"]
    edges, solid_lines, broken_lines = [], [], []
    for road_idx, road in enumerate(roads):
        road_type = road["type"]
        if ROAD_EDGE_TYPE_RANGE[0] <= road_type <= ROAD_EDGE_TYPE_RANGE[1]:
            edges.append(road_idx)
        elif ROAD_LINE_TYPE_RANGE[0] <= road_type <= ROAD_LINE_TYPE_RANGE[1]:
            (solid_lines if road_type in SOLID_ROAD_LINE_TYPES else broken_lines).append(road_idx)
    lanes = [road_idx for road_idx, road in enumerate(roads) if _is_lane(road["type"])]
    return {
        "edges": edges,
        "solid_lines": solid_lines,
        "broken_lines": broken_lines,
        "lanes": lanes,
        "left_links": _lateral_links(roads, navi, navi["left_lane"], LEFT_LINK_FRACTION),
        "right_links": _lateral_links(roads, navi, navi["right_lane"], RIGHT_LINK_FRACTION),
    }


def goal_lane_sets(data, navi, goal):
    """Navi-lane masks: reaches the goal lane by following exits only, and reaches it only with a lane change."""
    roads = data["roads"]
    lane_road_idx = navi["lane_road_idx"]
    road_to_navi = {int(road_idx): lane for lane, road_idx in enumerate(lane_road_idx)}
    predecessors = [[] for _ in range(navi["num_lanes"])]
    for lane, road_idx in enumerate(lane_road_idx):
        for exit_road in roads[road_idx]["exit_lanes"]:
            if exit_road in road_to_navi:
                predecessors[road_to_navi[exit_road]].append(lane)
    longitudinal = np.zeros(navi["num_lanes"], dtype=bool)
    longitudinal[goal] = True
    frontier = [goal]
    while frontier:
        lane = frontier.pop()
        for predecessor in predecessors[lane]:
            if not longitudinal[predecessor]:
                longitudinal[predecessor] = True
                frontier.append(predecessor)
    reachable = navi["distance_to_goal_m"][goal] < NAVI_UNREACHABLE_M
    return longitudinal, reachable & ~longitudinal


def default_goals(navi, goal_count=DEFAULT_GOAL_COUNT):
    return np.unique(np.linspace(0, navi["num_lanes"] - 1, min(goal_count, navi["num_lanes"])).round().astype(int))


def _goal_point(roads, navi, goal):
    road = roads[navi["lane_road_idx"][goal]]
    return road["x"][-1], road["y"][-1]


def load_map_and_navigation(map_path, navigation_dir=None):
    data = read_bin(Path(map_path))
    navi_path = navigation_path_for(map_path, navigation_dir)
    navi = read_navigation(navi_path)
    if (navi["map_crc32"], navi["map_size_bytes"]) != map_fingerprint(map_path):
        raise ValueError(f"{navi_path}: stale sidecar, map fingerprint mismatch (regenerate it)")
    if navi["num_roads"] != len(data["roads"]):
        raise ValueError(f"{navi_path}: num_roads {navi['num_roads']} != map {len(data['roads'])}")
    return data, navi


def _plotly_lines(go, roads, road_indices, name, color, width, dash="solid", visible=True, showlegend=True):
    xs, ys = [], []
    for road_idx in road_indices:
        line_x, line_y = _subsampled_xy(roads[road_idx])
        xs += line_x.tolist() + [None]
        ys += line_y.tolist() + [None]
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hoverinfo="skip",
        visible=visible,
        showlegend=showlegend,
    )


def _plotly_segments(go, segments, name, color):
    xs, ys = [], []
    for (x0, y0), (x1, y1) in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, line=dict(color=color, width=1.5), hoverinfo="skip")


def _lane_hover(navi, goal, lane):
    feasibility_row = navi["feasibility_center_m"][goal]
    left, right = navi["left_lane"][lane], navi["right_lane"][lane]
    left_m = feasibility_row[left] if left >= 0 else 0.0
    right_m = feasibility_row[right] if right >= 0 else 0.0
    return (
        f"lane {lane} (road {navi['lane_road_idx'][lane]})<br>"
        f"distance_to_goal {navi['distance_to_goal_m'][goal, lane]:.1f} m<br>"
        f"feasibility L/C/R {left_m:.0f} / {feasibility_row[lane]:.0f} / {right_m:.0f} m"
    )


def _goal_traces(go, data, navi, goal, visible):
    roads = data["roads"]
    longitudinal, lane_change_only = goal_lane_sets(data, navi, goal)
    lane_road_idx = navi["lane_road_idx"]
    distances = navi["distance_to_goal_m"][goal]
    reachable = longitudinal | lane_change_only
    mid_points = np.array([_point_at(roads[road_idx], 0.5) for road_idx in lane_road_idx])
    arrow_angles = np.array([90.0 - math.degrees(_mid_heading_rad(roads[road_idx])) for road_idx in lane_road_idx])
    reachable_lanes = np.nonzero(reachable)[0]
    unreachable_lanes = np.nonzero(~reachable)[0]
    max_distance = float(distances[reachable].max()) if reachable.any() else 1.0
    return [
        _plotly_lines(go, roads, lane_road_idx[longitudinal], "via exits only", NAV_BLUE, 3, visible=visible),
        _plotly_lines(
            go, roads, lane_road_idx[lane_change_only], "via lane change", LANE_CHANGE_VIOLET, 2, "dash", visible
        ),
        go.Scatter(
            x=mid_points[reachable_lanes, 0],
            y=mid_points[reachable_lanes, 1],
            mode="markers",
            name="distance_to_goal_m",
            marker=dict(
                symbol="arrow",
                size=10,
                angle=arrow_angles[reachable_lanes],
                color=distances[reachable_lanes],
                colorscale="Viridis",
                cmin=0.0,
                cmax=max_distance,
                colorbar=dict(title="m to goal"),
            ),
            text=[_lane_hover(navi, goal, lane) for lane in reachable_lanes],
            hoverinfo="text",
            visible=visible,
        ),
        go.Scatter(
            x=mid_points[unreachable_lanes, 0],
            y=mid_points[unreachable_lanes, 1],
            mode="markers",
            name="unreachable lanes",
            marker=dict(symbol="arrow", size=7, angle=arrow_angles[unreachable_lanes], color=UNREACHABLE_GRAY),
            text=[_lane_hover(navi, goal, lane) for lane in unreachable_lanes],
            hoverinfo="text",
            visible=visible,
        ),
        go.Scatter(
            x=[_goal_point(roads, navi, goal)[0]],
            y=[_goal_point(roads, navi, goal)[1]],
            mode="markers",
            name="selected goal",
            marker=dict(symbol="star", size=20, color=GOAL_RED, line=dict(color="black", width=1)),
            hovertext=f"goal lane {goal} (road {navi['lane_road_idx'][goal]})",
            hoverinfo="text",
            visible=visible,
        ),
    ]


def render_html(data, navi, output_path, title, goals):
    import plotly.graph_objects as go

    roads = data["roads"]
    layers = map_layers(data, navi)
    goal_points = np.array([_goal_point(roads, navi, goal) for goal in goals])
    static_traces = [
        _plotly_lines(go, roads, layers["lanes"], "lanes", LANE_COLOR, 1),
        _plotly_lines(go, roads, layers["edges"], "road edges", ROAD_EDGE_COLOR, 2),
        _plotly_lines(go, roads, layers["solid_lines"], "solid road lines", ROAD_LINE_COLOR, 1),
        _plotly_lines(go, roads, layers["broken_lines"], "broken road lines", ROAD_LINE_COLOR, 1, "dot"),
        _plotly_segments(go, layers["left_links"], "left lane change", LEFT_LINK_GREEN),
        _plotly_segments(go, layers["right_links"], "right lane change", RIGHT_LINK_ORANGE),
        go.Scatter(
            x=goal_points[:, 0],
            y=goal_points[:, 1],
            mode="markers+text",
            name="goals",
            text=[f"g{goal}" for goal in goals],
            textposition="top center",
            marker=dict(color=GOAL_RED, size=7),
            hovertext=[f"goal lane {goal} (road {navi['lane_road_idx'][goal]})" for goal in goals],
            hoverinfo="text",
        ),
    ]
    goal_traces = [_goal_traces(go, data, navi, goal, goal_slot == 0) for goal_slot, goal in enumerate(goals)]
    traces_per_goal = len(goal_traces[0])
    fig = go.Figure(static_traces + [trace for traces in goal_traces for trace in traces])

    buttons = []
    for goal_slot, goal in enumerate(goals):
        longitudinal, lane_change_only = goal_lane_sets(data, navi, goal)
        visibility = [True] * len(static_traces)
        for other_slot in range(len(goals)):
            visibility += [other_slot == goal_slot] * traces_per_goal
        label = f"goal {goal}: {int(longitudinal.sum())} exit-only / {int(lane_change_only.sum())} lane-change lanes"
        buttons.append(dict(label=label, method="restyle", args=[{"visible": visibility}]))
    fig.update_layout(
        title=title,
        height=HTML_HEIGHT_PX,
        plot_bgcolor="white",
        hovermode="closest",
        updatemenus=[dict(buttons=buttons, active=0, x=0.0, xanchor="left", y=1.08, yanchor="top")],
        legend=dict(itemsizing="constant"),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def render_svg(data, navi, output_path, title, goal):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    roads = data["roads"]
    layers = map_layers(data, navi)
    longitudinal, lane_change_only = goal_lane_sets(data, navi, goal)
    goal_x, goal_y = _goal_point(roads, navi, goal)

    def lines(road_indices):
        return [np.stack([roads[road_idx]["x"], roads[road_idx]["y"]], axis=1) for road_idx in road_indices]

    all_x = np.concatenate([roads[road_idx]["x"] for road_idx in layers["lanes"]])
    all_y = np.concatenate([roads[road_idx]["y"] for road_idx in layers["lanes"]])
    aspect = (np.ptp(all_y) + 1.0) / (np.ptp(all_x) + 1.0)
    height_in = float(np.clip(SVG_WIDTH_IN * aspect, *SVG_HEIGHT_RANGE_IN))
    fig, ax = plt.subplots(figsize=(SVG_WIDTH_IN, height_in))
    ax.add_collection(LineCollection(lines(layers["lanes"]), colors=LANE_COLOR, linewidths=0.6))
    ax.add_collection(LineCollection(lines(layers["edges"]), colors=ROAD_EDGE_COLOR, linewidths=1.0))
    ax.add_collection(LineCollection(lines(layers["solid_lines"]), colors=ROAD_LINE_COLOR, linewidths=0.6))
    ax.add_collection(
        LineCollection(lines(layers["broken_lines"]), colors=ROAD_LINE_COLOR, linewidths=0.6, linestyles="dotted")
    )
    lane_road_idx = navi["lane_road_idx"]
    ax.add_collection(LineCollection(lines(lane_road_idx[lane_change_only]), colors=LANE_CHANGE_VIOLET, linewidths=1.2))
    ax.add_collection(LineCollection(lines(lane_road_idx[longitudinal]), colors=NAV_BLUE, linewidths=1.8))
    ax.add_collection(LineCollection(layers["left_links"], colors=LEFT_LINK_GREEN, linewidths=0.8))
    ax.add_collection(LineCollection(layers["right_links"], colors=RIGHT_LINK_ORANGE, linewidths=0.8))
    ax.scatter([goal_x], [goal_y], s=180, marker="*", color=GOAL_RED, ec="k", zorder=5)
    ax.set_title(
        f"{title}\ngoal lane {goal}: blue = via exits only, violet = via lane change, "
        f"green/orange = legal left/right lane change",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_map(map_path, output_dir, navigation_dir=None, formats=FORMATS, goals=None, data=None, navi=None):
    """Write <output_dir>/<map_stem>.{html,svg}; loads map and sidecar unless both are given."""
    map_path = Path(map_path)
    if data is None or navi is None:
        data, navi = load_map_and_navigation(map_path, navigation_dir)
    goals = default_goals(navi) if not goals else [int(goal) for goal in goals]
    for goal in goals:
        if not 0 <= goal < navi["num_lanes"]:
            raise ValueError(f"goal lane {goal} out of range, {map_path.name} has {navi['num_lanes']} lanes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{map_path.stem}: {navi['num_lanes']} lanes"
    written = []
    if "html" in formats:
        written.append(output_dir / f"{map_path.stem}.html")
        render_html(data, navi, written[-1], title, goals)
    if "svg" in formats:
        written.append(output_dir / f"{map_path.stem}.svg")
        render_svg(data, navi, written[-1], title, goals[0])
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Map .bin file or directory of .bin maps")
    parser.add_argument("--navigation-dir", default=None, help="Sidecar directory (default: next to each map)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument(
        "--goals",
        type=int,
        nargs="+",
        default=None,
        help=f"Goal lanes (navi lane idx) to show; first is drawn in the SVG (default: {DEFAULT_GOAL_COUNT} evenly spaced)",
    )
    parser.add_argument("--max-maps", type=int, default=None, help="Only plot the first N maps (sorted)")
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    maps = [input_path] if input_path.is_file() else sorted(p for p in input_path.iterdir() if p.suffix == ".bin")
    failed = 0
    for map_path in maps[: args.max_maps]:
        try:
            written = render_map(map_path, args.output_dir, args.navigation_dir, args.format, args.goals)
            print(f"[OK] {map_path.name}: " + ", ".join(str(path) for path in written), flush=True)
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"[FAIL] {map_path.name}: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
