#!/usr/bin/env python3
"""Plan a collision-free waypoint route on a ROS occupancy map."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Waypoint:
    id: str
    x: float
    y: float
    theta: float | None = None


def read_map_yaml(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8")
    resolution_match = re.search(r"^resolution:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    origin_match = re.search(
        r"^origin:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,",
        text,
        re.MULTILINE,
    )
    if not resolution_match or not origin_match:
        raise SystemExit(f"Could not read resolution/origin from {path}.")
    return float(resolution_match.group(1)), float(origin_match.group(1)), float(origin_match.group(2))


def world_to_grid(x: float, y: float, height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    gx = int(round((x - origin_x) / resolution))
    gy = height - int(round((y - origin_y) / resolution))
    return gx, gy


def grid_to_world(gx: int, gy: int, height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[float, float]:
    x = origin_x + gx * resolution
    y = origin_y + (height - gy) * resolution
    return x, y


def inflate_obstacles(blocked: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return blocked.copy()
    height, width = blocked.shape
    result = blocked.copy()
    offsets = [
        (dy, dx)
        for dy in range(-radius_px, radius_px + 1)
        for dx in range(-radius_px, radius_px + 1)
        if dx * dx + dy * dy <= radius_px * radius_px
    ]
    ys, xs = np.nonzero(blocked)
    for y, x in zip(ys, xs, strict=False):
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < height and 0 <= xx < width:
                result[yy, xx] = True
    return result


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    width = blocked.shape[1]
    height = blocked.shape[0]
    if blocked[start[1], start[0]]:
        raise SystemExit(f"Start grid cell {start} is blocked after inflation.")
    if blocked[goal[1], goal[0]]:
        raise SystemExit(f"Goal grid cell {goal} is blocked after inflation.")

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]
    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_set:
        _priority, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < width and 0 <= ny < height) or blocked[ny, nx]:
                continue
            candidate = (nx, ny)
            tentative = g_score[current] + cost
            if tentative >= g_score.get(candidate, float("inf")):
                continue
            came_from[candidate] = current
            g_score[candidate] = tentative
            heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
            heapq.heappush(open_set, (tentative + heuristic, candidate))

    raise SystemExit(f"No path found from {start} to {goal}.")


def simplify_path(path: list[tuple[int, int]], step_px: int) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    last = path[0]
    for point in path[1:-1]:
        if math.hypot(point[0] - last[0], point[1] - last[1]) >= step_px:
            simplified.append(point)
            last = point
    simplified.append(path[-1])
    return simplified


def default_targets() -> list[Waypoint]:
    return []


def load_targets(path: Path | None) -> list[Waypoint]:
    if path is None:
        return default_targets()
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Waypoint(
            str(item.get("id", f"W{index + 1:02d}")),
            float(item["x"]),
            float(item["y"]),
            float(item["theta"]) if item.get("theta") is not None else None,
        )
        for index, item in enumerate(data)
    ]


def compute_orientations(points: list[tuple[float, float]]) -> list[Waypoint]:
    waypoints: list[Waypoint] = []
    for i, point in enumerate(points):
        if i < len(points) - 1:
            nxt = points[i + 1]
            theta = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
        elif waypoints:
            theta = waypoints[-1].theta
        else:
            theta = 0.0
        waypoints.append(Waypoint(f"R{i + 1:03d}", round(point[0], 3), round(point[1], 3), round(float(theta), 3)))
    return waypoints


def route_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:], strict=False))


def draw_route(
    map_image: Path,
    output: Path,
    path_px: list[tuple[int, int]],
    target_px: list[tuple[int, int]],
) -> None:
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    if len(path_px) > 1:
        draw.line(path_px, fill=(220, 20, 20), width=4)
    for index, point in enumerate(target_px, start=1):
        x, y = point
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(20, 90, 230))
        draw.text((x + 9, y - 9), str(index), fill=(20, 90, 230))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a route through the generic simulation map.")
    parser.add_argument("--map-image", type=Path, default=Path("output/generic_sim_map_32x26.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("output/generic_sim_map_32x26.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/generic_sim_route"))
    parser.add_argument("--robot-radius", type=float, default=0.35)
    parser.add_argument("--safety-margin", type=float, default=0.20)
    parser.add_argument("--waypoint-spacing", type=float, default=0.90)
    parser.add_argument("--targets", type=Path, help="Optional JSON list of target waypoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution, origin_x, origin_y = read_map_yaml(args.map_yaml)
    array = np.asarray(Image.open(args.map_image).convert("L"), dtype=np.uint8)
    blocked = array < 80
    inflation_px = int(math.ceil((args.robot_radius + args.safety_margin) / resolution))
    inflated = inflate_obstacles(blocked, inflation_px)

    targets = load_targets(args.targets)
    height = array.shape[0]
    target_grid = [
        world_to_grid(target.x, target.y, height, resolution, origin_x, origin_y)
        for target in targets
    ]
    full_path: list[tuple[int, int]] = []
    for start, goal in zip(target_grid[:-1], target_grid[1:], strict=False):
        segment = astar(inflated, start, goal)
        full_path.extend(segment if not full_path else segment[1:])

    simplified_grid = simplify_path(full_path, max(1, int(round(args.waypoint_spacing / resolution))))
    route_points = [
        grid_to_world(gx, gy, height, resolution, origin_x, origin_y)
        for gx, gy in simplified_grid
    ]
    route = compute_orientations(route_points)

    output_json = args.output.with_suffix(".json")
    output_png = args.output.with_suffix(".png")
    output_csv = args.output.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "map": str(args.map_yml if False else args.map_yaml),
                "robot_radius": args.robot_radius,
                "safety_margin": args.safety_margin,
                "inflation_radius": args.robot_radius + args.safety_margin,
                "route_length": round(route_length(route_points), 3),
                "target_waypoints": [asdict(target) for target in targets],
                "route_waypoints": [asdict(point) for point in route],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_csv.write_text(
        "id,x,y,theta\n"
        + "\n".join(f"{point.id},{point.x:.3f},{point.y:.3f},{point.theta:.3f}" for point in route)
        + "\n",
        encoding="utf-8",
    )
    draw_route(args.map_image, output_png, simplified_grid, target_grid)
    print(f"targets: {len(targets)}")
    print(f"route waypoints: {len(route)}")
    print(f"route length: {route_length(route_points):.2f} m")
    print(f"inflation radius: {args.robot_radius + args.safety_margin:.2f} m")
    print(f"wrote: {output_json}")
    print(f"wrote: {output_csv}")
    print(f"wrote: {output_png}")


if __name__ == "__main__":
    main()
