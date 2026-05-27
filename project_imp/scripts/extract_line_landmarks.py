#!/usr/bin/env python3
"""Extract line landmarks from an occupancy map with a simple RANSAC pass."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LineLandmark:
    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    center: tuple[float, float]
    length: float
    rms_error: float
    points: int


@dataclass(frozen=True)
class ObstacleLandmark:
    kind: str
    center: tuple[float, float]
    radius: float
    points: int


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


def load_occupied_points(map_image: Path, resolution: float, origin_x: float, origin_y: float) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(map_image).convert("L")
    array = np.asarray(image, dtype=np.uint8)
    ys, xs = np.nonzero(array < 80)
    height = array.shape[0]
    world_x = origin_x + (xs.astype(np.float64) + 0.5) * resolution
    world_y = origin_y + (height - ys.astype(np.float64) - 0.5) * resolution
    return np.column_stack([world_x, world_y]), array


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = np.array([-direction[1], direction[0]])
    distances = centered @ normal
    rms_error = float(math.sqrt(np.mean(distances * distances)))
    return center, direction, rms_error


def make_line_landmark(points: np.ndarray) -> LineLandmark:
    center, direction, rms_error = fit_line(points)
    projections = (points - center) @ direction
    start = center + projections.min() * direction
    end = center + projections.max() * direction
    length = float(np.linalg.norm(end - start))
    return LineLandmark(
        kind="line",
        start=(float(start[0]), float(start[1])),
        end=(float(end[0]), float(end[1])),
        center=(float(center[0]), float(center[1])),
        length=length,
        rms_error=rms_error,
        points=int(len(points)),
    )


def split_continuous_segments(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    direction: np.ndarray,
    max_gap: float,
) -> list[np.ndarray]:
    selected = points[candidate_indices]
    projections = selected @ direction
    order = np.argsort(projections)
    ordered_indices = candidate_indices[order]
    ordered_projections = projections[order]
    segments: list[list[int]] = [[int(ordered_indices[0])]]
    for previous, current, index in zip(ordered_projections[:-1], ordered_projections[1:], ordered_indices[1:], strict=False):
        if float(current - previous) > max_gap:
            segments.append([])
        segments[-1].append(int(index))
    return [np.asarray(segment, dtype=np.int32) for segment in segments if segment]


def ransac_lines(
    points: np.ndarray,
    iterations: int,
    distance_threshold: float,
    max_gap: float,
    min_points: int,
    min_length: float,
    max_error: float,
    max_lines: int,
    seed: int,
) -> tuple[list[LineLandmark], np.ndarray]:
    rng = random.Random(seed)
    remaining = np.arange(len(points), dtype=np.int32)
    lines: list[LineLandmark] = []

    while len(remaining) >= min_points and len(lines) < max_lines:
        best_segment: np.ndarray | None = None
        best_score = 0.0

        for _ in range(iterations):
            local_a, local_b = rng.sample(range(len(remaining)), 2)
            p1 = points[remaining[local_a]]
            p2 = points[remaining[local_b]]
            delta = p2 - p1
            norm = float(np.linalg.norm(delta))
            if norm < min_length:
                continue
            direction = delta / norm
            normal = np.array([-direction[1], direction[0]])
            distances = np.abs((points[remaining] - p1) @ normal)
            candidate_indices = remaining[distances <= distance_threshold]
            if len(candidate_indices) < min_points:
                continue

            segments = split_continuous_segments(points, candidate_indices, direction, max_gap)
            for segment in segments:
                if len(segment) < min_points:
                    continue
                landmark = make_line_landmark(points[segment])
                if landmark.length < min_length or landmark.rms_error > max_error:
                    continue
                score = len(segment) * landmark.length / max(landmark.rms_error, 0.01)
                if score > best_score:
                    best_score = score
                    best_segment = segment

        if best_segment is None:
            break

        line = make_line_landmark(points[best_segment])
        lines.append(line)
        remove = np.zeros(len(points), dtype=bool)
        remove[best_segment] = True
        remaining = remaining[~remove[remaining]]

    return lines, remaining


def cluster_obstacles(points: np.ndarray, indices: np.ndarray, join_distance: float, min_points: int) -> list[ObstacleLandmark]:
    if len(indices) == 0:
        return []
    selected = points[indices]
    cell_size = join_distance
    buckets: dict[tuple[int, int], list[int]] = {}
    for local_index, point in enumerate(selected):
        key = (int(math.floor(point[0] / cell_size)), int(math.floor(point[1] / cell_size)))
        buckets.setdefault(key, []).append(local_index)

    visited = np.zeros(len(selected), dtype=bool)
    obstacles: list[ObstacleLandmark] = []
    for start in range(len(selected)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            item = stack.pop()
            component.append(item)
            x, y = selected[item]
            key_x = int(math.floor(x / cell_size))
            key_y = int(math.floor(y / cell_size))
            for yy in range(key_y - 1, key_y + 2):
                for xx in range(key_x - 1, key_x + 2):
                    for neighbor in buckets.get((xx, yy), []):
                        if visited[neighbor]:
                            continue
                        if float(np.linalg.norm(selected[neighbor] - selected[item])) <= join_distance:
                            visited[neighbor] = True
                            stack.append(neighbor)

        if len(component) >= min_points:
            component_points = selected[component]
            center = component_points.mean(axis=0)
            radius = float(np.percentile(np.linalg.norm(component_points - center, axis=1), 90))
            obstacles.append(
                ObstacleLandmark(
                    kind="obstacle",
                    center=(float(center[0]), float(center[1])),
                    radius=radius,
                    points=int(len(component_points)),
                )
            )
    obstacles.sort(key=lambda item: item.points, reverse=True)
    return obstacles


def world_to_image(point: tuple[float, float], height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    x, y = point
    return int(round((x - origin_x) / resolution)), height - int(round((y - origin_y) / resolution))


def save_overlay(
    map_array: np.ndarray,
    lines: list[LineLandmark],
    obstacles: list[ObstacleLandmark],
    output: Path,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> None:
    image = Image.fromarray(map_array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    height = map_array.shape[0]
    for line in lines:
        start = world_to_image(line.start, height, resolution, origin_x, origin_y)
        end = world_to_image(line.end, height, resolution, origin_x, origin_y)
        draw.line([start, end], fill=(220, 20, 20), width=3)
        draw.ellipse((start[0] - 3, start[1] - 3, start[0] + 3, start[1] + 3), fill=(220, 20, 20))
        draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=(220, 20, 20))
    for obstacle in obstacles:
        center = world_to_image(obstacle.center, height, resolution, origin_x, origin_y)
        radius = max(3, int(round(obstacle.radius / resolution)))
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline=(20, 80, 220), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract straight line landmarks from an occupancy map.")
    parser.add_argument("map_image", type=Path)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/line_landmarks"))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--distance-threshold", type=float, default=0.07)
    parser.add_argument("--max-gap", type=float, default=0.20)
    parser.add_argument("--min-line-points", type=int, default=14)
    parser.add_argument("--min-line-length", type=float, default=0.35)
    parser.add_argument("--max-line-error", type=float, default=0.06)
    parser.add_argument("--max-lines", type=int, default=30)
    parser.add_argument("--obstacle-distance", type=float, default=0.15)
    parser.add_argument("--min-obstacle-points", type=int, default=18)
    parser.add_argument("--max-obstacles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution, origin_x, origin_y = read_map_yaml(args.map_yaml)
    points, map_array = load_occupied_points(args.map_image, resolution, origin_x, origin_y)
    if len(points) < args.min_line_points:
        raise SystemExit("Not enough occupied points.")

    lines, remaining = ransac_lines(
        points,
        iterations=args.iterations,
        distance_threshold=args.distance_threshold,
        max_gap=args.max_gap,
        min_points=args.min_line_points,
        min_length=args.min_line_length,
        max_error=args.max_line_error,
        max_lines=args.max_lines,
        seed=args.seed,
    )
    obstacles = cluster_obstacles(
        points,
        remaining,
        join_distance=args.obstacle_distance,
        min_points=args.min_obstacle_points,
    )[: args.max_obstacles]

    output_json = args.output.with_suffix(".json")
    output_png = args.output.with_suffix(".png")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps([asdict(item) for item in [*lines, *obstacles]], indent=2),
        encoding="utf-8",
    )
    save_overlay(map_array, lines, obstacles, output_png, resolution, origin_x, origin_y)
    print(f"occupied points: {len(points)}")
    print(f"lines: {len(lines)}")
    print(f"obstacles: {len(obstacles)}")
    print(f"remaining points: {len(remaining)}")
    print(f"wrote: {output_json}")
    print(f"wrote: {output_png}")


if __name__ == "__main__":
    main()
