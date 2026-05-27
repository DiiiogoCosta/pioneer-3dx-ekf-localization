#!/usr/bin/env python3
"""Build an occupancy map from simulated odometry CSV and LiDAR NPZ data."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Pose2D:
    t: float
    x: float
    y: float
    yaw: float


def bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


class OccupancyMap:
    def __init__(
        self,
        width_m: float,
        height_m: float,
        resolution: float,
        origin_x: float,
        origin_y: float,
        log_free: float,
        log_occ: float,
        log_min: float,
        log_max: float,
    ) -> None:
        self.resolution = resolution
        self.width = int(math.ceil(width_m / resolution))
        self.height = int(math.ceil(height_m / resolution))
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.log_free = log_free
        self.log_occ = log_occ
        self.log_min = log_min
        self.log_max = log_max
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(math.floor((x - self.origin_x) / self.resolution))
        gy = int(math.floor((y - self.origin_y) / self.resolution))
        return gx, gy

    def inside(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def add_ray(self, start: tuple[float, float], end: tuple[float, float], hit: bool) -> None:
        x0, y0 = self.world_to_grid(*start)
        x1, y1 = self.world_to_grid(*end)
        cells = list(bresenham(x0, y0, x1, y1))
        free_cells = cells[:-1] if hit else cells
        for gx, gy in free_cells:
            if self.inside(gx, gy):
                self.grid[gy, gx] = max(self.log_min, self.grid[gy, gx] + self.log_free)
        if hit and self.inside(x1, y1):
            self.grid[y1, x1] = min(self.log_max, self.grid[y1, x1] + self.log_occ)

    def to_image_array(self) -> np.ndarray:
        image = np.full((self.height, self.width), 205, dtype=np.uint8)
        image[self.grid < -0.4] = 254
        image[self.grid > 0.85] = 0
        return np.flipud(image)

    def save(self, output_prefix: Path) -> None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(self.to_image_array(), mode="L")
        png_path = output_prefix.with_suffix(".png")
        pgm_path = output_prefix.with_suffix(".pgm")
        yaml_path = output_prefix.with_suffix(".yaml")
        image.save(png_path)
        image.save(pgm_path)
        yaml_path.write_text(
            "\n".join(
                [
                    f"image: {pgm_path.name}",
                    f"resolution: {self.resolution:.6f}",
                    f"origin: [{self.origin_x:.6f}, {self.origin_y:.6f}, 0.0]",
                    "negate: 0",
                    "occupied_thresh: 0.65",
                    "free_thresh: 0.196",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def read_poses(path: Path) -> list[Pose2D]:
    poses: list[Pose2D] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            poses.append(Pose2D(float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"])))
    if not poses:
        raise SystemExit(f"No poses found in {path}.")
    return poses


def nearest_pose(times: list[float], poses: list[Pose2D], t: float, max_age: float) -> Pose2D | None:
    index = bisect_left(times, t)
    choices: list[Pose2D] = []
    if index < len(poses):
        choices.append(poses[index])
    if index > 0:
        choices.append(poses[index - 1])
    if not choices:
        return None
    best = min(choices, key=lambda pose: abs(pose.t - t))
    if abs(best.t - t) > max_age:
        return None
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map simulated LiDAR scans with simulated odometry.")
    parser.add_argument("--odom", type=Path, default=Path("output/simulated_sensors_starter/noisy_odometry.csv"))
    parser.add_argument("--lidar", type=Path, default=Path("output/simulated_sensors_starter/lidar_scans.npz"))
    parser.add_argument("--output", type=Path, default=Path("output/simulated_mapping/noisy_odom_map"))
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--width", type=float, default=32.0)
    parser.add_argument("--height", type=float, default=26.0)
    parser.add_argument("--origin-x", type=float, default=-16.0)
    parser.add_argument("--origin-y", type=float, default=-13.0)
    parser.add_argument("--max-odom-age", type=float, default=0.25)
    parser.add_argument("--max-range", type=float)
    parser.add_argument("--scan-step", type=int, default=1)
    parser.add_argument("--beam-stride", type=int, default=1)
    parser.add_argument("--log-free", type=float, default=-0.4)
    parser.add_argument("--log-occ", type=float, default=0.85)
    parser.add_argument("--log-min", type=float, default=-5.0)
    parser.add_argument("--log-max", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poses = read_poses(args.odom)
    pose_times = [pose.t for pose in poses]
    lidar = np.load(args.lidar)
    scan_times = lidar["scan_times"]
    angles = lidar["angles"]
    ranges = lidar["ranges"]
    range_min = float(lidar["range_min"])
    range_max = float(lidar["range_max"]) if args.max_range is None else min(float(lidar["range_max"]), args.max_range)

    occupancy = OccupancyMap(
        args.width,
        args.height,
        args.resolution,
        args.origin_x,
        args.origin_y,
        args.log_free,
        args.log_occ,
        args.log_min,
        args.log_max,
    )

    scans_used = 0
    beams_used = 0
    for scan_index in range(0, len(scan_times), args.scan_step):
        pose = nearest_pose(pose_times, poses, float(scan_times[scan_index]), args.max_odom_age)
        if pose is None:
            continue
        scan = ranges[scan_index]
        for beam_index in range(0, len(scan), args.beam_stride):
            distance = float(scan[beam_index])
            if not math.isfinite(distance) or distance < range_min:
                continue
            hit = distance < range_max - 1e-3
            ray_distance = min(distance, range_max)
            angle = pose.yaw + float(angles[beam_index])
            end = (pose.x + ray_distance * math.cos(angle), pose.y + ray_distance * math.sin(angle))
            occupancy.add_ray((pose.x, pose.y), end, hit)
            beams_used += 1
        scans_used += 1

    occupancy.save(args.output)
    print(f"poses: {len(poses)}")
    print(f"scans used: {scans_used}/{len(scan_times)}")
    print(f"beams used: {beams_used}")
    print(f"wrote: {args.output.with_suffix('.png')}")
    print(f"wrote: {args.output.with_suffix('.pgm')}")
    print(f"wrote: {args.output.with_suffix('.yaml')}")


if __name__ == "__main__":
    main()
