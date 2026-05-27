#!/usr/bin/env python3
"""Create a 2D occupancy grid map from a ROS bag with Odometry and LaserScan.

This script intentionally does not require a ROS installation. It reads ROS 1
.bag files and ROS 2 bag directories through the `rosbags` Python package.
"""

from __future__ import annotations

import argparse
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


LASER_TYPES = {"sensor_msgs/msg/LaserScan", "sensor_msgs/LaserScan"}
ODOM_TYPES = {"nav_msgs/msg/Odometry", "nav_msgs/Odometry"}


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class LaserMount:
    x: float
    y: float
    yaw: float


def yaw_from_quaternion(q: object) -> float:
    """Return planar yaw from a ROS geometry_msgs/Quaternion-like object."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_from_odom(msg: object) -> Pose2D:
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    return Pose2D(float(pos.x), float(pos.y), yaw_from_quaternion(ori))


def norm_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def find_topic(reader: AnyReader, msgtypes: set[str], requested: str | None, label: str) -> str:
    if requested:
        matches = [connection for connection in reader.connections if connection.topic == requested]
        if not matches:
            raise SystemExit(f"Topic {requested!r} was not found in the bag.")
        return requested

    candidates = sorted({connection.topic for connection in reader.connections if connection.msgtype in msgtypes})
    if not candidates:
        known = "\n".join(
            f"  {connection.topic}: {connection.msgtype}" for connection in reader.connections
        )
        raise SystemExit(f"No {label} topic was found. Bag topics:\n{known}")
    if len(candidates) > 1:
        formatted = ", ".join(candidates)
        raise SystemExit(f"Multiple {label} topics found: {formatted}. Pass --{label}-topic.")
    return candidates[0]


def bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[tuple[int, int]]:
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
        if not cells:
            return

        free_cells = cells[:-1] if hit else cells
        for gx, gy in free_cells:
            if self.inside(gx, gy):
                self.grid[gy, gx] = max(self.log_min, self.grid[gy, gx] + self.log_free)

        if hit and self.inside(x1, y1):
            self.grid[y1, x1] = min(self.log_max, self.grid[y1, x1] + self.log_occ)

    def to_image_array(self) -> np.ndarray:
        image = np.full((self.height, self.width), 205, dtype=np.uint8)
        occupied = self.grid > 0.85
        free = self.grid < -0.4
        image[free] = 254
        image[occupied] = 0
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


def nearest_pose(
    pose_times: list[int],
    poses: list[tuple[int, Pose2D]],
    timestamp: int,
    max_dt_ns: int,
) -> Pose2D | None:
    index = bisect_left(pose_times, timestamp)
    choices: list[tuple[int, Pose2D]] = []
    if index < len(poses):
        choices.append(poses[index])
    if index > 0:
        choices.append(poses[index - 1])
    if not choices:
        return None
    best_time, best_pose = min(choices, key=lambda item: abs(item[0] - timestamp))
    if abs(best_time - timestamp) > max_dt_ns:
        return None
    return best_pose


def laser_world_pose(robot: Pose2D, mount: LaserMount) -> Pose2D:
    cos_yaw = math.cos(robot.yaw)
    sin_yaw = math.sin(robot.yaw)
    x = robot.x + cos_yaw * mount.x - sin_yaw * mount.y
    y = robot.y + sin_yaw * mount.x + cos_yaw * mount.y
    return Pose2D(x, y, norm_angle(robot.yaw + mount.yaw))


def add_scan_to_map(
    occupancy_map: OccupancyMap,
    scan: object,
    robot_pose: Pose2D,
    mount: LaserMount,
    max_range: float | None,
    stride: int,
) -> int:
    laser_pose = laser_world_pose(robot_pose, mount)
    usable_max = float(scan.range_max)
    if max_range is not None:
        usable_max = min(usable_max, max_range)
    usable_min = float(scan.range_min)

    inserted = 0
    ranges = list(scan.ranges)
    for i in range(0, len(ranges), stride):
        distance = float(ranges[i])
        if math.isnan(distance) or distance < usable_min:
            continue

        hit = math.isfinite(distance) and distance <= usable_max
        ray_distance = distance if hit else usable_max
        angle = laser_pose.yaw + float(scan.angle_min) + i * float(scan.angle_increment)
        end = (
            laser_pose.x + ray_distance * math.cos(angle),
            laser_pose.y + ray_distance * math.sin(angle),
        )
        occupancy_map.add_ray((laser_pose.x, laser_pose.y), end, hit)
        inserted += 1
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a simple occupancy grid map from ROS bag odometry and LiDAR scans."
    )
    parser.add_argument("bag", type=Path, help="ROS 1 .bag file or ROS 2 bag directory.")
    parser.add_argument("--odom-topic", help="Odometry topic. Auto-detected when only one exists.")
    parser.add_argument("--scan-topic", help="LaserScan topic. Auto-detected when only one exists.")
    parser.add_argument("--output", type=Path, default=Path("output/map"), help="Output prefix.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution in meters/cell.")
    parser.add_argument("--width", type=float, default=30.0, help="Map width in meters.")
    parser.add_argument("--height", type=float, default=30.0, help="Map height in meters.")
    parser.add_argument("--origin-x", type=float, help="Map lower-left X. Defaults to centered on first odom.")
    parser.add_argument("--origin-y", type=float, help="Map lower-left Y. Defaults to centered on first odom.")
    parser.add_argument("--max-range", type=float, help="Clamp LiDAR range in meters.")
    parser.add_argument("--scan-step", type=int, default=1, help="Use every Nth scan message.")
    parser.add_argument("--beam-stride", type=int, default=1, help="Use every Nth LiDAR beam.")
    parser.add_argument("--max-odom-age", type=float, default=0.25, help="Max scan/odom timestamp difference in seconds.")
    parser.add_argument("--laser-x", type=float, default=0.0, help="LiDAR X offset in robot frame, meters.")
    parser.add_argument("--laser-y", type=float, default=0.0, help="LiDAR Y offset in robot frame, meters.")
    parser.add_argument("--laser-yaw", type=float, default=0.0, help="LiDAR yaw offset in robot frame, radians.")
    parser.add_argument("--log-free", type=float, default=-0.4, help="Log-odds update for free cells.")
    parser.add_argument("--log-occ", type=float, default=0.85, help="Log-odds update for occupied cells.")
    parser.add_argument("--log-min", type=float, default=-5.0, help="Minimum log-odds value.")
    parser.add_argument("--log-max", type=float, default=5.0, help="Maximum log-odds value.")
    parser.add_argument(
        "--ros-store",
        default="ros2_humble",
        choices=[store.value for store in Stores if store is not Stores.EMPTY],
        help="Built-in ROS message type store to use when a bag has no embedded type definitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_step < 1 or args.beam_stride < 1:
        raise SystemExit("--scan-step and --beam-stride must be >= 1.")

    typestore = get_typestore(Stores(args.ros_store))
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        odom_topic = find_topic(reader, ODOM_TYPES, args.odom_topic, "odom")
        scan_topic = find_topic(reader, LASER_TYPES, args.scan_topic, "scan")
        odom_connections = [connection for connection in reader.connections if connection.topic == odom_topic]
        scan_connections = [connection for connection in reader.connections if connection.topic == scan_topic]

        poses: list[tuple[int, Pose2D]] = []
        for connection, timestamp, rawdata in reader.messages(connections=odom_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            poses.append((timestamp, pose_from_odom(msg)))

        if not poses:
            raise SystemExit(f"No odometry messages found on {odom_topic!r}.")

        pose_times = [timestamp for timestamp, _pose in poses]
        origin_x = args.origin_x if args.origin_x is not None else poses[0][1].x - args.width / 2.0
        origin_y = args.origin_y if args.origin_y is not None else poses[0][1].y - args.height / 2.0
        occupancy_map = OccupancyMap(
            width_m=args.width,
            height_m=args.height,
            resolution=args.resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            log_free=args.log_free,
            log_occ=args.log_occ,
            log_min=args.log_min,
            log_max=args.log_max,
        )
        mount = LaserMount(args.laser_x, args.laser_y, args.laser_yaw)
        max_dt_ns = int(args.max_odom_age * 1_000_000_000)

        scans_seen = 0
        scans_used = 0
        beams_used = 0
        for connection, timestamp, rawdata in reader.messages(connections=scan_connections):
            scans_seen += 1
            if scans_seen % args.scan_step != 0:
                continue
            pose = nearest_pose(pose_times, poses, timestamp, max_dt_ns)
            if pose is None:
                continue
            scan = reader.deserialize(rawdata, connection.msgtype)
            beams_used += add_scan_to_map(
                occupancy_map=occupancy_map,
                scan=scan,
                robot_pose=pose,
                mount=mount,
                max_range=args.max_range,
                stride=args.beam_stride,
            )
            scans_used += 1

    occupancy_map.save(args.output)
    print(f"odom topic: {odom_topic}")
    print(f"scan topic: {scan_topic}")
    print(f"odometry poses: {len(poses)}")
    print(f"scans used: {scans_used}/{scans_seen}")
    print(f"beams inserted: {beams_used}")
    print(f"wrote: {args.output.with_suffix('.png')}")
    print(f"wrote: {args.output.with_suffix('.pgm')}")
    print(f"wrote: {args.output.with_suffix('.yaml')}")


if __name__ == "__main__":
    main()
