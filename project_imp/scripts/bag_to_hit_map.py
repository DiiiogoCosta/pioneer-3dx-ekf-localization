#!/usr/bin/env python3
"""Create a sharper obstacle-density image from a bag with Odometry and LaserScan.

Unlike bag_to_map.py, this script only accumulates LiDAR hit endpoints. It is
better for visually choosing wall/corner landmarks, but it is not a complete
ROS occupancy map because it does not encode free space.
"""

from __future__ import annotations

import argparse
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

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
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_from_odom(msg: object) -> Pose2D:
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    return Pose2D(float(pos.x), float(pos.y), yaw_from_quaternion(ori))


def find_topic(reader: AnyReader, msgtypes: set[str], requested: str | None, label: str) -> str:
    if requested:
        matches = [connection for connection in reader.connections if connection.topic == requested]
        if not matches:
            raise SystemExit(f"Topic {requested!r} was not found in the bag.")
        return requested
    candidates = sorted({connection.topic for connection in reader.connections if connection.msgtype in msgtypes})
    if not candidates:
        raise SystemExit(f"No {label} topic was found.")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple {label} topics found: {', '.join(candidates)}. Pass --{label}-topic.")
    return candidates[0]


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


def laser_pose(robot: Pose2D, mount: LaserMount) -> Pose2D:
    cos_yaw = math.cos(robot.yaw)
    sin_yaw = math.sin(robot.yaw)
    x = robot.x + cos_yaw * mount.x - sin_yaw * mount.y
    y = robot.y + sin_yaw * mount.x + cos_yaw * mount.y
    return Pose2D(x, y, robot.yaw + mount.yaw)


class HitMap:
    def __init__(self, width_m: float, height_m: float, resolution: float, origin_x: float, origin_y: float) -> None:
        self.width = int(math.ceil(width_m / resolution))
        self.height = int(math.ceil(height_m / resolution))
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.hits = np.zeros((self.height, self.width), dtype=np.float32)

    def add_hit(self, x: float, y: float) -> bool:
        gx = int(math.floor((x - self.origin_x) / self.resolution))
        gy = int(math.floor((y - self.origin_y) / self.resolution))
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return False
        self.hits[gy, gx] += 1.0
        return True

    @staticmethod
    def blur(array: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return array
        sigma = max(radius, 0.1)
        half_width = max(1, int(math.ceil(3 * sigma)))
        coords = np.arange(-half_width, half_width + 1, dtype=np.float32)
        kernel = np.exp(-(coords * coords) / (2 * sigma * sigma))
        kernel /= kernel.sum()

        padded_x = np.pad(array, ((0, 0), (half_width, half_width)), mode="edge")
        blurred_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x)
        padded_y = np.pad(blurred_x, ((half_width, half_width), (0, 0)), mode="edge")
        return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_y)

    def save(self, output_prefix: Path, percentile: float, blur_radius: float, min_hits: float) -> None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        hits = self.hits.copy()
        if blur_radius > 0:
            hits = self.blur(hits, blur_radius)

        nonzero = hits[hits > 0]
        threshold = max(min_hits, float(np.percentile(nonzero, percentile)) if nonzero.size else min_hits)
        obstacles = hits >= threshold

        image = np.full((self.height, self.width), 245, dtype=np.uint8)
        image[obstacles] = 0
        image = np.flipud(image)

        Image.fromarray(image, mode="L").save(output_prefix.with_suffix(".png"))
        Image.fromarray(image, mode="L").save(output_prefix.with_suffix(".pgm"))
        output_prefix.with_suffix(".yaml").write_text(
            "\n".join(
                [
                    f"image: {output_prefix.with_suffix('.pgm').name}",
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
        np.save(output_prefix.with_suffix(".npy"), self.hits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sharp LiDAR endpoint density map from a ROS bag.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--odom-topic")
    parser.add_argument("--scan-topic")
    parser.add_argument("--output", type=Path, default=Path("output/hit_map"))
    parser.add_argument("--resolution", type=float, default=0.03)
    parser.add_argument("--width", type=float, default=18.0)
    parser.add_argument("--height", type=float, default=18.0)
    parser.add_argument("--origin-x", type=float)
    parser.add_argument("--origin-y", type=float)
    parser.add_argument("--min-range", type=float, default=0.12)
    parser.add_argument("--max-range", type=float, default=4.0)
    parser.add_argument("--scan-step", type=int, default=1)
    parser.add_argument("--beam-stride", type=int, default=1)
    parser.add_argument("--max-odom-age", type=float, default=0.25)
    parser.add_argument("--laser-x", type=float, default=0.0)
    parser.add_argument("--laser-y", type=float, default=0.0)
    parser.add_argument("--laser-yaw", type=float, default=0.0)
    parser.add_argument("--percentile", type=float, default=70.0)
    parser.add_argument("--min-hits", type=float, default=2.0)
    parser.add_argument("--blur-radius", type=float, default=0.8)
    parser.add_argument(
        "--ros-store",
        default="ros2_humble",
        choices=[store.value for store in Stores if store is not Stores.EMPTY],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        hit_map = HitMap(args.width, args.height, args.resolution, origin_x, origin_y)
        mount = LaserMount(args.laser_x, args.laser_y, args.laser_yaw)
        max_dt_ns = int(args.max_odom_age * 1_000_000_000)

        scans_seen = 0
        scans_used = 0
        hits_used = 0
        for connection, timestamp, rawdata in reader.messages(connections=scan_connections):
            scans_seen += 1
            if scans_seen % args.scan_step != 0:
                continue
            robot_pose = nearest_pose(pose_times, poses, timestamp, max_dt_ns)
            if robot_pose is None:
                continue
            scan = reader.deserialize(rawdata, connection.msgtype)
            lp = laser_pose(robot_pose, mount)
            for i in range(0, len(scan.ranges), args.beam_stride):
                distance = float(scan.ranges[i])
                if not math.isfinite(distance):
                    continue
                if distance < max(args.min_range, float(scan.range_min)) or distance > min(args.max_range, float(scan.range_max)):
                    continue
                angle = lp.yaw + float(scan.angle_min) + i * float(scan.angle_increment)
                if hit_map.add_hit(lp.x + distance * math.cos(angle), lp.y + distance * math.sin(angle)):
                    hits_used += 1
            scans_used += 1

    hit_map.save(args.output, args.percentile, args.blur_radius, args.min_hits)
    print(f"odom topic: {odom_topic}")
    print(f"scan topic: {scan_topic}")
    print(f"scans used: {scans_used}/{scans_seen}")
    print(f"hits accumulated: {hits_used}")
    print(f"wrote: {args.output.with_suffix('.png')}")
    print(f"wrote: {args.output.with_suffix('.pgm')}")
    print(f"wrote: {args.output.with_suffix('.yaml')}")


if __name__ == "__main__":
    main()
