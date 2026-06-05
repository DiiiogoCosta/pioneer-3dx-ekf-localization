#!/usr/bin/env python3
"""Convert a ROS 1/2 bag with /odom and /scan to the simple CSV/NPZ format."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


LASER_TYPES = {"sensor_msgs/msg/LaserScan", "sensor_msgs/LaserScan"}
ODOM_TYPES = {"nav_msgs/msg/Odometry", "nav_msgs/Odometry"}


def yaw_from_quaternion(q: object) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def find_topic(reader: AnyReader, msgtypes: set[str], requested: str | None, label: str) -> str:
    if requested:
        if any(connection.topic == requested for connection in reader.connections):
            return requested
        raise SystemExit(f"Topic {requested!r} was not found in the bag.")
    candidates = sorted({connection.topic for connection in reader.connections if connection.msgtype in msgtypes})
    if not candidates:
        raise SystemExit(f"No {label} topic found.")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple {label} topics found: {candidates}. Pass --{label}-topic.")
    return candidates[0]


def timestamp_to_s(timestamp_ns: int, start_ns: int) -> float:
    return (timestamp_ns - start_ns) / 1_000_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ROS bag odom/scan topics to simple offline files.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/bag_simple"))
    parser.add_argument("--odom-topic")
    parser.add_argument("--scan-topic")
    parser.add_argument(
        "--ros-store",
        default="ros2_humble",
        choices=[store.value for store in Stores if store is not Stores.EMPTY],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    typestore = get_typestore(Stores(args.ros_store))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    odom_rows: list[list[float]] = []
    scan_times: list[float] = []
    scan_ranges: list[np.ndarray] = []
    angles: np.ndarray | None = None
    range_min = 0.0
    range_max = 0.0

    with AnyReader([args.bag], default_typestore=typestore) as reader:
        odom_topic = find_topic(reader, ODOM_TYPES, args.odom_topic, "odom")
        scan_topic = find_topic(reader, LASER_TYPES, args.scan_topic, "scan")
        connections = [connection for connection in reader.connections if connection.topic in {odom_topic, scan_topic}]
        start_ns: int | None = None
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if start_ns is None:
                start_ns = timestamp
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = timestamp_to_s(timestamp, start_ns)
            if connection.topic == odom_topic:
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                odom_rows.append([t, float(pos.x), float(pos.y), yaw_from_quaternion(ori)])
            elif connection.topic == scan_topic:
                ranges = np.asarray([float(value) for value in msg.ranges], dtype=np.float32)
                scan_ranges.append(ranges)
                scan_times.append(t)
                if angles is None:
                    angles = np.linspace(float(msg.angle_min), float(msg.angle_max), len(ranges), dtype=np.float32)
                    range_min = float(msg.range_min)
                    range_max = float(msg.range_max)

    if angles is None:
        raise SystemExit("No scan messages found.")

    odom_path = args.output_dir / "odometry.csv"
    lidar_path = args.output_dir / "lidar_scans.npz"
    with odom_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta"])
        writer.writerows(odom_rows)
    np.savez_compressed(
        lidar_path,
        scan_times=np.asarray(scan_times, dtype=np.float32),
        angles=angles,
        ranges=np.vstack(scan_ranges).astype(np.float32),
        range_min=np.float32(range_min),
        range_max=np.float32(range_max),
    )
    print(f"odom topic: {odom_topic}")
    print(f"scan topic: {scan_topic}")
    print(f"odom rows: {len(odom_rows)}")
    print(f"scan rows: {len(scan_ranges)}")
    print(f"wrote: {odom_path}")
    print(f"wrote: {lidar_path}")


if __name__ == "__main__":
    main()
