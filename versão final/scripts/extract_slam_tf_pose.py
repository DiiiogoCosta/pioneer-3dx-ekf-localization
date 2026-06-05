#!/usr/bin/env python3
"""Extract a SLAM pose trajectory from ROS2 TF.

The expected TF chain is:

    map -> odom -> base_link

The output CSV has the same columns used by the EKF scripts: t,x,y,theta.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


@dataclass(frozen=True)
class Transform2D:
    t: float
    parent: str
    child: str
    x: float
    y: float
    yaw: float


def yaw_from_quaternion(q: object) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(a: Transform2D, b: Transform2D) -> tuple[float, float, float]:
    cos_yaw = math.cos(a.yaw)
    sin_yaw = math.sin(a.yaw)
    x = a.x + cos_yaw * b.x - sin_yaw * b.y
    y = a.y + sin_yaw * b.x + cos_yaw * b.y
    return x, y, wrap_angle(a.yaw + b.yaw)


def nearest(transforms: list[Transform2D], t: float, max_age: float) -> Transform2D | None:
    if not transforms:
        return None
    times = np.asarray([item.t for item in transforms], dtype=np.float64)
    index = int(np.searchsorted(times, t))
    choices = []
    if index < len(transforms):
        choices.append(transforms[index])
    if index > 0:
        choices.append(transforms[index - 1])
    best = min(choices, key=lambda item: abs(item.t - t))
    return best if abs(best.t - t) <= max_age else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract map->base pose from a ROS2 bag containing TF.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument(
        "--ros-store",
        default="ros2_humble",
        choices=[store.value for store in Stores if store is not Stores.EMPTY],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    typestore = get_typestore(Stores(args.ros_store))
    map_to_odom: list[Transform2D] = []
    odom_to_base: list[Transform2D] = []

    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic in {"/tf", "/tf_static"}]
        if not connections:
            raise SystemExit(f"No /tf or /tf_static topics found in {args.bag}.")
        for connection, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            for transform in msg.transforms:
                parent = str(transform.header.frame_id)
                child = str(transform.child_frame_id)
                stamp = transform.header.stamp
                t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
                tr = transform.transform.translation
                rot = transform.transform.rotation
                item = Transform2D(t, parent, child, float(tr.x), float(tr.y), yaw_from_quaternion(rot))
                if parent == args.map_frame and child == args.odom_frame:
                    map_to_odom.append(item)
                elif parent == args.odom_frame and child == args.base_frame:
                    odom_to_base.append(item)

    map_to_odom.sort(key=lambda item: item.t)
    odom_to_base.sort(key=lambda item: item.t)
    if not map_to_odom:
        raise SystemExit(f"No {args.map_frame}->{args.odom_frame} transform found in {args.bag}.")
    if not odom_to_base:
        raise SystemExit(f"No {args.odom_frame}->{args.base_frame} transform found in {args.bag}.")

    rows_abs: list[tuple[float, float, float, float]] = []
    for odom_base in odom_to_base:
        map_odom = nearest(map_to_odom, odom_base.t, args.max_age)
        if map_odom is None:
            continue
        x, y, yaw = compose(map_odom, odom_base)
        rows_abs.append((odom_base.t, x, y, yaw))

    if not rows_abs:
        raise SystemExit("No synchronized map->base poses could be composed.")
    start_t = rows_abs[0][0]
    rows = [(t - start_t, x, y, yaw) for t, x, y, yaw in rows_abs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta"])
        for row in rows:
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
    print(f"map->odom transforms: {len(map_to_odom)}")
    print(f"odom->base transforms: {len(odom_to_base)}")
    print(f"poses written: {len(rows)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
