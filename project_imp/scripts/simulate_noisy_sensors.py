#!/usr/bin/env python3
"""Simulate noisy odometry and 2D LiDAR readings along a planned route."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    theta: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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


def load_route(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    if len(points) < 2:
        raise SystemExit("Route must contain at least two points.")
    return points


def interpolate_route(points: list[tuple[float, float]], speed: float, dt: float) -> list[Pose]:
    poses: list[Pose] = []
    t = 0.0
    for start, end in zip(points[:-1], points[1:], strict=False):
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            continue
        theta = math.atan2(dy, dx)
        steps = max(1, int(math.ceil(distance / (speed * dt))))
        for i in range(steps):
            alpha = i / steps
            poses.append(Pose(t, sx + alpha * dx, sy + alpha * dy, theta))
            t += dt
    last = points[-1]
    previous = points[-2]
    poses.append(Pose(t, last[0], last[1], math.atan2(last[1] - previous[1], last[0] - previous[0])))
    return poses


def simulate_odometry(
    true_poses: list[Pose],
    rng: np.random.Generator,
    translational_std_per_m: float,
    rotational_std_per_rad: float,
    heading_drift_std_per_s: float,
) -> list[Pose]:
    odom = [Pose(true_poses[0].t, true_poses[0].x, true_poses[0].y, true_poses[0].theta)]
    heading_bias = 0.0
    for previous_true, current_true in zip(true_poses[:-1], true_poses[1:], strict=False):
        previous_odom = odom[-1]
        dt = max(current_true.t - previous_true.t, 1e-6)
        dx = current_true.x - previous_true.x
        dy = current_true.y - previous_true.y
        ds = math.hypot(dx, dy)
        dtheta = wrap_angle(current_true.theta - previous_true.theta)

        noisy_ds = ds + rng.normal(0.0, translational_std_per_m * max(ds, 1e-4))
        noisy_dtheta = dtheta + rng.normal(0.0, rotational_std_per_rad * max(abs(dtheta), 0.02))
        heading_bias += rng.normal(0.0, heading_drift_std_per_s * math.sqrt(dt))

        theta_mid = previous_odom.theta + noisy_dtheta / 2.0 + heading_bias
        x = previous_odom.x + noisy_ds * math.cos(theta_mid)
        y = previous_odom.y + noisy_ds * math.sin(theta_mid)
        theta = wrap_angle(previous_odom.theta + noisy_dtheta + heading_bias * dt)
        odom.append(Pose(current_true.t, x, y, theta))
    return odom


def world_to_grid(x: float, y: float, height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    gx = int(math.floor((x - origin_x) / resolution))
    gy = height - 1 - int(math.floor((y - origin_y) / resolution))
    return gx, gy


def grid_to_image(x: float, y: float, height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    gx = int(round((x - origin_x) / resolution))
    gy = height - int(round((y - origin_y) / resolution))
    return gx, gy


def raycast(
    occupied: np.ndarray,
    pose: Pose,
    angle: float,
    range_min: float,
    range_max: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> float:
    height, width = occupied.shape
    step = resolution * 0.5
    distance = range_min
    while distance <= range_max:
        x = pose.x + distance * math.cos(angle)
        y = pose.y + distance * math.sin(angle)
        gx, gy = world_to_grid(x, y, height, resolution, origin_x, origin_y)
        if gx < 0 or gx >= width or gy < 0 or gy >= height:
            return range_max
        if occupied[gy, gx]:
            return distance
        distance += step
    return range_max


def simulate_lidar(
    occupied: np.ndarray,
    poses: list[Pose],
    rng: np.random.Generator,
    resolution: float,
    origin_x: float,
    origin_y: float,
    scan_period: float,
    angle_min: float,
    angle_max: float,
    beams: int,
    range_min: float,
    range_max: float,
    range_std: float,
    dropout_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    scan_indices = [0]
    last_t = poses[0].t
    for index, pose in enumerate(poses[1:], start=1):
        if pose.t - last_t >= scan_period:
            scan_indices.append(index)
            last_t = pose.t

    angles = np.linspace(angle_min, angle_max, beams, dtype=np.float32)
    scans = np.empty((len(scan_indices), beams), dtype=np.float32)
    for out_index, pose_index in enumerate(scan_indices):
        pose = poses[pose_index]
        for beam_index, relative_angle in enumerate(angles):
            true_range = raycast(
                occupied,
                pose,
                pose.theta + float(relative_angle),
                range_min,
                range_max,
                resolution,
                origin_x,
                origin_y,
            )
            if rng.random() < dropout_prob:
                measured = range_max
            else:
                measured = true_range + rng.normal(0.0, range_std)
            scans[out_index, beam_index] = np.clip(measured, range_min, range_max)
    return np.asarray(scan_indices, dtype=np.int32), scans


def save_pose_csv(path: Path, poses: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta"])
        for pose in poses:
            writer.writerow([f"{pose.t:.3f}", f"{pose.x:.6f}", f"{pose.y:.6f}", f"{pose.theta:.6f}"])


def draw_overlay(
    map_image: Path,
    output: Path,
    true_poses: list[Pose],
    odom_poses: list[Pose],
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> None:
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = image.size[1]
    true_px = [grid_to_image(pose.x, pose.y, height, resolution, origin_x, origin_y) for pose in true_poses]
    odom_px = [grid_to_image(pose.x, pose.y, height, resolution, origin_x, origin_y) for pose in odom_poses]
    draw.line(true_px, fill=(20, 160, 60), width=3)
    draw.line(odom_px, fill=(220, 20, 20), width=3)
    for label, pose, color in [
        ("S", true_poses[0], (20, 90, 230)),
        ("F", true_poses[-1], (20, 90, 230)),
    ]:
        x, y = grid_to_image(pose.x, pose.y, height, resolution, origin_x, origin_y)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.text((x + 9, y - 9), label, fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate noisy odometry and LiDAR for EKF experiments.")
    parser.add_argument("--map-image", type=Path, default=Path("output/generic_sim_map_32x26.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("output/generic_sim_map_32x26.yaml"))
    parser.add_argument("--route", type=Path, default=Path("output/generic_sim_route.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/simulated_sensors"))
    parser.add_argument("--speed", type=float, default=0.35, help="Robot nominal speed in m/s.")
    parser.add_argument("--dt", type=float, default=0.1, help="Odometry timestep in seconds.")
    parser.add_argument("--scan-period", type=float, default=0.2, help="LiDAR period in seconds.")
    parser.add_argument("--range-max", type=float, default=5.6)
    parser.add_argument("--range-min", type=float, default=0.02)
    parser.add_argument("--beams", type=int, default=181)
    parser.add_argument("--angle-min", type=float, default=-2.35619449)
    parser.add_argument("--angle-max", type=float, default=2.09234977)
    parser.add_argument("--odom-trans-std-per-m", type=float, default=0.025)
    parser.add_argument("--odom-rot-std-per-rad", type=float, default=0.04)
    parser.add_argument("--odom-heading-drift-std", type=float, default=0.002)
    parser.add_argument("--lidar-range-std", type=float, default=0.025)
    parser.add_argument("--lidar-dropout-prob", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    resolution, origin_x, origin_y = read_map_yaml(args.map_yaml)
    map_array = np.asarray(Image.open(args.map_image).convert("L"), dtype=np.uint8)
    occupied = map_array < 80
    route_points = load_route(args.route)
    true_poses = interpolate_route(route_points, args.speed, args.dt)
    odom_poses = simulate_odometry(
        true_poses,
        rng,
        translational_std_per_m=args.odom_trans_std_per_m,
        rotational_std_per_rad=args.odom_rot_std_per_rad,
        heading_drift_std_per_s=args.odom_heading_drift_std,
    )
    scan_indices, scans = simulate_lidar(
        occupied,
        true_poses,
        rng,
        resolution,
        origin_x,
        origin_y,
        scan_period=args.scan_period,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        beams=args.beams,
        range_min=args.range_min,
        range_max=args.range_max,
        range_std=args.lidar_range_std,
        dropout_prob=args.lidar_dropout_prob,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    true_csv = args.output_dir / "true_path.csv"
    odom_csv = args.output_dir / "noisy_odometry.csv"
    lidar_npz = args.output_dir / "lidar_scans.npz"
    summary_json = args.output_dir / "summary.json"
    overlay_png = args.output_dir / "trajectory_overlay.png"
    save_pose_csv(true_csv, true_poses)
    save_pose_csv(odom_csv, odom_poses)
    np.savez_compressed(
        lidar_npz,
        scan_indices=scan_indices,
        scan_times=np.asarray([true_poses[i].t for i in scan_indices], dtype=np.float32),
        angles=np.linspace(args.angle_min, args.angle_max, args.beams, dtype=np.float32),
        ranges=scans,
        range_min=np.float32(args.range_min),
        range_max=np.float32(args.range_max),
    )
    final_error = math.hypot(odom_poses[-1].x - true_poses[-1].x, odom_poses[-1].y - true_poses[-1].y)
    heading_error = wrap_angle(odom_poses[-1].theta - true_poses[-1].theta)
    summary = {
        "pose_count": len(true_poses),
        "scan_count": int(scans.shape[0]),
        "beams_per_scan": int(scans.shape[1]),
        "duration_s": round(true_poses[-1].t, 3),
        "range_min": args.range_min,
        "range_max": args.range_max,
        "odom_noise": {
            "trans_std_per_m": args.odom_trans_std_per_m,
            "rot_std_per_rad": args.odom_rot_std_per_rad,
            "heading_drift_std_per_s": args.odom_heading_drift_std,
        },
        "lidar_noise": {
            "range_std_m": args.lidar_range_std,
            "dropout_prob": args.lidar_dropout_prob,
        },
        "final_position_error_m": round(final_error, 3),
        "final_heading_error_rad": round(heading_error, 3),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_overlay(args.map_image, overlay_png, true_poses, odom_poses, resolution, origin_x, origin_y)

    print(f"duration: {true_poses[-1].t:.1f} s")
    print(f"poses: {len(true_poses)}")
    print(f"scans: {scans.shape[0]} x {scans.shape[1]}")
    print(f"final odom position error: {final_error:.2f} m")
    print(f"final odom heading error: {heading_error:.2f} rad")
    print(f"wrote: {true_csv}")
    print(f"wrote: {odom_csv}")
    print(f"wrote: {lidar_npz}")
    print(f"wrote: {summary_json}")
    print(f"wrote: {overlay_png}")


if __name__ == "__main__":
    main()
