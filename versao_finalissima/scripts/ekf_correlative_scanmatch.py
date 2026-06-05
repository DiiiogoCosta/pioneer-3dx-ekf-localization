#!/usr/bin/env python3
"""EKF localization with correlative scan-to-map matching as pose updates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    theta: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_poses(path: Path) -> list[Pose]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Pose(float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"]))
            for row in csv.DictReader(handle)
        ]


def save_poses(path: Path, poses: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta"])
        for pose in poses:
            writer.writerow([f"{pose.t:.6f}", f"{pose.x:.6f}", f"{pose.y:.6f}", f"{pose.theta:.6f}"])


def read_map_yaml(path: Path) -> tuple[float, float, float]:
    import re

    text = path.read_text(encoding="utf-8")
    resolution = float(re.search(r"^resolution:\s*([0-9.eE+-]+)", text, re.MULTILINE).group(1))
    origin = re.search(r"^origin:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,", text, re.MULTILINE)
    if origin is None:
        raise SystemExit(f"Could not parse origin from {path}")
    return resolution, float(origin.group(1)), float(origin.group(2))


class DistanceMap:
    def __init__(self, image_path: Path, yaml_path: Path, max_distance: float = 2.0) -> None:
        self.resolution, self.origin_x, self.origin_y = read_map_yaml(yaml_path)
        array = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        self.height, self.width = array.shape
        occupied = array < 80
        distances_px = distance_transform_edt(~occupied)
        self.distance = np.minimum(distances_px * self.resolution, max_distance).astype(np.float32)

    def world_to_grid_arrays(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = np.rint((x - self.origin_x) / self.resolution).astype(np.int32)
        gy = self.height - np.rint((y - self.origin_y) / self.resolution).astype(np.int32)
        inside = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        return gx, gy, inside

    def score_points(self, pose: np.ndarray, points_robot: np.ndarray, trim: float) -> tuple[float, int]:
        cos_t = math.cos(float(pose[2]))
        sin_t = math.sin(float(pose[2]))
        wx = pose[0] + cos_t * points_robot[:, 0] - sin_t * points_robot[:, 1]
        wy = pose[1] + sin_t * points_robot[:, 0] + cos_t * points_robot[:, 1]
        gx, gy, inside = self.world_to_grid_arrays(wx, wy)
        if int(np.count_nonzero(inside)) < 10:
            return float("inf"), int(np.count_nonzero(inside))
        distances = self.distance[gy[inside], gx[inside]]
        if trim < 1.0 and len(distances) > 8:
            keep = max(8, int(len(distances) * trim))
            distances = np.partition(distances, keep - 1)[:keep]
        return float(np.mean(distances)), int(len(distances))


def nearest_pose(times: list[float], poses: list[Pose], t: float, max_age: float) -> Pose | None:
    index = bisect_left(times, t)
    choices: list[Pose] = []
    if index < len(poses):
        choices.append(poses[index])
    if index > 0:
        choices.append(poses[index - 1])
    if not choices:
        return None
    best = min(choices, key=lambda pose: abs(pose.t - t))
    return best if abs(best.t - t) <= max_age else None


def predict_from_odom_delta(
    state: np.ndarray,
    covariance: np.ndarray,
    previous_odom: Pose,
    current_odom: Pose,
    trans_noise: float,
    rot_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx = current_odom.x - previous_odom.x
    dy = current_odom.y - previous_odom.y
    ds = math.hypot(dx, dy)
    dtheta = wrap_angle(current_odom.theta - previous_odom.theta)
    motion_heading = math.atan2(dy, dx) if ds > 1e-9 else previous_odom.theta
    local_heading = wrap_angle(motion_heading - previous_odom.theta)
    theta_motion = state[2] + local_heading
    state = state.copy()
    state[0] += ds * math.cos(theta_motion)
    state[1] += ds * math.sin(theta_motion)
    state[2] = wrap_angle(state[2] + dtheta)
    fx = np.eye(3)
    fx[0, 2] = -ds * math.sin(theta_motion)
    fx[1, 2] = ds * math.cos(theta_motion)
    q = np.diag(
        [
            (trans_noise * max(ds, 0.05)) ** 2,
            (trans_noise * max(ds, 0.05)) ** 2,
            (rot_noise * max(abs(dtheta), 0.02)) ** 2,
        ]
    )
    return state, fx @ covariance @ fx.T + q


def update_pose_measurement(
    state: np.ndarray,
    covariance: np.ndarray,
    measured: np.ndarray,
    std_xy: float,
    std_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = np.asarray([measured[0] - state[0], measured[1] - state[1], wrap_angle(measured[2] - state[2])])
    h = np.eye(3)
    r = np.diag([std_xy * std_xy, std_xy * std_xy, std_theta * std_theta])
    s = h @ covariance @ h.T + r
    k = covariance @ h.T @ np.linalg.inv(s)
    state = state + k @ innovation
    state[2] = wrap_angle(float(state[2]))
    covariance = (np.eye(3) - k @ h) @ covariance
    return state, covariance


def scan_points(ranges: np.ndarray, angles: np.ndarray, range_max: float, stride: int, max_points: int) -> np.ndarray:
    valid = np.isfinite(ranges) & (ranges > 0.12) & (ranges < range_max - 0.10)
    selected_ranges = ranges[valid][::stride]
    selected_angles = angles[valid][::stride]
    points = np.column_stack([selected_ranges * np.cos(selected_angles), selected_ranges * np.sin(selected_angles)])
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int32)
        points = points[indices]
    return points.astype(np.float64)


def correlative_match(
    predicted: np.ndarray,
    points_robot: np.ndarray,
    distance_map: DistanceMap,
    xy_window: float,
    xy_step: float,
    theta_window: float,
    theta_step: float,
    trim: float,
) -> tuple[np.ndarray, float, int]:
    dxs = np.arange(-xy_window, xy_window + 0.5 * xy_step, xy_step)
    dys = np.arange(-xy_window, xy_window + 0.5 * xy_step, xy_step)
    dts = np.arange(-theta_window, theta_window + 0.5 * theta_step, theta_step)
    best_pose = predicted.copy()
    best_score = float("inf")
    best_used = 0
    for dx, dy, dt in itertools.product(dxs, dys, dts):
        candidate = np.asarray([predicted[0] + dx, predicted[1] + dy, wrap_angle(predicted[2] + dt)])
        score, used = distance_map.score_points(candidate, points_robot, trim=trim)
        if score < best_score:
            best_score = score
            best_pose = candidate
            best_used = used
    return best_pose, best_score, best_used


def path_errors(reference: list[Pose], estimate: list[Pose]) -> tuple[float, float]:
    reference_times = [pose.t for pose in reference]
    errors: list[float] = []
    for pose in estimate:
        ref = nearest_pose(reference_times, reference, pose.t, max_age=0.25)
        if ref is None:
            continue
        errors.append(math.hypot(ref.x - pose.x, ref.y - pose.y))
    if not errors:
        return float("inf"), float("inf")
    return errors[-1], float(sum(errors) / len(errors))


def transform_pose_from_initial_alignment(pose: Pose, source_initial: Pose, target_initial: Pose) -> Pose:
    dx = pose.x - source_initial.x
    dy = pose.y - source_initial.y
    dtheta = wrap_angle(target_initial.theta - source_initial.theta)
    cos_t = math.cos(dtheta)
    sin_t = math.sin(dtheta)
    return Pose(
        pose.t,
        target_initial.x + cos_t * dx - sin_t * dy,
        target_initial.y + sin_t * dx + cos_t * dy,
        wrap_angle(pose.theta + dtheta),
    )


def align_to_reference_initial(poses: list[Pose], reference: list[Pose]) -> list[Pose]:
    if not poses or not reference:
        return poses
    return [transform_pose_from_initial_alignment(pose, poses[0], reference[0]) for pose in poses]


def world_to_px(point: tuple[float, float], height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    return int(round((point[0] - origin_x) / resolution)), height - int(round((point[1] - origin_y) / resolution))


def draw_overlay(map_image: Path, map_yaml: Path, slam: list[Pose], odom: list[Pose], ekf: list[Pose], output: Path) -> None:
    resolution, origin_x, origin_y = read_map_yaml(map_yaml)
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = image.height
    slam_px = [world_to_px((p.x, p.y), height, resolution, origin_x, origin_y) for p in slam]
    odom_px = [world_to_px((p.x, p.y), height, resolution, origin_x, origin_y) for p in odom]
    ekf_px = [world_to_px((p.x, p.y), height, resolution, origin_x, origin_y) for p in ekf]
    if len(slam_px) > 1:
        draw.line(slam_px, fill=(0, 150, 70), width=5)
    if len(odom_px) > 1:
        draw.line(odom_px, fill=(220, 30, 30), width=3)
    if len(ekf_px) > 1:
        draw.line(ekf_px, fill=(30, 90, 230), width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EKF with correlative scan matching pose corrections.")
    parser.add_argument("--odom", type=Path, required=True)
    parser.add_argument("--slam", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--map-image", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-odom-age", type=float, default=0.25)
    parser.add_argument("--scan-every", type=int, default=5)
    parser.add_argument("--beam-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=90)
    parser.add_argument("--xy-window", type=float, default=0.35)
    parser.add_argument("--xy-step", type=float, default=0.10)
    parser.add_argument("--theta-window", type=float, default=0.25)
    parser.add_argument("--theta-step", type=float, default=0.08)
    parser.add_argument("--score-gate", type=float, default=0.18)
    parser.add_argument("--min-used", type=int, default=25)
    parser.add_argument("--trim", type=float, default=0.75)
    parser.add_argument("--std-xy", type=float, default=0.20)
    parser.add_argument("--std-theta", type=float, default=0.08)
    parser.add_argument("--predict-trans-noise", type=float, default=0.08)
    parser.add_argument("--predict-rot-noise", type=float, default=0.05)
    parser.add_argument(
        "--align-odom-to-slam-initial",
        action="store_true",
        help="Diagnostic mode: transform odometry into the SLAM map frame by matching only the first pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    odom = read_poses(args.odom)
    slam = read_poses(args.slam)
    if args.align_odom_to_slam_initial:
        odom = align_to_reference_initial(odom, slam)
    lidar = np.load(args.lidar)
    scan_times = lidar["scan_times"]
    ranges = lidar["ranges"]
    angles = lidar["angles"]
    range_max = float(lidar["range_max"])
    distance_map = DistanceMap(args.map_image, args.map_yaml)

    state = np.asarray([odom[0].x, odom[0].y, odom[0].theta], dtype=np.float64)
    covariance = np.diag([0.05, 0.05, 0.03])
    estimates = [Pose(odom[0].t, float(state[0]), float(state[1]), float(state[2]))]
    scan_index = 0
    scan_updates = 0
    scan_candidates = 0
    rejected = 0

    for odom_index in range(1, len(odom)):
        state, covariance = predict_from_odom_delta(
            state,
            covariance,
            odom[odom_index - 1],
            odom[odom_index],
            args.predict_trans_noise,
            args.predict_rot_noise,
        )
        while scan_index < len(scan_times) and float(scan_times[scan_index]) <= odom[odom_index].t:
            if scan_index % args.scan_every == 0:
                pts = scan_points(
                    ranges[scan_index],
                    angles,
                    range_max,
                    stride=args.beam_stride,
                    max_points=args.max_points,
                )
                if len(pts) >= args.min_used:
                    matched, score, used = correlative_match(
                        state,
                        pts,
                        distance_map,
                        args.xy_window,
                        args.xy_step,
                        args.theta_window,
                        args.theta_step,
                        args.trim,
                    )
                    scan_candidates += 1
                    jump = math.hypot(float(matched[0] - state[0]), float(matched[1] - state[1]))
                    if score <= args.score_gate and used >= args.min_used and jump <= args.xy_window + 1e-6:
                        state, covariance = update_pose_measurement(
                            state,
                            covariance,
                            matched,
                            std_xy=args.std_xy,
                            std_theta=args.std_theta,
                        )
                        scan_updates += 1
                    else:
                        rejected += 1
            scan_index += 1
        estimates.append(Pose(odom[odom_index].t, float(state[0]), float(state[1]), float(state[2])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimate_csv = args.output_dir / "ekf_estimate.csv"
    overlay_png = args.output_dir / "overlay.png"
    summary_json = args.output_dir / "summary.json"
    save_poses(estimate_csv, estimates)
    draw_overlay(args.map_image, args.map_yaml, slam, odom, estimates, overlay_png)
    odom_final, odom_mean = path_errors(slam, odom)
    ekf_final, ekf_mean = path_errors(slam, estimates)
    summary = {
        "scan_candidates": scan_candidates,
        "scan_updates": scan_updates,
        "rejected": rejected,
        "odom_final_error_m": round(odom_final, 3),
        "odom_mean_error_m": round(odom_mean, 3),
        "ekf_final_error_m": round(ekf_final, 3),
        "ekf_mean_error_m": round(ekf_mean, 3),
        "params": vars(args) | {"odom": str(args.odom), "slam": str(args.slam), "lidar": str(args.lidar), "map_image": str(args.map_image), "map_yaml": str(args.map_yaml), "output_dir": str(args.output_dir)},
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {estimate_csv}")
    print(f"wrote: {overlay_png}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
