#!/usr/bin/env python3
"""Global relocalization test with a particle filter using LiDAR and odometry.

This script is intentionally separate from ekf_localization.py. It tests the
global recovery layer that can be used to reinitialize the EKF after kidnapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
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


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_error(a: float, b: float) -> float:
    return float(abs(math.atan2(math.sin(a - b), math.cos(a - b))))


def read_poses(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            poses.append(Pose(float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"])))
    return poses


def read_map_yaml(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8")
    resolution = float(next(line.split(":", 1)[1] for line in text.splitlines() if line.startswith("resolution:")))
    origin_text = next(line.split("[", 1)[1].split("]", 1)[0] for line in text.splitlines() if line.startswith("origin:"))
    origin_x, origin_y, *_ = [float(part.strip()) for part in origin_text.split(",")]
    return resolution, origin_x, origin_y


def world_to_grid(
    x: np.ndarray | float,
    y: np.ndarray | float,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.rint((np.asarray(x) - origin_x) / resolution).astype(np.int32)
    gy = height - np.rint((np.asarray(y) - origin_y) / resolution).astype(np.int32)
    return gx, gy


def world_to_px(
    x: float,
    y: float,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[int, int]:
    gx, gy = world_to_grid(x, y, height, resolution, origin_x, origin_y)
    return int(gx), int(gy)


def load_map(map_image: Path, map_yaml: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    resolution, origin_x, origin_y = read_map_yaml(map_yaml)
    image = np.asarray(Image.open(map_image).convert("L"))
    occupied = image < 128
    free = image > 200
    distance_map = distance_transform_edt(~occupied) * resolution
    return image, free, distance_map, resolution, origin_x, origin_y


def sample_free_particles(
    rng: np.random.Generator,
    free: np.ndarray,
    count: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    ys, xs = np.nonzero(free)
    chosen = rng.integers(0, len(xs), size=count)
    px = xs[chosen] + rng.uniform(-0.5, 0.5, size=count)
    py = ys[chosen] + rng.uniform(-0.5, 0.5, size=count)
    world_x = origin_x + px * resolution
    world_y = origin_y + (free.shape[0] - py) * resolution
    theta = rng.uniform(-math.pi, math.pi, size=count)
    return np.column_stack([world_x, world_y, theta])


def apply_odometry_delta(
    particles: np.ndarray,
    previous_odom: Pose,
    current_odom: Pose,
    rng: np.random.Generator,
    translation_noise: float,
    rotation_noise: float,
) -> None:
    dx = current_odom.x - previous_odom.x
    dy = current_odom.y - previous_odom.y
    translation = math.hypot(dx, dy)
    odom_heading = math.atan2(dy, dx) if translation > 1e-9 else previous_odom.theta
    local_heading = math.atan2(math.sin(odom_heading - previous_odom.theta), math.cos(odom_heading - previous_odom.theta))
    dtheta = math.atan2(math.sin(current_odom.theta - previous_odom.theta), math.cos(current_odom.theta - previous_odom.theta))

    noisy_translation = translation + rng.normal(0.0, translation_noise + 0.03 * translation, size=len(particles))
    noisy_local_heading = local_heading + rng.normal(0.0, rotation_noise, size=len(particles))
    noisy_dtheta = dtheta + rng.normal(0.0, rotation_noise + 0.01 * translation, size=len(particles))

    direction = particles[:, 2] + noisy_local_heading
    particles[:, 0] += noisy_translation * np.cos(direction)
    particles[:, 1] += noisy_translation * np.sin(direction)
    particles[:, 2] = wrap_angle(particles[:, 2] + noisy_dtheta)


def enforce_free_space(
    particles: np.ndarray,
    free: np.ndarray,
    rng: np.random.Generator,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> None:
    gx, gy = world_to_grid(particles[:, 0], particles[:, 1], free.shape[0], resolution, origin_x, origin_y)
    inside = (gx >= 0) & (gx < free.shape[1]) & (gy >= 0) & (gy < free.shape[0])
    valid = inside.copy()
    valid[inside] &= free[gy[inside], gx[inside]]
    bad = ~valid
    if np.any(bad):
        particles[bad] = sample_free_particles(rng, free, int(np.sum(bad)), resolution, origin_x, origin_y)


def choose_lidar_beams(
    ranges: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    max_beams: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(ranges) & (ranges > 0.12) & (ranges < range_max - 0.05)
    indices = np.nonzero(valid)[0]
    if len(indices) > max_beams:
        chosen = np.linspace(0, len(indices) - 1, max_beams).astype(np.int32)
        indices = indices[chosen]
    return ranges[indices], angles[indices]


def sensor_update(
    particles: np.ndarray,
    weights: np.ndarray,
    ranges: np.ndarray,
    angles: np.ndarray,
    distance_map: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    sigma_hit: float,
) -> tuple[np.ndarray, float]:
    if len(ranges) == 0:
        return weights, 0.0

    log_weights = np.zeros(len(particles), dtype=np.float64)
    height, width = distance_map.shape
    for scan_range, scan_angle in zip(ranges, angles, strict=False):
        beam_angle = particles[:, 2] + scan_angle
        hit_x = particles[:, 0] + scan_range * np.cos(beam_angle)
        hit_y = particles[:, 1] + scan_range * np.sin(beam_angle)
        gx, gy = world_to_grid(hit_x, hit_y, height, resolution, origin_x, origin_y)
        inside = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        distances = np.full(len(particles), 2.0, dtype=np.float64)
        distances[inside] = distance_map[gy[inside], gx[inside]]
        clipped = np.minimum(distances, 2.0)
        log_weights += -0.5 * (clipped / sigma_hit) ** 2

    log_weights -= np.max(log_weights)
    likelihood = np.exp(log_weights)
    weights *= likelihood + 1e-300
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        weights[:] = 1.0 / len(weights)
    else:
        weights /= total
    return weights, float(np.mean(log_weights))


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights * weights))


def systematic_resample(rng: np.random.Generator, particles: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    indexes = np.searchsorted(cumulative, positions, side="left")
    particles = particles[indexes].copy()
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    return particles, weights


def cluster_indices_around_pose(
    particles: np.ndarray,
    pose: np.ndarray,
    radius: float,
    heading_radius: float,
) -> np.ndarray:
    distance = np.hypot(particles[:, 0] - pose[0], particles[:, 1] - pose[1])
    heading = np.abs(wrap_angle(particles[:, 2] - pose[2]))
    return np.nonzero((distance < radius) & (heading < heading_radius))[0]


def weighted_pose_from_indices(particles: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    local_weights = weights[indices]
    total = float(np.sum(local_weights))
    if total <= 1e-300 or not np.isfinite(total):
        local_weights = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
    else:
        local_weights = local_weights / total
    x = float(np.sum(particles[indices, 0] * local_weights))
    y = float(np.sum(particles[indices, 1] * local_weights))
    sin_theta = float(np.sum(np.sin(particles[indices, 2]) * local_weights))
    cos_theta = float(np.sum(np.cos(particles[indices, 2]) * local_weights))
    return np.asarray([x, y, math.atan2(sin_theta, cos_theta)], dtype=np.float64)


def estimate_best_cluster(
    particles: np.ndarray,
    weights: np.ndarray,
    top_fraction: float,
    cluster_radius: float,
    cluster_heading: float,
) -> tuple[np.ndarray, float, int, int, float]:
    best = int(np.argmax(weights))
    local = cluster_indices_around_pose(particles, particles[best], cluster_radius, cluster_heading)
    if len(local) >= 10:
        top = local
    else:
        count = max(10, int(len(particles) * top_fraction))
        top = np.argsort(weights)[-count:]
    pose = weighted_pose_from_indices(particles, weights, top)
    return pose, float(np.sum(weights[top])), int(len(top)), best, float(weights[best])


def true_particle_diagnostics(
    particles: np.ndarray,
    weights: np.ndarray,
    true_pose: Pose,
    cluster_radius: float,
    cluster_heading: float,
) -> tuple[float, int, float, int]:
    true_vector = np.asarray([true_pose.x, true_pose.y, true_pose.theta], dtype=np.float64)
    distance = np.hypot(particles[:, 0] - true_pose.x, particles[:, 1] - true_pose.y)
    heading = np.abs(wrap_angle(particles[:, 2] - true_pose.theta))
    combined = distance + heading
    nearest = int(np.argmin(combined))
    order = np.argsort(weights)[::-1]
    rank = int(np.nonzero(order == nearest)[0][0]) + 1
    local = cluster_indices_around_pose(particles, true_vector, cluster_radius, cluster_heading)
    mass = float(np.sum(weights[local])) if len(local) else 0.0
    return float(weights[nearest]), rank, mass, int(len(local))


def select_with_hysteresis(
    particles: np.ndarray,
    weights: np.ndarray,
    candidate_pose: np.ndarray,
    candidate_mass: float,
    previous_pose: np.ndarray | None,
    margin: float,
    cluster_radius: float,
    cluster_heading: float,
) -> tuple[np.ndarray, float, bool, float, int]:
    if previous_pose is None:
        return candidate_pose, candidate_mass, True, 0.0, 0
    previous_indices = cluster_indices_around_pose(particles, previous_pose, cluster_radius, cluster_heading)
    if len(previous_indices) == 0:
        return candidate_pose, candidate_mass, True, 0.0, 0
    previous_mass = float(np.sum(weights[previous_indices]))
    if candidate_mass > previous_mass + margin:
        return candidate_pose, candidate_mass, True, previous_mass, int(len(previous_indices))
    previous_estimate = weighted_pose_from_indices(particles, weights, previous_indices)
    return previous_estimate, previous_mass, False, previous_mass, int(len(previous_indices))


def poses_are_close(a: np.ndarray, b: np.ndarray, radius: float, heading_radius: float) -> bool:
    distance = math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
    heading = angle_error(float(a[2]), float(b[2]))
    return distance < radius and heading < heading_radius


def draw_overlay(
    map_image: Path,
    map_yaml: Path,
    true_poses: list[Pose],
    estimates: list[tuple[int, np.ndarray, float]],
    particles: np.ndarray,
    output: Path,
) -> None:
    resolution, origin_x, origin_y = read_map_yaml(map_yaml)
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = image.size[1]
    true_px = [world_to_px(pose.x, pose.y, height, resolution, origin_x, origin_y) for pose in true_poses]
    draw.line(true_px, fill=(20, 160, 60), width=4)
    estimate_px = [world_to_px(float(pose[0]), float(pose[1]), height, resolution, origin_x, origin_y) for _i, pose, _e in estimates]
    if len(estimate_px) > 1:
        draw.line(estimate_px, fill=(30, 90, 230), width=3)
    for _scan_index, pose, error in estimates[:: max(1, len(estimates) // 50)]:
        x, y = world_to_px(float(pose[0]), float(pose[1]), height, resolution, origin_x, origin_y)
        color = (30, 90, 230) if error < 0.75 else (230, 130, 20)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    for particle in particles[:: max(1, len(particles) // 400)]:
        x, y = world_to_px(float(particle[0]), float(particle[1]), height, resolution, origin_x, origin_y)
        draw.point((x, y), fill=(170, 40, 160))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Particle filter global relocalization test.")
    parser.add_argument("--map-image", type=Path, default=Path("project_imp/maps/floorplan_walkable_map.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("project_imp/maps/floorplan_walkable_map.yaml"))
    parser.add_argument("--lidar", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/lidar_scans.npz"))
    parser.add_argument("--odom", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/noisy_odometry.csv"))
    parser.add_argument("--true", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/true_path.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("project_imp/results/particle_filter_strong"))
    parser.add_argument("--particles", type=int, default=2500)
    parser.add_argument("--max-beams", type=int, default=48)
    parser.add_argument("--scan-step", type=int, default=2)
    parser.add_argument("--sigma-hit", type=float, default=0.22)
    parser.add_argument("--translation-noise", type=float, default=0.035)
    parser.add_argument("--rotation-noise", type=float, default=0.025)
    parser.add_argument("--random-particle-rate", type=float, default=0.03)
    parser.add_argument("--resample-ratio", type=float, default=0.55)
    parser.add_argument("--estimate-top-fraction", type=float, default=0.08)
    parser.add_argument("--cluster-radius", type=float, default=1.25)
    parser.add_argument("--cluster-heading", type=float, default=0.80)
    parser.add_argument("--hysteresis-margin", type=float, default=0.0)
    parser.add_argument(
        "--hysteresis-window",
        type=int,
        default=1,
        help="Use a rolling mean over this many scans before accepting a different global hypothesis.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    _image, free, distance_map, resolution, origin_x, origin_y = load_map(args.map_image, args.map_yaml)
    odom_poses = read_poses(args.odom)
    true_poses = read_poses(args.true)
    lidar = np.load(args.lidar)
    all_ranges = lidar["ranges"]
    angles = lidar["angles"]
    range_max = float(lidar["range_max"])
    scan_indices = lidar["scan_indices"] if "scan_indices" in lidar else np.arange(len(all_ranges), dtype=np.int32)

    particles = sample_free_particles(rng, free, args.particles, resolution, origin_x, origin_y)
    weights = np.full(args.particles, 1.0 / args.particles, dtype=np.float64)

    rows: list[dict[str, float | int]] = []
    estimates: list[tuple[int, np.ndarray, float]] = []
    previous_odom_index = int(scan_indices[0])
    selected_pose: np.ndarray | None = None
    pending_pose: np.ndarray | None = None
    pending_candidate_masses: deque[float] = deque(maxlen=max(1, args.hysteresis_window))
    pending_selected_masses: deque[float] = deque(maxlen=max(1, args.hysteresis_window))
    hypothesis_switches = 0

    for scan_index in range(0, len(all_ranges), args.scan_step):
        odom_index = int(scan_indices[scan_index])
        if odom_index != previous_odom_index:
            apply_odometry_delta(
                particles,
                odom_poses[previous_odom_index],
                odom_poses[odom_index],
                rng,
                args.translation_noise,
                args.rotation_noise,
            )
            previous_odom_index = odom_index
            enforce_free_space(particles, free, rng, resolution, origin_x, origin_y)

        beam_ranges, beam_angles = choose_lidar_beams(all_ranges[scan_index], angles, range_max, args.max_beams)
        weights, mean_log_likelihood = sensor_update(
            particles,
            weights,
            beam_ranges,
            beam_angles,
            distance_map,
            resolution,
            origin_x,
            origin_y,
            args.sigma_hit,
        )

        ess = effective_sample_size(weights)
        true_pose = true_poses[min(odom_index, len(true_poses) - 1)]
        candidate_pose, candidate_mass, candidate_count, best_particle, best_particle_weight = estimate_best_cluster(
            particles,
            weights,
            args.estimate_top_fraction,
            args.cluster_radius,
            args.cluster_heading,
        )
        switched = False
        if args.hysteresis_window <= 1:
            estimate, selected_mass, switched, previous_mass, previous_count = select_with_hysteresis(
                particles,
                weights,
                candidate_pose,
                candidate_mass,
                selected_pose,
                args.hysteresis_margin,
                args.cluster_radius,
                args.cluster_heading,
            )
            if selected_pose is not None and switched:
                hypothesis_switches += 1
            selected_pose = estimate
            pending_mean_candidate = candidate_mass
            pending_mean_selected = previous_mass
            pending_count = 0
        else:
            if selected_pose is None:
                estimate = candidate_pose
                selected_pose = estimate
                selected_mass = candidate_mass
                previous_mass = 0.0
                previous_count = 0
                pending_mean_candidate = candidate_mass
                pending_mean_selected = 0.0
                pending_count = 0
            else:
                selected_indices = cluster_indices_around_pose(
                    particles,
                    selected_pose,
                    args.cluster_radius,
                    args.cluster_heading,
                )
                previous_count = int(len(selected_indices))
                previous_mass = float(np.sum(weights[selected_indices])) if len(selected_indices) else 0.0
                if len(selected_indices):
                    held_pose = weighted_pose_from_indices(particles, weights, selected_indices)
                else:
                    held_pose = selected_pose

                if poses_are_close(candidate_pose, selected_pose, args.cluster_radius, args.cluster_heading):
                    estimate = candidate_pose
                    selected_pose = estimate
                    selected_mass = candidate_mass
                    pending_pose = None
                    pending_candidate_masses.clear()
                    pending_selected_masses.clear()
                    pending_mean_candidate = candidate_mass
                    pending_mean_selected = previous_mass
                    pending_count = 0
                else:
                    if pending_pose is None or not poses_are_close(
                        candidate_pose,
                        pending_pose,
                        args.cluster_radius,
                        args.cluster_heading,
                    ):
                        pending_pose = candidate_pose
                        pending_candidate_masses.clear()
                        pending_selected_masses.clear()
                    pending_candidate_masses.append(candidate_mass)
                    pending_selected_masses.append(previous_mass)
                    pending_mean_candidate = float(np.mean(pending_candidate_masses))
                    pending_mean_selected = float(np.mean(pending_selected_masses))
                    pending_count = len(pending_candidate_masses)
                    if (
                        pending_count >= args.hysteresis_window
                        and pending_mean_candidate > pending_mean_selected + args.hysteresis_margin
                    ):
                        estimate = candidate_pose
                        selected_pose = estimate
                        selected_mass = candidate_mass
                        switched = True
                        hypothesis_switches += 1
                        pending_pose = None
                        pending_candidate_masses.clear()
                        pending_selected_masses.clear()
                    else:
                        estimate = held_pose
                        selected_pose = estimate
                        selected_mass = previous_mass
        true_nearest_weight, true_nearest_rank, true_cluster_mass, true_cluster_count = true_particle_diagnostics(
            particles,
            weights,
            true_pose,
            args.cluster_radius,
            args.cluster_heading,
        )
        position_error = math.hypot(float(estimate[0]) - true_pose.x, float(estimate[1]) - true_pose.y)
        heading_error = angle_error(float(estimate[2]), true_pose.theta)
        estimates.append((scan_index, estimate, position_error))
        rows.append(
            {
                "scan_index": scan_index,
                "t": true_pose.t,
                "x": float(estimate[0]),
                "y": float(estimate[1]),
                "theta": float(estimate[2]),
                "true_x": true_pose.x,
                "true_y": true_pose.y,
                "true_theta": true_pose.theta,
                "position_error_m": position_error,
                "heading_error_rad": heading_error,
                "ess": ess,
                "beams": len(beam_ranges),
                "mean_log_likelihood": mean_log_likelihood,
                "best_particle": best_particle,
                "best_particle_weight": best_particle_weight,
                "candidate_cluster_mass": candidate_mass,
                "candidate_cluster_count": candidate_count,
                "selected_cluster_mass": selected_mass,
                "selected_switched": int(switched),
                "previous_cluster_mass": previous_mass,
                "previous_cluster_count": previous_count,
                "pending_mean_candidate_mass": pending_mean_candidate,
                "pending_mean_selected_mass": pending_mean_selected,
                "pending_count": pending_count,
                "hypothesis_switches": hypothesis_switches,
                "true_nearest_particle_weight": true_nearest_weight,
                "true_nearest_particle_rank": true_nearest_rank,
                "true_cluster_mass": true_cluster_mass,
                "true_cluster_count": true_cluster_count,
            }
        )

        if ess < args.resample_ratio * args.particles:
            particles, weights = systematic_resample(rng, particles, weights)
            random_count = int(round(args.random_particle_rate * args.particles))
            if random_count > 0:
                replace = rng.choice(args.particles, size=random_count, replace=False)
                particles[replace] = sample_free_particles(rng, free, random_count, resolution, origin_x, origin_y)
                weights[:] = 1.0 / args.particles

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "particle_filter_estimate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.asarray([float(row["position_error_m"]) for row in rows], dtype=np.float64)
    heading_errors = np.asarray([float(row["heading_error_rad"]) for row in rows], dtype=np.float64)
    converged_indices = np.nonzero((errors < 0.75) & (heading_errors < 0.60))[0]
    summary = {
        "particles": args.particles,
        "scans": len(rows),
        "mean_position_error_m": float(np.mean(errors)),
        "median_position_error_m": float(np.median(errors)),
        "final_position_error_m": float(errors[-1]),
        "final_heading_error_rad": float(heading_errors[-1]),
        "converged": bool(len(converged_indices)),
        "first_converged_scan": int(rows[int(converged_indices[0])]["scan_index"]) if len(converged_indices) else None,
        "first_converged_t": float(rows[int(converged_indices[0])]["t"]) if len(converged_indices) else None,
        "converged_fraction": float(np.mean((errors < 0.75) & (heading_errors < 0.60))),
        "hypothesis_switches": hypothesis_switches,
        "hysteresis_margin": args.hysteresis_margin,
        "hysteresis_window": args.hysteresis_window,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_overlay(args.map_image, args.map_yaml, true_poses, estimates, particles, args.output_dir / "particle_filter_overlay.png")

    print(json.dumps(summary, indent=2))
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {args.output_dir / 'particle_filter_overlay.png'}")


if __name__ == "__main__":
    main()
