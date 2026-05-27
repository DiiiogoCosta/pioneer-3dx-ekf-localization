#!/usr/bin/env python3
"""Particle-filter global localization for a real bag without initial pose.

This script does not need ground truth or an aligned initial pose. It uses:

- a known occupancy map,
- relative wheel odometry,
- a LiDAR scan file with the correct laser yaw offset.

It accepts a global pose only from confidence/stability metrics, not from true
position error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from particle_filter_relocalization_test import (
    Pose,
    apply_odometry_delta,
    choose_lidar_beams,
    cluster_indices_around_pose,
    effective_sample_size,
    enforce_free_space,
    estimate_best_cluster,
    load_map,
    poses_are_close,
    read_map_yaml,
    read_poses,
    sample_free_particles,
    sensor_update,
    systematic_resample,
    weighted_pose_from_indices,
    world_to_px,
)


def angle_error(a: float, b: float) -> float:
    return float(abs(math.atan2(math.sin(a - b), math.cos(a - b))))


def nearest_reference(reference: list[Pose], t: float) -> Pose:
    index = min(range(len(reference)), key=lambda i: abs(reference[i].t - t))
    return reference[index]


def draw_overlay(
    map_image: Path,
    map_yaml: Path,
    estimates: list[tuple[int, np.ndarray, bool]],
    particles: np.ndarray,
    output: Path,
    reference: list[Pose] | None = None,
) -> None:
    resolution, origin_x, origin_y = read_map_yaml(map_yaml)
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = image.size[1]
    if reference:
        ref_px = [world_to_px(pose.x, pose.y, height, resolution, origin_x, origin_y) for pose in reference[::10]]
        if len(ref_px) > 1:
            draw.line(ref_px, fill=(220, 20, 20), width=3)
    estimate_px = [world_to_px(float(pose[0]), float(pose[1]), height, resolution, origin_x, origin_y) for _i, pose, _a in estimates]
    if len(estimate_px) > 1:
        draw.line(estimate_px, fill=(30, 90, 230), width=3)
    for _scan_index, pose, accepted in estimates[:: max(1, len(estimates) // 60)]:
        x, y = world_to_px(float(pose[0]), float(pose[1]), height, resolution, origin_x, origin_y)
        color = (20, 180, 80) if accepted else (230, 130, 20)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    for particle in particles[:: max(1, len(particles) // 500)]:
        x, y = world_to_px(float(particle[0]), float(particle[1]), height, resolution, origin_x, origin_y)
        draw.point((x, y), fill=(170, 40, 160))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global localization for real bag using Particle Filter.")
    parser.add_argument("--map-image", type=Path, default=Path("project_imp/real_bag_main/map/main_real_map.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("project_imp/real_bag_main/map/main_real_map.yaml"))
    parser.add_argument("--odom", type=Path, default=Path("project_imp/real_bag_main/data/raw_odometry.csv"))
    parser.add_argument("--lidar", type=Path, default=Path("project_imp/real_bag_main/data/lidar_yaw_minus90.npz"))
    parser.add_argument("--reference", type=Path, help="Optional aligned odometry, used only for offline evaluation/plots.")
    parser.add_argument("--output-dir", type=Path, default=Path("project_imp/real_bag_main/results/particle_filter_global"))
    parser.add_argument("--particles", type=int, default=6000)
    parser.add_argument("--max-beams", type=int, default=72)
    parser.add_argument("--scan-step", type=int, default=2)
    parser.add_argument("--sigma-hit", type=float, default=0.22)
    parser.add_argument("--translation-noise", type=float, default=0.035)
    parser.add_argument("--rotation-noise", type=float, default=0.025)
    parser.add_argument("--random-particle-rate", type=float, default=0.02)
    parser.add_argument("--resample-ratio", type=float, default=0.45)
    parser.add_argument("--estimate-top-fraction", type=float, default=0.05)
    parser.add_argument("--cluster-radius", type=float, default=0.85)
    parser.add_argument("--cluster-heading", type=float, default=0.55)
    parser.add_argument("--hysteresis-margin", type=float, default=0.10)
    parser.add_argument("--hysteresis-window", type=int, default=6)
    parser.add_argument("--accept-mass", type=float, default=0.72)
    parser.add_argument("--accept-window", type=int, default=8)
    parser.add_argument("--max-accepted-ess-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    _image, free, distance_map, resolution, origin_x, origin_y = load_map(args.map_image, args.map_yaml)
    odom_poses = read_poses(args.odom)
    reference = read_poses(args.reference) if args.reference else None
    lidar = np.load(args.lidar)
    all_ranges = lidar["ranges"]
    angles = lidar["angles"]
    range_max = float(lidar["range_max"])
    scan_times = lidar["scan_times"]

    particles = sample_free_particles(rng, free, args.particles, resolution, origin_x, origin_y)
    weights = np.full(args.particles, 1.0 / args.particles, dtype=np.float64)

    rows: list[dict[str, float | int | str]] = []
    estimates: list[tuple[int, np.ndarray, bool]] = []
    selected_pose: np.ndarray | None = None
    pending_pose: np.ndarray | None = None
    pending_candidate_masses: deque[float] = deque(maxlen=max(1, args.hysteresis_window))
    pending_selected_masses: deque[float] = deque(maxlen=max(1, args.hysteresis_window))
    accept_history: deque[bool] = deque(maxlen=args.accept_window)
    first_accepted: dict[str, float | int] | None = None
    hypothesis_switches = 0
    previous_odom_index = 0

    odom_times = np.asarray([pose.t for pose in odom_poses], dtype=np.float64)
    for scan_index in range(0, len(all_ranges), args.scan_step):
        scan_t = float(scan_times[scan_index])
        odom_index = int(np.searchsorted(odom_times, scan_t, side="left"))
        odom_index = min(max(odom_index, 0), len(odom_poses) - 1)
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
        candidate_pose, candidate_mass, candidate_count, best_particle, best_particle_weight = estimate_best_cluster(
            particles,
            weights,
            args.estimate_top_fraction,
            args.cluster_radius,
            args.cluster_heading,
        )

        switched = False
        previous_mass = 0.0
        previous_count = 0
        pending_mean_candidate = candidate_mass
        pending_mean_selected = 0.0
        pending_count = 0
        if selected_pose is None:
            estimate = candidate_pose
            selected_pose = estimate
            selected_mass = candidate_mass
        else:
            selected_indices = cluster_indices_around_pose(
                particles,
                selected_pose,
                args.cluster_radius,
                args.cluster_heading,
            )
            previous_count = int(len(selected_indices))
            previous_mass = float(np.sum(weights[selected_indices])) if len(selected_indices) else 0.0
            held_pose = weighted_pose_from_indices(particles, weights, selected_indices) if len(selected_indices) else selected_pose

            if poses_are_close(candidate_pose, selected_pose, args.cluster_radius, args.cluster_heading):
                estimate = candidate_pose
                selected_pose = estimate
                selected_mass = candidate_mass
                pending_pose = None
                pending_candidate_masses.clear()
                pending_selected_masses.clear()
            else:
                if pending_pose is None or not poses_are_close(candidate_pose, pending_pose, args.cluster_radius, args.cluster_heading):
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

        stable_now = selected_mass >= args.accept_mass and ess <= args.max_accepted_ess_ratio * args.particles
        accept_history.append(bool(stable_now))
        accepted = len(accept_history) == args.accept_window and all(accept_history)
        if accepted and first_accepted is None:
            first_accepted = {
                "scan_index": scan_index,
                "t": scan_t,
                "x": float(estimate[0]),
                "y": float(estimate[1]),
                "theta": float(estimate[2]),
                "selected_cluster_mass": selected_mass,
                "ess": ess,
            }

        row: dict[str, float | int | str] = {
            "scan_index": scan_index,
            "t": scan_t,
            "odom_index": odom_index,
            "x": float(estimate[0]),
            "y": float(estimate[1]),
            "theta": float(estimate[2]),
            "selected_cluster_mass": selected_mass,
            "candidate_cluster_mass": candidate_mass,
            "candidate_cluster_count": candidate_count,
            "best_particle": best_particle,
            "best_particle_weight": best_particle_weight,
            "previous_cluster_mass": previous_mass,
            "previous_cluster_count": previous_count,
            "pending_mean_candidate_mass": pending_mean_candidate,
            "pending_mean_selected_mass": pending_mean_selected,
            "pending_count": pending_count,
            "selected_switched": int(switched),
            "hypothesis_switches": hypothesis_switches,
            "ess": ess,
            "ess_ratio": ess / args.particles,
            "beams": len(beam_ranges),
            "mean_log_likelihood": mean_log_likelihood,
            "stable_now": int(stable_now),
            "accepted": int(accepted),
        }
        if reference is not None:
            ref = nearest_reference(reference, scan_t)
            row["reference_x"] = ref.x
            row["reference_y"] = ref.y
            row["reference_theta"] = ref.theta
            row["reference_position_error_m"] = math.hypot(float(estimate[0]) - ref.x, float(estimate[1]) - ref.y)
            row["reference_heading_error_rad"] = angle_error(float(estimate[2]), ref.theta)
        rows.append(row)
        estimates.append((scan_index, estimate.copy(), accepted))

        if ess < args.resample_ratio * args.particles:
            particles, weights = systematic_resample(rng, particles, weights)
            random_count = int(round(args.random_particle_rate * args.particles))
            if random_count > 0:
                replace = rng.choice(args.particles, size=random_count, replace=False)
                particles[replace] = sample_free_particles(rng, free, random_count, resolution, origin_x, origin_y)
                weights[:] = 1.0 / args.particles

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "particle_filter_global_estimate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accepted_count = sum(int(row["accepted"]) for row in rows)
    summary: dict[str, object] = {
        "particles": args.particles,
        "scans": len(rows),
        "accepted_count": accepted_count,
        "first_accepted": first_accepted,
        "hypothesis_switches": hypothesis_switches,
        "accept_mass": args.accept_mass,
        "accept_window": args.accept_window,
        "max_accepted_ess_ratio": args.max_accepted_ess_ratio,
    }
    if reference is not None:
        errors = np.asarray([float(row["reference_position_error_m"]) for row in rows], dtype=np.float64)
        heading_errors = np.asarray([float(row["reference_heading_error_rad"]) for row in rows], dtype=np.float64)
        summary.update(
            {
                "reference_mean_position_error_m": float(np.mean(errors)),
                "reference_median_position_error_m": float(np.median(errors)),
                "reference_final_position_error_m": float(errors[-1]),
                "reference_final_heading_error_rad": float(heading_errors[-1]),
            }
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_overlay(
        args.map_image,
        args.map_yaml,
        estimates,
        particles,
        args.output_dir / "particle_filter_global_overlay.png",
        reference=reference,
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")
    print(f"wrote: {args.output_dir / 'particle_filter_global_overlay.png'}")


if __name__ == "__main__":
    main()
