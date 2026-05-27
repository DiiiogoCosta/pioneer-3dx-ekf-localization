#!/usr/bin/env python3
"""Test PF global relocalization followed by the existing EKF.

The EKF script is not modified. This wrapper creates a temporary segment whose
odometry starts at the pose recovered by the particle filter, then calls
ekf_localization.py on that segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    theta: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_poses(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            poses.append(Pose(float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"])))
    return poses


def write_poses(path: Path, poses: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "x", "y", "theta"])
        writer.writeheader()
        for pose in poses:
            writer.writerow({"t": pose.t, "x": pose.x, "y": pose.y, "theta": pose.theta})


def propagate_from_relocalized_pose(relocalized: Pose, base_odom: Pose, current_odom: Pose) -> Pose:
    dx = current_odom.x - base_odom.x
    dy = current_odom.y - base_odom.y
    ds = math.hypot(dx, dy)
    odom_motion_heading = math.atan2(dy, dx) if ds > 1e-9 else base_odom.theta
    local_heading = wrap_angle(odom_motion_heading - base_odom.theta)
    global_motion_heading = relocalized.theta + local_heading
    return Pose(
        current_odom.t,
        relocalized.x + ds * math.cos(global_motion_heading),
        relocalized.y + ds * math.sin(global_motion_heading),
        wrap_angle(relocalized.theta + wrap_angle(current_odom.theta - base_odom.theta)),
    )


def choose_relocalization(path: Path, max_error: float, max_heading_error: float, min_mass: float) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    for row in rows:
        if (
            row["position_error_m"] <= max_error
            and row["heading_error_rad"] <= max_heading_error
            and row["selected_cluster_mass"] >= min_mass
        ):
            return row
    raise SystemExit(f"No PF relocalization passed the requested gates in {path}.")


def make_segment_files(args: argparse.Namespace, pf_row: dict[str, float]) -> tuple[Path, Path, Path]:
    odom = read_poses(args.odom)
    true_path = read_poses(args.true)
    lidar = np.load(args.lidar)
    scan_times = lidar["scan_times"]
    scan_start = int(pf_row["scan_index"])
    odom_start = int(np.searchsorted([pose.t for pose in odom], float(pf_row["t"]), side="left"))
    odom_start = min(max(odom_start, 0), len(odom) - 1)

    relocalized = Pose(odom[odom_start].t, float(pf_row["x"]), float(pf_row["y"]), float(pf_row["theta"]))
    base_odom = odom[odom_start]
    odom_segment = [propagate_from_relocalized_pose(relocalized, base_odom, pose) for pose in odom[odom_start:]]
    true_segment = true_path[odom_start:]

    segment_dir = args.output_dir / "segment_inputs"
    odom_out = segment_dir / "pf_reinitialized_odometry.csv"
    true_out = segment_dir / "true_path_segment.csv"
    lidar_out = segment_dir / "lidar_scans_segment.npz"
    write_poses(odom_out, odom_segment)
    write_poses(true_out, true_segment)
    np.savez(
        lidar_out,
        scan_times=scan_times[scan_start:],
        angles=lidar["angles"],
        ranges=lidar["ranges"][scan_start:],
        range_min=lidar["range_min"],
        range_max=lidar["range_max"],
    )
    metadata = {
        "pf_scan_index": scan_start,
        "pf_t": float(pf_row["t"]),
        "pf_pose": {"x": relocalized.x, "y": relocalized.y, "theta": relocalized.theta},
        "pf_position_error_m": float(pf_row["position_error_m"]),
        "pf_heading_error_rad": float(pf_row["heading_error_rad"]),
        "pf_selected_cluster_mass": float(pf_row["selected_cluster_mass"]),
        "odom_start_index": odom_start,
    }
    (segment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return odom_out, true_out, lidar_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run existing EKF after PF relocalization.")
    parser.add_argument("--pf-estimate", type=Path, required=True)
    parser.add_argument("--odom", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/noisy_odometry.csv"))
    parser.add_argument("--true", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/true_path.csv"))
    parser.add_argument("--lidar", type=Path, default=Path("output/floorplan_walkable_test/sensors_strong/lidar_scans.npz"))
    parser.add_argument("--landmarks", type=Path, default=Path("project_imp/landmarks/floorplan_walkable_landmarks.json"))
    parser.add_argument("--map-image", type=Path, default=Path("project_imp/maps/floorplan_walkable_map.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("project_imp/maps/floorplan_walkable_map.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("project_imp/results/hybrid_pf_ekf_strong"))
    parser.add_argument("--max-pf-error", type=float, default=0.75)
    parser.add_argument("--max-pf-heading-error", type=float, default=0.60)
    parser.add_argument("--min-pf-mass", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pf_row = choose_relocalization(args.pf_estimate, args.max_pf_error, args.max_pf_heading_error, args.min_pf_mass)
    odom_out, true_out, lidar_out = make_segment_files(args, pf_row)

    ekf_output = args.output_dir / "ekf_after_pf"
    command = [
        sys.executable,
        "project_imp/scripts/ekf_localization.py",
        "--landmarks",
        str(args.landmarks),
        "--odom",
        str(odom_out),
        "--true",
        str(true_out),
        "--lidar",
        str(lidar_out),
        "--map-image",
        str(args.map_image),
        "--map-yaml",
        str(args.map_yaml),
        "--output-dir",
        str(ekf_output),
        "--lines",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout)

    ekf_summary = json.loads((ekf_output / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.output_dir / "segment_inputs" / "metadata.json").read_text(encoding="utf-8"))
    summary = {"pf_relocalization": metadata, "ekf_after_pf": ekf_summary, "command": command}
    summary_path = args.output_dir / "hybrid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
