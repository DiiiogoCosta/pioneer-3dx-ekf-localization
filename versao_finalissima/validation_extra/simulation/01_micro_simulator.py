#!/usr/bin/env python3
"""Micro-simulator for EKF Localization validation.

Generates synthetic ground-truth trajectories for a differential-drive robot
in a known 2D map, simulates noisy odometry and LiDAR scans, runs an EKF with
correlative scan matching, and reports ATE-RMSE and NEES statistics over
multiple Monte Carlo runs.

Includes a kidnapping experiment: at a configurable time step, the true robot
pose jumps. The script reports how quickly NIS spikes and whether the filter
recovers.

Usage:
    python 01_micro_simulator.py --runs 10 --trajectory loop --noise medium
    python 01_micro_simulator.py --runs 10 --trajectory loop --kidnap-at 50

Outputs CSV with per-run metrics and PNG plots in ./sim_output/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- map & ground truth --------------------

@dataclass(frozen=True)
class GridMap:
    occupied: np.ndarray   # bool array, True = wall
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int: return self.occupied.shape[0]
    @property
    def width(self) -> int: return self.occupied.shape[1]


def make_corridor_map(width_m: float = 12.0, height_m: float = 4.0, res: float = 0.05) -> GridMap:
    w = int(width_m / res); h = int(height_m / res)
    occ = np.zeros((h, w), dtype=bool)
    occ[0, :] = True; occ[-1, :] = True
    occ[:, 0] = True; occ[:, -1] = True
    # two pillars to break symmetry
    py1, px1 = h // 2, w // 3
    py2, px2 = h // 2, 2 * w // 3
    occ[py1-3:py1+3, px1-3:px1+3] = True
    occ[py2-3:py2+3, px2-3:px2+3] = True
    return GridMap(occ, res, 0.0, 0.0)


def make_room_map(size_m: float = 8.0, res: float = 0.05) -> GridMap:
    s = int(size_m / res)
    occ = np.zeros((s, s), dtype=bool)
    occ[0, :] = True; occ[-1, :] = True
    occ[:, 0] = True; occ[:, -1] = True
    # internal walls
    occ[s//3, :2*s//3] = True
    occ[2*s//3, s//3:] = True
    return GridMap(occ, res, 0.0, 0.0)


def make_loop_trajectory(n_steps: int = 400) -> np.ndarray:
    """Rectangular loop inside the corridor map."""
    poses = []
    # start at (2, 1), go right to (10, 1), up to (10, 3), left to (2, 3), down to (2,1)
    legs = [
        ((2.0, 1.0), (10.0, 1.0)),
        ((10.0, 1.0), (10.0, 3.0)),
        ((10.0, 3.0), (2.0, 3.0)),
        ((2.0, 3.0), (2.0, 1.0)),
    ]
    per_leg = n_steps // 4
    for (sx, sy), (ex, ey) in legs:
        for k in range(per_leg):
            alpha = k / per_leg
            x = sx + alpha * (ex - sx)
            y = sy + alpha * (ey - sy)
            theta = math.atan2(ey - sy, ex - sx)
            poses.append([x, y, theta])
    return np.asarray(poses, dtype=np.float64)


def make_lshape_trajectory(n_steps: int = 300) -> np.ndarray:
    poses = []
    legs = [((1.0, 1.0), (6.0, 1.0)), ((6.0, 1.0), (6.0, 3.0))]
    per_leg = n_steps // 2
    for (sx, sy), (ex, ey) in legs:
        for k in range(per_leg):
            a = k / per_leg
            x = sx + a * (ex - sx); y = sy + a * (ey - sy)
            theta = math.atan2(ey - sy, ex - sx)
            poses.append([x, y, theta])
    return np.asarray(poses, dtype=np.float64)


def make_curve_trajectory(n_steps: int = 400) -> np.ndarray:
    cx, cy, r = 6.0, 2.0, 1.5
    poses = []
    for k in range(n_steps):
        ang = 2 * math.pi * k / n_steps
        x = cx + r * math.cos(ang); y = cy + r * math.sin(ang)
        poses.append([x, y, ang + math.pi / 2])
    return np.asarray(poses, dtype=np.float64)


# -------------------- noisy sensor simulation --------------------

NOISE_LEVELS = {
    "low":    {"alpha_trans": 0.02, "alpha_rot": 0.01, "lidar_sigma": 0.02},
    "medium": {"alpha_trans": 0.08, "alpha_rot": 0.05, "lidar_sigma": 0.05},
    "high":   {"alpha_trans": 0.20, "alpha_rot": 0.12, "lidar_sigma": 0.10},
}


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def simulate_noisy_odometry(gt: np.ndarray, alpha_trans: float, alpha_rot: float, rng: np.random.Generator) -> np.ndarray:
    """Returns noisy odometry trajectory by integrating noisy increments."""
    odom = np.zeros_like(gt)
    odom[0] = gt[0]
    for i in range(1, len(gt)):
        dx = gt[i, 0] - gt[i-1, 0]
        dy = gt[i, 1] - gt[i-1, 1]
        ds = math.hypot(dx, dy)
        dtheta = wrap(gt[i, 2] - gt[i-1, 2])
        ds_n = ds + rng.normal(0, alpha_trans * max(ds, 0.01))
        dtheta_n = dtheta + rng.normal(0, alpha_rot * max(abs(dtheta), 0.005) + 0.002)
        odom[i, 2] = wrap(odom[i-1, 2] + dtheta_n)
        heading = odom[i-1, 2] + 0.5 * dtheta_n
        odom[i, 0] = odom[i-1, 0] + ds_n * math.cos(heading)
        odom[i, 1] = odom[i-1, 1] + ds_n * math.sin(heading)
    return odom


def raytrace(pose: np.ndarray, gmap: GridMap, n_beams: int = 60, max_range: float = 5.0,
             fov: float = 2 * math.pi, sigma: float = 0.02, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    if rng is None: rng = np.random.default_rng()
    angles = np.linspace(-fov/2, fov/2, n_beams) + pose[2]
    ranges = np.full(n_beams, max_range, dtype=np.float64)
    step = gmap.resolution * 0.5
    n_steps = int(max_range / step)
    for bi, ang in enumerate(angles):
        cx = math.cos(ang); sx = math.sin(ang)
        for s in range(1, n_steps):
            d = s * step
            wx = pose[0] + d * cx
            wy = pose[1] + d * sx
            gx = int((wx - gmap.origin_x) / gmap.resolution)
            gy = int((wy - gmap.origin_y) / gmap.resolution)
            if gx < 0 or gy < 0 or gx >= gmap.width or gy >= gmap.height:
                ranges[bi] = d; break
            if gmap.occupied[gy, gx]:
                ranges[bi] = d + rng.normal(0, sigma); break
    return ranges, np.linspace(-fov/2, fov/2, n_beams)


# -------------------- EKF with correlative scan match --------------------

from scipy.ndimage import distance_transform_edt


def distance_map(gmap: GridMap, max_dist: float = 2.0) -> np.ndarray:
    px = distance_transform_edt(~gmap.occupied)
    return np.minimum(px * gmap.resolution, max_dist).astype(np.float32)


def score_pose(pose: np.ndarray, pts_robot: np.ndarray, dmap: np.ndarray, gmap: GridMap, trim: float = 0.75) -> tuple[float, int]:
    ct = math.cos(pose[2]); st = math.sin(pose[2])
    wx = pose[0] + ct * pts_robot[:, 0] - st * pts_robot[:, 1]
    wy = pose[1] + st * pts_robot[:, 0] + ct * pts_robot[:, 1]
    gx = np.rint((wx - gmap.origin_x) / gmap.resolution).astype(np.int32)
    gy = np.rint((wy - gmap.origin_y) / gmap.resolution).astype(np.int32)
    inside = (gx >= 0) & (gx < gmap.width) & (gy >= 0) & (gy < gmap.height)
    if int(np.count_nonzero(inside)) < 8: return float("inf"), 0
    d = dmap[gy[inside], gx[inside]]
    if trim < 1.0 and len(d) > 8:
        k = max(8, int(len(d) * trim))
        d = np.partition(d, k - 1)[:k]
    return float(np.mean(d)), int(len(d))


def correlative_match(predicted: np.ndarray, pts_robot: np.ndarray, dmap: np.ndarray, gmap: GridMap,
                      xy_win: float = 0.3, xy_step: float = 0.1, th_win: float = 0.15, th_step: float = 0.05) -> tuple[np.ndarray, float]:
    dxs = np.arange(-xy_win, xy_win + 0.5*xy_step, xy_step)
    dys = np.arange(-xy_win, xy_win + 0.5*xy_step, xy_step)
    dts = np.arange(-th_win, th_win + 0.5*th_step, th_step)
    best = predicted.copy(); best_score = float("inf")
    for dx in dxs:
        for dy in dys:
            for dt in dts:
                cand = np.asarray([predicted[0]+dx, predicted[1]+dy, wrap(predicted[2]+dt)])
                s, _ = score_pose(cand, pts_robot, dmap, gmap)
                if s < best_score: best_score = s; best = cand
    return best, best_score


def ekf_predict(state: np.ndarray, cov: np.ndarray, odom_prev: np.ndarray, odom_curr: np.ndarray,
                alpha_t: float, alpha_r: float) -> tuple[np.ndarray, np.ndarray]:
    """Thrun-style odometry motion model: rot1, trans, rot2."""
    dx = odom_curr[0] - odom_prev[0]; dy = odom_curr[1] - odom_prev[1]
    trans = math.hypot(dx, dy)
    rot1 = wrap(math.atan2(dy, dx) - odom_prev[2]) if trans > 1e-6 else 0.0
    rot2 = wrap(odom_curr[2] - odom_prev[2] - rot1)
    th = state[2]
    new = np.array([
        state[0] + trans * math.cos(th + rot1),
        state[1] + trans * math.sin(th + rot1),
        wrap(th + rot1 + rot2),
    ])
    # Jacobian w.r.t. state
    G = np.array([
        [1, 0, -trans * math.sin(th + rot1)],
        [0, 1,  trans * math.cos(th + rot1)],
        [0, 0,  1],
    ])
    # Process noise from odom uncertainty
    sigma_t = alpha_t * max(trans, 0.02)
    sigma_r = alpha_r * (abs(rot1) + abs(rot2) + 0.01)
    Q = np.diag([sigma_t**2, sigma_t**2, sigma_r**2])
    return new, G @ cov @ G.T + Q


def ekf_update(state: np.ndarray, cov: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns new state, cov, NIS scalar."""
    H = np.eye(3)
    y = np.array([z[0]-state[0], z[1]-state[1], wrap(z[2]-state[2])])
    S = H @ cov @ H.T + R
    nis = float(y @ np.linalg.solve(S, y))
    K = cov @ H.T @ np.linalg.inv(S)
    new = state + K @ y
    new[2] = wrap(new[2])
    new_cov = (np.eye(3) - K @ H) @ cov
    return new, new_cov, nis


# -------------------- single run --------------------

def run_once(gt: np.ndarray, gmap: GridMap, noise_cfg: dict, rng: np.random.Generator,
             kidnap_at: int | None = None, kidnap_dx: float = 3.0, mahalanobis_gate: float = 11.34) -> dict:
    odom = simulate_noisy_odometry(gt, noise_cfg["alpha_trans"], noise_cfg["alpha_rot"], rng)
    dmap = distance_map(gmap)

    state = gt[0].copy()
    cov = np.diag([0.05, 0.05, 0.02])
    R = np.diag([0.15**2, 0.15**2, 0.08**2])

    estimates = [state.copy()]
    nis_log: list[float] = []
    accepted = 0; rejected = 0

    for i in range(1, len(gt)):
        state, cov = ekf_predict(state, cov, odom[i-1], odom[i], noise_cfg["alpha_trans"], noise_cfg["alpha_rot"])

        # KIDNAP: at step kidnap_at, push true pose so subsequent scans come from new place
        true_pose = gt[i].copy()
        if kidnap_at is not None and i >= kidnap_at:
            true_pose[0] += kidnap_dx

        # scan every 3 steps
        if i % 3 == 0:
            ranges, angles = raytrace(true_pose, gmap, n_beams=60, sigma=noise_cfg["lidar_sigma"], rng=rng)
            valid = ranges < 4.95
            pts = np.column_stack([ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid])])
            if len(pts) >= 10:
                matched, score = correlative_match(state, pts, dmap, gmap)
                if score < 0.15:
                    H = np.eye(3)
                    y = np.array([matched[0]-state[0], matched[1]-state[1], wrap(matched[2]-state[2])])
                    S = H @ cov @ H.T + R
                    nis_now = float(y @ np.linalg.solve(S, y))
                    nis_log.append(nis_now)
                    if nis_now < mahalanobis_gate:
                        state, cov, _ = ekf_update(state, cov, matched, R)
                        accepted += 1
                    else:
                        rejected += 1
        estimates.append(state.copy())

    est = np.asarray(estimates)
    err = np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1])
    odom_err = np.hypot(odom[:, 0] - gt[:, 0], odom[:, 1] - gt[:, 1])
    heading_err = np.abs(np.array([wrap(est[i, 2] - gt[i, 2]) for i in range(len(gt))]))
    return {
        "ate_rmse": float(np.sqrt(np.mean(err**2))),
        "ate_mean": float(np.mean(err)),
        "ate_max": float(np.max(err)),
        "ate_final": float(err[-1]),
        "odom_rmse": float(np.sqrt(np.mean(odom_err**2))),
        "heading_rmse_deg": float(math.degrees(np.sqrt(np.mean(heading_err**2)))),
        "nis_log": nis_log,
        "accepted": accepted,
        "rejected": rejected,
        "estimates": est,
        "odom": odom,
        "gt": gt,
    }


# -------------------- main: monte carlo + reporting --------------------

TRAJECTORIES = {"loop": (make_corridor_map, make_loop_trajectory),
                "lshape": (make_corridor_map, make_lshape_trajectory),
                "circle": (make_room_map, make_curve_trajectory)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--trajectory", choices=list(TRAJECTORIES), default="loop")
    p.add_argument("--noise", choices=list(NOISE_LEVELS), default="medium")
    p.add_argument("--kidnap-at", type=int, default=None, help="step at which true pose jumps")
    p.add_argument("--kidnap-dx", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=Path("sim_output"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_fn, traj_fn = TRAJECTORIES[args.trajectory]
    gmap = map_fn(); gt = traj_fn()
    noise_cfg = NOISE_LEVELS[args.noise]

    metrics = []
    last = None
    for r in range(args.runs):
        rng = np.random.default_rng(args.seed + r)
        out = run_once(gt, gmap, noise_cfg, rng, kidnap_at=args.kidnap_at, kidnap_dx=args.kidnap_dx)
        metrics.append({k: out[k] for k in ("ate_rmse", "ate_mean", "ate_max", "ate_final", "odom_rmse", "heading_rmse_deg", "accepted", "rejected")})
        last = out
        print(f"run {r+1}/{args.runs}: ATE-RMSE={out['ate_rmse']:.3f} m  odom-RMSE={out['odom_rmse']:.3f} m  heading-RMSE={out['heading_rmse_deg']:.2f} deg")

    arr = {k: np.array([m[k] for m in metrics]) for k in metrics[0] if k not in ("accepted", "rejected")}
    summary = {
        "trajectory": args.trajectory,
        "noise": args.noise,
        "runs": args.runs,
        "kidnap_at": args.kidnap_at,
        "ate_rmse_mean": float(arr["ate_rmse"].mean()),
        "ate_rmse_std": float(arr["ate_rmse"].std()),
        "odom_rmse_mean": float(arr["odom_rmse"].mean()),
        "odom_rmse_std": float(arr["odom_rmse"].std()),
        "heading_rmse_deg_mean": float(arr["heading_rmse_deg"].mean()),
        "improvement_factor": float(arr["odom_rmse"].mean() / max(arr["ate_rmse"].mean(), 1e-6)),
    }
    (args.output_dir / f"summary_{args.trajectory}_{args.noise}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    # plots from last run
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(gmap.occupied, origin="lower", extent=[gmap.origin_x, gmap.origin_x + gmap.width*gmap.resolution,
                                                         gmap.origin_y, gmap.origin_y + gmap.height*gmap.resolution],
                 cmap="gray_r", alpha=0.4)
    ax[0].plot(last["gt"][:, 0], last["gt"][:, 1], "g-", label="ground truth", lw=2)
    ax[0].plot(last["odom"][:, 0], last["odom"][:, 1], "r--", label="raw odom", lw=1)
    ax[0].plot(last["estimates"][:, 0], last["estimates"][:, 1], "b-", label="EKF", lw=1.5)
    ax[0].set_title(f"{args.trajectory} / {args.noise} noise (run {args.runs})"); ax[0].legend(); ax[0].axis("equal")

    if last["nis_log"]:
        ax[1].plot(last["nis_log"], "b-")
        ax[1].axhline(11.34, color="r", linestyle="--", label="χ²_{3,0.99}=11.34")
        ax[1].set_title("NIS over scan-match updates"); ax[1].set_xlabel("update #"); ax[1].set_ylabel("NIS"); ax[1].legend()

    plt.tight_layout()
    fname = args.output_dir / f"plot_{args.trajectory}_{args.noise}_kidnap{args.kidnap_at}.png"
    plt.savefig(fname, dpi=120); plt.close()
    print(f"plot: {fname}")

    # csv
    csv_path = args.output_dir / f"runs_{args.trajectory}_{args.noise}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader(); w.writerows(metrics)
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
