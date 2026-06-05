#!/usr/bin/env python3
"""EKF v2 — improvements over the original ekf_correlative_scanmatch.py.

Three improvements over v1:

1. Thrun-style odometry motion model (rot1, trans, rot2) instead of the
   hybrid atan2(dy,dx) approach. Standard formulation from Probabilistic
   Robotics §5.4 — easier to defend in the discussion.

2. Mahalanobis gating on the scan-match measurement:
       NIS = y^T S^{-1} y
   The update is rejected if NIS > chi^2_{3, 0.99} = 11.34. This gives a
   principled rejection criterion based on filter consistency, rather than
   ad-hoc thresholds on jump magnitude.

3. Adaptive measurement covariance R: the scan-match score is treated as the
   negative log-likelihood. R is rescaled by (1 + alpha*score) so that worse
   matches are trusted less. Reduces over-confidence on edge cases.

Drop-in replacement: same CLI as v1. Produces summary.json with NIS history
included for consistency analysis.
"""

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


CHI2_3_99 = 11.345  # chi-squared 3 dof, p=0.99


@dataclass(frozen=True)
class Pose:
    t: float; x: float; y: float; theta: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_poses(path: Path) -> list[Pose]:
    with path.open(newline="", encoding="utf-8") as h:
        return [Pose(float(r["t"]), float(r["x"]), float(r["y"]), float(r["theta"])) for r in csv.DictReader(h)]


def save_poses(path: Path, poses: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h); w.writerow(["t", "x", "y", "theta"])
        for p in poses: w.writerow([f"{p.t:.6f}", f"{p.x:.6f}", f"{p.y:.6f}", f"{p.theta:.6f}"])


def read_map_yaml(path: Path) -> tuple[float, float, float]:
    import re
    txt = path.read_text(encoding="utf-8")
    res = float(re.search(r"^resolution:\s*([0-9.eE+-]+)", txt, re.MULTILINE).group(1))
    o = re.search(r"^origin:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,", txt, re.MULTILINE)
    return res, float(o.group(1)), float(o.group(2))


class DistanceMap:
    def __init__(self, img: Path, yaml: Path, max_d: float = 2.0):
        self.resolution, self.origin_x, self.origin_y = read_map_yaml(yaml)
        arr = np.asarray(Image.open(img).convert("L"), dtype=np.uint8)
        self.height, self.width = arr.shape
        occ = arr < 80
        d_px = distance_transform_edt(~occ)
        self.distance = np.minimum(d_px * self.resolution, max_d).astype(np.float32)

    def world_to_grid(self, x, y):
        gx = np.rint((x - self.origin_x) / self.resolution).astype(np.int32)
        gy = self.height - np.rint((y - self.origin_y) / self.resolution).astype(np.int32)
        inside = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        return gx, gy, inside

    def score(self, pose: np.ndarray, pts: np.ndarray, trim: float) -> tuple[float, int]:
        ct = math.cos(pose[2]); st = math.sin(pose[2])
        wx = pose[0] + ct*pts[:, 0] - st*pts[:, 1]
        wy = pose[1] + st*pts[:, 0] + ct*pts[:, 1]
        gx, gy, inside = self.world_to_grid(wx, wy)
        if int(np.count_nonzero(inside)) < 10: return float("inf"), int(np.count_nonzero(inside))
        d = self.distance[gy[inside], gx[inside]]
        if trim < 1.0 and len(d) > 8:
            k = max(8, int(len(d)*trim)); d = np.partition(d, k-1)[:k]
        return float(np.mean(d)), int(len(d))


def predict_thrun_odom(state: np.ndarray, cov: np.ndarray, prev: Pose, curr: Pose,
                        alpha1: float, alpha2: float, alpha3: float, alpha4: float) -> tuple[np.ndarray, np.ndarray]:
    """Probabilistic Robotics §5.4 odometry motion model."""
    dx = curr.x - prev.x; dy = curr.y - prev.y
    trans = math.hypot(dx, dy)
    rot1 = wrap_angle(math.atan2(dy, dx) - prev.theta) if trans > 1e-6 else 0.0
    rot2 = wrap_angle(curr.theta - prev.theta - rot1)

    th = state[2]
    new = np.array([
        state[0] + trans * math.cos(th + rot1),
        state[1] + trans * math.sin(th + rot1),
        wrap_angle(th + rot1 + rot2),
    ])
    G = np.array([
        [1, 0, -trans * math.sin(th + rot1)],
        [0, 1,  trans * math.cos(th + rot1)],
        [0, 0,  1],
    ])
    # Motion noise (Thrun §5.4 eq 5.42)
    sigma_rot1 = math.sqrt(alpha1 * rot1**2 + alpha2 * trans**2 + 1e-6)
    sigma_trans = math.sqrt(alpha3 * trans**2 + alpha4 * (rot1**2 + rot2**2) + 1e-6)
    sigma_rot2 = math.sqrt(alpha1 * rot2**2 + alpha2 * trans**2 + 1e-6)
    # Map control noise into state space (approximation: diagonal)
    V = np.array([
        [-trans * math.sin(th + rot1), math.cos(th + rot1), 0],
        [ trans * math.cos(th + rot1), math.sin(th + rot1), 0],
        [1, 0, 1],
    ])
    M = np.diag([sigma_rot1**2, sigma_trans**2, sigma_rot2**2])
    R_motion = V @ M @ V.T
    return new, G @ cov @ G.T + R_motion


def adaptive_R(score: float, base_xy: float, base_th: float, alpha: float = 4.0) -> np.ndarray:
    f = 1.0 + alpha * max(score, 0.0)
    return np.diag([(base_xy*f)**2, (base_xy*f)**2, (base_th*f)**2])


def update_with_mahalanobis(state: np.ndarray, cov: np.ndarray, z: np.ndarray, R: np.ndarray,
                             gate: float = CHI2_3_99) -> tuple[np.ndarray, np.ndarray, float, bool]:
    H = np.eye(3)
    y = np.array([z[0]-state[0], z[1]-state[1], wrap_angle(z[2]-state[2])])
    S = H @ cov @ H.T + R
    nis = float(y @ np.linalg.solve(S, y))
    if nis > gate:
        return state, cov, nis, False
    K = cov @ H.T @ np.linalg.inv(S)
    new = state + K @ y
    new[2] = wrap_angle(float(new[2]))
    new_cov = (np.eye(3) - K @ H) @ cov
    return new, new_cov, nis, True


def scan_points(ranges, angles, range_max, stride, max_points):
    valid = np.isfinite(ranges) & (ranges > 0.12) & (ranges < range_max - 0.10)
    sr = ranges[valid][::stride]; sa = angles[valid][::stride]
    pts = np.column_stack([sr*np.cos(sa), sr*np.sin(sa)])
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, max_points).astype(np.int32); pts = pts[idx]
    return pts.astype(np.float64)


def correlative_match(predicted, pts, dmap, xy_w, xy_s, th_w, th_s, trim):
    dxs = np.arange(-xy_w, xy_w + 0.5*xy_s, xy_s)
    dys = np.arange(-xy_w, xy_w + 0.5*xy_s, xy_s)
    dts = np.arange(-th_w, th_w + 0.5*th_s, th_s)
    best = predicted.copy(); best_s = float("inf"); best_used = 0
    for dx, dy, dt in itertools.product(dxs, dys, dts):
        cand = np.asarray([predicted[0]+dx, predicted[1]+dy, wrap_angle(predicted[2]+dt)])
        s, used = dmap.score(cand, pts, trim=trim)
        if s < best_s: best_s = s; best = cand; best_used = used
    return best, best_s, best_used


def path_errors(ref, est):
    times = [p.t for p in ref]; errs = []
    for p in est:
        idx = bisect_left(times, p.t)
        cands = []
        if idx < len(ref): cands.append(ref[idx])
        if idx > 0: cands.append(ref[idx-1])
        if not cands: continue
        best = min(cands, key=lambda r: abs(r.t-p.t))
        if abs(best.t - p.t) <= 0.25:
            errs.append(math.hypot(best.x-p.x, best.y-p.y))
    if not errs: return float("inf"), float("inf")
    return errs[-1], float(sum(errs)/len(errs))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--odom", type=Path, required=True)
    p.add_argument("--slam", type=Path, required=True)
    p.add_argument("--lidar", type=Path, required=True)
    p.add_argument("--map-image", type=Path, required=True)
    p.add_argument("--map-yaml", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scan-every", type=int, default=5)
    p.add_argument("--beam-stride", type=int, default=5)
    p.add_argument("--max-points", type=int, default=80)
    p.add_argument("--xy-window", type=float, default=0.35)
    p.add_argument("--xy-step", type=float, default=0.10)
    p.add_argument("--theta-window", type=float, default=0.18)
    p.add_argument("--theta-step", type=float, default=0.06)
    p.add_argument("--score-gate", type=float, default=0.20, help="absolute score upper bound")
    p.add_argument("--mahalanobis-gate", type=float, default=CHI2_3_99)
    p.add_argument("--min-used", type=int, default=25)
    p.add_argument("--trim", type=float, default=0.7)
    p.add_argument("--base-std-xy", type=float, default=0.18)
    p.add_argument("--base-std-theta", type=float, default=0.08)
    p.add_argument("--adaptive-alpha", type=float, default=4.0)
    p.add_argument("--alpha1", type=float, default=0.02)
    p.add_argument("--alpha2", type=float, default=0.02)
    p.add_argument("--alpha3", type=float, default=0.05)
    p.add_argument("--alpha4", type=float, default=0.01)
    return p.parse_args()


def main():
    a = parse_args()
    odom = read_poses(a.odom); slam = read_poses(a.slam)
    lidar = np.load(a.lidar)
    scan_t = lidar["scan_times"]; ranges = lidar["ranges"]; angles = lidar["angles"]
    range_max = float(lidar["range_max"])
    dmap = DistanceMap(a.map_image, a.map_yaml)

    state = np.array([odom[0].x, odom[0].y, odom[0].theta], dtype=np.float64)
    cov = np.diag([0.05, 0.05, 0.03])
    estimates = [Pose(odom[0].t, state[0], state[1], state[2])]
    nis_log = []; acc = 0; rej_score = 0; rej_maha = 0; cand = 0; si = 0

    for i in range(1, len(odom)):
        state, cov = predict_thrun_odom(state, cov, odom[i-1], odom[i], a.alpha1, a.alpha2, a.alpha3, a.alpha4)
        while si < len(scan_t) and float(scan_t[si]) <= odom[i].t:
            if si % a.scan_every == 0:
                pts = scan_points(ranges[si], angles, range_max, a.beam_stride, a.max_points)
                if len(pts) >= a.min_used:
                    matched, score, used = correlative_match(state, pts, dmap,
                        a.xy_window, a.xy_step, a.theta_window, a.theta_step, a.trim)
                    cand += 1
                    if score > a.score_gate or used < a.min_used:
                        rej_score += 1
                    else:
                        R = adaptive_R(score, a.base_std_xy, a.base_std_theta, a.adaptive_alpha)
                        state, cov, nis, accepted = update_with_mahalanobis(state, cov, matched, R, a.mahalanobis_gate)
                        nis_log.append(nis)
                        if accepted: acc += 1
                        else: rej_maha += 1
            si += 1
        estimates.append(Pose(odom[i].t, state[0], state[1], state[2]))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_poses(a.output_dir / "ekf_v2_estimate.csv", estimates)
    np.savetxt(a.output_dir / "nis_log.csv", np.asarray(nis_log), header="nis", comments="")

    odom_f, odom_m = path_errors(slam, odom)
    ekf_f, ekf_m = path_errors(slam, estimates)
    nis_arr = np.asarray(nis_log) if nis_log else np.array([0.0])
    summary = {
        "version": "ekf_v2_improved",
        "scan_candidates": cand, "accepted": acc,
        "rejected_score": rej_score, "rejected_mahalanobis": rej_maha,
        "odom_final_error_m": round(odom_f, 3), "odom_mean_error_m": round(odom_m, 3),
        "ekf_final_error_m": round(ekf_f, 3), "ekf_mean_error_m": round(ekf_m, 3),
        "nis_mean": float(nis_arr.mean()), "nis_p95": float(np.percentile(nis_arr, 95)),
        "nis_above_gate_pct": float(100.0 * np.mean(nis_arr > a.mahalanobis_gate)),
        "consistency_note": "for a well-tuned 3-DOF EKF, NIS mean should be near 3.0 and ~99% below 11.34",
    }
    (a.output_dir / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
