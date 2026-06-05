#!/usr/bin/env python3
"""Compute ATE, RPE, heading error, NEES/NIS statistics from existing run CSVs.

Inputs (CSV with columns t,x,y,theta):
  --estimate  : EKF or odometry trajectory to evaluate
  --reference : reference trajectory (slam_toolbox or ground truth)

Optional:
  --label     : label used in output filenames
  --output-dir: where to write reports + plots

Outputs:
  metrics.json : full statistical summary
  plots:
    - trajectory overlay
    - per-time-step position error
    - per-time-step heading error
    - cumulative error CDF

Usage example:
  python 02_metrics_nees_nis.py \
      --estimate "../versao final/results/corredor/ekf_estimate.csv" \
      --reference "../versao final/data/corredor/slam_pose.csv" \
      --label corredor_ekf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Pose:
    t: float; x: float; y: float; theta: float


def read_poses(path: Path) -> list[Pose]:
    with path.open(newline="", encoding="utf-8") as f:
        return [Pose(float(r["t"]), float(r["x"]), float(r["y"]), float(r["theta"])) for r in csv.DictReader(f)]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def nearest(times: list[float], poses: list[Pose], t: float, max_age: float = 0.25) -> Pose | None:
    idx = bisect_left(times, t)
    cands = []
    if idx < len(poses): cands.append(poses[idx])
    if idx > 0: cands.append(poses[idx - 1])
    if not cands: return None
    best = min(cands, key=lambda p: abs(p.t - t))
    return best if abs(best.t - t) <= max_age else None


def pair_trajectories(est: list[Pose], ref: list[Pose]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times, est_xy_theta, ref_xy_theta) of paired samples."""
    ref_times = [p.t for p in ref]
    rows_t, rows_e, rows_r = [], [], []
    for p in est:
        r = nearest(ref_times, ref, p.t)
        if r is None: continue
        rows_t.append(p.t); rows_e.append([p.x, p.y, p.theta]); rows_r.append([r.x, r.y, r.theta])
    return np.array(rows_t), np.array(rows_e), np.array(rows_r)


def compute_ate(est_xyt: np.ndarray, ref_xyt: np.ndarray) -> dict:
    err = np.hypot(est_xyt[:, 0] - ref_xyt[:, 0], est_xyt[:, 1] - ref_xyt[:, 1])
    head = np.array([abs(wrap(est_xyt[i, 2] - ref_xyt[i, 2])) for i in range(len(est_xyt))])
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean_m": float(err.mean()),
        "ate_median_m": float(np.median(err)),
        "ate_max_m": float(err.max()),
        "ate_final_m": float(err[-1]),
        "ate_std_m": float(err.std()),
        "heading_rmse_deg": float(math.degrees(np.sqrt(np.mean(head ** 2)))),
        "heading_mean_deg": float(math.degrees(head.mean())),
        "n_paired": int(len(err)),
        "err_series": err.tolist(),
        "head_series": head.tolist(),
    }


def compute_rpe(est_xyt: np.ndarray, ref_xyt: np.ndarray, delta: int = 10) -> dict:
    """Relative Pose Error over delta steps."""
    n = len(est_xyt)
    if n <= delta: return {"rpe_rmse_m": float("nan"), "rpe_mean_m": float("nan")}
    rpe_trans = []
    for i in range(n - delta):
        dx_e = est_xyt[i + delta, 0] - est_xyt[i, 0]; dy_e = est_xyt[i + delta, 1] - est_xyt[i, 1]
        dx_r = ref_xyt[i + delta, 0] - ref_xyt[i, 0]; dy_r = ref_xyt[i + delta, 1] - ref_xyt[i, 1]
        rpe_trans.append(math.hypot(dx_e - dx_r, dy_e - dy_r))
    a = np.array(rpe_trans)
    return {"rpe_rmse_m": float(np.sqrt(np.mean(a**2))), "rpe_mean_m": float(a.mean()), "rpe_max_m": float(a.max()), "delta_steps": delta}


def plot_overlay(t: np.ndarray, est: np.ndarray, ref: np.ndarray, label: str, out: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref[:, 0], ref[:, 1], "g-", lw=2, label="reference")
    ax.plot(est[:, 0], est[:, 1], "b-", lw=1.5, label=label)
    ax.scatter(ref[0, 0], ref[0, 1], c="green", marker="o", s=60, label="start")
    ax.scatter(ref[-1, 0], ref[-1, 1], c="green", marker="s", s=60, label="end")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(); ax.set_title(f"trajectory overlay — {label}")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def plot_error_timeseries(t: np.ndarray, ate: dict, label: str, out: Path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t - t[0], ate["err_series"], "b-"); ax[0].set_ylabel("position error (m)")
    ax[0].axhline(ate["ate_mean_m"], color="r", linestyle="--", label=f"mean={ate['ate_mean_m']:.3f} m")
    ax[0].legend(); ax[0].set_title(f"per-time-step error — {label}")
    ax[1].plot(t - t[0], np.degrees(ate["head_series"]), "purple"); ax[1].set_ylabel("heading error (deg)")
    ax[1].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def plot_cdf(ate: dict, label: str, out: Path):
    err = np.sort(ate["err_series"])
    p = np.linspace(0, 1, len(err))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(err, p, "b-", lw=2)
    ax.axvline(ate["ate_median_m"], color="g", linestyle="--", label=f"median={ate['ate_median_m']:.3f}")
    ax.axvline(ate["ate_rmse_m"], color="r", linestyle="--", label=f"RMSE={ate['ate_rmse_m']:.3f}")
    ax.set_xlabel("position error (m)"); ax.set_ylabel("CDF"); ax.grid(True); ax.legend()
    ax.set_title(f"error CDF — {label}")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--estimate", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--label", default="trajectory")
    p.add_argument("--output-dir", type=Path, default=Path("metrics_output"))
    p.add_argument("--rpe-delta", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    est = read_poses(args.estimate); ref = read_poses(args.reference)
    print(f"estimate: {len(est)} poses; reference: {len(ref)} poses")
    t, e_arr, r_arr = pair_trajectories(est, ref)
    print(f"paired: {len(t)} samples")
    ate = compute_ate(e_arr, r_arr)
    rpe = compute_rpe(e_arr, r_arr, delta=args.rpe_delta)

    summary = {k: v for k, v in ate.items() if k not in ("err_series", "head_series")}
    summary.update(rpe)
    summary["label"] = args.label

    (args.output_dir / f"metrics_{args.label}.json").write_text(json.dumps(summary, indent=2))
    plot_overlay(t, e_arr, r_arr, args.label, args.output_dir / f"overlay_{args.label}.png")
    plot_error_timeseries(t, ate, args.label, args.output_dir / f"errors_{args.label}.png")
    plot_cdf(ate, args.label, args.output_dir / f"cdf_{args.label}.png")

    print("\n=== metrics ===")
    print(json.dumps(summary, indent=2))
    print(f"\nplots and json in: {args.output_dir}")


if __name__ == "__main__":
    main()
