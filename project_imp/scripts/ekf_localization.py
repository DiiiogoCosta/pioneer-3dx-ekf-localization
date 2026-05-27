#!/usr/bin/env python3
"""EKF localization using odometry prediction and circular LiDAR landmarks."""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class CircleLandmark:
    id: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class PointLandmark:
    id: str
    x: float
    y: float
    kind: str


@dataclass(frozen=True)
class LineLandmark:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Observation:
    range: float
    bearing: float
    radius: float
    points: int
    error: float


@dataclass(frozen=True)
class AssociationModel:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    score_weight: float
    feature_expansion: str


@dataclass(frozen=True)
class IcpConfidenceModel:
    initial_logit: float
    stumps: list[dict[str, float]]
    feature_names: list[str]
    threshold: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def load_association_model(path: Path, threshold: float, score_weight: float) -> AssociationModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "standardized_logistic_regression":
        raise SystemExit(f"Unsupported association model type in {path}: {data.get('type')}")
    weights = np.asarray(data["weights"], dtype=np.float64)
    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)
    if len(weights) != len(mean) + 1 or len(mean) != len(std):
        raise SystemExit(f"Invalid association model dimensions in {path}.")
    std = np.where(std < 1e-9, 1.0, std)
    return AssociationModel(
        weights=weights,
        mean=mean,
        std=std,
        threshold=threshold,
        score_weight=score_weight,
        feature_expansion=data.get("feature_expansion", "identity"),
    )


def association_features(state: np.ndarray, observation: Observation, landmark: CircleLandmark) -> np.ndarray:
    dx = landmark.x - state[0]
    dy = landmark.y - state[1]
    predicted_range = math.hypot(dx, dy)
    predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
    range_error = abs(observation.range - predicted_range)
    bearing_error = abs(wrap_angle(observation.bearing - predicted_bearing))
    radius_error = abs(observation.radius - landmark.radius)
    return np.asarray(
        [
            range_error,
            bearing_error,
            radius_error,
            observation.range,
            abs(observation.bearing),
            observation.radius,
            min(observation.points, 40) / 40.0,
            observation.error,
            predicted_range,
        ],
        dtype=np.float64,
    )


def expand_association_features(features: np.ndarray, expansion: str) -> np.ndarray:
    if expansion == "identity":
        return features
    if expansion != "poly2":
        raise SystemExit(f"Unsupported association feature expansion: {expansion}")
    values = [features, features * features]
    interactions = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            interactions.append(features[i] * features[j])
    return np.concatenate(values + [np.asarray(interactions, dtype=np.float64)])


def association_probability(model: AssociationModel, features: np.ndarray) -> float:
    expanded = expand_association_features(features, model.feature_expansion)
    normalized = (expanded - model.mean) / model.std
    vector = np.concatenate([[1.0], normalized])
    logit = float(np.clip(vector @ model.weights, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-logit))


def landmark_observation_errors(
    state: np.ndarray,
    observation: Observation,
    landmark: CircleLandmark,
) -> tuple[float, float, float]:
    dx = landmark.x - state[0]
    dy = landmark.y - state[1]
    predicted_range = math.hypot(dx, dy)
    predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
    return (
        abs(observation.range - predicted_range),
        abs(wrap_angle(observation.bearing - predicted_bearing)),
        abs(observation.radius - landmark.radius),
    )


def landmark_match_cost(
    state: np.ndarray,
    observation: Observation,
    landmark: CircleLandmark,
    gate_range: float,
    gate_bearing: float,
    radius_gate: float = 0.35,
) -> float | None:
    range_error, bearing_error, radius_error = landmark_observation_errors(state, observation, landmark)
    if range_error > gate_range or bearing_error > gate_bearing or radius_error > radius_gate:
        return None
    return range_error / gate_range + bearing_error / gate_bearing + radius_error / radius_gate


def score_landmark_pose(
    state: np.ndarray,
    observations: list[Observation],
    landmarks: list[CircleLandmark],
    gate_range: float,
    gate_bearing: float,
) -> tuple[int, float]:
    candidates: list[tuple[float, int, int]] = []
    for obs_index, observation in enumerate(observations):
        for landmark_index, landmark in enumerate(landmarks):
            cost = landmark_match_cost(state, observation, landmark, gate_range, gate_bearing)
            if cost is not None:
                candidates.append((cost, obs_index, landmark_index))
    candidates.sort(key=lambda item: item[0])
    used_obs: set[int] = set()
    used_landmarks: set[int] = set()
    total_cost = 0.0
    matches = 0
    for cost, obs_index, landmark_index in candidates:
        if obs_index in used_obs or landmark_index in used_landmarks:
            continue
        used_obs.add(obs_index)
        used_landmarks.add(landmark_index)
        total_cost += cost
        matches += 1
    if matches == 0:
        return 0, float("inf")
    return matches, total_cost / matches


def load_icp_confidence_model(path: Path, threshold: float) -> IcpConfidenceModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "gradient_boosted_decision_stumps":
        raise SystemExit(f"Unsupported ICP confidence model type in {path}: {data.get('type')}")
    return IcpConfidenceModel(
        initial_logit=float(data["initial_logit"]),
        stumps=data["stumps"],
        feature_names=list(data["feature_names"]),
        threshold=threshold,
    )


def predict_icp_confidence(model: IcpConfidenceModel, features: np.ndarray) -> float:
    score = model.initial_logit
    for stump in model.stumps:
        value = features[int(stump["feature"])]
        score += stump["left"] if value <= stump["threshold"] else stump["right"]
    score = float(np.clip(score, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-score))


def icp_confidence_features(
    state: np.ndarray,
    icp_pose: np.ndarray,
    icp_rmse: float,
    icp_used: int,
    points_robot_count: int,
    scan_ranges: np.ndarray,
) -> np.ndarray:
    jump_x = float(icp_pose[0] - state[0])
    jump_y = float(icp_pose[1] - state[1])
    jump_theta = wrap_angle(float(icp_pose[2] - state[2]))
    finite = np.isfinite(scan_ranges)
    valid = finite & (scan_ranges > 0.08)
    valid_ranges = scan_ranges[valid]
    if len(valid_ranges):
        mean_range = float(np.mean(valid_ranges))
        std_range = float(np.std(valid_ranges))
        min_range = float(np.min(valid_ranges))
    else:
        mean_range = 0.0
        std_range = 0.0
        min_range = 0.0
    return np.asarray(
        [
            float(icp_rmse if np.isfinite(icp_rmse) else 99.0),
            float(icp_used),
            float(icp_used / max(points_robot_count, 1)),
            jump_x,
            jump_y,
            jump_theta,
            math.hypot(jump_x, jump_y),
            abs(jump_theta),
            mean_range,
            std_range,
            min_range,
            float(np.mean(valid)) if len(scan_ranges) else 0.0,
        ],
        dtype=np.float64,
    )


ICP_CONFIDENCE_FEATURE_NAMES = [
    "icp_rmse",
    "icp_used",
    "icp_used_ratio",
    "jump_x",
    "jump_y",
    "jump_theta",
    "jump_xy",
    "abs_jump_theta",
    "scan_mean_range",
    "scan_std_range",
    "scan_min_range",
    "scan_valid_ratio",
]


def read_poses(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            poses.append(Pose(float(row["t"]), float(row["x"]), float(row["y"]), float(row["theta"])))
    if len(poses) < 2:
        raise SystemExit(f"Need at least two poses in {path}.")
    return poses


def read_circle_landmarks(path: Path) -> list[CircleLandmark]:
    data = json.loads(path.read_text(encoding="utf-8"))
    landmarks = [
        CircleLandmark(item["id"], float(item["center"][0]), float(item["center"][1]), float(item["radius"]))
        for item in data
        if item["type"] == "circle"
    ]
    return landmarks


def read_corner_landmarks(path: Path) -> list[PointLandmark]:
    data = json.loads(path.read_text(encoding="utf-8"))
    landmarks = [
        PointLandmark(item["id"], float(item["position"][0]), float(item["position"][1]), "corner")
        for item in data
        if item["type"] == "corner"
    ]
    return landmarks


def read_line_landmarks(path: Path) -> list[LineLandmark]:
    data = json.loads(path.read_text(encoding="utf-8"))
    landmarks = [
        LineLandmark(
            item["id"],
            float(item["start"][0]),
            float(item["start"][1]),
            float(item["end"][0]),
            float(item["end"][1]),
        )
        for item in data
        if item["type"] == "line"
    ]
    return landmarks


def fit_circle(points: np.ndarray) -> tuple[float, float, float, float] | None:
    if len(points) < 4:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones(len(points))])
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = c + cx * cx + cy * cy
    if radius_sq <= 0:
        return None
    radius = math.sqrt(float(radius_sq))
    residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    rms = math.sqrt(float(np.mean(residuals * residuals)))
    return float(cx), float(cy), radius, rms


def segment_scan(points: np.ndarray, valid: np.ndarray, max_gap: float) -> list[np.ndarray]:
    segments: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for point, is_valid in zip(points, valid, strict=False):
        if not is_valid:
            if current:
                segments.append(current)
                current = []
            previous = None
            continue
        if previous is not None and float(np.linalg.norm(point - previous)) > max_gap:
            if current:
                segments.append(current)
            current = []
        current.append(point)
        previous = point
    if current:
        segments.append(current)
    return [np.asarray(segment, dtype=np.float64) for segment in segments]


def detect_circle_observations(
    ranges: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    segment_gap: float,
    min_points: int,
    radius_min: float,
    radius_max: float,
    max_fit_error: float,
) -> list[Observation]:
    valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < range_max - 0.05)
    points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
    observations: list[Observation] = []
    for segment in segment_scan(points, valid, segment_gap):
        if len(segment) < min_points:
            continue
        fitted = fit_circle(segment)
        if fitted is None:
            continue
        cx, cy, radius, error = fitted
        if not (radius_min <= radius <= radius_max) or error > max_fit_error:
            continue
        obs_range = math.hypot(cx, cy)
        obs_bearing = math.atan2(cy, cx)
        observations.append(Observation(obs_range, obs_bearing, radius, len(segment), error))
    observations.sort(key=lambda obs: (obs.error, -obs.points))
    return observations


def fit_line_segment(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    if len(points) < 5:
        return None
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    direction = eigenvectors[:, order[0]]
    normal = np.asarray([-direction[1], direction[0]])
    projections = centered @ direction
    length = float(projections.max() - projections.min())
    errors = centered @ normal
    rms = math.sqrt(float(np.mean(errors * errors)))
    return center, direction, length, rms


def line_intersection(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray | None:
    matrix = np.column_stack([d1, -d2])
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-6:
        return None
    try:
        ts = np.linalg.solve(matrix, p2 - p1)
    except np.linalg.LinAlgError:
        return None
    return p1 + ts[0] * d1


def detect_corner_observations(
    ranges: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    segment_gap: float,
    min_line_points: int,
    min_line_length: float,
    max_line_error: float,
    angle_tolerance: float,
    max_corner_range: float,
) -> list[Observation]:
    valid = np.isfinite(ranges) & (ranges > 0.08) & (ranges < range_max - 0.05)
    points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
    segments = segment_scan(points, valid, segment_gap)
    lines: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for segment in segments:
        if len(segment) < min_line_points:
            continue
        fitted = fit_line_segment(segment)
        if fitted is None:
            continue
        center, direction, length, rms = fitted
        if length >= min_line_length and rms <= max_line_error:
            lines.append((center, direction, length, rms))

    observations: list[Observation] = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i + 1 :]:
            center_a, dir_a, len_a, err_a = line_a
            center_b, dir_b, len_b, err_b = line_b
            angle = abs(math.acos(float(np.clip(abs(np.dot(dir_a, dir_b)), 0.0, 1.0))))
            if abs(angle - math.pi / 2.0) > angle_tolerance:
                continue
            point = line_intersection(center_a, dir_a, center_b, dir_b)
            if point is None:
                continue
            corner_range = float(np.linalg.norm(point))
            if corner_range > max_corner_range or corner_range < 0.2:
                continue
            # Keep intersections close to both observed segments, not far-away crossing points.
            if np.linalg.norm(point - center_a) > len_a * 0.75 + 0.35:
                continue
            if np.linalg.norm(point - center_b) > len_b * 0.75 + 0.35:
                continue
            observations.append(
                Observation(
                    range=corner_range,
                    bearing=math.atan2(float(point[1]), float(point[0])),
                    radius=0.0,
                    points=int(len_a + len_b),
                    error=float(err_a + err_b),
                )
            )
    observations.sort(key=lambda obs: (obs.error, obs.range))
    return observations


def detect_line_observations(
    ranges: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    segment_gap: float,
    min_line_points: int,
    min_line_length: float,
    max_line_error: float,
) -> list[tuple[float, float, float, int, float]]:
    valid = np.isfinite(ranges) & (ranges > 0.08) & (ranges < range_max - 0.05)
    points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
    observations: list[tuple[float, float, float, int, float]] = []
    for segment in segment_scan(points, valid, segment_gap):
        if len(segment) < min_line_points:
            continue
        fitted = fit_line_segment(segment)
        if fitted is None:
            continue
        center, direction, length, rms = fitted
        if length < min_line_length or rms > max_line_error:
            continue
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        rho = float(center @ normal)
        alpha = math.atan2(float(normal[1]), float(normal[0]))
        if rho < 0.0:
            rho = -rho
            alpha = wrap_angle(alpha + math.pi)
        observations.append((rho, alpha, length, len(segment), rms))
    observations.sort(key=lambda item: (item[4], -item[2]))
    return observations


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
    covariance = fx @ covariance @ fx.T + q
    return state, covariance


def update_with_landmark(
    state: np.ndarray,
    covariance: np.ndarray,
    observation: Observation,
    landmark: CircleLandmark,
    range_std: float,
    bearing_std: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dx = landmark.x - state[0]
    dy = landmark.y - state[1]
    q = dx * dx + dy * dy
    if q < 1e-9:
        return state, covariance, float("inf")
    predicted_range = math.sqrt(q)
    predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
    innovation = np.asarray(
        [
            observation.range - predicted_range,
            wrap_angle(observation.bearing - predicted_bearing),
        ]
    )
    h = np.asarray(
        [
            [-dx / predicted_range, -dy / predicted_range, 0.0],
            [dy / q, -dx / q, -1.0],
        ]
    )
    r = np.diag([range_std * range_std, bearing_std * bearing_std])
    s = h @ covariance @ h.T + r
    try:
        s_inv = np.linalg.inv(s)
    except np.linalg.LinAlgError:
        return state, covariance, float("inf")
    mahalanobis = float(innovation.T @ s_inv @ innovation)
    k = covariance @ h.T @ s_inv
    state = state + k @ innovation
    state[2] = wrap_angle(float(state[2]))
    covariance = (np.eye(3) - k @ h) @ covariance
    return state, covariance, mahalanobis


def update_with_point_landmark(
    state: np.ndarray,
    covariance: np.ndarray,
    observation: Observation,
    landmark: PointLandmark,
    range_std: float,
    bearing_std: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dx = landmark.x - state[0]
    dy = landmark.y - state[1]
    q = dx * dx + dy * dy
    if q < 1e-9:
        return state, covariance, float("inf")
    predicted_range = math.sqrt(q)
    predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
    innovation = np.asarray(
        [
            observation.range - predicted_range,
            wrap_angle(observation.bearing - predicted_bearing),
        ]
    )
    h = np.asarray(
        [
            [-dx / predicted_range, -dy / predicted_range, 0.0],
            [dy / q, -dx / q, -1.0],
        ]
    )
    r = np.diag([range_std * range_std, bearing_std * bearing_std])
    s = h @ covariance @ h.T + r
    try:
        s_inv = np.linalg.inv(s)
    except np.linalg.LinAlgError:
        return state, covariance, float("inf")
    mahalanobis = float(innovation.T @ s_inv @ innovation)
    k = covariance @ h.T @ s_inv
    state = state + k @ innovation
    state[2] = wrap_angle(float(state[2]))
    covariance = (np.eye(3) - k @ h) @ covariance
    return state, covariance, mahalanobis


def line_params_in_map(landmark: LineLandmark) -> tuple[float, float]:
    dx = landmark.x2 - landmark.x1
    dy = landmark.y2 - landmark.y1
    alpha = math.atan2(dy, dx) + math.pi / 2.0
    rho = landmark.x1 * math.cos(alpha) + landmark.y1 * math.sin(alpha)
    if rho < 0.0:
        rho = -rho
        alpha += math.pi
    return rho, wrap_angle(alpha)


def update_with_line_landmark(
    state: np.ndarray,
    covariance: np.ndarray,
    observation: tuple[float, float, float, int, float],
    landmark: LineLandmark,
    rho_std: float,
    alpha_std: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    obs_rho, obs_alpha, _length, _points, _error = observation
    map_rho, map_alpha = line_params_in_map(landmark)
    predicted_rho = map_rho - state[0] * math.cos(map_alpha) - state[1] * math.sin(map_alpha)
    predicted_alpha = wrap_angle(map_alpha - state[2])
    if predicted_rho < 0.0:
        predicted_rho = -predicted_rho
        predicted_alpha = wrap_angle(predicted_alpha + math.pi)
    innovation = np.asarray(
        [
            obs_rho - predicted_rho,
            wrap_angle(obs_alpha - predicted_alpha),
        ],
        dtype=np.float64,
    )
    sign = -1.0 if map_rho - state[0] * math.cos(map_alpha) - state[1] * math.sin(map_alpha) >= 0.0 else 1.0
    h = np.asarray(
        [
            [sign * math.cos(map_alpha), sign * math.sin(map_alpha), 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    r = np.diag([rho_std * rho_std, alpha_std * alpha_std])
    s = h @ covariance @ h.T + r
    try:
        s_inv = np.linalg.inv(s)
    except np.linalg.LinAlgError:
        return state, covariance, float("inf")
    mahalanobis = float(innovation.T @ s_inv @ innovation)
    k = covariance @ h.T @ s_inv
    state = state + k @ innovation
    state[2] = wrap_angle(float(state[2]))
    covariance = (np.eye(3) - k @ h) @ covariance
    return state, covariance, mahalanobis


def associate_observation(
    state: np.ndarray,
    observation: Observation,
    landmarks: list[CircleLandmark],
    used: set[str],
    gate_range: float,
    gate_bearing: float,
    min_score_ratio: float = 1.0,
    association_model: AssociationModel | None = None,
) -> tuple[CircleLandmark, float | None] | None:
    candidates: list[tuple[float, CircleLandmark, float | None]] = []
    for landmark in landmarks:
        if landmark.id in used:
            continue
        dx = landmark.x - state[0]
        dy = landmark.y - state[1]
        predicted_range = math.hypot(dx, dy)
        predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
        range_error = abs(observation.range - predicted_range)
        bearing_error = abs(wrap_angle(observation.bearing - predicted_bearing))
        radius_error = abs(observation.radius - landmark.radius)
        if range_error > gate_range or bearing_error > gate_bearing or radius_error > 0.35:
            continue
        score = range_error / gate_range + bearing_error / gate_bearing + radius_error
        probability = None
        if association_model is not None:
            probability = association_probability(association_model, association_features(state, observation, landmark))
            if probability < association_model.threshold:
                continue
            score -= association_model.score_weight * probability
        candidates.append((score, landmark, probability))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and min_score_ratio > 1.0:
        best_score = max(candidates[0][0], 1e-6)
        if candidates[1][0] / best_score < min_score_ratio:
            return None
    return candidates[0][1], candidates[0][2]


def globally_associate_observations(
    state: np.ndarray,
    observations: list[Observation],
    landmarks: list[CircleLandmark],
    gate_range: float,
    gate_bearing: float,
    min_score_ratio: float = 1.0,
    association_model: AssociationModel | None = None,
) -> list[tuple[Observation, CircleLandmark, float | None]]:
    """Choose a unique observation->landmark assignment with minimum total cost.

    There are few observed circle landmarks, so a small brute-force search is
    clearer and safer than a greedy local assignment.
    """
    obs_subset = observations[:8]
    candidates_by_obs: list[list[tuple[float, CircleLandmark, float | None]]] = []
    for observation in obs_subset:
        candidates: list[tuple[float, CircleLandmark, float | None]] = []
        for landmark in landmarks:
            cost = landmark_match_cost(state, observation, landmark, gate_range, gate_bearing)
            if cost is None:
                continue
            probability = None
            if association_model is not None:
                probability = association_probability(association_model, association_features(state, observation, landmark))
                if probability < association_model.threshold:
                    continue
                cost -= association_model.score_weight * probability
            candidates.append((cost, landmark, probability))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and min_score_ratio > 1.0:
            best_score = max(candidates[0][0], 1e-6)
            if candidates[1][0] / best_score < min_score_ratio:
                candidates = []
        candidates_by_obs.append(candidates)

    best: tuple[int, float, list[tuple[Observation, CircleLandmark, float | None]]] | None = None

    def search(index: int, used: set[str], total: float, chosen: list[tuple[Observation, CircleLandmark, float | None]]) -> None:
        nonlocal best
        if index >= len(obs_subset):
            key = (len(chosen), -total)
            if best is None or key > (best[0], -best[1]):
                best = (len(chosen), total, list(chosen))
            return
        # Skip is allowed, because not every cluster is a valid landmark.
        search(index + 1, used, total, chosen)
        observation = obs_subset[index]
        for cost, landmark, probability in candidates_by_obs[index]:
            if landmark.id in used:
                continue
            used.add(landmark.id)
            chosen.append((observation, landmark, probability))
            search(index + 1, used, total + cost, chosen)
            chosen.pop()
            used.remove(landmark.id)

    search(0, set(), 0.0, [])
    return best[2] if best is not None else []


def associate_point_observation(
    state: np.ndarray,
    observation: Observation,
    landmarks: list[PointLandmark],
    used: set[str],
    gate_range: float,
    gate_bearing: float,
    min_score_ratio: float = 1.0,
) -> PointLandmark | None:
    candidates: list[tuple[float, PointLandmark]] = []
    for landmark in landmarks:
        if landmark.id in used:
            continue
        dx = landmark.x - state[0]
        dy = landmark.y - state[1]
        predicted_range = math.hypot(dx, dy)
        predicted_bearing = wrap_angle(math.atan2(dy, dx) - state[2])
        range_error = abs(observation.range - predicted_range)
        bearing_error = abs(wrap_angle(observation.bearing - predicted_bearing))
        if range_error > gate_range or bearing_error > gate_bearing:
            continue
        score = range_error / gate_range + bearing_error / gate_bearing
        candidates.append((score, landmark))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and min_score_ratio > 1.0:
        best_score = max(candidates[0][0], 1e-6)
        if candidates[1][0] / best_score < min_score_ratio:
            return None
    return candidates[0][1]


def associate_line_observation(
    state: np.ndarray,
    observation: tuple[float, float, float, int, float],
    landmarks: list[LineLandmark],
    used: set[str],
    gate_rho: float,
    gate_alpha: float,
) -> LineLandmark | None:
    obs_rho, obs_alpha, _length, _points, _error = observation
    candidates: list[tuple[float, LineLandmark]] = []
    for landmark in landmarks:
        if landmark.id in used:
            continue
        map_rho, map_alpha = line_params_in_map(landmark)
        predicted_rho = map_rho - state[0] * math.cos(map_alpha) - state[1] * math.sin(map_alpha)
        predicted_alpha = wrap_angle(map_alpha - state[2])
        if predicted_rho < 0.0:
            predicted_rho = -predicted_rho
            predicted_alpha = wrap_angle(predicted_alpha + math.pi)
        rho_error = abs(obs_rho - predicted_rho)
        alpha_error = abs(wrap_angle(obs_alpha - predicted_alpha))
        if rho_error > gate_rho or alpha_error > gate_alpha:
            continue
        candidates.append((rho_error / gate_rho + alpha_error / gate_alpha, landmark))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def circle_intersection_pose_candidates(
    observations: list[Observation],
    landmarks: list[CircleLandmark],
    max_pair_range_error: float,
) -> list[tuple[float, np.ndarray]]:
    candidates: list[tuple[float, np.ndarray]] = []
    if len(observations) < 2:
        return candidates

    for i, obs_a in enumerate(observations):
        for obs_b in observations[i + 1 :]:
            obs_bearing_delta = wrap_angle(obs_b.bearing - obs_a.bearing)
            if abs(obs_bearing_delta) < 0.15:
                continue
            for lm_a in landmarks:
                for lm_b in landmarks:
                    if lm_a.id == lm_b.id:
                        continue
                    landmark_distance = math.hypot(lm_b.x - lm_a.x, lm_b.y - lm_a.y)
                    observed_distance = math.sqrt(
                        obs_a.range * obs_a.range
                        + obs_b.range * obs_b.range
                        - 2.0 * obs_a.range * obs_b.range * math.cos(obs_bearing_delta)
                    )
                    if abs(landmark_distance - observed_distance) > max_pair_range_error:
                        continue

                    map_angle = math.atan2(lm_b.y - lm_a.y, lm_b.x - lm_a.x)
                    obs_angle = math.atan2(
                        obs_b.range * math.sin(obs_b.bearing) - obs_a.range * math.sin(obs_a.bearing),
                        obs_b.range * math.cos(obs_b.bearing) - obs_a.range * math.cos(obs_a.bearing),
                    )
                    theta = wrap_angle(map_angle - obs_angle)
                    robot_from_a = np.asarray(
                        [
                            lm_a.x - obs_a.range * math.cos(theta + obs_a.bearing),
                            lm_a.y - obs_a.range * math.sin(theta + obs_a.bearing),
                            theta,
                        ]
                    )
                    robot_from_b = np.asarray(
                        [
                            lm_b.x - obs_b.range * math.cos(theta + obs_b.bearing),
                            lm_b.y - obs_b.range * math.sin(theta + obs_b.bearing),
                            theta,
                        ]
                    )
                    candidate = robot_from_a.copy()
                    candidate[0:2] = (robot_from_a[0:2] + robot_from_b[0:2]) / 2.0

                    residual = 0.0
                    matches = 0
                    for obs in observations:
                        best_error = float("inf")
                        for lm in landmarks:
                            dx = lm.x - candidate[0]
                            dy = lm.y - candidate[1]
                            pred_range = math.hypot(dx, dy)
                            pred_bearing = wrap_angle(math.atan2(dy, dx) - candidate[2])
                            error = abs(pred_range - obs.range) + 2.0 * abs(wrap_angle(pred_bearing - obs.bearing))
                            best_error = min(best_error, error)
                        if best_error < 1.0:
                            residual += best_error
                            matches += 1
                    if matches >= 2:
                        candidates.append((residual / matches, candidate))
    candidates.sort(key=lambda item: item[0])
    return candidates


def maybe_global_recover(
    state: np.ndarray,
    covariance: np.ndarray,
    observations: list[Observation],
    landmarks: list[CircleLandmark],
    covariance_trace_trigger: float,
    max_position_jump: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if np.trace(covariance[:2, :2]) < covariance_trace_trigger and observations:
        # Even with low covariance, allow recovery if the best geometric pose is close enough.
        pass
    candidates = circle_intersection_pose_candidates(observations[:8], landmarks, max_pair_range_error=0.8)
    if not candidates:
        return state, covariance, False
    _score, candidate = candidates[0]
    jump = math.hypot(candidate[0] - state[0], candidate[1] - state[1])
    if jump > max_position_jump:
        return state, covariance, False
    state = state.copy()
    state[0] = (1.0 - blend) * state[0] + blend * candidate[0]
    state[1] = (1.0 - blend) * state[1] + blend * candidate[1]
    state[2] = wrap_angle((1.0 - blend) * state[2] + blend * candidate[2])
    covariance = covariance.copy()
    covariance[0, 0] = min(covariance[0, 0], 0.25)
    covariance[1, 1] = min(covariance[1, 1], 0.25)
    covariance[2, 2] = min(covariance[2, 2], 0.08)
    return state, covariance, True


def pose_from_observation_pair(
    obs_a: Observation,
    obs_b: Observation,
    lm_a: CircleLandmark,
    lm_b: CircleLandmark,
) -> np.ndarray | None:
    if lm_a.id == lm_b.id:
        return None
    obs_bearing_delta = wrap_angle(obs_b.bearing - obs_a.bearing)
    if abs(obs_bearing_delta) < 0.12:
        return None
    landmark_distance = math.hypot(lm_b.x - lm_a.x, lm_b.y - lm_a.y)
    observed_distance_sq = (
        obs_a.range * obs_a.range
        + obs_b.range * obs_b.range
        - 2.0 * obs_a.range * obs_b.range * math.cos(obs_bearing_delta)
    )
    if observed_distance_sq <= 1e-9:
        return None
    observed_distance = math.sqrt(observed_distance_sq)
    if abs(landmark_distance - observed_distance) > 1.0:
        return None
    map_angle = math.atan2(lm_b.y - lm_a.y, lm_b.x - lm_a.x)
    obs_angle = math.atan2(
        obs_b.range * math.sin(obs_b.bearing) - obs_a.range * math.sin(obs_a.bearing),
        obs_b.range * math.cos(obs_b.bearing) - obs_a.range * math.cos(obs_a.bearing),
    )
    theta = wrap_angle(map_angle - obs_angle)
    robot_from_a = np.asarray(
        [
            lm_a.x - obs_a.range * math.cos(theta + obs_a.bearing),
            lm_a.y - obs_a.range * math.sin(theta + obs_a.bearing),
            theta,
        ],
        dtype=np.float64,
    )
    robot_from_b = np.asarray(
        [
            lm_b.x - obs_b.range * math.cos(theta + obs_b.bearing),
            lm_b.y - obs_b.range * math.sin(theta + obs_b.bearing),
            theta,
        ],
        dtype=np.float64,
    )
    candidate = robot_from_a.copy()
    candidate[:2] = (robot_from_a[:2] + robot_from_b[:2]) / 2.0
    return candidate


def maybe_global_landmark_relocalize(
    state: np.ndarray,
    covariance: np.ndarray,
    observations: list[Observation],
    landmarks: list[CircleLandmark],
    gate_range: float,
    gate_bearing: float,
    min_matches: int,
    min_match_gain: int,
    min_score_gain: float,
    max_jump: float,
    blend: float,
    distance_map: DistanceMap | None = None,
    points_robot: np.ndarray | None = None,
    map_rmse_gate: float = 0.35,
    map_min_used: int = 25,
) -> tuple[np.ndarray, np.ndarray, bool]:
    obs_subset = observations[:8]
    if len(obs_subset) < 2:
        return state, covariance, False
    current_matches, current_score = score_landmark_pose(state, obs_subset, landmarks, gate_range, gate_bearing)
    best_candidate: np.ndarray | None = None
    best_matches = 0
    best_score = float("inf")
    best_map_rmse = float("inf")
    for obs_a, obs_b in itertools.permutations(obs_subset, 2):
        for lm_a, lm_b in itertools.permutations(landmarks, 2):
            candidate = pose_from_observation_pair(obs_a, obs_b, lm_a, lm_b)
            if candidate is None:
                continue
            jump = math.hypot(float(candidate[0] - state[0]), float(candidate[1] - state[1]))
            if jump > max_jump:
                continue
            matches, score = score_landmark_pose(candidate, obs_subset, landmarks, gate_range, gate_bearing)
            if matches < min_matches:
                continue
            map_rmse = 0.0
            if distance_map is not None and points_robot is not None:
                map_rmse, map_used = scan_to_map_rmse(candidate, points_robot, distance_map, max_corr_distance=1.2)
                if map_used < map_min_used or map_rmse > map_rmse_gate:
                    continue
            if (
                matches > best_matches
                or (matches == best_matches and map_rmse < best_map_rmse - 1e-6)
                or (matches == best_matches and abs(map_rmse - best_map_rmse) <= 1e-6 and score < best_score)
            ):
                best_candidate = candidate
                best_matches = matches
                best_score = score
                best_map_rmse = map_rmse
    if best_candidate is None:
        return state, covariance, False
    enough_more_matches = best_matches >= current_matches + min_match_gain
    enough_better_score = best_matches >= current_matches and best_score + min_score_gain < current_score
    if not enough_more_matches and not enough_better_score:
        return state, covariance, False
    state = state.copy()
    state[0] = (1.0 - blend) * state[0] + blend * best_candidate[0]
    state[1] = (1.0 - blend) * state[1] + blend * best_candidate[1]
    state[2] = wrap_angle(state[2] + blend * wrap_angle(best_candidate[2] - state[2]))
    covariance = covariance.copy()
    covariance[0, 0] = min(covariance[0, 0], 0.35)
    covariance[1, 1] = min(covariance[1, 1], 0.35)
    covariance[2, 2] = min(covariance[2, 2], 0.10)
    return state, covariance, True


def save_estimate(path: Path, estimates: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta"])
        for pose in estimates:
            writer.writerow([f"{pose.t:.3f}", f"{pose.x:.6f}", f"{pose.y:.6f}", f"{pose.theta:.6f}"])


def read_map_yaml(path: Path) -> tuple[float, float, float]:
    import re

    text = path.read_text(encoding="utf-8")
    resolution = float(re.search(r"^resolution:\s*([0-9.eE+-]+)", text, re.MULTILINE).group(1))
    origin = re.search(r"^origin:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,", text, re.MULTILINE)
    return resolution, float(origin.group(1)), float(origin.group(2))


class DistanceMap:
    def __init__(self, map_image: Path, map_yaml: Path, max_distance: float = 2.0) -> None:
        self.resolution, self.origin_x, self.origin_y = read_map_yaml(map_yaml)
        array = np.asarray(Image.open(map_image).convert("L"), dtype=np.uint8)
        self.height, self.width = array.shape
        occupied = array < 80
        ys, xs = np.nonzero(occupied)
        self.occupied_points = np.column_stack(
            [
                self.origin_x + xs.astype(np.float64) * self.resolution,
                self.origin_y + (self.height - ys.astype(np.float64)) * self.resolution,
            ]
        )
        self.dist = np.full((self.height, self.width), max_distance, dtype=np.float32)
        self.grad_x = np.zeros((self.height, self.width), dtype=np.float32)
        self.grad_y = np.zeros((self.height, self.width), dtype=np.float32)
        queue: list[tuple[float, int, int]] = []
        for y, x in zip(ys, xs, strict=False):
            self.dist[y, x] = 0.0
            heapq.heappush(queue, (0.0, int(y), int(x)))

        neighbors = [
            (-1, 0, self.resolution),
            (1, 0, self.resolution),
            (0, -1, self.resolution),
            (0, 1, self.resolution),
            (-1, -1, self.resolution * math.sqrt(2.0)),
            (-1, 1, self.resolution * math.sqrt(2.0)),
            (1, -1, self.resolution * math.sqrt(2.0)),
            (1, 1, self.resolution * math.sqrt(2.0)),
        ]
        while queue:
            distance, y, x = heapq.heappop(queue)
            if distance > float(self.dist[y, x]) or distance >= max_distance:
                continue
            for dy, dx, step in neighbors:
                yy = y + dy
                xx = x + dx
                if not (0 <= yy < self.height and 0 <= xx < self.width):
                    continue
                candidate = distance + step
                if candidate < float(self.dist[yy, xx]) and candidate <= max_distance:
                    self.dist[yy, xx] = candidate
                    heapq.heappush(queue, (candidate, yy, xx))
        self.grad_y, self.grad_x = np.gradient(self.dist, self.resolution, self.resolution)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        gx = int(round((x - self.origin_x) / self.resolution))
        gy = self.height - int(round((y - self.origin_y) / self.resolution))
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return gx, gy
        return None

    def sample(self, x: float, y: float) -> tuple[float, float, float] | None:
        cell = self.world_to_grid(x, y)
        if cell is None:
            return None
        gx, gy = cell
        return float(self.dist[gy, gx]), float(self.grad_x[gy, gx]), float(self.grad_y[gy, gx])


def scan_points_for_icp(
    ranges: np.ndarray,
    angles: np.ndarray,
    range_max: float,
    stride: int,
    max_points: int,
) -> np.ndarray:
    valid = np.isfinite(ranges) & (ranges > 0.08) & (ranges < range_max - 0.10)
    selected_ranges = ranges[valid][::stride]
    selected_angles = angles[valid][::stride]
    points = np.column_stack([selected_ranges * np.cos(selected_angles), selected_ranges * np.sin(selected_angles)])
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int32)
        points = points[indices]
    return points.astype(np.float64)


def icp_scan_to_map(
    state: np.ndarray,
    points_robot: np.ndarray,
    distance_map: DistanceMap,
    iterations: int,
    max_corr_distance: float,
    damping: float,
) -> tuple[np.ndarray, float, int]:
    pose = state.copy()
    if len(points_robot) < 10:
        return pose, float("inf"), 0
    used_count = 0
    last_rmse = float("inf")
    for _ in range(iterations):
        cos_t = math.cos(pose[2])
        sin_t = math.sin(pose[2])
        h = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        errors: list[float] = []
        used_count = 0
        for px, py in points_robot:
            wx = pose[0] + cos_t * px - sin_t * py
            wy = pose[1] + sin_t * px + cos_t * py
            sample = distance_map.sample(wx, wy)
            if sample is None:
                continue
            dist, grad_x, grad_y = sample
            if dist > max_corr_distance:
                continue
            dwdtheta_x = -sin_t * px - cos_t * py
            dwdtheta_y = cos_t * px - sin_t * py
            jac = np.asarray([grad_x, grad_y, grad_x * dwdtheta_x + grad_y * dwdtheta_y])
            h += np.outer(jac, jac)
            b += jac * dist
            errors.append(dist * dist)
            used_count += 1
        if used_count < 10:
            return pose, float("inf"), used_count
        h += np.diag([damping, damping, damping * 0.2])
        try:
            delta = -np.linalg.solve(h, b)
        except np.linalg.LinAlgError:
            break
        delta[0] = float(np.clip(delta[0], -0.25, 0.25))
        delta[1] = float(np.clip(delta[1], -0.25, 0.25))
        delta[2] = float(np.clip(delta[2], -0.08, 0.08))
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] = wrap_angle(pose[2] + delta[2])
        last_rmse = math.sqrt(float(np.mean(errors)))
        if np.linalg.norm(delta) < 1e-4:
            break
    return pose, last_rmse, used_count


def scan_to_map_rmse(
    pose: np.ndarray,
    points_robot: np.ndarray,
    distance_map: DistanceMap,
    max_corr_distance: float,
) -> tuple[float, int]:
    if len(points_robot) == 0:
        return float("inf"), 0
    cos_t = math.cos(pose[2])
    sin_t = math.sin(pose[2])
    errors: list[float] = []
    for px, py in points_robot:
        wx = pose[0] + cos_t * px - sin_t * py
        wy = pose[1] + sin_t * px + cos_t * py
        sample = distance_map.sample(wx, wy)
        if sample is None:
            continue
        dist = sample[0]
        if dist > max_corr_distance:
            continue
        errors.append(dist * dist)
    if not errors:
        return float("inf"), 0
    return math.sqrt(float(np.mean(errors))), len(errors)


def update_with_pose_measurement(
    state: np.ndarray,
    covariance: np.ndarray,
    measured_pose: np.ndarray,
    std_xy: float,
    std_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = measured_pose.copy()
    innovation = np.asarray([z[0] - state[0], z[1] - state[1], wrap_angle(z[2] - state[2])])
    h = np.eye(3)
    r = np.diag([std_xy * std_xy, std_xy * std_xy, std_theta * std_theta])
    s = h @ covariance @ h.T + r
    k = covariance @ h.T @ np.linalg.inv(s)
    state = state + k @ innovation
    state[2] = wrap_angle(float(state[2]))
    covariance = (np.eye(3) - k @ h) @ covariance
    return state, covariance


def world_to_px(point: tuple[float, float], height: int, resolution: float, origin_x: float, origin_y: float) -> tuple[int, int]:
    return int(round((point[0] - origin_x) / resolution)), height - int(round((point[1] - origin_y) / resolution))


def draw_overlay(
    map_image: Path,
    map_yaml: Path,
    true_path: list[Pose],
    odom_path: list[Pose],
    ekf_path: list[Pose],
    output: Path,
) -> None:
    resolution, origin_x, origin_y = read_map_yaml(map_yaml)
    image = Image.open(map_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = image.size[1]
    true_px = [world_to_px((pose.x, pose.y), height, resolution, origin_x, origin_y) for pose in true_path]
    odom_px = [world_to_px((pose.x, pose.y), height, resolution, origin_x, origin_y) for pose in odom_path]
    ekf_px = [world_to_px((pose.x, pose.y), height, resolution, origin_x, origin_y) for pose in ekf_path]
    draw.line(true_px, fill=(0, 150, 70), width=5)
    draw.line(odom_px, fill=(220, 30, 30), width=3)
    draw.line(ekf_px, fill=(30, 90, 230), width=3)
    draw.rectangle((10, 10, 230, 78), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(22, 25), (70, 25)], fill=(0, 150, 70), width=5)
    draw.text((78, 17), "real", fill=(0, 0, 0))
    draw.line([(22, 45), (70, 45)], fill=(220, 30, 30), width=3)
    draw.text((78, 37), "odometria", fill=(0, 0, 0))
    draw.line([(22, 65), (70, 65)], fill=(30, 90, 230), width=3)
    draw.text((78, 57), "EKF", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def path_errors(reference: list[Pose], estimate: list[Pose]) -> tuple[float, float]:
    count = min(len(reference), len(estimate))
    errors = [math.hypot(reference[i].x - estimate[i].x, reference[i].y - estimate[i].y) for i in range(count)]
    return errors[-1], sum(errors) / len(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EKF localization with circular LiDAR landmarks.")
    parser.add_argument("--landmarks", type=Path, default=Path("output/generic_sim_landmarks.json"))
    parser.add_argument("--odom", type=Path, default=Path("output/simulated_sensors_stable/noisy_odometry.csv"))
    parser.add_argument("--true", type=Path, default=Path("output/simulated_sensors_stable/true_path.csv"))
    parser.add_argument("--lidar", type=Path, default=Path("output/simulated_sensors_stable/lidar_scans.npz"))
    parser.add_argument("--map-image", type=Path, default=Path("output/generic_sim_map_32x26.pgm"))
    parser.add_argument("--map-yaml", type=Path, default=Path("output/generic_sim_map_32x26.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ekf_stable"))
    parser.add_argument("--segment-gap", type=float, default=0.28)
    parser.add_argument("--min-segment-points", type=int, default=5)
    parser.add_argument("--radius-min", type=float, default=0.28)
    parser.add_argument("--radius-max", type=float, default=0.85)
    parser.add_argument("--max-fit-error", type=float, default=0.08)
    parser.add_argument("--corners", action="store_true", help="Also use 90-degree wall corners as point landmarks.")
    parser.add_argument("--corner-min-line-points", type=int, default=8)
    parser.add_argument("--corner-min-line-length", type=float, default=0.45)
    parser.add_argument("--corner-max-line-error", type=float, default=0.055)
    parser.add_argument("--corner-angle-tolerance", type=float, default=0.35)
    parser.add_argument("--corner-max-range", type=float, default=5.6)
    parser.add_argument("--corner-gate-range", type=float, default=1.2)
    parser.add_argument("--corner-gate-bearing", type=float, default=0.45)
    parser.add_argument("--corner-range-std", type=float, default=0.25)
    parser.add_argument("--corner-bearing-std", type=float, default=0.14)
    parser.add_argument("--lines", action="store_true", help="Also use existing wall line landmarks.")
    parser.add_argument("--line-min-points", type=int, default=10)
    parser.add_argument("--line-min-length", type=float, default=0.6)
    parser.add_argument("--line-max-error", type=float, default=0.06)
    parser.add_argument("--line-gate-rho", type=float, default=0.8)
    parser.add_argument("--line-gate-alpha", type=float, default=0.45)
    parser.add_argument("--line-rho-std", type=float, default=0.25)
    parser.add_argument("--line-alpha-std", type=float, default=0.14)
    parser.add_argument("--gate-range", type=float, default=1.5)
    parser.add_argument("--gate-bearing", type=float, default=0.45)
    parser.add_argument("--range-std", type=float, default=0.18)
    parser.add_argument("--bearing-std", type=float, default=0.10)
    parser.add_argument("--predict-trans-noise", type=float, default=0.06)
    parser.add_argument("--predict-rot-noise", type=float, default=0.08)
    parser.add_argument("--mahalanobis-gate", type=float, default=9.21)
    parser.add_argument("--association-ratio", type=float, default=1.0)
    parser.add_argument("--global-association", action="store_true", help="Assign all visible circle landmarks jointly.")
    parser.add_argument("--global-relocalization", action="store_true", help="Relocalize from landmark-pair geometry when EKF is inconsistent.")
    parser.add_argument("--global-relocalization-gate-range", type=float, default=1.6)
    parser.add_argument("--global-relocalization-gate-bearing", type=float, default=0.65)
    parser.add_argument("--global-relocalization-min-matches", type=int, default=2)
    parser.add_argument("--global-relocalization-min-match-gain", type=int, default=1)
    parser.add_argument("--global-relocalization-min-score-gain", type=float, default=0.35)
    parser.add_argument("--global-relocalization-max-jump", type=float, default=40.0)
    parser.add_argument("--global-relocalization-blend", type=float, default=0.85)
    parser.add_argument("--global-relocalization-cooldown", type=int, default=0)
    parser.add_argument("--global-relocalization-map-check", action="store_true")
    parser.add_argument("--global-relocalization-map-rmse-gate", type=float, default=0.35)
    parser.add_argument("--global-relocalization-map-min-used", type=int, default=25)
    parser.add_argument("--association-model", type=Path, help="JSON model trained by train_landmark_association_model.py.")
    parser.add_argument("--association-model-threshold", type=float, default=0.50)
    parser.add_argument(
        "--association-model-score-weight",
        type=float,
        default=0.0,
        help="How strongly the ML probability changes candidate ranking. Default 0 means reject-only gate.",
    )
    parser.add_argument(
        "--association-model-adaptive-r",
        action="store_true",
        help="Use ML confidence to scale landmark measurement covariance instead of only accepting/rejecting.",
    )
    parser.add_argument("--association-model-min-confidence", type=float, default=0.25)
    parser.add_argument("--association-model-max-r-scale", type=float, default=4.0)
    parser.add_argument("--recovery", action="store_true", help="Use geometric landmark recovery for large odometry drift.")
    parser.add_argument("--recovery-max-jump", type=float, default=8.0)
    parser.add_argument("--recovery-blend", type=float, default=0.45)
    parser.add_argument("--icp", action="store_true", help="Use local scan-to-map ICP as an auxiliary EKF pose update.")
    parser.add_argument("--icp-every", type=int, default=5)
    parser.add_argument("--icp-iterations", type=int, default=8)
    parser.add_argument("--icp-stride", type=int, default=2)
    parser.add_argument("--icp-max-points", type=int, default=120)
    parser.add_argument("--icp-max-corr-distance", type=float, default=0.45)
    parser.add_argument("--icp-rmse-gate", type=float, default=0.22)
    parser.add_argument("--icp-std-xy", type=float, default=0.25)
    parser.add_argument("--icp-std-theta", type=float, default=0.12)
    parser.add_argument("--icp-model", type=Path, help="Gradient-boosted ICP confidence model JSON.")
    parser.add_argument("--icp-model-threshold", type=float, default=0.50)
    parser.add_argument("--icp-model-adaptive-r", action="store_true")
    parser.add_argument("--icp-model-min-confidence", type=float, default=0.20)
    parser.add_argument("--icp-model-max-r-scale", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    landmarks = read_circle_landmarks(args.landmarks)
    corner_landmarks = read_corner_landmarks(args.landmarks) if args.corners else []
    line_landmarks = read_line_landmarks(args.landmarks) if args.lines else []
    association_model = (
        load_association_model(
            args.association_model,
            args.association_model_threshold,
            args.association_model_score_weight,
        )
        if args.association_model
        else None
    )
    odom = read_poses(args.odom)
    true_path = read_poses(args.true)
    lidar = np.load(args.lidar)
    scan_times = lidar["scan_times"]
    ranges = lidar["ranges"]
    angles = lidar["angles"]
    range_max = float(lidar["range_max"])
    distance_map = DistanceMap(args.map_image, args.map_yaml) if args.icp or args.global_relocalization_map_check else None
    icp_model = load_icp_confidence_model(args.icp_model, args.icp_model_threshold) if args.icp_model else None

    state = np.asarray([odom[0].x, odom[0].y, odom[0].theta], dtype=np.float64)
    covariance = np.diag([0.05, 0.05, 0.03])
    estimates = [Pose(odom[0].t, float(state[0]), float(state[1]), float(state[2]))]
    scan_index = 0
    accepted_updates = 0
    accepted_corner_updates = 0
    recovery_updates = 0
    global_relocalization_updates = 0
    last_global_relocalization_scan = -10**9
    icp_updates = 0
    detected_observations = 0
    detected_corner_observations = 0
    detected_line_observations = 0
    accepted_line_updates = 0

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
            observations = detect_circle_observations(
                ranges[scan_index],
                angles,
                range_max,
                args.segment_gap,
                args.min_segment_points,
                args.radius_min,
                args.radius_max,
                args.max_fit_error,
            )
            detected_observations += len(observations)
            if (
                args.global_relocalization
                and len(observations) >= 2
                and scan_index - last_global_relocalization_scan >= args.global_relocalization_cooldown
            ):
                relocalization_points = None
                if args.global_relocalization_map_check:
                    relocalization_points = scan_points_for_icp(
                        ranges[scan_index],
                        angles,
                        range_max,
                        stride=args.icp_stride,
                        max_points=args.icp_max_points,
                    )
                state, covariance, global_relocalized = maybe_global_landmark_relocalize(
                    state,
                    covariance,
                    observations,
                    landmarks,
                    args.global_relocalization_gate_range,
                    args.global_relocalization_gate_bearing,
                    args.global_relocalization_min_matches,
                    args.global_relocalization_min_match_gain,
                    args.global_relocalization_min_score_gain,
                    args.global_relocalization_max_jump,
                    args.global_relocalization_blend,
                    distance_map=distance_map,
                    points_robot=relocalization_points,
                    map_rmse_gate=args.global_relocalization_map_rmse_gate,
                    map_min_used=args.global_relocalization_map_min_used,
                )
                if global_relocalized:
                    global_relocalization_updates += 1
                    last_global_relocalization_scan = scan_index
            if args.recovery and len(observations) >= 2:
                state, covariance, recovered = maybe_global_recover(
                    state,
                    covariance,
                    observations,
                    landmarks,
                    covariance_trace_trigger=2.0,
                    max_position_jump=args.recovery_max_jump,
                    blend=args.recovery_blend,
                )
                if recovered:
                    recovery_updates += 1
            if args.icp and scan_index % args.icp_every == 0 and distance_map is not None:
                points_robot = scan_points_for_icp(
                    ranges[scan_index],
                    angles,
                    range_max,
                    stride=args.icp_stride,
                    max_points=args.icp_max_points,
                )
                icp_pose, icp_rmse, icp_used = icp_scan_to_map(
                    state,
                    points_robot,
                    distance_map,
                    iterations=args.icp_iterations,
                    max_corr_distance=args.icp_max_corr_distance,
                    damping=1e-3,
                )
                icp_jump = math.hypot(icp_pose[0] - state[0], icp_pose[1] - state[1])
                icp_confidence = None
                if icp_model is not None:
                    icp_features = icp_confidence_features(
                        state,
                        icp_pose,
                        icp_rmse,
                        icp_used,
                        len(points_robot),
                        ranges[scan_index],
                    )
                    icp_confidence = predict_icp_confidence(icp_model, icp_features)
                icp_ok = icp_used >= 25 and icp_rmse <= args.icp_rmse_gate and icp_jump <= 1.0
                if icp_model is not None:
                    icp_ok = icp_ok and icp_confidence is not None and icp_confidence >= icp_model.threshold
                if icp_ok:
                    icp_std_xy = args.icp_std_xy
                    icp_std_theta = args.icp_std_theta
                    if args.icp_model_adaptive_r and icp_confidence is not None:
                        confidence = max(float(icp_confidence), args.icp_model_min_confidence)
                        scale = min(1.0 / confidence, args.icp_model_max_r_scale)
                        icp_std_xy *= scale
                        icp_std_theta *= scale
                    state, covariance = update_with_pose_measurement(
                        state,
                        covariance,
                        icp_pose,
                        std_xy=icp_std_xy,
                        std_theta=icp_std_theta,
                    )
                    icp_updates += 1
            used_landmarks: set[str] = set()
            if args.global_association:
                associated_observations = globally_associate_observations(
                    state,
                    observations,
                    landmarks,
                    args.gate_range,
                    args.gate_bearing,
                    args.association_ratio,
                    association_model,
                )
            else:
                associated_observations = []
                for observation in observations[:8]:
                    associated = associate_observation(
                        state,
                        observation,
                        landmarks,
                        used_landmarks,
                        args.gate_range,
                        args.gate_bearing,
                        args.association_ratio,
                        association_model,
                    )
                    if associated is None:
                        continue
                    landmark, probability = associated
                    used_landmarks.add(landmark.id)
                    associated_observations.append((observation, landmark, probability))
            for observation, landmark, probability in associated_observations:
                range_std = args.range_std
                bearing_std = args.bearing_std
                if args.association_model_adaptive_r and probability is not None:
                    confidence = max(float(probability), args.association_model_min_confidence)
                    scale = min(1.0 / confidence, args.association_model_max_r_scale)
                    range_std *= scale
                    bearing_std *= scale
                updated_state, updated_covariance, maha = update_with_landmark(
                    state,
                    covariance,
                    observation,
                    landmark,
                    range_std,
                    bearing_std,
                )
                if maha <= args.mahalanobis_gate:
                    state, covariance = updated_state, updated_covariance
                    accepted_updates += 1
            if args.corners and corner_landmarks:
                corner_observations = detect_corner_observations(
                    ranges[scan_index],
                    angles,
                    range_max,
                    args.segment_gap,
                    args.corner_min_line_points,
                    args.corner_min_line_length,
                    args.corner_max_line_error,
                    args.corner_angle_tolerance,
                    args.corner_max_range,
                )
                detected_corner_observations += len(corner_observations)
                used_corners: set[str] = set()
                for observation in corner_observations[:6]:
                    landmark = associate_point_observation(
                        state,
                        observation,
                        corner_landmarks,
                        used_corners,
                        args.corner_gate_range,
                        args.corner_gate_bearing,
                        args.association_ratio,
                    )
                    if landmark is None:
                        continue
                    updated_state, updated_covariance, maha = update_with_point_landmark(
                        state,
                        covariance,
                        observation,
                        landmark,
                        args.corner_range_std,
                        args.corner_bearing_std,
                    )
                    if maha <= args.mahalanobis_gate:
                        state, covariance = updated_state, updated_covariance
                        used_corners.add(landmark.id)
                        accepted_corner_updates += 1
            if args.lines and line_landmarks:
                line_observations = detect_line_observations(
                    ranges[scan_index],
                    angles,
                    range_max,
                    args.segment_gap,
                    args.line_min_points,
                    args.line_min_length,
                    args.line_max_error,
                )
                detected_line_observations += len(line_observations)
                used_lines: set[str] = set()
                for observation in line_observations[:6]:
                    landmark = associate_line_observation(
                        state,
                        observation,
                        line_landmarks,
                        used_lines,
                        args.line_gate_rho,
                        args.line_gate_alpha,
                    )
                    if landmark is None:
                        continue
                    updated_state, updated_covariance, maha = update_with_line_landmark(
                        state,
                        covariance,
                        observation,
                        landmark,
                        args.line_rho_std,
                        args.line_alpha_std,
                    )
                    if maha <= args.mahalanobis_gate:
                        state, covariance = updated_state, updated_covariance
                        used_lines.add(landmark.id)
                        accepted_line_updates += 1
            scan_index += 1
        estimates.append(Pose(odom[odom_index].t, float(state[0]), float(state[1]), float(state[2])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimate_csv = args.output_dir / "ekf_estimate.csv"
    overlay_png = args.output_dir / "ekf_overlay.png"
    summary_json = args.output_dir / "summary.json"
    save_estimate(estimate_csv, estimates)
    draw_overlay(args.map_image, args.map_yaml, true_path, odom, estimates, overlay_png)
    odom_final, odom_mean = path_errors(true_path, odom)
    ekf_final, ekf_mean = path_errors(true_path, estimates)
    summary = {
        "landmarks": len(landmarks),
        "corner_landmarks": len(corner_landmarks),
        "line_landmarks": len(line_landmarks),
        "detected_observations": detected_observations,
        "detected_corner_observations": detected_corner_observations,
        "detected_line_observations": detected_line_observations,
        "accepted_updates": accepted_updates,
        "accepted_corner_updates": accepted_corner_updates,
        "accepted_line_updates": accepted_line_updates,
        "association_model": str(args.association_model) if args.association_model else None,
        "association_model_threshold": args.association_model_threshold if args.association_model else None,
        "association_model_score_weight": args.association_model_score_weight if args.association_model else None,
        "association_model_adaptive_r": args.association_model_adaptive_r if args.association_model else None,
        "recovery_updates": recovery_updates,
        "global_association": args.global_association,
        "global_relocalization_updates": global_relocalization_updates,
        "icp_updates": icp_updates,
        "icp_model": str(args.icp_model) if args.icp_model else None,
        "icp_model_threshold": args.icp_model_threshold if args.icp_model else None,
        "icp_model_adaptive_r": args.icp_model_adaptive_r if args.icp_model else None,
        "odom_final_error_m": round(odom_final, 3),
        "odom_mean_error_m": round(odom_mean, 3),
        "ekf_final_error_m": round(ekf_final, 3),
        "ekf_mean_error_m": round(ekf_mean, 3),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {estimate_csv}")
    print(f"wrote: {overlay_png}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
