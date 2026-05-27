#!/usr/bin/env python3
"""Denoise a ROS-style occupancy map image while preserving room boundaries."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    offsets = disk_offsets(radius)
    for y, x in zip(ys, xs, strict=False):
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < height and 0 <= xx < width:
                result[yy, xx] = True
    return result


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    offsets = disk_offsets(radius)
    for dy, dx in offsets:
        result &= padded[
            radius + dy : radius + dy + height,
            radius + dx : radius + dx + width,
        ]
    return result


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def open_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return dilate(erode(mask, radius), radius)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for sy, sx in zip(*np.nonzero(mask), strict=False):
        if visited[sy, sx]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        points: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for dy, dx in neighbors:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    queue.append((yy, xx))
        components.append(np.asarray(points, dtype=np.int32))
    return components


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for component in connected_components(mask):
        if len(component) >= min_pixels:
            result[component[:, 0], component[:, 1]] = True
    return result


def keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    if max_components <= 0:
        return mask
    components = sorted(connected_components(mask), key=len, reverse=True)
    result = np.zeros_like(mask, dtype=bool)
    for component in components[:max_components]:
        result[component[:, 0], component[:, 1]] = True
    return result


def copy_yaml(input_yaml: Path, output_yaml: Path, output_image: Path) -> None:
    text = input_yaml.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.startswith("image:"):
            lines.append(f"image: {output_image.name}")
        else:
            lines.append(line)
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean noisy occupancy-map obstacle pixels.")
    parser.add_argument("map_image", type=Path)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/clean_map"))
    parser.add_argument("--obstacle-threshold", type=int, default=80)
    parser.add_argument("--min-component-pixels", type=int, default=10)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--open-radius", type=int, default=1)
    parser.add_argument("--thicken-radius", type=int, default=1)
    parser.add_argument("--largest-components", type=int, default=0)
    parser.add_argument("--unknown-value", type=int, default=205)
    parser.add_argument("--free-value", type=int, default=254)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.map_image).convert("L")
    array = np.asarray(image, dtype=np.uint8)

    occupied = array < args.obstacle_threshold
    cleaned = remove_small_components(occupied, args.min_component_pixels)
    cleaned = close(cleaned, args.close_radius)
    cleaned = open_mask(cleaned, args.open_radius)
    cleaned = remove_small_components(cleaned, args.min_component_pixels)
    cleaned = keep_largest_components(cleaned, args.largest_components)
    cleaned = dilate(cleaned, args.thicken_radius)

    output_array = np.full(array.shape, args.unknown_value, dtype=np.uint8)
    output_array[array > 240] = args.free_value
    output_array[cleaned] = 0

    output_png = args.output.with_suffix(".png")
    output_pgm = args.output.with_suffix(".pgm")
    output_yaml = args.output.with_suffix(".yaml")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_array, mode="L").save(output_png)
    Image.fromarray(output_array, mode="L").save(output_pgm)
    copy_yaml(args.map_yaml, output_yaml, output_pgm)

    print(f"input occupied pixels: {int(occupied.sum())}")
    print(f"clean occupied pixels: {int(cleaned.sum())}")
    print(f"wrote: {output_png}")
    print(f"wrote: {output_pgm}")
    print(f"wrote: {output_yaml}")


if __name__ == "__main__":
    main()
