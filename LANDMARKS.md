# Generic Simulation Landmarks

Files:

- `output/generic_sim_landmarks.json`: landmark definitions in metric map coordinates.
- `output/generic_sim_landmarks.png`: visual overlay.

Legend:

- Green crosses: corner landmarks.
- Red segments: line landmarks.
- Blue circles: circular/pillar landmarks.

## Recommended Detection Order

For a first EKF implementation, start with circular landmarks (`type: circle`):

1. Segment consecutive LiDAR beams where range changes smoothly.
2. Convert each segment to local Cartesian points.
3. Fit a circle or detect convex clusters with radius near the expected landmark radius.
4. Use the observed center bearing/range as the measurement.

Then add line landmarks (`type: line`):

1. Segment scan points by range discontinuities.
2. Fit lines with least squares or RANSAC.
3. Compare each observed line with map lines using angle and distance.

Corners (`type: corner`) are useful but should come later:

1. Detect two line segments that intersect near 90 degrees.
2. Use their intersection as the observed landmark.

## Practical EKF Measurements

Simple point/circle landmark measurement:

```text
z = [range, bearing]
range = sqrt((lx - rx)^2 + (ly - ry)^2)
bearing = atan2(ly - ry, lx - rx) - robot_yaw
```

Line landmark measurement can be represented as:

```text
z = [rho, alpha]
```

where `rho` is perpendicular distance from the robot to the line and `alpha` is the line normal angle in the robot frame.

## Suggested First Subset

Use these first because they are easy to see and well separated:

- `P01`
- `P02`
- `P03`
- `P04`
- `P05`
- `L01`
- `L02`
- `L04`
- `L06`
- `L08`

Avoid using all landmarks immediately. Add them gradually after the data association works.
