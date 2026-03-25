# Strawberry Tracker & Pose Estimator

A computer vision system for strawberry tracking and camera pose estimation using deep learning (YOLOv8) and global features (DISK/LightGlue).

## Core Features

- **Detection & Segmentation**: Leveraging YOLOv8 for berry detection.
- **Global Localization**: Camera rotation and translation recovery using background landmarks.
- **Motion Robustness**: Built-in "Small Motion Fallback" to prevent pose errors during robot pauses.
- **Descriptor Gallery**: Persistent ID tracking for berries across occlusions and leaf overlaps.

## Project Structure

- `strawberry_tracker.py`: Core logic module containing the `StrawberryTracker` class.
- `config.yaml`: Centralized configuration (thresholds, paths, camera intrinsics).
- `test_on_coco.py`: Validation script for processing image sequences.
- `run_tracker.py`: Entry point for real-time integration.

## Quick Start
Managed via [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run python test_on_coco.py
```

## API Usage

### Initialization

```python
from strawberry_tracker import StrawberryTracker
from ultralytics import YOLO
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

yolo = YOLO(config['yolo_weights_path'])
tracker = StrawberryTracker(yolo, device='cuda', config=config)
```

### The `forward()` Method

The primary method for processing frames. It returns all necessary data for navigation and state estimation.

```python
# K is the 3x3 camera intrinsic matrix
R, t, ids, boxes, matches1, matches2, inliers1, inliers2, debug = tracker.forward(img_rgb, K)
```

**Output Breakdown:**

- `R`, `t`: Rotation matrix and translation vector (relative to the previous frame).
- `ids`: List of global identities for each detected berry.
- `boxes`: Bounding boxes `[x1, y1, x2, y2]`.
- `inliers1`, `inliers2`: Geometrically verified keypoints (for pose visualization).
- `debug`: Metrics (inlier count, translation magnitude, etc.).

## Configuration (config.yaml)

- `yolo_conf`: Confidence threshold (set to `0.5` to avoid background false positives).
- `min_motion_thresh`: Displacement threshold (pixels) below which the camera is assumed static.
- `ransac_threshold`: Projection error tolerance for RANSAC/MAGSAC.
- `pose_solver`: Selection of geometry solver (`RANSAC` or `MAGSAC`).

## Important Notes

1. **Camera Calibration**: Ensure `focal_length` and `center_x/y` in `config.yaml` match your hardware. Inaccurate intrinsics will corrupt motion estimation.
2. **Inlier Color Coding**: In the debug visualizations, **Cyan** represents raw matches, while **Red** represents verified inliers.
3. **Static Detection**: If the title shows `STATIC`, the displacement was below `min_motion_thresh`. The system skips the solver to avoid degenerate matrix calculations and assumes Identity motion.

