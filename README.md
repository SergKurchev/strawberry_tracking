# Strawberry Tracker & Pose Estimator

A system for strawberry tracking and camera pose estimation based on deep learning (YOLOv8) and feature extraction (DISK/LightGlue). Designed for processing synthetic data from Isaac Sim and preparing measurements for the GTSAM factor graph.

## Core Features and Architecture (Dual-Pipeline)

The project uses a hybrid approach to resolve the conflict between high-quality tracking of small objects and reliable visual odometry:

1. **Phase 1: Focused Tracking (Local Features)**
   - YOLOv8 neural network detects berries.
   - The frame background is masked (filled with black), forcing the DISK extractor to generate keypoints *exclusively* on the strawberry texture.
   - **Result:** Ideal performance of the Hungarian algorithm (Re-ID) without background objects "stealing" features.
2. **Phase 2: Global Localization (Global Features)**
   - Simultaneously, DISK scans the entire frame (berries, leaves, table, walls).
   - **Result:** Rich scene geometry eliminates the "Planar Degeneracy" problem, allowing the MAGSAC algorithm to accurately calculate 3D rotation and camera translation.
3. **Phase 3: Long-term Memory (Re-ID Gallery)**
   - If a berry is occluded by a leaf and disappears, its features are stored in the `gallery` for a specified number of frames.
   - Upon reappearance, the tracker matches it with the gallery and restores the original ID.
4. **Resilience to Pauses (Small Motion Fallback)**
   - If the average point displacement between frames is less than `min_motion_thresh` (e.g., 1.0 pixel), the system detects a camera stop and returns an identity rotation matrix, protecting the algorithm from mathematical instability.

---

## Mathematics and Camera Parameters

For accurate localization, the algorithm uses the laws of epipolar geometry.

### 1. Camera Intrinsic Matrix ($K$)
Physical camera parameters from Isaac Sim (millimeters) are converted to pixel values for OpenCV:

$$f_{px} = f_{mm} \cdot \frac{W_{px}}{A_{horiz}}$$

For a resolution of $1920 \times 1080$, a focal length of $18.0$ mm, and a horizontal aperture of $20.955$ mm, the intrinsic matrix $K$ looks like this:

$$K = \begin{bmatrix} 1649.25 & 0 & 960.0 \\ 0 & 1649.27 & 540.0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 2. Odometry Calculation (Essential Matrix)
Based on global keypoint matches, an Essential Matrix $E$ is calculated, satisfying the condition:
$$x_2^T E x_1 = 0$$
where $x_1, x_2$ are normalized coordinates of matched points on adjacent frames. 
To remove noise, a robust optimization algorithm is applied (default is `MAGSAC`). The matrix $E$ is then decomposed into the final:
- **$R$**: Rotation matrix ($3 \times 3$).
- **$t$**: Translation vector ($3 \times 1$, normalized).

---

## Installation and Launch

The project uses the modern `uv` package manager.

```bash
uv sync
uv run python test_on_coco.py
```

---

## Usage (Tracker API)

Example of integrating the `StrawberryTracker` into your pipeline with Dual-Pipeline support:

```python
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from strawberry_tracker import StrawberryTracker

# 1. Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Initialize models
yolo = YOLO(config['yolo_weights_path'])
tracker = StrawberryTracker(yolo, device='cuda', config=config)

# 3. Camera Intrinsics (1920x1080, Isaac Sim)
K = np.array([
    [1649.25, 0, 960.0],
    [0, 1649.27, 540.0],
    [0, 0, 1]
], dtype=np.float32)

# 4. Process frame (Input must be in RGB format!)
img_rgb = ... # Load frame
R, t, ids, boxes, matches1, matches2, inliers1, inliers2, debug = tracker.forward(img_rgb, K)
```

**`forward` Output Format:**
- `R`, `t`: Relative camera rotation and translation.
- `ids`: Array of global berry IDs (e.g., `[0, 4, 12]`).
- `boxes`: Bounding boxes `[x1, y1, x2, y2]`.
- `matches1`, `matches2`: All found keypoint matches (raw data).
- `inliers1`, `inliers2`: Scene point coordinates that passed MAGSAC filtering (ideal for odometry visualization).

---

## Data Export (GTSAM Integration)

The `config.yaml` file includes an `export_data: true` mode. When enabled, the `test_on_coco.py` script automatically aggregates all scene information and saves it to the `exported_data/` directory:

1. **`poses.csv`**: Rotation matrices ($R$) and translation vectors ($t$) line-by-line (frame, r11-r33, tx, ty, tz).
2. **`tracking.json`**: Berry IDs and their box coordinates (for MOTA/IDF1 metrics and point factors in GTSAM).
3. **`inliers.json`**: Precise coordinates of point pairs that passed RANSAC (ready-to-use landmarks for Loop Closures).

### Example Data Loading and Analysis (Python / Pandas)

```python
import json
import pandas as pd
import numpy as np

# 1. Loading camera trajectory (Odometry)
poses_df = pd.read_csv('exported_data/poses.csv')
print("First 5 frames of odometry:")
print(poses_df.head())

# Reconstruction of the Rotation Matrix R (3x3) for frame 0
row = poses_df.iloc[0]
R = np.array([
    [row['r11'], row['r12'], row['r13']],
    [row['r21'], row['r22'], row['r23']],
    [row['r31'], row['r32'], row['r33']]
])

# 2. Loading berry tracking
with open('exported_data/tracking.json', 'r') as f:
    tracking = json.load(f)

# Practical example: collecting the trajectory of a single berry (ID 0)
berry_0_trajectory = []
for frame_name, detections in tracking.items():
    for det in detections:
        if det['id'] == 0:
            # Bounding Box center calculation
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            berry_0_trajectory.append((frame_name, center_x, center_y))
            
print(f"\nBerry ID 0 successfully tracked in {len(berry_0_trajectory)} frames.")
```
