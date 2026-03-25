# Strawberry Tracker & Pose Estimator

A visual odometry and long-term tracking system optimized for synthetic Isaac Sim data. The system generates measurements for the GTSAM factor graph.

## Mathematical Architecture (Dual-Pipeline)

The project implements a hybrid feature extraction method to simultaneously solve object Re-ID and SLAM tasks:

### 1. Detection and Segmentation
YOLOv8 is used to generate a set of bounding boxes $B = \{b_1, ..., b_n\}$. To eliminate duplicates, NMS is applied with a threshold $\tau_{iou}$. A local mask is associated with each box for berry texture segmentation.

### 2. Feature Extraction (DISK)
Separate keypoint extraction is applied for two different tasks:
* **Feature-Track (Local):** Extraction of points $p_{local}$ inside the masks $B$. Mathematically, this is filtering the input image $I$ through a binary object mask $M$: $I_{masked} = I \odot M$. This ensures that descriptors belong only to berries.
* **SLAM-Track (Global):** Extraction of points $p_{global}$ from the entire frame $I$ to capture scene geometry (walls, floor, leaves), which is critical for odometry stability when there are few berries in the frame.

### 3. Matching and Re-ID (Hungarian Algorithm)
To associate berries between frames $t-1$ and $t$, a weight matrix $W$ is constructed, where $w_{ij}$ is the number of LightGlue matches between boxes $i$ and $j$. The linear assignment problem is solved:
$$\max \sum_{i,j} w_{ij} x_{ij}, \quad \text{s.t.} \sum_i x_{ij} = 1, \sum_j x_{ij} = 1$$
The result is the preservation of a stable `global_id` for each object.

### 4. Pose Recovery (Essential Matrix & MAGSAC)
Based on global inliers $x_1, x_2$, the Essential Matrix $E$ is calculated by minimizing the reprojection error:
$$x_2^T E x_1 = 0, \quad E = [t]_{\times} R$$
The **MAGSAC** algorithm is used to filter outliers. If the average point displacement $\Delta p < \tau_{motion}$, the motion is recognized as zero ($R=I, t=0$) to protect against drift in static scenes.

---

## API Usage Instructions

### Main Loop in Code
The `forward` method encapsulates all frame processing logic.

```python
from strawberry_tracker import StrawberryTracker
from ultralytics import YOLO

# 1. Initialization
tracker = StrawberryTracker(YOLO('best.pt'), device='cuda', config=config)

# 2. K Matrix (Intrinsics from Isaac Sim)
K = np.array([[1649.25, 0, 960], [0, 1649.25, 540], [0, 0, 1]])

# 3. Processing (in a loop)
# img_rgb: [H, W, 3] array
R, t, ids, boxes, m1, m2, i1, i2, debug = tracker.forward(img_rgb, K)
```

**Return Values:**
* `R`, `t`: Relative camera transformation (Rotation 3x3, Translation 3x1).
* `ids`: List of global IDs for berries in the current frame.
* `boxes`: Box coordinates `[x1, y1, x2, y2]`.
* `i1, i2`: Inlier point coordinates (geometrically correct matches).

---

## Running Tests and Data Export

To run mass processing of the COCO dataset and generate files for GTSAM, use the `test_on_coco.py` script.

### 1. Preparation
Ensure that `config.yaml` is configured with:
- `test_frame_range: [start, end]` — the range of frames.
- `export_data: true` — the results saving flag.

### 2. Launch
```bash
uv run python test_on_coco.py
```

### 3. Export Results (`exported_data/`)
After the run, three files will appear in the folder:
* `poses.csv`: Table of relative movements (odometry).
* `tracking.json`: Movement history for each berry (ID + BBoxes).
* `inliers.json`: Pairs of 2D points for each frame (Landmarks).

---

## Data Re-import

Example of how to load exported data back for analysis or passing to GTSAM:

```python
import pandas as pd
import json
import numpy as np

# Load odometry (CSV)
df = pd.read_csv('exported_data/poses.csv')
for _, row in df.iterrows():
    # Extract translation vector
    translation = np.array([row['tx'], row['ty'], row['tz']])
    # Extract rotation matrix
    rotation = row[['r11','r12','r13','r21','r22','r23','r31','r32','r33']].values.reshape(3,3)

# Load tracking (JSON)
with open('exported_data/tracking.json', 'r') as f:
    data = json.load(f)
    # Get boxes for frame 065
    frame_boxes = data['frame_065']
    for obj in frame_boxes:
        print(f"ID: {obj['id']}, BBox: {obj['bbox']}")
```

---

## Environment Requirements
The project is managed via `uv`. Main dependencies: `torch`, `torchvision`, `kornia`, `ultralytics`, `opencv-python`, `pyyaml`.

```bash
uv sync
```
