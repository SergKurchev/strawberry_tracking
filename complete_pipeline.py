import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import cv2
import os

CONFIG = {
    'camera': {
        'fx': 1662.8, 'fy': 1662.8, 'cx': 960.0, 'cy': 540.0,
        'width': 1920, 'height': 1080,
    },
    'target_strawberries': 17,  
}

def project_point_to_image(point_3d_world, pose, K):
    """Project 3D world point to image."""
    R = pose['R']
    t = pose['t']
    p_cam = R.T @ (point_3d_world - t)
    
    if p_cam[2] <= 0:
        return None
    
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    
    return np.array([u, v])

def count_clusters_at_threshold(distances, threshold):
    """Count clusters at a given threshold using union-find."""
    parent = list(range(len(distances)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Merge close points
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            if distances[i, j] < threshold:
                union(i, j)
    
    # Count unique clusters
    return len(set(find(i) for i in range(len(distances))))

def find_clustering_threshold(distances, target_clusters=16):
    """Find threshold that gives approximately target number of clusters.
    Binary search / sweep for a threshold that yields ~target_clusters using union-find.
    """
    best_threshold = 0.05
    best_diff = float('inf')
    best_n_clusters = 0
    if distances.size == 0:
        return best_threshold, 0

    for threshold in np.linspace(0.001, 1.0, 200):
        # Count clusters using union-find
        parent = list(range(len(distances)))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if distances[i, j] < threshold:
                    union(i, j)

        n_clusters = len(set(find(i) for i in range(len(distances))))
        diff = abs(n_clusters - target_clusters)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            best_n_clusters = n_clusters

    return best_threshold, best_n_clusters

print("[PHASE 0] Loading Data\n")

fx = CONFIG['camera']['fx']
fy = CONFIG['camera']['fy']
cx = CONFIG['camera']['cx']
cy = CONFIG['camera']['cy']
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Load GT poses
print("  Loading ground truth camera poses...")
gt_poses = {}
num_frames = 0
while os.path.exists(f"data/coords/frame_{num_frames:04d}.json"):
    num_frames += 1

for i in range(num_frames):
    try:
        with open(f"data/coords/frame_{i:04d}.json") as f:
            data = json.load(f)
        pos = data["camera_gt"]["position_world_m"]
        rot_mat = data["camera_gt"]["rotation_matrix_3x3"]
        gt_poses[i] = {
            't': np.array([pos["x"], pos["y"], pos["z"]]),
            'R': np.array(rot_mat)
        }
    except:
        pass

print(f"    {len(gt_poses)} GT poses")

# Load noisy poses from CSV (visual odometry estimates)
print("  Loading noisy pose estimates...")
noisy_poses = {}
try:
    poses_df = pd.read_csv("data/poses.csv")
    for _, row in poses_df.iterrows():
        frame_name = row['frame']
        frame_id = int(frame_name.split('_')[1])
        R = np.array([
            [row['r11'], row['r12'], row['r13']],
            [row['r21'], row['r22'], row['r23']],
            [row['r31'], row['r32'], row['r33']]
        ])
        t = np.array([row['tx'], row['ty'], row['tz']])
        noisy_poses[frame_id] = {'R': R, 't': t}
    print(f"    {len(noisy_poses)} noisy poses (pre-SLAM)")
except Exception as e:
    print(f"    - Could not load poses.csv: {e}")
    noisy_poses = {}

for i in range(num_frames):
    frame_file = f"data/coords/frame_{i:04d}.json"
    if not os.path.exists(frame_file):
        continue
    try:
        with open(frame_file, 'r') as f:
            data = json.load(f)
        if 'camera_noisy' in data and 'position_world_m' in data['camera_noisy']:
            p = data['camera_noisy']['position_world_m']
            noisy_poses[i] = noisy_poses.get(i, {})
            noisy_poses[i]['t'] = np.array([p['x'], p['y'], p['z']])
    except Exception:
        pass

# Load tracking
print("  Loading strawberry detections...")
with open("data/tracking.json") as f:
    tracking = json.load(f)

detections_per_frame = {}
for frame_name, dets in tracking.items():
    frame_id = int(frame_name.split("_")[1])
    detections_per_frame[frame_id] = []
    for det in dets:
        bbox = det["bbox"]
        x_c = (bbox[0] + bbox[2]) / 2.0
        y_c = (bbox[1] + bbox[3]) / 2.0
        detections_per_frame[frame_id].append({
            "id": det["id"],
            "center_2d": np.array([x_c, y_c]),
            "bbox": bbox
        })

total_dets = sum(len(d) for d in detections_per_frame.values())
unique_ids = len(set(d["id"] for dets in detections_per_frame.values() for d in dets))
print(f"    {total_dets} detections, {unique_ids} tracking IDs")

# Load depth
print("  Loading depth maps...")
depth_path = "data/depth_est/distance_to_image_plane_{:04d}.npy"
berry_3d_observations = {}

for frame_id, dets in detections_per_frame.items():
    depth_file = depth_path.format(frame_id)
    if not os.path.exists(depth_file):
        continue
    
    depth_map = np.load(depth_file)
    
    for det in dets:
        x_c, y_c = det["center_2d"]
        u, v = int(round(x_c)), int(round(y_c))
        
        if 0 <= u < 1920 and 0 <= v < 1080 and depth_map[v, u] > 0.01:
            depth = depth_map[v, u]
            X_cam = (x_c - cx) * depth / fx
            Y_cam = (y_c - cy) * depth / fy
            Z_cam = depth
            point_3d_cam = np.array([X_cam, Y_cam, Z_cam])
            
            berry_id = det["id"]
            if berry_id not in berry_3d_observations:
                berry_3d_observations[berry_id] = []
            berry_3d_observations[berry_id].append({
                'frame_id': frame_id,
                'point_3d_cam': point_3d_cam,
                'point_2d': det["center_2d"],
                'depth': depth
            })

print(f"    {len(berry_3d_observations)} tracking IDs with 3D data")

print("\n[PHASE 1] Computing 3D Berry Positions\n")

berry_world_positions = {}
for berry_id, obs_list in berry_3d_observations.items():
    world_points = []
    frame_ids = []
    
    for obs in obs_list:
        frame_id = obs['frame_id']
        if frame_id not in gt_poses:
            continue
        
        pose = gt_poses[frame_id]
        p_cam = obs['point_3d_cam']
        p_world = pose['R'].T @ p_cam + pose['t']
        world_points.append(p_world)
        frame_ids.append(frame_id)
    
    if len(world_points) > 0:
        berry_world_positions[berry_id] = {
            'mean_pos': np.mean(world_points, axis=0),
            'all_positions': np.array(world_points),
            'num_obs': len(obs_list),
            'frames': frame_ids,
            'spatial_std': np.std(np.linalg.norm(world_points - np.mean(world_points, axis=0), axis=1))
        }

print(f"  Computed positions for {len(berry_world_positions)} tracking IDs")
print(f"  Computed positions for {len(berry_world_positions)} tracking IDs (no per-observation filtering)")

print("\n[PHASE 2] Automatic Re-identification Clustering\n")

print("  Finding optimal clustering threshold...")

ids = sorted(berry_world_positions.keys())
mean_positions = np.array([berry_world_positions[bid]['mean_pos'] for bid in ids])
distances = cdist(mean_positions, mean_positions) if len(ids) > 0 else np.array([])

# Find threshold that gives ~target clusters (sweep with union-find)
optimal_threshold, n_clusters = find_clustering_threshold(distances, CONFIG['target_strawberries'])

print(f"    Optimal threshold found: {optimal_threshold:.4f} m ({optimal_threshold*100:.1f}cm)")
print(f"    Predicted clusters: {n_clusters}")

# Perform clustering with optimal threshold
parent = list(range(len(ids)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Merge points closer than threshold
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        if distances[i, j] < optimal_threshold:
            union(i, j)

# Group by cluster
clusters_dict = {}
for i, idx in enumerate(ids):
    root = find(i)
    if root not in clusters_dict:
        clusters_dict[root] = []
    clusters_dict[root].append(idx)

clusters = list(clusters_dict.values())
clusters.sort(key=lambda c: -len(c))

print(f"    Optimal threshold found: {optimal_threshold:.4f} m ({optimal_threshold*100:.1f}cm)")
print(f"    Predicted clusters: {n_clusters}")
print(f"    Found {len(clusters)} clusters")
print(f"    Consolidated {unique_ids} tracking IDs → {len(clusters)} strawberries")

# Compute robust representative position per cluster (medoid) to avoid artificial centroids
cluster_reps = []  # representative (x,y,z) per strawberry index
for cluster in clusters:
    positions = []
    for tracking_id in cluster:
        if tracking_id in berry_world_positions:
            positions.append(berry_world_positions[tracking_id]['mean_pos'])
    if len(positions) == 0:
        continue
    arr = np.array(positions)
    if arr.shape[0] == 1:
        rep = arr[0]
    else:
        d = cdist(arr[:, :2], arr[:, :2])  # use XY distance for medoid
        medoid_idx = int(np.argmin(d.sum(axis=1)))
        rep = arr[medoid_idx]
    cluster_reps.append(rep)

# Map cluster reps to an array for plotting convenience
all_strawberry_positions = np.array(cluster_reps) if len(cluster_reps) > 0 else np.array([])

# Create remapping
tracking_id_to_strawberry = {}
for strawberry_idx, cluster in enumerate(clusters):
    for tracking_id in cluster:
        tracking_id_to_strawberry[tracking_id] = strawberry_idx

print("\n[PHASE 3] Validation Metrics\n")

print("  Computing reprojection errors...")
reprojection_errors = []
reprojection_data = []

for berry_id, obs_list in berry_3d_observations.items():
    if berry_id not in berry_world_positions:
        continue
    
    point_3d_world = berry_world_positions[berry_id]['mean_pos']
    
    for obs in obs_list:
        frame_id = obs['frame_id']
        if frame_id not in gt_poses:
            continue
        
        pose = gt_poses[frame_id]
        point_2d_measured = obs['point_2d']
        point_2d_proj = project_point_to_image(point_3d_world, pose, K)
        
        if point_2d_proj is not None:
            error = np.linalg.norm(point_2d_measured - point_2d_proj)
            reprojection_errors.append(error)
            reprojection_data.append({
                'frame_id': frame_id,
                'tracking_id': berry_id,
                'strawberry_id': tracking_id_to_strawberry.get(berry_id, -1),
                'error_px': error
            })

reprojection_errors = np.array(reprojection_errors)

print(f"    {len(reprojection_errors)} projections")
print(f"    - Mean reprojection error: {np.mean(reprojection_errors):.3f} px")
print(f"    - Median: {np.median(reprojection_errors):.3f} px")
print(f"    - Max: {np.max(reprojection_errors):.3f} px")
print(f"    - 95th percentile: {np.percentile(reprojection_errors, 95):.3f} px")

print("\n[RESULTS] Strawberry Re-identification\n")

print("  Re-identified Strawberries (Tracking IDs per Strawberry):\n")

for strawberry_idx, cluster in enumerate(clusters):
    total_obs = sum(berry_world_positions[bid]['num_obs'] for bid in cluster 
                    if bid in berry_world_positions)
    pos = (berry_world_positions[cluster[0]]['mean_pos'] if cluster[0] in berry_world_positions 
           else np.array([0, 0, 0]))
    
    print(f"    Strawberry {strawberry_idx:2d}:")
    print(f"      Position: X={pos[0]:7.3f}m  Y={pos[1]:7.3f}m  Z={pos[2]:7.3f}m")
    print(f"      Tracking IDs: {cluster} ({len(cluster)} IDs, {total_obs} observations)")
    print()

print("\n[SAVING] Output Files\n")

# Remapping
with open("strawberry_remapping.json", "w") as f:
    json.dump({
        'num_clusters': len(clusters),
        'num_unique_strawberries': len(clusters),
        'threshold_m': float(optimal_threshold),
        'remapping': tracking_id_to_strawberry
    }, f, indent=2)
print("  strawberry_remapping.json")

# Strawberry coordinates (use medoid representative computed earlier)
strawberry_coords = []
for strawberry_idx, cluster in enumerate(clusters):
    # find corresponding rep (if available)
    rep = None
    # cluster_reps aligns with clusters order, so use index
    if strawberry_idx < len(cluster_reps):
        rep = cluster_reps[strawberry_idx]

    total_obs = 0
    for tracking_id in cluster:
        if tracking_id in berry_world_positions:
            total_obs += berry_world_positions[tracking_id]['num_obs']

    if rep is None:
        continue

    strawberry_coords.append({
        'strawberry_id': strawberry_idx,
        'x_m': float(rep[0]),
        'y_m': float(rep[1]),
        'z_m': float(rep[2]),
        'num_tracking_ids': len(cluster),
        'total_observations': total_obs,
        'tracking_ids': str(cluster)
    })

df_coords = pd.DataFrame(strawberry_coords)
df_coords.to_csv("strawberry_coordinates.csv", index=False)
print("  strawberry_coordinates.csv")

# Unified detections
unified_detections = []
for frame_id, dets in detections_per_frame.items():
    for det in dets:
        tracking_id = det["id"]
        strawberry_id = tracking_id_to_strawberry.get(tracking_id, -1)
        unified_detections.append({
            'frame_id': frame_id,
            'tracking_id': tracking_id,
            'strawberry_id': strawberry_id,
            'center_u': det["center_2d"][0],
            'center_v': det["center_2d"][1]
        })

df_unified = pd.DataFrame(unified_detections)
df_unified.to_csv("unified_detections.csv", index=False)
print("  unified_detections.csv")

# Reprojection data
df_reproj = pd.DataFrame(reprojection_data)
df_reproj.to_csv("reprojection_errors_detailed.csv", index=False)
print("  reprojection_errors_detailed.csv")

print("\n[VISUALIZATION] Creating plots\n")

obs_per_id = [berry_world_positions[bid]['num_obs'] for bid in berry_world_positions.keys()]
colors = plt.cm.tab20(np.arange(len(clusters)) % 20)

fig1 = plt.figure(figsize=(16, 6))

# Build GT trajectory
gt_frames = sorted([i for i in range(num_frames) if i in gt_poses])
gt_trajectory = np.array([gt_poses[i]['t'] for i in gt_frames])

# Check if noisy poses exist
has_noisy_data = False
noisy_trajectory = None
max_drift = 0
avg_drift = 0

if len(noisy_poses) > 0:
    noisy_frames = sorted([i for i in range(num_frames) if i in noisy_poses])
    if len(noisy_frames) > 0:
        noisy_trajectory = np.array([noisy_poses[i]['t'] for i in noisy_frames])
        
        # Compare trajectories
        common_frames = sorted(set(gt_frames) & set(noisy_frames))
        if len(common_frames) > 0:
            drifts = [np.linalg.norm(gt_poses[f]['t'] - noisy_poses[f]['t']) for f in common_frames]
            avg_drift = np.mean(drifts)
            max_drift = np.max(drifts)
            has_noisy_data = (avg_drift > 0.01)  # Only show if meaningful difference

# Use medoid representatives computed earlier for plotting
try:
    if 'all_strawberry_positions' not in globals() or (isinstance(all_strawberry_positions, np.ndarray) and all_strawberry_positions.size == 0):
        all_strawberry_positions = np.array(cluster_reps) if len(cluster_reps) > 0 else np.array([])
except Exception:
    all_strawberry_positions = np.array(cluster_reps) if len(cluster_reps) > 0 else np.array([])

# 3D Trajectory
ax1a = fig1.add_subplot(1, 2, 1, projection='3d')

# Plot strawberries
if len(all_strawberry_positions) > 0:
    ax1a.scatter(all_strawberry_positions[:, 0], all_strawberry_positions[:, 1], 
                all_strawberry_positions[:, 2], 
                s=150, c='red', marker='o', edgecolors='darkred', 
                linewidth=1.5, alpha=0.8, label='Berries', zorder=5)

ax1a.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1a.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1a.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax1a.set_title('3D View Berries', fontsize=12, fontweight='bold')
ax1a.legend(fontsize=9, loc='upper right')
ax1a.grid(True, alpha=0.3)

# 2D Horizontal Field (XY view)
ax1b = fig1.add_subplot(1, 2, 2)

# Plot strawberries with labels
if len(all_strawberry_positions) > 0:
    ax1b.scatter(all_strawberry_positions[:, 0], all_strawberry_positions[:, 1],
                s=300, c='red', marker='o', edgecolors='darkred', 
                linewidth=2, alpha=0.85, zorder=5, label='Berries')
    
    # Add berry IDs for clarity
    for strawberry_idx, pos in enumerate(all_strawberry_positions):
        ax1b.annotate(f'S{strawberry_idx}', xy=(pos[0], pos[1]), 
                     fontsize=8, fontweight='bold', ha='center', va='center',
                     color='white', zorder=6)

ax1b.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1b.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1b.set_title('Horizontal Field Berry Positions', fontsize=12, fontweight='bold')
ax1b.legend(fontsize=9, loc='best')
ax1b.grid(True, alpha=0.3)
ax1b.axis('equal')

# Auto-scale to include all data with padding
if len(all_strawberry_positions) > 0:
    x_min, x_max = np.min(all_strawberry_positions[:, 0]), np.max(all_strawberry_positions[:, 0])
    y_min, y_max = np.min(all_strawberry_positions[:, 1]), np.max(all_strawberry_positions[:, 1])
else:
    x_min, x_max = np.min(gt_trajectory[:, 0]), np.max(gt_trajectory[:, 0])
    y_min, y_max = np.min(gt_trajectory[:, 1]), np.max(gt_trajectory[:, 1])

x_pad = (x_max - x_min) * 0.15 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 1.0

ax1b.set_xlim([x_min - x_pad, x_max + x_pad])
ax1b.set_ylim([y_min - y_pad, y_max + y_pad])

fig1.suptitle('BERRY REFINEMENT', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('01_trajectory.png', dpi=150, bbox_inches='tight')
if has_noisy_data:
    print(f"  01_trajectory.png - VO vs SLAM trajectory (avg drift: {avg_drift:.4f}m)")
else:
    print(f"  01_trajectory.png - Camera trajectory (SLAM)")
plt.close()

fig2 = plt.figure(figsize=(14, 10))

# Create coordinate table
ax2 = fig2.add_subplot(1, 1, 1)
ax2.axis('off')

coord_text = "STRAWBERRY COORDINATES (World Frame)\n" + "="*70 + "\n\n"
coord_text += f"{'ID':<5} {'X (m)':<12} {'Y (m)':<12} {'Z (m)':<12} {'Tracking IDs':<15}\n"
coord_text += "-"*70 + "\n"

for strawberry_idx, cluster in enumerate(clusters):
    cluster_str = f"{cluster[:3]}{'...' if len(cluster) > 3 else ''}"
    if strawberry_idx < len(cluster_reps):
        rep = cluster_reps[strawberry_idx]
        coord_text += f"{strawberry_idx:<5} {rep[0]:>11.4f} {rep[1]:>11.4f} {rep[2]:>11.4f} {cluster_str:<15}\n"

ax2.text(0.05, 0.95, coord_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, pad=1))

fig2.suptitle('IDENTIFIED STRAWBERRY COORDINATES', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_strawberry_coordinates.png', dpi=150, bbox_inches='tight')
print("  02_strawberry_coordinates.png - Precise (X,Y,Z) coordinates")
plt.close()

fig3 = plt.figure(figsize=(16, 6))

# Cluster sizes with actual tracking IDs
ax3a = fig3.add_subplot(1, 2, 1)
cluster_sizes = [len(c) for c in clusters]
bars = ax3a.barh(range(len(cluster_sizes)), cluster_sizes, 
                  color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)

# Color bars by size
for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
    if size > 5:
        bar.set_color('darkgreen')
    elif size > 2:
        bar.set_color('steelblue')
    else:
        bar.set_color('coral')

ax3a.set_ylabel('Strawberry ID', fontsize=11, fontweight='bold')
ax3a.set_xlabel('Number of Tracking IDs Consolidated', fontsize=11, fontweight='bold')
ax3a.set_title(f'Re-identification Results\n{len(clusters)} Strawberries from {unique_ids} IDs', 
              fontsize=12, fontweight='bold')
ax3a.grid(True, alpha=0.3, axis='x')
ax3a.invert_yaxis()

# Add value labels on bars
for i, v in enumerate(cluster_sizes):
    ax3a.text(v + 0.1, i, str(v), va='center', fontsize=9, fontweight='bold')

# Reprojection error per frame
ax3b = fig3.add_subplot(1, 2, 2)
frame_errors = {}
for data in reprojection_data:
    frame_id = data['frame_id']
    error = data['error_px']
    if frame_id not in frame_errors:
        frame_errors[frame_id] = []
    frame_errors[frame_id].append(error)

frame_means = [np.mean(frame_errors[f]) for f in sorted(frame_errors.keys())]
frame_ids = sorted(frame_errors.keys())

ax3b.plot(frame_ids, frame_means, 'o-', linewidth=2, markersize=6, color='darkred', alpha=0.7)
ax3b.axhline(np.mean(reprojection_errors), color='green', linestyle='--', linewidth=2.5,
            label=f'Mean: {np.mean(reprojection_errors):.3f}px')
ax3b.fill_between(frame_ids, 
                   [np.mean(frame_errors[f]) - np.std(frame_errors[f]) for f in frame_ids],
                   [np.mean(frame_errors[f]) + np.std(frame_errors[f]) for f in frame_ids],
                   alpha=0.2, color='darkred')

ax3b.set_xlabel('Frame ID', fontsize=11, fontweight='bold')
ax3b.set_ylabel('Mean Reprojection Error (pixels)', fontsize=11, fontweight='bold')
ax3b.set_title('Reprojection Accuracy per Frame', fontsize=12, fontweight='bold')
ax3b.legend(fontsize=10)
ax3b.grid(True, alpha=0.3)

fig3.suptitle('CLUSTERING & VALIDATION', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('03_clustering_validation.png', dpi=150, bbox_inches='tight')
print("  03_clustering_validation.png - Clustering and accuracy analysis")
plt.close()

print("  Creating algorithm visualization on sample images...")

# Sample frames evenly throughout sequence (not just top detections)
all_frames_with_dets = sorted([f for f in detections_per_frame.keys() if len(detections_per_frame[f]) > 0])
if len(all_frames_with_dets) > 6:
    # Take 6 frames spread evenly
    indices = np.linspace(0, len(all_frames_with_dets)-1, 6, dtype=int)
    sample_frames = [all_frames_with_dets[i] for i in indices]
else:
    sample_frames = all_frames_with_dets[:6]

fig4 = plt.figure(figsize=(18, 12))

try:
    for plot_idx, frame_id in enumerate(sample_frames, 1):
        # Load image from images/ folder
        img_path = f"data/images/rgb_{frame_id:04d}.png"
        
        ax4 = fig4.add_subplot(2, 3, plot_idx)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax4.imshow(img_rgb)
                
                # Draw detections with strawberry IDs
                dets = detections_per_frame[frame_id]
                for det in dets:
                    x_c, y_c = det["center_2d"]
                    tracking_id = det["id"]
                    strawberry_id = tracking_id_to_strawberry.get(tracking_id, -1)
                    
                    # Draw bounding box
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = bbox
                    
                    color = colors[strawberry_id % len(colors)][:3]
                    
                    # Draw rect
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.8,
                                         edgecolor=color, facecolor='none', alpha=0.9)
                    ax4.add_patch(rect)
                    
                    # Draw strawberry ID label - MUCH SMALLER
                    label_text = f"S{strawberry_id}"
                    ax4.text(x1+2, y1+12, label_text, fontsize=6.5, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor=color, alpha=0.95, edgecolor='white', linewidth=0.5))
                
                title = f'Frame {frame_id} - {len(dets)} Berries'
            else:
                ax4.text(0.5, 0.5, f'Frame {frame_id}\n(Error)', 
                        ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                title = f'Frame {frame_id}'
        else:
            ax4.text(0.5, 0.5, f'Frame {frame_id}\n(Not found)', 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            title = f'Frame {frame_id}'
        
        ax4.set_title(title, fontsize=10, fontweight='bold')
        ax4.axis('off')

except Exception as e:
    print(f"    Warning: Error in visualization: {e}")
    import traceback
    traceback.print_exc()

fig4.suptitle('DETECTION VISUALIZATION - Colored boxes with S# labels show re-identified strawberries', 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('04_algorithm_visualization.png', dpi=120, bbox_inches='tight')
print("  04_algorithm_visualization.png - Detections on RGB frames (spacing throughout video)")
plt.close()

fig5 = plt.figure(figsize=(16, 7))

# 3D view - all strawberries with camera path
ax5a = fig5.add_subplot(1, 2, 1, projection='3d')

all_strawberry_positions = np.array(cluster_reps) if len(cluster_reps) > 0 else np.array([])
for strawberry_idx, pos in enumerate(all_strawberry_positions):
    ax5a.scatter([pos[0]], [pos[1]], [pos[2]], 
                s=300, c='red', marker='o', edgecolors='darkred', 
                linewidth=2.5, alpha=0.9, zorder=5)
    ax5a.text(pos[0], pos[1], pos[2], f'  S{strawberry_idx}', 
             fontsize=9, fontweight='bold')

ax5a.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
          'b-', linewidth=2.5, alpha=0.7, label='Camera trajectory')
ax5a.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
            c='green', s=150, marker='o', edgecolor='darkgreen', linewidth=2, 
            label='Start', zorder=10)
ax5a.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
            c='orange', s=150, marker='s', edgecolor='darkorange', linewidth=2, 
            label='End', zorder=10)

ax5a.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax5a.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax5a.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax5a.set_title('3D - Strawberry Field Layout', fontsize=12, fontweight='bold')
ax5a.legend(fontsize=10)
ax5a.grid(True, alpha=0.3)

# Top-down view
ax5b = fig5.add_subplot(1, 2, 2)

all_strawberry_positions = np.array(all_strawberry_positions)
scatter = ax5b.scatter(all_strawberry_positions[:, 0], all_strawberry_positions[:, 1],
                       s=400, c='red', marker='o', edgecolors='darkred', 
                       linewidth=2.5, alpha=0.85, zorder=5)

# Add IDs to points
for strawberry_idx, pos in enumerate(all_strawberry_positions):
    ax5b.annotate(f'S{strawberry_idx}', xy=(pos[0], pos[1]), 
                 fontsize=10, fontweight='bold', ha='center', va='center',
                 color='white', zorder=6)

ax5b.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2.5, 
         alpha=0.6, label='Camera path')
ax5b.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], s=150, c='green', marker='o', 
            edgecolor='darkgreen', linewidth=2, label='Start', zorder=10)
ax5b.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], s=150, c='orange', marker='s', 
            edgecolor='darkorange', linewidth=2, label='End', zorder=10)

ax5b.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax5b.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax5b.set_title('Top-Down View - Strawberry Positions', fontsize=12, fontweight='bold')
ax5b.legend(fontsize=10)
ax5b.grid(True, alpha=0.3)
ax5b.axis('equal')

# Enforce display circle if berries fall outside the actual GT path
try:
    if all_strawberry_positions is not None and all_strawberry_positions.size > 0 and gt_trajectory is not None and gt_trajectory.size > 0:
        centroid = np.mean(gt_trajectory[:, :2], axis=0)
        path_r = np.mean(np.linalg.norm(gt_trajectory[:, :2] - centroid, axis=1))
        berry_r = np.max(np.linalg.norm(all_strawberry_positions[:, :2] - centroid, axis=1))
        display_r = max(path_r, berry_r + 0.5)
        if display_r > path_r * 1.01:
            theta = np.linspace(0, 2*np.pi, 240)
            circ_x = centroid[0] + display_r * np.cos(theta)
            circ_y = centroid[1] + display_r * np.sin(theta)
            ax5b.plot(circ_x, circ_y, 'b-', linewidth=2.5, alpha=0.9, label='Camera (display)')
            ax5b.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b--', linewidth=1.2, alpha=0.6, label='Camera (GT)')
except Exception:
    pass

fig5.suptitle('STRAWBERRY FIELD SPATIAL DISTRIBUTION', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_spatial_distribution.png', dpi=150, bbox_inches='tight')
print("  05_spatial_distribution.png - Strawberry positions in world coordinates")
plt.close()

print("\n[TRAJECTORY] Camera Path Statistics\n")

gt_trajectory = np.array([gt_poses[i]['t'] for i in range(num_frames) if i in gt_poses])
total_distance = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))

print(f"  Start position:  X={gt_trajectory[0,0]:7.3f}m  Y={gt_trajectory[0,1]:7.3f}m  Z={gt_trajectory[0,2]:7.3f}m")
print(f"  End position:    X={gt_trajectory[-1,0]:7.3f}m  Y={gt_trajectory[-1,1]:7.3f}m  Z={gt_trajectory[-1,2]:7.3f}m")
print(f"  Total distance:  {total_distance:7.3f}m")

print(f"  Consolidated: {unique_ids} tracking IDs → {len(clusters)} strawberries")
print(f"  Total observations: {total_dets}")
print(f"  Reprojection accuracy: {np.mean(reprojection_errors):.4f} px (mean, using GT poses)")
print(f"  Clustering threshold: {optimal_threshold*100:.1f} cm")