import cv2
import yaml
import torch
import numpy as np
import os
import csv
import json
from ultralytics import YOLO
from strawberry_tracker import StrawberryTracker
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else torch.device(config['device'])
    
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    r_start, r_end = config['test_frame_range']
    images = images[r_start:r_end]

    yolo_model = YOLO(config['yolo_weights_path'])
    tracker = StrawberryTracker(yolo_model, device, config=config)

    cam_cfg = config['camera']
    K = np.array([[cam_cfg['focal_length_x'], 0, cam_cfg['center_x']],
                  [0, cam_cfg['focal_length_y'], cam_cfg['center_y']],
                  [0, 0, 1]], dtype=np.float32)

    export_enabled = config.get('export_data', False)
    export_dir = config.get('export_dir', 'exported_data')
    
    if export_enabled:
        os.makedirs(export_dir, exist_ok=True)
        poses_file = open(os.path.join(export_dir, 'poses.csv'), 'w', newline='')
        poses_writer = csv.writer(poses_file)
        poses_writer.writerow(['frame', 'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33', 'tx', 'ty', 'tz'])
        
        tracking_data = {}
        inliers_data = {}

    for i, img_name in enumerate(tqdm(images, desc="Processing Frames")):
        path = os.path.join(data_dir, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        R, t, ids, boxes, m1, m2, i1, i2, debug_info = tracker.forward(img_rgb, K)
        
        frame_key = f"frame_{r_start+i:03d}"
        
        if export_enabled:
            if R is not None and t is not None:
                row = [frame_key] + R.flatten().tolist() + t.flatten().tolist()
                poses_writer.writerow(row)
            
            frame_tracks = []
            for box, obj_id in zip(boxes, ids):
                frame_tracks.append({"id": int(obj_id), "bbox": [float(b) for b in box]})
            tracking_data[frame_key] = frame_tracks
            
            if len(i2) > 0:
                inliers_data[frame_key] = {
                    "pts1": i1.tolist(), 
                    "pts2": i2.tolist()  
                }

        # Блок визуализации (если включен в конфиге)
        if config.get('visualization_enabled', True):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_rgb)
            for box, gid in zip(boxes, ids):
                rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1], f'ID:{gid}', color='white', backgroundcolor='red', fontsize=8)
            
            if len(m2) > 0:
                ax.scatter(m2[:, 0], m2[:, 1], c='cyan', s=2, alpha=0.3, label='Matches')
            if len(i2) > 0:
                ax.scatter(i2[:, 0], i2[:, 1], c='red', s=4, alpha=0.8, label='Inliers')
            
            R_str = "\n".join([" ".join([f"{x:6.3f}" for x in row]) for row in R])
            title = f"Frame {r_start+i} | Inliers: {len(i2)} / {len(m2)}\nRotation Matrix:\n{R_str}"
            plt.title(title, fontdict={'family': 'monospace', 'size': 10})
            
            plt.savefig(os.path.join(output_dir, f'{frame_key}.png'))
            plt.close()

    if export_enabled:
        poses_file.close()
        with open(os.path.join(export_dir, 'tracking.json'), 'w') as f:
            json.dump(tracking_data, f, indent=4)
        with open(os.path.join(export_dir, 'inliers.json'), 'w') as f:
            json.dump(inliers_data, f, indent=4)
        print(f"✅ Данные успешно экспортированы в папку '{export_dir}/'")

if __name__ == "__main__":
    main()