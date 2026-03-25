import cv2
import yaml
import torch
import numpy as np
import os
import csv
from ultralytics import YOLO
from strawberry_tracker import StrawberryTracker
import matplotlib.pyplot as plt

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
    K = np.array([[cam_cfg['focal_length'], 0, cam_cfg['center_x']],
                  [0, cam_cfg['focal_length'], cam_cfg['center_y']],
                  [0, 0, 1]], dtype=np.float32)

    stats = []

    for i, img_name in enumerate(images):
        path = os.path.join(data_dir, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Process frame
        # res: R, t, ids, boxes, raw_m1, raw_m2, inl1, inl2, debug_info
        R, t, ids, boxes, m1, m2, i1, i2, debug_info = tracker.forward(img_rgb, K, 
                                                                       iou_thresh=config['iou_threshold'],
                                                                       match_thresh=config['match_threshold'])
        
        if debug_info:
            debug_info['frame'] = r_start + i
            stats.append(debug_info)

        if i > 0 and config['visualization_enabled']:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_rgb)
            for box, gid in zip(boxes, ids):
                rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1], f'ID:{gid}', color='white', backgroundcolor='red', fontsize=8)
            
            # Show ALL matches in Cyan
            if len(m2) > 0:
                ax.scatter(m2[:, 0], m2[:, 1], c='cyan', s=2, alpha=0.3, label='Matches')
            # Show INLIERS in Red
            if len(i2) > 0:
                ax.scatter(i2[:, 0], i2[:, 1], c='red', s=4, alpha=0.8, label='Inliers')
            
            # Display Rotation Matrix
            R_str = "\n".join([" ".join([f"{x:6.3f}" for x in row]) for row in R])
            title = f"Frame {r_start+i} | Inliers: {len(i2)} / {len(m2)}\nRotation Matrix:\n{R_str}"
            plt.title(title, fontdict={'family': 'monospace', 'size': 10})
            
            plt.savefig(os.path.join(output_dir, f'frame_{r_start+i:03d}.png'))
            plt.close()

    if stats:
        keys = stats[0].keys()
        report_path = os.path.join(output_dir, 'adequacy_report.csv')
        with open(report_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats)

if __name__ == "__main__":
    main()
