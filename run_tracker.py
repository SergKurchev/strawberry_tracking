import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from strawberry_tracker import StrawberryTracker, plot_results

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda_is_available() else 'cpu') if config['device'] == 'auto' else torch.device(config['device'])
    print(f"Using device: {device}")

    # 2. Initialize Models
    yolo_model = YOLO(config['yolo_weights_path'])
    tracker = StrawberryTracker(yolo_model, device, config=config) # Updated tracker initialization

    # 3. Simulate Image Stream
    
    # Camera Intrinsic Matrix (K)
    cam_cfg = config['camera']
    K = np.array([
        [cam_cfg['focal_length'], 0, cam_cfg['center_x']],
        [0, cam_cfg['focal_length'], cam_cfg['center_y']],
        [0, 0, 1]
    ], dtype=np.float32)

    prev_img = None
    
    print("\nStarting Stream Processing...")
    # Process images from config data_dir
    images = sorted([f for f in os.listdir(config['data_dir']) if f.endswith('.png')])
    for i, img_name in enumerate(images):
        img_path = os.path.join(config['data_dir'], img_name)
        print(f"\n--- Processing Frame {i}: {img_path} ---")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Skipping {img_path} (not found)")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 4. Use the new forward method
        # Returns R, t, ids, boxes, matched_kp1, matched_kp2, debug_info
        res = tracker.forward(img_rgb, K)
        R, t, ids, boxes, m1, m2, debug_info = res # Updated return unpacking

        print(f"Detections: {len(boxes)}")
        print(f"Active IDs: {ids}")
        if i > 0:
            print(f"Pose R (rotation): {R.ravel()[:3]}...") # Partial print
            print(f"Pose t (translation): {t.ravel()}")

        # 5. Optional Visualization
        if config['visualization_enabled'] and prev_img is not None:
            plot_results(prev_img, img_rgb, m_kp1, m_kp2, tracker.prev_frame['boxes'], boxes, ids, None)
        
        prev_img = img_rgb

if __name__ == "__main__":
    main()
