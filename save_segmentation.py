import cv2
import yaml
import torch
import os
from ultralytics import YOLO

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load config and device
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else torch.device(config['device'])
    
    data_dir = config['data_dir']
    output_dir = os.path.join(config['output_dir'], 'segmentation_only')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Get frames from config range
    images = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    r_start, r_end = config['test_frame_range']
    images = images[r_start:r_end]

    # 3. Load YOLO model
    model = YOLO(config['yolo_weights_path'])
    
    print(f"Processing {len(images)} frames from {r_start} to {r_end}...")
    print(f"Saving results to: {output_dir}")

    for i, img_name in enumerate(images):
        path = os.path.join(data_dir, img_name)
        
        # 4. Predict (with segmentation masks)
        results = model.predict(path, iou=config['iou_threshold'], conf=config['yolo_conf'], verbose=False)
        
        # 5. Save the plotted results (masks + boxes)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            save_path = os.path.join(output_dir, f'seg_{r_start+i:03d}.png')
            cv2.imwrite(save_path, im_array)

    print("Done! Check the results in 'test_results/segmentation_only'.")

if __name__ == "__main__":
    main()
