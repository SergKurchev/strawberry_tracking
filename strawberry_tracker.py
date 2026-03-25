import cv2
import torch
import numpy as np
import kornia.feature as KF
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Helper class for merging features from memory
class DummyFeats:
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

class StrawberryTracker:
    def __init__(self, yolo_model, device, config):
        self.device = device
        self.yolo = yolo_model
        
        # Hyperparameters from config
        self.num_features = config.get('num_features', 4096)
        self.gallery_ttl = config.get('gallery_ttl', 10)
        self.debug = config.get('debug_mode', False)
        self.use_global = config.get('use_global_features', True)
        self.yolo_conf = config.get('yolo_conf', 0.5)
        self.ransac_prob = config.get('ransac_prob', 0.999)
        self.ransac_threshold = config.get('ransac_threshold', 3.0)
        self.min_motion_thresh = config.get('min_motion_thresh', 1.0)
        self.pose_solver = config.get('pose_solver', 'MAGSAC')
        
        if self.debug:
            print(f"Initializing StrawberryTracker (Global: {self.use_global}, Solver: {self.pose_solver})")
            
        self.disk = KF.DISK.from_pretrained('depth').to(device)
        self.lg = KF.LightGlue('disk').to(device).eval()

        self.prev_frame = None
        self.gallery = {}
        self.global_id_counter = 0
        self.current_frame_idx = 0

    def prepare_kornia_tensor(self, img_rgb):
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t.to(self.device).contiguous()

    def get_best_box_for_point(self, p, boxes):
        best_idx = -1
        min_area = float('inf')
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    best_idx = idx
        return best_idx

    def _extract_features(self, img_rgb, boxes):
        kimg = self.prepare_kornia_tensor(img_rgb)
        if not self.use_global:
            mask = torch.zeros_like(kimg)
            _, _, H, W = kimg.shape
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                mask[:, :, y1:y2, x1:x2] = 1.0
            kimg = kimg * mask
        with torch.inference_mode():
            features = self.disk(kimg, n=self.num_features, pad_if_not_divisible=True)
            return features[0]

    def _match_and_associate(self, feats1, feats2, boxes1, boxes2, img_rgb_shape, match_thresh=3):
        if len(feats1.keypoints) == 0 or len(feats2.keypoints) == 0:
            return [], np.array([]), np.array([]), []
            
        image0 = {"keypoints": feats1.keypoints[None], "descriptors": feats1.descriptors[None],
                  "image_size": torch.tensor([[img_rgb_shape[1], img_rgb_shape[0]]], device=self.device).float()}
        image1 = {"keypoints": feats2.keypoints[None], "descriptors": feats2.descriptors[None],
                  "image_size": torch.tensor([[img_rgb_shape[1], img_rgb_shape[0]]], device=self.device).float()}
                  
        with torch.inference_mode():
            matches_data = self.lg({"image0": image0, "image1": image1})
            
        kp1, kp2 = feats1.keypoints.cpu().numpy(), feats2.keypoints.cpu().numpy()
        
        if 'matches' in matches_data and len(matches_data['matches'][0]) > 0:
            m_indices = matches_data['matches'][0].cpu().numpy()
        elif 'matches0' in matches_data:
            matches_indices = matches_data['matches0'][0].cpu().numpy()
            valid = matches_indices != -1
            if np.any(valid): 
                m_indices = np.stack([np.where(valid)[0], matches_indices[valid]], axis=1)
            else: return [], np.array([]), np.array([]), []
        else: return [], np.array([]), np.array([]), []
        
        m_kp1, m_kp2 = kp1[m_indices[:, 0]], kp2[m_indices[:, 1]]
        match_counts = np.zeros((len(boxes1), len(boxes2)))
        
        for i in range(len(m_kp1)):
            id1 = self.get_best_box_for_point(m_kp1[i], boxes1)
            id2 = self.get_best_box_for_point(m_kp2[i], boxes2)
            if id1 != -1 and id2 != -1: 
                match_counts[id1, id2] += 1
                
        row_ind, col_ind = linear_sum_assignment(-match_counts)
        associations = [(r, c) for r, c in zip(row_ind, col_ind) if match_counts[r, c] >= match_thresh]
        return associations, m_kp1, m_kp2, m_indices

    def forward(self, img_rgb, K_matrix, iou_thresh=0.45, match_thresh=3):
        res = self.yolo.predict(img_rgb, verbose=False, iou=iou_thresh, conf=self.yolo_conf)[0]
        curr_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
        curr_feats = self._extract_features(img_rgb, curr_boxes)
        curr_ids = [-1] * len(curr_boxes)
        raw_m1, raw_m2 = np.array([]), np.array([])
        inl1, inl2 = np.array([]), np.array([])
        R, t = np.eye(3), np.zeros((3, 1))
        debug_info = {}

        if self.prev_frame is not None:
            assoc, raw_m1, raw_m2, _ = self._match_and_associate(
                self.prev_frame['feats'], curr_feats, 
                self.prev_frame['boxes'], curr_boxes, 
                img_rgb.shape, match_thresh
            )
            
            for b_prev_idx, b_curr_idx in assoc:
                curr_ids[b_curr_idx] = self.prev_frame['ids'][b_prev_idx]

            if len(raw_m1) >= 5:
                dist = np.linalg.norm(raw_m2 - raw_m1, axis=1)
                avg_dist = np.mean(dist)
                
                if avg_dist < self.min_motion_thresh:
                    R, t = np.eye(3), np.zeros((3, 1))
                    inl1, inl2 = raw_m1, raw_m2
                    if self.debug: print(f"| Frame {self.current_frame_idx} | STATIC (Dist: {avg_dist:.2f}px)")
                else:
                    R, t, mask, inliers1, inliers2 = localize_camera(
                        raw_m1, raw_m2, K_matrix, 
                        prob=self.ransac_prob, thresh=self.ransac_threshold, solver=self.pose_solver
                    )
                    inl1, inl2 = inliers1, inliers2
                
                if self.debug and len(inl1) > 0:
                    debug_info.update({'total_matches': len(raw_m1), 'inliers': len(inl1)})

            unmatched_curr = [i for i, cid in enumerate(curr_ids) if cid == -1]
            active_ids = set(curr_ids)
            lost_gallery = {gid: data for gid, data in self.gallery.items() if gid not in active_ids}

            if len(unmatched_curr) > 0 and len(lost_gallery) > 0:
                gal_kps, gal_descs, gal_boxes, gal_ids = [], [], [], []
                for gid, data in lost_gallery.items():
                    gal_kps.append(data['feats']['keypoints'])
                    gal_descs.append(data['feats']['descriptors'])
                    gal_boxes.append(data['box'])
                    gal_ids.append(gid)
                
                gal_feats = DummyFeats(torch.cat(gal_kps, dim=0), torch.cat(gal_descs, dim=0))
                
                gal_assoc, _, _, _ = self._match_and_associate(
                    gal_feats, curr_feats, gal_boxes, curr_boxes, img_rgb.shape, match_thresh
                )
                
                recovered = 0
                for b_gal_idx, b_curr_idx in gal_assoc:
                    if curr_ids[b_curr_idx] == -1: 
                        curr_ids[b_curr_idx] = gal_ids[b_gal_idx]
                        recovered += 1
                
                if self.debug and recovered > 0:
                    print(f"| Frame {self.current_frame_idx} | ↺ Recovered from strawberry memory: {recovered}")

        for i in range(len(curr_ids)):
            if curr_ids[i] == -1:
                curr_ids[i] = self.global_id_counter
                self.global_id_counter += 1

        for i, gid in enumerate(curr_ids):
            box = curr_boxes[i]
            kp_curr = curr_feats.keypoints.cpu().numpy()
            box_indices = [k for k, p in enumerate(kp_curr) if self.get_best_box_for_point(p, curr_boxes) == i]
            
            if box_indices:
                self.gallery[gid] = {
                    'feats': {'keypoints': curr_feats.keypoints[box_indices], 'descriptors': curr_feats.descriptors[box_indices]},
                    'last_seen': self.current_frame_idx, 'box': box
                }
            elif gid in self.gallery:
                self.gallery[gid]['last_seen'] = self.current_frame_idx
                self.gallery[gid]['box'] = box

        expired = [gid for gid, data in list(self.gallery.items()) if self.current_frame_idx - data['last_seen'] > self.gallery_ttl]
        for gid in expired: 
            del self.gallery[gid]

        self.prev_frame = {'feats': curr_feats, 'boxes': curr_boxes, 'ids': curr_ids, 'img_shape': img_rgb.shape}
        self.current_frame_idx += 1
        
        return R, t, curr_ids, curr_boxes, raw_m1, raw_m2, inl1, inl2, debug_info

def localize_camera(kp1, kp2, K_matrix, prob=0.999, thresh=1.0, solver='MAGSAC'):
    if len(kp1) < 5: return np.eye(3), np.zeros((3, 1)), None, np.array([]), np.array([])
    pts1, pts2 = kp1.astype(np.float64), kp2.astype(np.float64)
    
    method = cv2.USAC_MAGSAC if solver == 'MAGSAC' else cv2.LMEDS if solver == 'LMEDS' else cv2.RANSAC

    print(f"LOC: pts={len(pts1)}, solver={solver}, thresh={thresh}, K={K_matrix.astype(np.float32).ravel()}")

    
    E, mask = cv2.findEssentialMat(pts1, pts2, K_matrix.astype(np.float64), method, prob, thresh)
    if E is None or E.shape != (3,3): 
        return np.eye(3), np.zeros((3, 1)), None, np.array([]), np.array([])
        
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_matrix.astype(np.float64), mask=mask)
    return R, t, mask, pts1[mask.ravel()==1], pts2[mask.ravel()==1]

