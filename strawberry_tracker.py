import cv2
import torch
import numpy as np
import kornia.feature as KF
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

class DummyFeats:
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

class StrawberryTracker:
    def __init__(self, yolo_model, device, config):
        self.device = device
        self.yolo = yolo_model
        
        self.num_features = config.get('num_features', 4096)
        self.gallery_ttl = config.get('gallery_ttl', 10)
        self.debug = config.get('debug_mode', False)
        self.yolo_conf = config.get('yolo_conf', 0.5)
        self.ransac_prob = config.get('ransac_prob', 0.999)
        self.ransac_threshold = config.get('ransac_threshold', 3.0)
        self.min_motion_thresh = config.get('min_motion_thresh', 1.0)
        self.pose_solver = config.get('pose_solver', 'MAGSAC')
        
        if self.debug:
            print(f"Initializing StrawberryTracker (Dual-Pipeline, Solver: {self.pose_solver})")
            
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

    def _extract_features(self, img_rgb, boxes=None, global_extraction=False):
        """Извлекает фичи. Если global_extraction=False, маскирует фон."""
        kimg = self.prepare_kornia_tensor(img_rgb)
        
        if not global_extraction and boxes is not None and len(boxes) > 0:
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

    def _match_features(self, feats1, feats2, img_rgb_shape):
        """Только нейросетевой матчинг точек (LightGlue)."""
        if len(feats1.keypoints) == 0 or len(feats2.keypoints) == 0:
            return np.array([]), np.array([]), np.array([])
            
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
            else: return np.array([]), np.array([]), np.array([])
        else: return np.array([]), np.array([]), np.array([])
        
        m_kp1, m_kp2 = kp1[m_indices[:, 0]], kp2[m_indices[:, 1]]
        return m_kp1, m_kp2, m_indices

    def _associate_boxes(self, m_kp1, m_kp2, boxes1, boxes2, match_thresh):
        """Венгерский алгоритм для привязки точек к боксам."""
        if len(m_kp1) == 0 or len(boxes1) == 0 or len(boxes2) == 0: return []
        
        match_counts = np.zeros((len(boxes1), len(boxes2)))
        for i in range(len(m_kp1)):
            id1 = self.get_best_box_for_point(m_kp1[i], boxes1)
            id2 = self.get_best_box_for_point(m_kp2[i], boxes2)
            if id1 != -1 and id2 != -1: 
                match_counts[id1, id2] += 1
                
        row_ind, col_ind = linear_sum_assignment(-match_counts)
        return [(r, c) for r, c in zip(row_ind, col_ind) if match_counts[r, c] >= match_thresh]

    def forward(self, img_rgb, K_matrix, iou_thresh=0.45, match_thresh=3):
        # 1. FORMATTING FOR YOLO (BGR)
        img_bgr_for_yolo = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        res = self.yolo.predict(img_bgr_for_yolo, verbose=False, iou=iou_thresh, conf=self.yolo_conf, imgsz=640)[0]
        curr_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
        curr_ids = [-1] * len(curr_boxes)

        # 2. FEATURE EXTRACTION (DUAL-PIPELINE)
        # For tracking strawberries: the background is filled with black, all points go to the strawberries
        curr_feats_track = self._extract_features(img_rgb, curr_boxes, global_extraction=False)
        # For SLAM: scan the entire frame (with strawberries, walls and leaves)
        curr_feats_slam = self._extract_features(img_rgb, None, global_extraction=True)
        
        raw_m1, raw_m2, inl1, inl2 = np.array([]), np.array([]), np.array([]), np.array([])
        R, t = np.eye(3), np.zeros((3, 1))
        debug_info = {}

        if self.prev_frame is not None:
            # === PHASE 1.1: TRACKING OF STRAWBERRIES (LOCAL FEATURES) ===
            m_kp1_t, m_kp2_t, _ = self._match_features(self.prev_frame['feats_track'], curr_feats_track, img_rgb.shape)
            assoc = self._associate_boxes(m_kp1_t, m_kp2_t, self.prev_frame['boxes'], curr_boxes, match_thresh)
            
            for b_prev_idx, b_curr_idx in assoc:
                curr_ids[b_curr_idx] = self.prev_frame['ids'][b_prev_idx]

            # === PHASE 1.2: SLAM ODOMETRY (GLOBAL FEATURES) ===
            raw_m1, raw_m2, _ = self._match_features(self.prev_frame['feats_slam'], curr_feats_slam, img_rgb.shape)
            
            if len(raw_m1) >= 5:
                avg_dist = np.mean(np.linalg.norm(raw_m2 - raw_m1, axis=1))
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

            # === PHASE 2: RE-ID ===
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
                
                m_kp1_g, m_kp2_g, _ = self._match_features(gal_feats, curr_feats_track, img_rgb.shape)
                gal_assoc = self._associate_boxes(m_kp1_g, m_kp2_g, gal_boxes, curr_boxes, match_thresh)
                
                recovered = 0
                for b_gal_idx, b_curr_idx in gal_assoc:
                    if curr_ids[b_curr_idx] == -1: 
                        curr_ids[b_curr_idx] = gal_ids[b_gal_idx]
                        recovered += 1
                
                if self.debug and recovered > 0:
                    print(f"| Frame {self.current_frame_idx} | ↺ Recovered from strawberry memory: {recovered}")

        # === PHASE 3: UPDATING OF TRACKED OBJECTS AND ID ===
        for i in range(len(curr_ids)):
            if curr_ids[i] == -1:
                curr_ids[i] = self.global_id_counter
                self.global_id_counter += 1

        for i, gid in enumerate(curr_ids):
            box = curr_boxes[i]
            kp_curr = curr_feats_track.keypoints.cpu().numpy()
            box_indices = [k for k, p in enumerate(kp_curr) if self.get_best_box_for_point(p, curr_boxes) == i]
            
            if box_indices:
                self.gallery[gid] = {
                    'feats': {'keypoints': curr_feats_track.keypoints[box_indices], 'descriptors': curr_feats_track.descriptors[box_indices]},
                    'last_seen': self.current_frame_idx, 'box': box
                }
            elif gid in self.gallery:
                self.gallery[gid]['last_seen'] = self.current_frame_idx
                self.gallery[gid]['box'] = box

        expired = [gid for gid, data in list(self.gallery.items()) if self.current_frame_idx - data['last_seen'] > self.gallery_ttl]
        for gid in expired: 
            del self.gallery[gid]

        self.prev_frame = {
            'feats_track': curr_feats_track, 
            'feats_slam': curr_feats_slam, 
            'boxes': curr_boxes, 
            'ids': curr_ids, 
            'img_shape': img_rgb.shape
        }
        self.current_frame_idx += 1
        
        return R, t, curr_ids, curr_boxes, raw_m1, raw_m2, inl1, inl2, debug_info

def localize_camera(kp1, kp2, K_matrix, prob=0.999, thresh=1.0, solver='MAGSAC'):
    if len(kp1) < 5: return np.eye(3), np.zeros((3, 1)), None, np.array([]), np.array([])
    pts1, pts2 = kp1.astype(np.float64), kp2.astype(np.float64)
    
    method = cv2.USAC_MAGSAC if solver == 'MAGSAC' else cv2.LMEDS if solver == 'LMEDS' else cv2.RANSAC
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K_matrix.astype(np.float64), method, prob, thresh)
    if E is None or E.shape != (3,3): 
        return np.eye(3), np.zeros((3, 1)), None, np.array([]), np.array([])
        
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_matrix.astype(np.float64), mask=mask)
    return R, t, mask, pts1[mask.ravel()==1], pts2[mask.ravel()==1]