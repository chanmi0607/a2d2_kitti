import numpy as np

# --- (원본 함수) GT 박스 로더 ---
# def load_gt_boxes(gt_txt_path):
#     if not gt_txt_path.exists():
#         return np.array([]), []
        
#     gt_boxes, gt_names = [], []
#     with open(gt_txt_path, 'r') as f:
#         for line in f.readlines():
#             parts = line.strip().split()
#             cls_name = parts[0]
#             try:
#                 # A2D2: x(2), y(3), z(4), l(5), w(6), h(7), yaw(8)
#                 x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
#                 l, w, h, yaw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
#                 gt_boxes.append([x, y, z, l, w, h, yaw])
#                 gt_names.append(cls_name)
#             except ValueError:
#                 print(f"[경고] {gt_txt_path} 파일의 라인을 파싱할 수 없습니다: {line}")
                
#     gt_boxes = np.array(gt_boxes, dtype=np.float32)

#     # ⭐ 핵심: 하단(z_min) 기준 → 중심(z_center) 기준
#     if gt_boxes.shape[0] > 0:
#         gt_boxes[:, 2] += gt_boxes[:, 5] / 2.0

#     return gt_boxes, gt_names

