import torch
import numpy as np

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# 2. RPN box 출력
    # data_dict가 Bounding box 인거임
    # NMS을 하는데, score 되게 작게해서 많은 박스를 최대한 살리거나 후처리 조금 추가
    # 후처리 알고리즘 구현 필요시 작성
def extract_point_features_cpu(points_np, boxes_np, min_points_in_box, logger):
    """
    CPU(Numpy)와 OpenPCDet 유틸리티(roiaware_pool3d)를 사용하여
    각 박스 내부 포인트로부터 특징 추출 (최적화 버전)
    (원본 스크립트의 함수와 동일)
    """
    num_boxes = boxes_np.shape[0]
    point_features_list = []
    valid_box_indices = [] # 특징 추출에 성공한 박스의 인덱스

    try:
        points_tensor = torch.from_numpy(points_np[:, 0:3]).float()
        boxes_tensor = torch.from_numpy(boxes_np).float()
        
        point_indices_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
            points_tensor, boxes_tensor
        ).numpy() # (num_boxes, num_points)
        
    except Exception as e:
        logger.error(f"  Fatal Error during roiaware_pool3d_utils.points_in_boxes_cpu: {e}")
        return np.array([]), np.array([], dtype=int)

    # 2. 각 박스를 순회하며 특징 계산 (NumPy)
    for i in range(num_boxes):
        try:
            mask = point_indices_mask[i]
            points_in_box = points_np[mask.astype(bool)]
        except Exception as e:
            continue

        if points_in_box.shape[0] < min_points_in_box:
            continue # 포인트 부족 시 건너뛰기

        valid_box_indices.append(i) # 유효한 박스 인덱스 추가

        try:
            num_points = points_in_box.shape[0]

            min_coords = np.min(points_in_box[:, :3], axis=0)
            max_coords = np.max(points_in_box[:, :3], axis=0)
            dims = max_coords - min_coords
            width = dims[0]       # 포인트 분포의 너비 (x)
            length = dims[1]      # 포인트 분포의 길이 (y)
            height = dims[2]      # 포인트 분포의 높이 (z)
            
            box_l, box_w, box_h = boxes_np[i, 3], boxes_np[i, 4], boxes_np[i, 5]
            box_volume = (box_l * box_w * box_h) + 1e-6 
            density = num_points / box_volume 

            aspect_ratio = width / (length + 1e-6) 
            mean_z = np.mean(points_in_box[:, 2])
            std_z = np.std(points_in_box[:, 2])
            mean_intensity = np.mean(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0
            std_intensity = np.std(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0

            features = [
                num_points, 
                width, length, height,
                density, aspect_ratio, 
                mean_z, std_z, mean_intensity, std_intensity
            ]
            point_features_list.append(features)
        except Exception as e:
            if i in valid_box_indices:
                 valid_box_indices.pop()

    if not point_features_list:
        return np.array([]), np.array([], dtype=int)

    return np.array(point_features_list, dtype=np.float32), np.array(valid_box_indices, dtype=int)
