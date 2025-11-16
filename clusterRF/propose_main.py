# main.py

import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
import json
from collections import defaultdict
# 로컬 모듈에서 필요한 함수들을 import 합니다.
#from pcdet.datasets.processor.ground_removal import read_kitti_bin, custom_extract_roi, custom_ransac_plane_fitting
from ground_removal_onlyz import read_kitti_bin, remove_ground_by_z_axis
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from clustering_utils import cluster_bev_image
from classifier_utils import load_model, extract_features

def filter_points_by_z_spread(points: np.ndarray, x_range: tuple, y_range: tuple, resolution: float, z_spread_threshold: float) -> np.ndarray:
    """
    BEV 그리드 셀 내 포인트들의 Z값 분포를 기반으로 벽이나 나무 같은 수직 객체를 필터링합니다.

    Args:
        points (np.ndarray): 필터링할 원본 포인트 클라우드 (N, 3+).
        x_range (tuple): BEV X축 범위.
        y_range (tuple): BEV Y축 범위.
        resolution (float): BEV 그리드 해상도.
        z_spread_threshold (float): Z값의 최대-최소 차이 임계값. 이 값보다 크면 수직 객체로 간주.

    Returns:
        np.ndarray: 수직 객체 포인트들이 제거된 포인트 클라우드.
    """
    if points.shape[0] == 0:
        return points

    # 각 그리드 셀에 어떤 포인트들이 속하는지 저장할 딕셔너리
    grid_cells = defaultdict(list)

    # 포인트를 해당하는 그리드 셀에 할당
    for i, point in enumerate(points):
        x, y = point[0], point[1]
        # BEV 범위 밖의 포인트는 무시
        if not (x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]):
            continue
        
        # 포인트의 그리드 인덱스 계산
        ix = int((x - x_range[0]) / resolution)
        iy = int((y - y_range[0]) / resolution)
        grid_cells[(ix, iy)].append(i)

    valid_point_indices = []
    # 각 그리드 셀을 순회하며 z값 분포 확인
    for indices in grid_cells.values():
        if not indices:
            continue
        
        cell_points = points[indices]
        z_values = cell_points[:, 2]
        
        # Z값의 최대-최소 차이(Peak to Peak) 계산
        z_spread = np.ptp(z_values)
        
        # Z값의 차이가 임계값보다 작을 경우에만 유효한 포인트로 추가
        if z_spread < z_spread_threshold:
            valid_point_indices.extend(indices)
    unique_indices = np.unique(valid_point_indices).astype(int)
    return points[unique_indices]

def main():
    """메인 실행 함수"""
    
    # ========================== 설정 ==========================
    BIN_DATA_DIR = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    GROUND_HEIGHT_THRESHOLD = -2.7  # Z축 지면 제거 임계값 (단위: 미터)

    BEV_X_RANGE = (0, 70.4)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.2
    MIN_CLUSTER_AREA = 15
    Z_SPREAD_THRESHOLD = 3.5

    MODEL_PATH = "cluster+rf/car_detector.joblib"
    MAPPING_PATH = "class_mapping.json"
    # ========================================================

    classifier = load_model(MODEL_PATH)
    if classifier is None:
        print(f"'{MODEL_PATH}'를 로드할 수 없습니다. train_model.py를 먼저 실행하여 모델을 생성하세요.")
        return
        
    try:
        with open(MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        reverse_mapping = {str(v): k for k, v in class_mapping.items()}
    except FileNotFoundError:
        print(f"'{MAPPING_PATH}'를 찾을 수 없습니다.")
        return

    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 128), (128, 255, 128),
        (128, 128, 255), (200, 200, 200) 
    ]

    bin_files = sorted([f for f in os.listdir(BIN_DATA_DIR) if f.endswith(".bin")])
    if not bin_files:
        print(f"Error: '{BIN_DATA_DIR}' 디렉토리에서 .bin 파일을 찾을 수 없습니다.")
        return

    for bin_name in tqdm(bin_files, desc="파일 처리 중"):
        file_path = os.path.join(BIN_DATA_DIR, bin_name)

        # ========================= [수정된 부분 시작] =========================
        # 2. 포인트 클라우드 처리 (지면 제거)
        # 기존의 복잡한 RANSAC 기반 지면 제거 로직을 Z축 필터링으로 대체
        
        # 2-1. 데이터 로드
        pcd_original = read_kitti_bin(file_path)
        if len(pcd_original.points) == 0:
            continue

        # 2-2. Z축 높이 기준으로 지면 제거
        pcd_non_ground = remove_ground_by_z_axis(pcd_original, GROUND_HEIGHT_THRESHOLD)
        non_ground_points = np.asarray(pcd_non_ground.points)
        
        if non_ground_points.shape[0] == 0:
            continue
        
        # 2-3. Z축 퍼짐 기준으로 벽/나무 등 수직 객체 필터링 (기존 로직 유지)
        filtered_points = filter_points_by_z_spread(
            points=non_ground_points,
            x_range=BEV_X_RANGE,
            y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION,
            z_spread_threshold=Z_SPREAD_THRESHOLD
        )
        
        if filtered_points.shape[0] == 0:
            continue

        # 2-4. 필터링된 포인트를 사용하여 BEV 이미지 생성
        bev_image = pointcloud_to_bev(
            points=filtered_points,
            x_range=BEV_X_RANGE, 
            y_range=BEV_Y_RANGE, 
            resolution=BEV_RESOLUTION
        )
        # ========================= [수정된 부분 끝] ===========================
        
        if bev_image is None:
            continue
            
        kernel = np.ones((5, 5), np.uint8)
        closed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
        
        clusters, clustered_bev_image = cluster_bev_image(
            closed_bev_image, min_area_threshold=MIN_CLUSTER_AREA
        )
        
        if clusters:
            features = extract_features(clusters, resolution=BEV_RESOLUTION)
            predictions = classifier.predict(features)

            for i, (x, y, w, h) in enumerate(clusters):
                prediction_int = predictions[i]
                label_text = reverse_mapping.get(str(prediction_int), "Unknown")
                color = color_map[prediction_int % len(color_map)]
                
                cv2.rectangle(clustered_bev_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(clustered_bev_image, label_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(clustered_bev_image, bin_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Prediction Result (Press any key to continue, 'q' to quit)", clustered_bev_image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()