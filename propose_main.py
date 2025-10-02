# main.py

import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
import json

# 로컬 모듈에서 필요한 함수들을 import 합니다.
from pcdet.datasets.processor.ground_removal import read_kitti_bin, custom_extract_roi, custom_ransac_plane_fitting
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from pcdet.models.dense_heads.clustering_utils import cluster_bev_image
from pcdet.models.dense_heads.classifier_utils import load_model, extract_features

def main():
    """메인 실행 함수"""
    
    # ========================== 설정 ==========================
    BIN_DATA_DIR = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    
    # ROI, RANSAC, BEV, 클러스터링 파라미터
    ROI_X_RANGE = (-70.4, 80.4)
    ROI_Y_RANGE = (-40, 40)
    ROI_Z_RANGE = (-3.73, -1.30)
    RANSAC_DIST_THRESH = 0.20
    RANSAC_MAX_ITER = 1000
    RANSAC_MIN_SAMPLES = 3
    BEV_X_RANGE = (0, 70.4)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.1  # 해상도 (필요에 따라 조정)
    MIN_CLUSTER_AREA = 15

    # 학습된 모델 및 클래스 정보 파일 경로
    MODEL_PATH = "car_detector.joblib"
    MAPPING_PATH = "class_mapping.json"
    # ========================================================

    # 1. 모델과 클래스 매핑 정보 로드
    classifier = load_model(MODEL_PATH)
    if classifier is None:
        print(f"'{MODEL_PATH}'를 로드할 수 없습니다. train_model.py를 먼저 실행하여 모델을 생성하세요.")
        return
        
    try:
        with open(MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        # 예측 결과(숫자)를 클래스 이름(문자)으로 바꾸기 위한 역방향 맵 생성
        # 예: {"0": "Car", "1": "Pedestrian", ...}
        reverse_mapping = {str(v): k for k, v in class_mapping.items()}
    except FileNotFoundError:
        print(f"'{MAPPING_PATH}'를 찾을 수 없습니다. train_model.py를 먼저 실행하여 클래스 맵 파일을 생성하세요.")
        return

    # 각 클래스별로 다른 색상을 지정 (BGR 순서)
    # 색상 수는 클래스 매핑의 클래스 수 이상으로 준비하는 것이 좋습니다.
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

        # 2. 포인트 클라우드 처리 (지면 제거, BEV 변환, 클러스터링)
        pcd = read_kitti_bin(file_path)
        points = np.asarray(pcd.points)
        roi_pcd, _ = custom_extract_roi(pcd, ROI_X_RANGE, ROI_Y_RANGE, ROI_Z_RANGE)
        plane_coeffs, ground_indices, _ = custom_ransac_plane_fitting(
            roi_pcd, max_iterations=RANSAC_MAX_ITER, distance_threshold=RANSAC_DIST_THRESH, min_samples=RANSAC_MIN_SAMPLES
        )
        if plane_coeffs is None: continue
        
        roi_mask = (
            (points[:, 0] >= ROI_X_RANGE[0]) & (points[:, 0] <= ROI_X_RANGE[1]) &
            (points[:, 1] >= ROI_Y_RANGE[0]) & (points[:, 1] <= ROI_Y_RANGE[1]) &
            (points[:, 2] >= ROI_Z_RANGE[0]) & (points[:, 2] <= ROI_Z_RANGE[1])
        )
        original_indices_in_roi = np.where(roi_mask)[0]
        original_indices_of_ground = original_indices_in_roi[ground_indices]
        all_indices = np.arange(len(points))
        non_ground_indices = np.setdiff1d(all_indices, original_indices_of_ground)
        non_ground_points = points[non_ground_indices]
        
        if non_ground_points.shape[0] == 0: continue
        
        bev_image = pointcloud_to_bev(
            points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION
        )
        if bev_image is None: continue
        # Closing에 사용할 커널(kernel) 정의. (3, 3) 또는 (5, 5) 시도
        kernel = np.ones((5, 5), np.uint8)

        # OpenCV의 morphologyEx 함수를 사용해 Closing 연산 수행
        closed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
        
        clusters, clustered_bev_image = cluster_bev_image(
            closed_bev_image, min_area_threshold=MIN_CLUSTER_AREA
        )
        
        
        # 3. 감지된 클러스터가 있을 경우 예측 및 시각화 수행
        if clusters:
            # 특징 추출 시 resolution 값을 함께 전달하여 단위를 '미터'로 통일
            features = extract_features(clusters, resolution=BEV_RESOLUTION)
            predictions = classifier.predict(features)

            # 예측 결과를 BGR 이미지에 시각화
            for i, (x, y, w, h) in enumerate(clusters):
                prediction_int = predictions[i]
                
                # 예측된 숫자 라벨을 클래스 이름으로 변환
                label_text = reverse_mapping.get(str(prediction_int), "Unknown")
                # 클래스에 맞는 색상 가져오기
                color = color_map[prediction_int % len(color_map)]
                
                # 바운딩 박스와 라벨 텍스트를 이미지에 그리기
                cv2.rectangle(clustered_bev_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(clustered_bev_image, label_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. 최종 결과 창에 출력
        cv2.putText(clustered_bev_image, bin_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Prediction Result (Press any key to continue, 'q' to quit)", clustered_bev_image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("모든 파일 처리가 완료되었습니다.")


if __name__ == "__main__":
    main()