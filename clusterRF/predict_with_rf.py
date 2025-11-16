# create pkl 파일

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle # pkl 저장을 위해 import
import joblib # 학습된 RF 모델 로드를 위해 import

# 기존 스크립트들에서 필요한 함수들을 가져옵니다.
from ground_removal_onlyz import read_kitti_bin, remove_ground_by_z_axis
from propose_main import filter_points_by_z_spread
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from clustering_utils import cluster_bev_image
# from pcdet.models.dense_heads.main_gt_labeling import associate_clusters_with_gt # GT는 필요 없음
# from pcdet.models.dense_heads.rf_gt_utils import get_a2d2_gt_boxes # GT는 필요 없음
import cv2

# --- 설정 ---
# 1. 학습된 모델 및 인코더 경로
RF_MODEL_PATH = 'random_forest_model_from_csv.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# 2. 검증할 프레임 ID 목록 (val.txt 또는 train.txt)
#    (visualize_frame.py의 FRAME_ID가 포함된 파일을 선택하세요)
SPLIT_FILE_PATH = 'data/a2d2/ImageSets/val.txt' # 👈 검증하고 싶은 목록 (train.txt or val.txt)

# 3. 데이터 경로
BASE_DIR = "data/a2d2/training"
LIDAR_DIR = os.path.join(BASE_DIR, "velodyne")

# 4. 생성될 PKL 파일 경로
OUTPUT_PKL_PATH = 'output/rf_prediction_results.pkl' # 👈 시각화 스크립트가 읽을 경로

# 5. 전처리 파라미터 (create_rf_training_data.py와 동일하게 유지)
GROUND_HEIGHT_THRESHOLD = -2.7
BEV_X_RANGE = (0, 30)
BEV_Y_RANGE = (-40, 40)
BEV_RESOLUTION = 0.1
MIN_CLUSTER_AREA = 1
Z_SPREAD_THRESHOLD = 13.0
# --------------------


def extract_features_and_box(points):
    """
    포인트 묶음(snippet)으로부터 특징 벡터(X)와 3D 박스(box_lidar)를 추출합니다.
    """
    if points.shape[0] < 5:
        return None, None

    # --- 1. 특징 벡터(X) 추출 (기존과 동일) ---
    num_points = points.shape[0]
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # [x, y, z] 순서로 계산
    dims = max_coords - min_coords
    width = dims[0]   # X
    length = dims[1]  # Y
    height = dims[2]  # Z

    density = num_points / ((width * length * height) + 1e-6)
    aspect_ratio = width / (length + 1e-6)
    mean_z = np.mean(points[:, 2])
    std_z = np.std(points[:, 2])

    # DataFrame 순서와 동일하게 특징 구성
    features = {
        'num_points': num_points,
        'width': width,
        'length': length,
        'height': height,
        'density': density,
        'aspect_ratio': aspect_ratio,
        'mean_z': mean_z,
        'std_z': std_z
    }
    # DataFrame의 컬럼 순서대로 값을 정렬 (중요!)
    feature_vector = [features[col] for col in [
        'num_points', 'width', 'length', 'height', 'density', 
        'aspect_ratio', 'mean_z', 'std_z'
    ]]

    # --- 2. 3D 박스(boxes_lidar) 계산 ---
    # (AABB - Axis-Aligned Bounding Box)
    center = (min_coords + max_coords) / 2.0
    
    # [x, y, z, l, w, h, yaw] 형식 (l,w,h 순서 주의, RF는 yaw=0으로 가정)
    # create_rf_training_data.py의 width(x), length(y), height(z) 정의를 따름
    box_lidar = np.array([
        center[0], center[1], center[2], 
        width, length, height, 0.0
    ])
    
    return feature_vector, box_lidar


def load_frame_ids(file_path):
    """ .txt 파일에서 프레임 ID 목록을 읽어옵니다. """
    try:
        with open(file_path, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"❗️ Error: '{file_path}' 파일을 찾을 수 없습니다.")
        return None

def main():
    # 1. 학습된 모델 로드
    try:
        model = joblib.load(RF_MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print(f"✅ '{RF_MODEL_PATH}'에서 모델을 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❗️ Error: '{RF_MODEL_PATH}' 또는 '{LABEL_ENCODER_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 randomforest.py 스크립트를 실행하여 모델을 학습/저장해야 합니다.")
        return

    # 2. 검증할 프레임 ID 목록 로드
    frame_ids = load_frame_ids(SPLIT_FILE_PATH)
    if frame_ids is None:
        return
    print(f"'{SPLIT_FILE_PATH}'에서 {len(frame_ids)}개의 프레임 ID를 로드했습니다.")

    all_predictions_list = [] # 최종 pkl 파일에 저장될 리스트

    # 3. 각 프레임에 대해 전처리 및 예측 수행
    for frame_id in tqdm(frame_ids, desc="프레임 예측 중"):
        bin_path = os.path.join(LIDAR_DIR, f"{frame_id}.bin")
        if not os.path.exists(bin_path):
            continue

        # (a) 전처리 파이프라인 (create_rf_training_data.py와 동일)
        pcd_original = read_kitti_bin(bin_path)
        pcd_non_ground = remove_ground_by_z_axis(pcd_original, GROUND_HEIGHT_THRESHOLD)
        non_ground_points = np.asarray(pcd_non_ground.points)
        if non_ground_points.shape[0] == 0: continue

        filtered_points = filter_points_by_z_spread(
            points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION, z_spread_threshold=Z_SPREAD_THRESHOLD
        )
        if filtered_points.shape[0] == 0: continue
        
        bev_image = pointcloud_to_bev(points=filtered_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION)
        if bev_image is None: continue
        kernel = np.ones((3, 3), np.uint8)
        processed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
        clusters, _ = cluster_bev_image(processed_bev_image, min_area_threshold=MIN_CLUSTER_AREA)
        
        # (b) 각 클러스터에 대해 특징 및 3D 박스 추출
        frame_features_list = []
        frame_boxes_list = []
        
        for cluster_info in clusters:
            #x, y, w, h = cluster_info['box']
            x, y, w, h = cluster_info
            lidar_x_max = BEV_X_RANGE[1] - (y * BEV_RESOLUTION)
            lidar_x_min = BEV_X_RANGE[1] - ((y + h) * BEV_RESOLUTION)
            lidar_y_max = BEV_Y_RANGE[1] - (x * BEV_RESOLUTION)
            lidar_y_min = BEV_Y_RANGE[1] - ((x + w) * BEV_RESOLUTION)

            mask = (
                (filtered_points[:, 0] >= lidar_x_min) & (filtered_points[:, 0] <= lidar_x_max) &
                (filtered_points[:, 1] >= lidar_y_min) & (filtered_points[:, 1] <= lidar_y_max)
            )
            object_points = filtered_points[mask]

            feature_vector, box_lidar = extract_features_and_box(object_points)
            
            if feature_vector is not None:
                frame_features_list.append(feature_vector)
                frame_boxes_list.append(box_lidar)

        # (c) 프레임에 클러스터가 있으면 일괄 예측
        if frame_features_list:
            X_frame = np.array(frame_features_list)
            
            # 클래스 예측 (숫자 인덱스)
            pred_class_indices = model.predict(X_frame)
            # 클래스별 확률 예측 (N, num_classes)
            pred_scores_all = model.predict_proba(X_frame)
            # 예측된 클래스의 신뢰도 점수만 추출
            pred_scores = pred_scores_all[np.arange(len(pred_class_indices)), pred_class_indices]
            # 숫자 인덱스를 실제 클래스 이름으로 변환
            pred_names = label_encoder.inverse_transform(pred_class_indices)
            
            # 'DontCare' 또는 'Background' 클래스 제외 (이름이 일치해야 함)
            # (label_encoder에 'DontCare'가 없다면 이 필터는 작동하지 않음)
            valid_mask = (pred_names != 'DontCare')
            
            final_boxes = np.array(frame_boxes_list)[valid_mask]
            final_scores = pred_scores[valid_mask]
            final_names = pred_names[valid_mask]

        else: # 클러스터가 없는 경우 빈 배열
            final_boxes = np.array([])
            final_scores = np.array([])
            final_names = np.array([])

        # (d) 최종 딕셔너리 생성
        frame_result = {
            'frame_id': frame_id,
            'boxes_lidar': final_boxes,
            'score': final_scores,
            'name': final_names
        }
        all_predictions_list.append(frame_result)

    # 4. 모든 예측 결과를 .pkl 파일로 저장
    output_dir = os.path.dirname(OUTPUT_PKL_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump(all_predictions_list, f)
        
    print(f"\n✅ 예측 완료. 결과가 '{OUTPUT_PKL_PATH}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()