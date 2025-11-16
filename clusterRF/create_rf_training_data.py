# 랜덤 포레스트 학습 데이터 (csv) 생성

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 기존 스크립트들에서 필요한 함수들을 가져옵니다.
from ground_removal_onlyz import read_kitti_bin, remove_ground_by_z_axis
from propose_main import filter_points_by_z_spread
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from clustering_utils import cluster_bev_image
from pcdet.models.dense_heads.main_gt_labeling import associate_clusters_with_gt # main_gt_labeling.py에서 함수 임포트
from rf_gt_utils import get_a2d2_gt_boxes
import cv2

def extract_features_from_points(points):
    """
    주어진 포인트 묶음(snippet)으로부터 RandomForest가 학습할 특징 벡터를 추출합니다.
    (이 함수는 수정되지 않았습니다.)
    """
    if points.shape[0] < 5:
        return None
    num_points = points.shape[0]
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    width = max_coords[0] - min_coords[0]
    length = max_coords[1] - min_coords[1]
    height = max_coords[2] - min_coords[2]
    density = num_points / ((width * length * height) + 1e-6)
    aspect_ratio = width / (length + 1e-6)
    mean_z = np.mean(points[:, 2])
    std_z = np.std(points[:, 2])
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
    return features


def load_frame_ids_from_file(file_path):
    """
    [신규] .txt 파일에서 프레임 ID 목록을 읽어오는 헬퍼 함수
    """
    if not os.path.exists(file_path):
        print(f"경고: '{file_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")
        return set()
    try:
        with open(file_path, 'r') as f:
            # 각 줄의 공백(개행문자 등)을 제거하고 set에 추가
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"'{file_path}' 파일 로드 중 오류 발생: {e}")
        return set()


def main():
    """메인 실행 함수"""
    
    # --- 경로 및 파라미터 설정 ---
    BASE_DIR = "/home/a/OpenPCDet/data/a2d2/training"
    
    # [수정] train.txt와 val.txt가 있는 ImageSets 디렉터리 경로
    IMAGESETS_DIR = "/home/a/OpenPCDet/data/a2d2/ImageSets"
    
    # [수정] 모든 데이터를 저장할 출력 파일 이름
    OUTPUT_FILE = "/home/a/OpenPCDet/data/a2d2/rf_dataset_all.csv" 
    
    # ... (기타 파라미터는 동일) ...
    GROUND_HEIGHT_THRESHOLD = -2.7
    BEV_X_RANGE = (0, 30)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.1
    MIN_CLUSTER_AREA = 1
    Z_SPREAD_THRESHOLD = 13.0
    DISTANCE_THRESHOLD_PIXELS = 5
    # ----------------------------

    all_features = [] # 모든 파일에서 추출된 특징들을 저장할 리스트

    # --- [수정] train.txt와 val.txt의 ID를 모두 로드 ---
    train_split_file = os.path.join(IMAGESETS_DIR, "train.txt")
    val_split_file = os.path.join(IMAGESETS_DIR, "val.txt")
    
    train_ids = load_frame_ids_from_file(train_split_file)
    val_ids = load_frame_ids_from_file(val_split_file)
    
    allowed_frame_ids = train_ids.union(val_ids) # 두 set을 합침
    
    if not allowed_frame_ids:
        print(f"오류: '{IMAGESETS_DIR}'에서 'train.txt' 또는 'val.txt' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{IMAGESETS_DIR}'에서 총 {len(allowed_frame_ids)}개의 프레임 ID (train+val)를 로드했습니다.")
    # --- [수정] 끝 ---


    # 2. 전체 .bin 파일 목록을 가져온 후, 허용된 ID 목록을 기반으로 필터링합니다.
    bin_dir = os.path.join(BASE_DIR, "velodyne")
    all_bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    
    # [수정] 변수 이름 변경 (train_bin_files -> target_bin_files)
    target_bin_files = [
        f for f in all_bin_files if os.path.splitext(f)[0] in allowed_frame_ids
    ]

    print(f"'{bin_dir}'에서 총 {len(all_bin_files)}개의 파일을 발견했습니다.")
    print(f"이 중 {len(target_bin_files)}개가 처리 대상으로 필터링되었습니다.")

    # 3. 필터링된 파일 목록(target_bin_files)을 사용하여 메인 루프를 실행합니다.
    for bin_name in tqdm(target_bin_files, desc="전체 데이터 생성 중"): # [수정] tqdm 설명 변경
        file_base = os.path.splitext(bin_name)[0]
        bin_path = os.path.join(BASE_DIR, "velodyne", bin_name)
        label_path = os.path.join(BASE_DIR, "label_2", f"{file_base}.txt")
        calib_path = os.path.join(BASE_DIR, "calib", f"{file_base}.txt")

        # 1. 포인트 클라우드 처리 (기존 로직과 동일)
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
        
        gt_boxes = get_a2d2_gt_boxes(label_path, calib_path, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)
        final_objects = associate_clusters_with_gt(clusters, gt_boxes, DISTANCE_THRESHOLD_PIXELS)

        # 2. 각 객체 박스에 대해 포인트들을 찾고 특징 추출
        for obj in final_objects:
            # ... (좌표 변환 로직은 동일) ...
            x, y, w, h = obj['box']
            lidar_x_max = BEV_X_RANGE[1] - (y * BEV_RESOLUTION)
            lidar_x_min = BEV_X_RANGE[1] - ((y + h) * BEV_RESOLUTION)
            lidar_y_max = BEV_Y_RANGE[1] - (x * BEV_RESOLUTION)
            lidar_y_min = BEV_Y_RANGE[1] - ((x + w) * BEV_RESOLUTION)

            mask = (
                (filtered_points[:, 0] >= lidar_x_min) & (filtered_points[:, 0] <= lidar_x_max) &
                (filtered_points[:, 1] >= lidar_y_min) & (filtered_points[:, 1] <= lidar_y_max)
            )
            object_points = filtered_points[mask]

            features = extract_features_from_points(object_points)
            
            if features:
                features['class'] = obj['class'] 
                features['frame_id'] = file_base  # frame_id는 이미 추가되어 있음
                all_features.append(features)

    # 3. 모든 특징들을 DataFrame으로 변환하고 CSV 파일로 저장
    if all_features:
        df = pd.DataFrame(all_features)
        
        # [수정] 컬럼 순서 재정렬 (frame_id, class를 맨 앞으로)
        cols_to_move = ['frame_id', 'class']
        other_cols = [col for col in df.columns if col not in cols_to_move]
        df = df[cols_to_move + other_cols]
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n총 {len(df)}개의 객체 특징 추출 완료. '{OUTPUT_FILE}' 파일로 저장되었습니다.")
        print(df.head())
    else:
        print("추출된 특징이 없습니다.")


if __name__ == "__main__":
    main()