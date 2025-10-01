# main.py

import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

# 로컬 모듈에서 필요한 함수들을 import 합니다.
from pcdet.datasets.processor.ground_removal import read_kitti_bin, custom_extract_roi, custom_ransac_plane_fitting, custom_visualize_ground_removal
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from pcdet.models.dense_heads.clustering_utils import cluster_bev_image

def main():
    """메인 실행 함수"""
    
    # ========================== 설정 ==========================
    # 데이터셋 경로를 지정하세요.
    BIN_DATA_DIR = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    
    # ROI 파라미터
    ROI_X_RANGE = (-70.4, 80.4)
    ROI_Y_RANGE = (-40, 40)
    ROI_Z_RANGE = (-3.73, -1.30)
    
    # RANSAC 파라미터
    RANSAC_DIST_THRESH = 0.20
    RANSAC_MAX_ITER = 1000
    RANSAC_MIN_SAMPLES = 3

    # BEV 변환 파라미터
    BEV_X_RANGE = (0, 70.4)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.1

    # 클러스터링 노이즈 제거를 위한 최소 면적 설정
    MIN_CLUSTER_AREA = 10
    # ========================================================

    bin_files = sorted([f for f in os.listdir(BIN_DATA_DIR) if f.endswith(".bin")])
    if not bin_files:
        print(f"Error: '{BIN_DATA_DIR}' 디렉토리에서 .bin 파일을 찾을 수 없습니다.")
        return

    for bin_name in tqdm(bin_files, desc="파일 처리 중"):
        file_path = os.path.join(BIN_DATA_DIR, bin_name)

        # 1. 포인트 클라우드 로드 (o3d.PointCloud 객체)
        pcd = read_kitti_bin(file_path)
        points = np.asarray(pcd.points) # NumPy 배열로도 변환해 둠

        # 2. ROI 영역 추출
        roi_pcd, _ = custom_extract_roi(pcd, ROI_X_RANGE, ROI_Y_RANGE, ROI_Z_RANGE)

        # 3. RANSAC으로 지면 평면 찾기
        plane_coeffs, ground_indices, object_indices = custom_ransac_plane_fitting(
            roi_pcd, 
            max_iterations=RANSAC_MAX_ITER,
            distance_threshold=RANSAC_DIST_THRESH,
            min_samples=RANSAC_MIN_SAMPLES
        )

        if plane_coeffs is None:
            print(f"[{bin_name}] 지면 평면을 찾지 못했습니다. 건너뜁니다.")
            continue
        
        # (선택) RANSAC 결과 시각화
        # custom_visualize_ground_removal(roi_pcd, plane_coeffs, ground_indices, object_indices)

        # 4. 전체 포인트 클라우드에서 지면 제거
        # ROI 내 인덱스를 전체 인덱스로 변환하는 과정
        roi_mask = (
            (points[:, 0] >= ROI_X_RANGE[0]) & (points[:, 0] <= ROI_X_RANGE[1]) &
            (points[:, 1] >= ROI_Y_RANGE[0]) & (points[:, 1] <= ROI_Y_RANGE[1]) &
            (points[:, 2] >= ROI_Z_RANGE[0]) & (points[:, 2] <= ROI_Z_RANGE[1])
        )
        original_indices_in_roi = np.where(roi_mask)[0]
        original_indices_of_ground = original_indices_in_roi[ground_indices]
        
        # 원본 포인트에서 지면 인덱스를 제외
        all_indices = np.arange(len(points))
        non_ground_indices = np.setdiff1d(all_indices, original_indices_of_ground)
        non_ground_points = points[non_ground_indices]

        if non_ground_points.shape[0] == 0:
            print(f"[{bin_name}] 지면 제거 후 남은 포인트가 없습니다. 건너뜁니다.")
            continue

        # 5. BEV 이미지로 변환
        bev_image = pointcloud_to_bev(
            points=non_ground_points,
            x_range=BEV_X_RANGE,
            y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION
        )

        if bev_image is None:
            print(f"[{bin_name}] BEV 변환 후 이미지가 비어있습니다. 건너뜁니다.")
            continue

        clusters, clustered_bev_image = cluster_bev_image(
            bev_image, 
            min_area_threshold=MIN_CLUSTER_AREA
        )
        print(f"[{bin_name}] {len(clusters)}개의 클러스터를 감지했습니다.")
            
        # 6. 최종 결과(BEV) 시각화
        cv2.putText(bev_image, bin_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Bird's-Eye View (Press any key to continue, 'q' to quit)", clustered_bev_image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()