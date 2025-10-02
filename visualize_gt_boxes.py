

import os
import numpy as np
import cv2

# 필요한 헬퍼 함수들을 import 합니다.
from pcdet.datasets.processor.ground_removal import read_kitti_bin
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from tqdm import tqdm

class A2D2Calibration:
    """
    A2D2 보정 파일을 읽고 좌표 변환을 수행하는 클래스
    """
    def __init__(self, calib_path):
        self.data = self._load_calib_file(calib_path)
        self.R0_rect = self._get_matrix('R0_rect')
        self.Tr_velo_to_cam = self._get_matrix('Tr_velo_to_cam')

    def _load_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def _get_matrix(self, key):
        matrix = self.data[key].reshape(3, -1)
        if matrix.shape[1] == 3: # R0_rect
            homo_matrix = np.eye(4)
            homo_matrix[:3, :3] = matrix
            return homo_matrix
        elif matrix.shape[1] == 4: # Tr_velo_to_cam
            homo_matrix = np.vstack([matrix, [0, 0, 0, 1]])
            return homo_matrix
        return matrix

    def cam_to_lidar(self, points_cam):
        """카메라 좌표계 포인트를 LiDAR 좌표계로 변환"""
        points_cam_homo = np.hstack((points_cam, np.ones((len(points_cam), 1))))
        # (R_rect * Tr_velo_to_cam)^-1 * P_cam
        transform_matrix = np.linalg.inv(self.R0_rect @ self.Tr_velo_to_cam)
        points_lidar = points_cam_homo @ transform_matrix.T
        return points_lidar[:, :3]

def load_and_project_a2d2_gt_boxes(label_path, calibration, bev_x_range, bev_y_range, resolution):
    """
    A2D2 라벨을 읽고, '보정' 및 '회전'을 거쳐 BEV 2D 박스의 네 꼭짓점을 반환합니다.
    """
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if line[0] in ['DontCare', 'Misc']: continue

            class_name = line[0]
            h, w, l = float(line[8]), float(line[9]), float(line[10])
            cx_cam, cy_cam, cz_cam = float(line[11]), float(line[12]), float(line[13])
            ry_cam = float(line[14])

            center_cam = np.array([[cx_cam, cy_cam - h / 2, cz_cam]])
            center_lidar = calibration.cam_to_lidar(center_cam)[0]
            
            # ========================================================
            # ✅ 1. 회전 각도 변환 및 네 꼭짓점 계산
            # ========================================================
            # 카메라 좌표계의 Y축 회전을 LiDAR 좌표계의 Z축 회전(yaw)으로 변환
            ry_lidar = -ry_cam - np.pi / 2

            # 2D BEV 평면에서의 네 꼭짓점 좌표 계산
            corners_bev = np.array([
                [l / 2, w / 2],
                [l / 2, -w / 2],
                [-l / 2, -w / 2],
                [-l / 2, w / 2],
            ])

            # 회전 변환 행렬
            rotation_matrix = np.array([
                [np.cos(ry_lidar), -np.sin(ry_lidar)],
                [np.sin(ry_lidar), np.cos(ry_lidar)]
            ])

            # 꼭짓점 회전 및 중심점 이동
            rotated_corners = corners_bev @ rotation_matrix.T
            rotated_corners += center_lidar[:2] # x, y 좌표만 사용

            # ========================================================
            # ✅ 2. 네 꼭짓점을 BEV 픽셀 좌표로 변환
            # ========================================================
            pixel_corners = []
            for corner in rotated_corners:
                px = corner[0]
                py = corner[1]
                
                pixel_y = int((bev_x_range[1] - px) / resolution)
                pixel_x = int((bev_y_range[1] - py) / resolution)
                pixel_corners.append([pixel_x, pixel_y])

            gt_boxes.append({'corners': np.array(pixel_corners), 'class': class_name})
            
    return gt_boxes

def debug_single_a2d2_frame(bin_path, label_path, calib_path):
    """단일 A2D2 프레임에 대한 GT 박스 시각화 디버깅 함수"""

    # --- 설정 ---
    BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION = (0, 70.4), (-40, 40), 0.1
    
    # 1. 보정 파일 로드
    calibration = A2D2Calibration(calib_path)

    # 2. BEV 이미지 생성
    pcd = read_kitti_bin(bin_path)
    points = np.asarray(pcd.points)
    bev_image = pointcloud_to_bev(points=points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION)
    vis_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)

    # 3. GT 박스 로드 및 시각화
    gt_boxes = load_and_project_a2d2_gt_boxes(label_path, calibration, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)

    for gt in gt_boxes:
        # ========================================================
        # ✅ 3. cv2.rectangle 대신 cv2.drawContours로 그리기
        # ========================================================
        corners = gt['corners']
        # cv2.drawContours는 [꼭짓점 배열] 형태의 리스트를 입력으로 받음
        cv2.drawContours(vis_image, [corners.astype(np.int32)], -1, (0, 255, 0), 2)
        
        # 텍스트 위치는 첫 번째 꼭짓점을 기준으로 표시
        text_pos = tuple(corners[0])
        cv2.putText(vis_image, gt['class'], (text_pos[0], text_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 최종 이미지 보여주기
    cv2.imshow("A2D2 GT Visualization (Rotation Corrected)", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ========================================================
    # ✅ 이 부분이 수정되었습니다.
    # ========================================================
    
    # --- 데이터 기본 경로 설정 ---
    BASE_DIR = "/home/a/OpenPCDet/data/a2d2/training"
    
    velo_dir = os.path.join(BASE_DIR, "velodyne")
    label_dir = os.path.join(BASE_DIR, "label_2")
    calib_dir = os.path.join(BASE_DIR, "calib")
    
    # velodyne 폴더의 모든 .bin 파일을 순서대로 가져옴
    if not os.path.isdir(velo_dir):
        print(f"❗️ Error: Velodyne 디렉토리를 찾을 수 없습니다: {velo_dir}")
    else:
        bin_files = sorted(os.listdir(velo_dir))
        
        # 모든 파일을 순회하며 시각화
        for bin_filename in tqdm(bin_files, desc="Processing frames"):
            if not bin_filename.endswith('.bin'):
                continue

            file_name_base = os.path.splitext(bin_filename)[0]
            
            # 각 .bin 파일에 해당하는 .txt 파일들의 전체 경로 생성
            target_bin_path = os.path.join(velo_dir, f"{file_name_base}.bin")
            target_label_path = os.path.join(label_dir, f"{file_name_base}.txt")
            target_calib_path = os.path.join(calib_dir, f"{file_name_base}.txt")
            
            # 라벨과 보정 파일이 모두 존재할 경우에만 실행
            if os.path.exists(target_label_path) and os.path.exists(target_calib_path):
                key = debug_single_a2d2_frame(target_bin_path, target_label_path, target_calib_path)

                # 'q' 키를 누르면 루프 종료
                if key == ord('q'):
                    print("사용자에 의해 프로그램이 종료되었습니다.")
                    break
            else:
                print(f"Skipping {file_name_base}: 라벨 또는 보정 파일이 없습니다.")
    
    print("모든 파일 처리가 완료되었습니다.")