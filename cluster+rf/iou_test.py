import os
import numpy as np
import cv2

# 필요한 헬퍼 함수들을 import 합니다.
# 이 함수들이 별도 파일에 있다면 경로가 맞는지 확인해야 합니다.
# 여기서는 사용자가 제공한 코드에 포함된 것으로 가정합니다.
from pcdet.datasets.processor.ground_removal import read_kitti_bin
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from tqdm import tqdm
from typing import List, Tuple

def calculate_iou_polygons(poly1_corners: np.ndarray, rect2: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
    """
    회전된 폴리곤(GT)과 축 정렬 사각형(클러스터) 간의 IoU를 계산합니다.
    
    Args:
        poly1_corners (np.ndarray): 첫 번째 폴리곤의 꼭짓점 배열 (N, 2). GT 박스.
        rect2 (Tuple[int, int, int, int]): 두 번째 사각형의 (x, y, w, h). 클러스터 박스.
        image_shape (Tuple[int, int]): 마스크를 생성할 이미지의 (높이, 너비).

    Returns:
        float: 계산된 IoU 값.
    """
    # 1. 두 개의 빈 마스크 생성
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)

    # 2. 각 마스크에 폴리곤과 사각형을 채워서 그리기
    cv2.fillPoly(mask1, [poly1_corners.astype(np.int32)], 255)
    x, y, w, h = rect2
    cv2.rectangle(mask2, (x, y), (x + w, y + h), 255, -1)

    # 3. 교집합(Intersection)과 합집합(Union) 계산
    intersection = np.sum(cv2.bitwise_and(mask1, mask2) == 255)
    union = np.sum(cv2.bitwise_or(mask1, mask2) == 255)

    # 4. IoU 계산 (0으로 나누는 경우 방지)
    iou = intersection / union if union > 0 else 0.0
    return iou


# ========================================================
# ✅ 1. 클러스터링 함수를 스크립트에 추가
# ========================================================
def cluster_bev_image(
    bev_image: np.ndarray, 
    min_area_threshold: int = 5
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    OpenCV의 연결 요소 레이블링을 사용해 BEV 이미지 내 객체들을 클러스터링합니다.

    Args:
        bev_image (np.ndarray): 흑백 BEV 이미지 (uint8).
        min_area_threshold (int): 클러스터로 인정할 최소 픽셀 면적. 
                                   노이즈를 제거하는 데 사용됩니다.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
            - clusters: 감지된 클러스터들의 바운딩 박스 리스트 [(x, y, w, h), ...].
            - visual_image: 바운딩 박스가 그려진 컬러 시각화 이미지.
    """
    # 1. 연결 요소 분석 수행
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bev_image, connectivity=8
    )

    # 2. 시각화를 위해 흑백 이미지를 컬러(BGR) 이미지로 변환
    visual_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
    
    clusters = []
    # 라벨 0은 배경이므로, 1부터 순회
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area_threshold:
            clusters.append((x, y, w, h))
            
            # 3. 시각화 이미지에 바운딩 박스 그리기 (빨간색으로 변경)
            cv2.rectangle(visual_image, (x, y), (x + w, y + h), (0, 0, 255), 2) # 빨간색
            cv2.putText(visual_image, f"C{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1) # 파란색 텍스트

    return clusters, visual_image


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
            
            ry_lidar = -ry_cam - np.pi / 2

            corners_bev = np.array([
                [l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2],
            ])
            rotation_matrix = np.array([
                [np.cos(ry_lidar), -np.sin(ry_lidar)],
                [np.sin(ry_lidar), np.cos(ry_lidar)]
            ])
            rotated_corners = corners_bev @ rotation_matrix.T
            rotated_corners += center_lidar[:2]

            pixel_corners = []
            for corner in rotated_corners:
                px, py = corner[0], corner[1]
                pixel_y = int((bev_x_range[1] - px) / resolution)
                pixel_x = int((bev_y_range[1] - py) / resolution)
                pixel_corners.append([pixel_x, pixel_y])

            gt_boxes.append({'corners': np.array(pixel_corners), 'class': class_name})
            
    return gt_boxes

def debug_single_a2d2_frame(bin_path, label_path, calib_path):
    """단일 A2D2 프레임에 대한 GT 박스 및 클러스터링 결과 시각화 함수"""

    # --- 설정 ---
    BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION = (0, 70.4), (-40, 40), 0.1
    IOU_THRESHOLD = 0.001
    
    # 1. 보정 파일 로드
    calibration = A2D2Calibration(calib_path)
    points = np.asarray(read_kitti_bin(bin_path).points)
    bev_image = pointcloud_to_bev(points=points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION)
    
    # 시각화를 위한 컬러 이미지 준비
    vis_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
     # Closing에 사용할 커널(kernel) 정의. (3, 3) 또는 (5, 5) 시도
    kernel = np.ones((10, 10), np.uint8)

        # OpenCV의 morphologyEx 함수를 사용해 Closing 연산 수행
    closed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
    clusters, vis_image = cluster_bev_image(closed_bev_image, min_area_threshold=15)

    # GT 박스와 클러스터링 결과 가져오기
    gt_boxes = load_and_project_a2d2_gt_boxes(label_path, calibration, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)
    clusters, _ = cluster_bev_image(bev_image, min_area_threshold=15)

    # ========================================================
    # ✅ NEW: 매칭 로직 (GT와 클러스터 간 IoU 계산)
    # ========================================================
    # 각 클러스터에 가장 잘 맞는 GT 정보를 저장할 리스트
    cluster_matches = [{'label': None, 'iou': 0.0} for _ in clusters]

    for gt_box in gt_boxes:
        gt_corners = gt_box['corners']
        gt_label = gt_box['class']

        for i, cluster_rect in enumerate(clusters):
            iou = calculate_iou_polygons(gt_corners, cluster_rect, bev_image.shape)

            # 현재 클러스터에 대한 기존 매칭보다 IoU가 높으면 정보 업데이트
            if iou > cluster_matches[i]['iou']:
                cluster_matches[i]['iou'] = iou
                cluster_matches[i]['label'] = gt_label

    for i, cluster_rect in enumerate(clusters):
        x, y, w, h = cluster_rect
        match_info = cluster_matches[i]
        
        # 기본 박스 색상 (빨간색)
        box_color = (0, 0, 255)
        
        # IoU 임계값을 넘는 매칭이 있는 경우
        if match_info['iou'] >= IOU_THRESHOLD:
            # 매칭 성공 시 박스 색상 변경 (자주색) 및 라벨 표시
            box_color = (255, 0, 255) 
            label = match_info['label']
            iou_text = f"{match_info['iou']:.2f}"
            
            # GT 라벨과 IoU 값을 함께 표시
            display_text = f"{label} ({iou_text})"
            cv2.putText(vis_image, display_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), box_color, 2)

    # ========================================================
    # ✅ 2. BEV 이미지로 클러스터링 수행 및 결과 이미지 가져오기
    # ========================================================
    for gt in gt_boxes:
        # ========================================================
        # ✅ 3. 클러스터링 이미지 위에 GT 박스(초록색) 겹쳐 그리기
        # ========================================================
        corners = gt['corners']
        # cv2.drawContours는 [꼭짓점 배열] 형태의 리스트를 입력으로 받음
        cv2.drawContours(vis_image, [corners.astype(np.int32)], -1, (0, 255, 0), 2) # 초록색
        
        # 텍스트 위치는 첫 번째 꼭짓점을 기준으로 표시
        text_pos = tuple(corners[0])
        cv2.putText(vis_image, gt['class'], (text_pos[0], text_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 최종 이미지 보여주기
    cv2.imshow("A2D2: GT(Green), Cluster(Red/Purple), Match(Purple)", vis_image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key
    # ========================================================
    # ✅ 4. 'q' 키 입력을 반환하여 메인 루프에서 종료할 수 있도록 수정
    # ========================================================
    return key


if __name__ == '__main__':
    BASE_DIR = "/home/a/OpenPCDet/data/a2d2/training"
    
    velo_dir = os.path.join(BASE_DIR, "velodyne")
    label_dir = os.path.join(BASE_DIR, "label_2")
    calib_dir = os.path.join(BASE_DIR, "calib")
    
    if not os.path.isdir(velo_dir):
        print(f"❗️ Error: Velodyne 디렉토리를 찾을 수 없습니다: {velo_dir}")
    else:
        bin_files = sorted(os.listdir(velo_dir))
        
        for bin_filename in tqdm(bin_files, desc="Processing frames"):
            if not bin_filename.endswith('.bin'):
                continue

            file_name_base = os.path.splitext(bin_filename)[0]
            
            target_bin_path = os.path.join(velo_dir, f"{file_name_base}.bin")
            target_label_path = os.path.join(label_dir, f"{file_name_base}.txt")
            target_calib_path = os.path.join(calib_dir, f"{file_name_base}.txt")
            
            if os.path.exists(target_label_path) and os.path.exists(target_calib_path):
                # debug 함수가 key를 반환하도록 수정되었으므로 변수에 저장
                key = debug_single_a2d2_frame(target_bin_path, target_label_path, target_calib_path)

                if key == ord('q'):
                    print("사용자에 의해 프로그램이 종료되었습니다.")
                    break
            else:
                print(f"Skipping {file_name_base}: 라벨 또는 보정 파일이 없습니다.")
    
    print("모든 파일 처리가 완료되었습니다.")