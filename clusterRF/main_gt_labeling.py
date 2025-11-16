# clustering 후 GT 라벨과 겹쳐 병합하는 유틸리티 함수

import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

# 기존 모듈에서 필요한 함수들을 가져옴
from ground_removal_onlyz import read_kitti_bin, remove_ground_by_z_axis
from propose_main import filter_points_by_z_spread
from bev_utils import pointcloud_to_bev # 포인트 클라우드를 bev 이미지로 변환
from clustering_utils import cluster_bev_image # bev 이미지에서 클러스터링 수행
from rf_gt_utils import get_a2d2_gt_boxes # a2d2 GT 박스 로드
# ========================= [수정된 부분 시작] =========================
def associate_clusters_with_gt(clusters, gt_boxes, distance_threshold=0):
    """
    각 GT 박스 내에 중심점이 포함되는 모든 클러스터를 찾아 하나의 큰 박스로 병합합니다.

    Args:
        clusters (list): (x, y, w, h) 형태의 클러스터 바운딩 박스 리스트.
        gt_boxes (list): {'corners': ndarray, 'class': str} 형태의 GT 박스 리스트.

    Returns:
        list: {'box': [x, y, w, h], 'class': str} 형태의 최종 병합 및 라벨링된 객체 리스트.
    """
    # 각 GT 박스(인덱스 기준)에 어떤 클러스터들이 속하는지 저장할 딕셔너리
    gt_to_clusters_map = defaultdict(list)

    # 1. 모든 클러스터를 순회하며 어떤 GT 박스에 속하는지 매핑
    for cluster_box in clusters:
        cx, cy, cw, ch = cluster_box
        cluster_center = (float(cx + cw / 2), float(cy + ch / 2))

        for i, gt_box in enumerate(gt_boxes):
            distance = cv2.pointPolygonTest(gt_box['corners'], cluster_center, True)
            # 클러스터의 중심점이 GT 박스(다각형) 내에 있는지 확인
            if distance >= -distance_threshold:
                gt_to_clusters_map[i].append(cluster_box)
                break


    final_labeled_objects = []
    # 2. GT 박스별로 매핑된 클러스터들을 하나로 병합
    for gt_index, associated_clusters in gt_to_clusters_map.items():
        if not associated_clusters:
            continue

        # 현재 GT 박스의 클래스 이름을 가져옴
        gt_class = gt_boxes[gt_index]['class']
        
        # 포함된 모든 클러스터를 감싸는 하나의 큰 박스를 계산
        # (x, y, w, h) -> (x1, y1, x2, y2) 형태로 변환하여 min/max 계산
        points = []
        for (x, y, w, h) in associated_clusters:
            points.append([x, y])          # top-left
            points.append([x + w, y + h])  # bottom-right
        
        points_np = np.array(points)
        min_x = np.min(points_np[:, 0])
        min_y = np.min(points_np[:, 1])
        max_x = np.max(points_np[:, 0])
        max_y = np.max(points_np[:, 1])

        # 최종 병합된 박스 (x, y, w, h)
        merged_box = [min_x, min_y, max_x - min_x, max_y - min_y]
        final_labeled_objects.append({'box': merged_box, 'class': gt_class})

    return final_labeled_objects
# ========================= [수정된 부분 끝] ===========================


def main():
    """메인 실행 함수"""
    
    # --- 🔧 경로 설정 ---ㄱ
    BASE_DIR = "/home/a/OpenPCDet/data/a2d2/training"
    BIN_DATA_DIR = os.path.join(BASE_DIR, "velodyne")
    LABEL_DIR = os.path.join(BASE_DIR, "label_2")
    CALIB_DIR = os.path.join(BASE_DIR, "calib")
    # --------------------

    # --- 🔧 파라미터 설정 ---
    GROUND_HEIGHT_THRESHOLD = -2.7
    BEV_X_RANGE = (0, 30)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.1
    MIN_CLUSTER_AREA = 1
    Z_SPREAD_THRESHOLD = 13.0
    DISTANCE_THRESHOLD_PIXELS = 5
    # ------------------------

    # 클래스별 색상 지정 (BGR 순서)
    CLASS_COLOR_MAP = {
        'Car': (0, 0, 255), 'Van': (0, 0, 255), 'Truck': (0, 0, 255), 
        'Pedestrian': (0, 0, 255), 'Cyclist': (255, 255, 0),
        'Default': (200, 200, 200)
    }

    bin_files = sorted([f for f in os.listdir(BIN_DATA_DIR) if f.endswith(".bin")])
    if not bin_files:
        print(f"Error: '{BIN_DATA_DIR}' 디렉토리에서 .bin 파일을 찾을 수 없습니다.")
        return

    for bin_name in tqdm(bin_files, desc="파일 처리 중"):
        file_base = os.path.splitext(bin_name)[0]
        bin_path = os.path.join(BIN_DATA_DIR, bin_name)
        label_path = os.path.join(LABEL_DIR, f"{file_base}.txt")
        calib_path = os.path.join(CALIB_DIR, f"{file_base}.txt")

        # 1. 포인트 클라우드 처리하여 클러스터 추출
        pcd_original = read_kitti_bin(bin_path)
        pcd_non_ground = remove_ground_by_z_axis(pcd_original, GROUND_HEIGHT_THRESHOLD)
        non_ground_points = np.asarray(pcd_non_ground.points)
        if non_ground_points.shape[0] == 0: continue

        filtered_points = filter_points_by_z_spread(
            points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION, z_spread_threshold=Z_SPREAD_THRESHOLD
        )
        if filtered_points.shape[0] == 0: continue

        bev_image = pointcloud_to_bev(
            points=filtered_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION
        )
        if bev_image is None: continue

        kernel = np.ones((3, 3), np.uint8)

        #dilated_bev_image = cv2.dilate(bev_image, kernel, iterations=2)

        closed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
        
        clusters, clustered_bev_image = cluster_bev_image(closed_bev_image, min_area_threshold=MIN_CLUSTER_AREA)
        
        # 2. GT 박스 정보 로드
        gt_boxes = get_a2d2_gt_boxes(label_path, calib_path, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)

        # 3. 클러스터와 GT를 연관시켜 라벨링 및 병합 수행
        final_objects = associate_clusters_with_gt(clusters, gt_boxes, DISTANCE_THRESHOLD_PIXELS)
        
        # 4. 결과 시각화
        # GT 박스 그리기 (녹색)
        for gt_box in gt_boxes:
            cv2.polylines(clustered_bev_image, [gt_box['corners']], isClosed=True, color=(0, 255, 0), thickness=2)

        # ========================= [수정된 부분 시작] =========================
        
        # 최종 라벨링된 객체 그리기 (빨간색으로 통일)
        color_red = (255, 0, 0)  # BGR 순서에서 빨간색
        
        for obj in final_objects:
            # box 좌표를 int로 변환
            x, y, w, h = map(int, obj['box'])
            
            # 요청대로 모든 박스를 'color_red'로 그림
            cv2.rectangle(clustered_bev_image, (x, y), (x + w, y + h), color_red, 2)
            
            # 요청대로 클래스 이름 출력(putText) 부분은 주석 처리
            class_name = obj['class']
            cv2.putText(clustered_bev_image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_red, 2)
        
        # ========================= [수정된 부분 끝] ===========================

        # 최종 라벨링된 객체 그리기 (클래스별 색상)
        for obj in final_objects:
            # box 좌표를 int로 변환해야 cv2.rectangle에서 오류가 발생하지 않습니다.
            x, y, w, h = map(int, obj['box'])
            class_name = obj['class']
            color = CLASS_COLOR_MAP.get(class_name, CLASS_COLOR_MAP['Default'])
            cv2.rectangle(clustered_bev_image, (x, y), (x + w, y + h), color, 2)
            # 1. 원본 BEV (Morphology 적용 전)
            #    bev_image는 0 또는 1 이상의 값을 가질 수 있으므로, 
            #    시각화를 위해 0보다 크면 255(흰색)로 변환합니다.
            vis_bev_image = (bev_image > 0).astype(np.uint8) * 255
            cv2.imshow("1. Original BEV (Before Morphology)", vis_bev_image)

            # 2. Morphology 적용 후
            vis_closed_bev_image = (closed_bev_image > 0).astype(np.uint8) * 255
            cv2.imshow("2. After Morphology (Close)", vis_closed_bev_image)
            cv2.putText(clustered_bev_image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(clustered_bev_image, bin_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("GT Labeling Result (GT: Green)", clustered_bev_image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()