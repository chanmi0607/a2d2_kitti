# full_pipeline_eval.py

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# --- 기존 모듈들을 모두 import 합니다 ---
# (파일 실제 위치에 맞게 경로를 다시 한번 확인해주세요)
from pcdet.datasets.processor.ground_removal import remove_ground_open3d
from pcdet.datasets.processor.ground_removal import read_kitti_bin as read_bin_to_pcd # 이름 충돌을 피하기 위해 별칭 사용
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from pcdet.models.dense_heads.clustering_utils import cluster_bev_image
from pcdet.models.dense_heads.classifier_utils import load_model, extract_features

# --- 헬퍼 함수 ---

def calculate_iou(boxA, boxB):
    """두 바운딩 박스(x,y,w,h)의 IoU를 계산합니다."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def project_3d_box_to_bev(obj_info, bev_x_range, bev_y_range, resolution):
    """JSON의 3D 박스를 BEV 2D 박스로 변환합니다."""
    center_3d = obj_info.get('center')
    size_3d = obj_info.get('size')
    
    if center_3d is None or size_3d is None:
        return None
        
    l, w, h = size_3d
    cx, cy, _ = center_3d
    
    x_min_world, x_max_world = cx - l/2, cx + l/2
    y_min_world, y_max_world = cy - w/2, cy + w/2

    px_y1 = int((bev_x_range[1] - x_max_world) / resolution)
    px_y2 = int((bev_x_range[1] - x_min_world) / resolution)
    px_x1 = int((bev_y_range[1] - y_max_world) / resolution)
    px_x2 = int((bev_y_range[1] - y_min_world) / resolution)
    
    return (px_x1, px_y1, px_x2 - px_x1, px_y2 - px_y1)

# --- 메인 평가 함수 ---

def full_evaluation():
    # ========================== 설정 ==========================
    BIN_DATA_DIR = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    JSON_BASE_DIR = "/home/a/OpenPCDet_real/data/a2d2/camera_lidar_semantic_bboxes"
    
    MODEL_PATH = "car_detector.joblib"
    MAPPING_PATH = "class_mapping.json"
    AVERAGES_PATH = "class_averages.json" # 특징 추출에 필요한 평균값 파일

    IOU_THRESHOLD = 0.1
    VISUALIZE = True # 시각화 기능 활성화
    
    BEV_X_RANGE = (0, 70.4)
    BEV_Y_RANGE = (-40, 40)
    BEV_RESOLUTION = 0.1
    MIN_CLUSTER_AREA = 10
    # ========================================================
    
    # 모델 및 클래스 정보 로드
    classifier = load_model(MODEL_PATH)
    try:
        with open(MAPPING_PATH, 'r') as f: class_mapping = json.load(f)
        with open(AVERAGES_PATH, 'r') as f: class_averages = json.load(f)
        reverse_mapping = {str(v): k for k, v in class_mapping.items()}
        num_classes = len(class_mapping)
    except FileNotFoundError as e:
        print(f"필수 파일을 찾을 수 없습니다: {e}. train_model.py를 먼저 실행했는지 확인하세요.")
        return

    stats = {'tp': np.zeros(num_classes), 'fp': np.zeros(num_classes), 'fn': np.zeros(num_classes)}
    bin_files = sorted([f for f in os.listdir(BIN_DATA_DIR) if f.endswith(".bin")])
    
    for bin_name in tqdm(bin_files, desc="전체 파이프라인 평가 중"):
        # --- 1. 예측 박스 생성 (main.py 로직과 동일) ---
        bin_path = os.path.join(BIN_DATA_DIR, bin_name)
        pcd = read_bin_to_pcd(bin_path)
        original_points = np.asarray(pcd.points)
        
        non_ground_points = remove_ground_open3d(original_points)
        if non_ground_points.shape[0] == 0: continue
            
        bev_image = pointcloud_to_bev(points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION)
        if bev_image is None: continue

        kernel = np.ones((3, 3), np.uint8)
        processed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
            
        clusters, _ = cluster_bev_image(processed_bev_image, min_area_threshold=MIN_CLUSTER_AREA)
        
        predicted_boxes = []
        if clusters:
            features = extract_features(clusters, BEV_RESOLUTION, class_averages, num_classes, non_ground_points, BEV_X_RANGE, BEV_Y_RANGE)
            if features.shape[0] > 0:
                predictions = classifier.predict(features)
                for i, box in enumerate(clusters):
                    predicted_boxes.append({'box': box, 'class_id': predictions[i], 'status': 'fp'}) # status 기본값 fp

        # --- 2. 정답(GT) 박스 로드 ---
        try:
            scene = bin_name.split('_')[0]
            filename_base = os.path.splitext(bin_name)[0]
            json_path = os.path.join(JSON_BASE_DIR, scene, 'label3D', 'cam_front_center', f'{filename_base.replace("_velodyne", "_label3D_frontcenter")}.json')
            with open(json_path, 'r') as f: data = json.load(f)
        except (FileNotFoundError, IndexError):
            continue
            
        gt_boxes = []
        for obj_info in data.values():
            if not isinstance(obj_info, dict): continue
            class_label_str = obj_info.get('class')
            # 클래스 통합
            if class_label_str in ['VanSUV', 'EmergencyVehicle', 'CaravanTransporter']: class_label_str = 'Car'
            if class_label_str == 'Motorcycle': class_label_str = 'MotorBiker'

            if class_label_str in class_mapping:
                box_2d = project_3d_box_to_bev(obj_info, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)
                if box_2d:
                    gt_boxes.append({'box': box_2d, 'class_id': class_mapping[class_label_str], 'used': False})

        # --- 3. 예측 박스와 정답 박스 매칭 ---
        for pred in predicted_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if pred['class_id'] == gt['class_id'] and not gt['used']:
                    iou = calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou > IOU_THRESHOLD:
                stats['tp'][pred['class_id']] += 1
                gt_boxes[best_gt_idx]['used'] = True
                pred['status'] = 'tp'
            else:
                stats['fp'][pred['class_id']] += 1
        
        for gt in gt_boxes:
            if not gt['used']:
                stats['fn'][gt['class_id']] += 1

        # --- 4. 시각화 ---
        if VISUALIZE:
            vis_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
            # 정답(GT) 박스 그리기 (초록색, 얇은 선)
            for gt in gt_boxes:
                x, y, w, h = gt['box']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 예측(Predicted) 박스 그리기 (TP: 파랑, FP: 빨강, 굵은 선)
            for pred in predicted_boxes:
                x, y, w, h = pred['box']
                class_name = reverse_mapping.get(str(pred['class_id']), "Unknown")
                color = (255, 0, 0) if pred['status'] == 'tp' else (0, 0, 255)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(f"IoU Visualization - {bin_name}", vis_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.destroyWindow(f"IoU Visualization - {bin_name}")

    # --- 5. 최종 리포트 계산 및 출력 ---
    print("\n" + "="*50)
    print("📊 전체 파이프라인 성능 평가 리포트 (실전 점수)")
    print("="*50)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*50)

    class_names = list(class_mapping.keys())
    for i, class_name in enumerate(class_names):
        tp, fp, fn = stats['tp'][i], stats['fp'][i], stats['fn'][i]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{class_name:<20} {precision:<10.2f} {recall:<10.2f} {f1_score:<10.2f}")

if __name__ == '__main__':
    full_evaluation()