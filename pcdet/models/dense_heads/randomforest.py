import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm
from collections import Counter
import cv2
from multiprocessing import Pool, cpu_count

# --- 필요한 모듈만 import ---
from pcdet.datasets.processor.ground_removal import remove_ground_open3d, read_kitti_bin
from pcdet.models.detectors.bev_utils import pointcloud_to_bev
from pcdet.models.dense_heads.clustering_utils import cluster_bev_image
from rf_gt_utils import get_a2d2_gt_boxes # KITTI GT 로더 (이전 파일 이름 그대로 사용)

# --- 헬퍼 함수 정의 ---
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

def extract_features_from_clusters(clusters, resolution, non_ground_points, bev_x_range, bev_y_range):
    """클러스터에서 3D 통계 특징을 포함한 벡터를 추출합니다."""
    all_features, valid_clusters = [], []
    for (x, y, w_pixels, h_pixels) in clusters:
        x_max_world = bev_x_range[1] - (y * resolution)
        x_min_world = bev_x_range[1] - ((y + h_pixels) * resolution)
        y_max_world = bev_y_range[1] - (x * resolution)
        y_min_world = bev_y_range[1] - ((x + w_pixels) * resolution)
        mask = np.where(
            (non_ground_points[:, 0] >= x_min_world) & (non_ground_points[:, 0] < x_max_world) &
            (non_ground_points[:, 1] >= y_min_world) & (non_ground_points[:, 1] < y_max_world)
        )
        points_in_box = non_ground_points[mask]
        if len(points_in_box) < 2: continue
        
        w, l = w_pixels * resolution, h_pixels * resolution
        h = np.max(points_in_box[:, 2]) - np.min(points_in_box[:, 2])
        z_std = np.std(points_in_box[:, 2])
        
        feature_vector = [w, l, len(points_in_box), h, z_std]
        all_features.append(feature_vector)
        valid_clusters.append((x, y, w_pixels, h_pixels))
    return valid_clusters, np.array(all_features) if all_features else np.array([])

# ========================== 설정 (전역 변수) ==========================
BASE_DIR = "/home/a/OpenPCDet/data/a2d2/training"
BIN_DATA_DIR = os.path.join(BASE_DIR, "velodyne")
LABEL_DIR = os.path.join(BASE_DIR, "label_2")
CALIB_DIR = os.path.join(BASE_DIR, "calib")

MODEL_SAVE_PATH = "advanced_model.joblib"
MAPPING_SAVE_PATH = "advanced_class_mapping.json"
IOU_THRESHOLD = 0.3
VISUALIZE_MATCHING = True
VISUALIZATION_OUTPUT_DIR = "training_visualizations_a2d2"

CLASS_MAPPING = {
    'Car': 0, 'Pedestrian': 1, 'Truck': 2, 'Cyclist': 3, 'Bicycle': 4,
    'Bus': 5, 'UtilityVehicle': 6, 'Trailer': 7, 'MotorBiker': 8, 'Background': 9
}
BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION, MIN_CLUSTER_AREA = (0, 70.4), (-40, 40), 0.1, 15
# ======================================================================

def process_file(bin_name):
    """단일 .bin 파일을 처리하는 작업자 함수 (KITTI 기준)"""
    try:
        # --- 0. 파일 경로 준비 ---
        file_name_base = os.path.splitext(bin_name)[0]
        bin_path = os.path.join(BIN_DATA_DIR, f"{file_name_base}.bin")
        label_path = os.path.join(LABEL_DIR, f"{file_name_base}.txt")
        calib_path = os.path.join(CALIB_DIR, f"{file_name_base}.txt")

        if not (os.path.exists(label_path) and os.path.exists(calib_path)): return None

        # --- 1. 파이프라인 실행: 클러스터 후보 생성 ---
        pcd = read_kitti_bin(bin_path)
        original_points = np.asarray(pcd.points)
        if original_points.shape[0] == 0: return None

        non_ground_points = remove_ground_open3d(original_points)
        if non_ground_points.shape[0] == 0: return None
        
        bev_image = pointcloud_to_bev(points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, resolution=BEV_RESOLUTION)
        if bev_image is None: return None

        kernel = np.ones((3, 3), np.uint8)
        processed_bev_image = cv2.morphologyEx(bev_image, cv2.MORPH_CLOSE, kernel)
        
        clusters, _ = cluster_bev_image(processed_bev_image, min_area_threshold=MIN_CLUSTER_AREA)
        if not clusters: return None

        # --- 2. 정답(GT) 박스 로드 (rf_gt_utils 사용) ---
        gt_boxes_with_corners = get_a2d2_gt_boxes(label_path, calib_path, BEV_X_RANGE, BEV_Y_RANGE, BEV_RESOLUTION)
        if not gt_boxes_with_corners: return None
        
        # --- 3. 클러스터로부터 특징 추출 ---
        valid_clusters, features_from_clusters = extract_features_from_clusters(
            clusters, BEV_RESOLUTION, non_ground_points, BEV_X_RANGE, BEV_Y_RANGE
        )
        if features_from_clusters.shape[0] == 0: return None

        # --- 4. IoU 기반 자동 라벨링 ---
        file_features, file_labels, clusters_for_vis = [], [], []
        for i, cluster_box in enumerate(valid_clusters):
            best_iou, best_gt_class_id = 0.0, -1
            for gt in gt_boxes_with_corners:
                gt_aligned_box = cv2.boundingRect(gt['corners'])
                iou = calculate_iou(cluster_box, gt_aligned_box)
                if iou > best_iou:
                    gt_class_str = gt['class']
                    if gt_class_str in ['Van']: gt_class_str = 'Car' # KITTI 클래스 통합
                    
                    if gt_class_str in CLASS_MAPPING:
                        best_iou, best_gt_class_id = iou, CLASS_MAPPING[gt_class_str]

            label_to_assign = CLASS_MAPPING['Background']
            if best_iou >= IOU_THRESHOLD:
                label_to_assign = best_gt_class_id
            
            file_features.append(features_from_clusters[i])
            file_labels.append(label_to_assign)
            clusters_for_vis.append({'box': cluster_box, 'label': label_to_assign})

        # --- 5. 최종 결과 반환 ---
        return {
            "bin_name": bin_name, "bev_image": processed_bev_image,
            "gt_boxes": gt_boxes_with_corners, "clusters_for_vis": clusters_for_vis,
            "features": file_features, "labels": file_labels
        }
    except Exception:
        return None


def train_classifier():
    if VISUALIZE_MATCHING:
        os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}

    all_features, all_labels = [], []
    bin_files = sorted([f for f in os.listdir(BIN_DATA_DIR) if f.endswith(".bin")])

    # --- 병렬 처리 ---
    num_processes = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"{num_processes}개의 프로세스를 사용하여 병렬 처리를 시작합니다...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, bin_files), total=len(bin_files), desc="학습 데이터 생성 중"))

    # --- 결과 취합 & 시각화 ---
    print("\n결과 취합 및 시각화 이미지 저장 중...")
    for result in tqdm(results, desc="결과 처리 중"):
        if result is None: continue
        all_features.extend(result["features"])
        all_labels.extend(result["labels"])

        if VISUALIZE_MATCHING:
            vis_image = cv2.cvtColor(result["bev_image"], cv2.COLOR_GRAY2BGR)
            for gt in result["gt_boxes"]:
                # 회전된 GT 박스 그리기
                cv2.drawContours(vis_image, [gt['corners'].astype(np.int32)], -1, (0, 255, 0), 1)
            for cluster in result["clusters_for_vis"]:
                x, y, w, h = cluster['box']
                label_int = cluster['label']
                class_name = reverse_mapping.get(label_int, "Unknown")
                color = (0, 0, 255) if class_name == 'Background' else (255, 0, 0)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            save_path = os.path.join(VISUALIZATION_OUTPUT_DIR, f"{os.path.splitext(result['bin_name'])[0]}.png")
            cv2.imwrite(save_path, vis_image)

    # --- 모델 학습 및 평가 ---
    if not all_features:
        print("❗️ Error: 생성된 학습 데이터가 없습니다. 경로 및 파라미터를 확인하세요.")
        return
        
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print("\n" + "="*40)
    print("클래스별 데이터 개수:")
    for label_int, count in sorted(Counter(y).items()):
        print(f"- {reverse_mapping.get(label_int, 'Unknown')}: {count}개")
    print("="*40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nRandomForest 모델 학습을 시작합니다...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=10)
    model.fit(X_train, y_train)
    print("✅ 모델 학습 완료.")

    y_pred = model.predict(X_test)
    print("\n" + "="*40)
    print("📊 모델 성능 평가 리포트:")
    print(classification_report(y_test, y_pred, labels=list(CLASS_MAPPING.values()), target_names=list(CLASS_MAPPING.keys()), zero_division=0))
    print("="*40)

    # --- 모델 및 매핑 정보 저장 ---
    joblib.dump(model, MODEL_SAVE_PATH)
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump(CLASS_MAPPING, f, indent=4)
        
    print(f"✅ 모델 저장 완료: {MODEL_SAVE_PATH}")
    print(f"✅ 클래스 맵 저장 완료: {MAPPING_SAVE_PATH}")


if __name__ == '__main__':
    # Python의 multiprocessing을 안전하게 사용하기 위해 필수적인 구문
    train_classifier()