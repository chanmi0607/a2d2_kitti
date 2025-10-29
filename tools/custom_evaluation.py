import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_utils, calibration_kitti


# --- 설정값 (사용자 환경에 맞게 수정) ---
# 1. 생성된 예측 결과 pkl 파일 경로
PRED_INFO_PATH = 'result.pkl' # 👈 본인 경로로 수정!
# 2. Ground-Truth 라벨이 있는 디렉터리 경로
LABEL_PATH = 'data/a2d2/training/label_2' # 👈 본인 경로로 수정!
CALIB_PATH = 'data/a2d2/training/calib'

VAL_FILE_PATH = 'data/a2d2/ImageSets/val.txt'
# 3. 평가 파라미터
#CLASS_NAMES = ['Car', 'Truck', 'UtilityVehicle', 'Cyclist', 'Bicycle', 'MotorBiker', 'Bus', 'Trailer', 'Pedestrian']
CLASS_NAMES = ['Car', 'Truck', 'UtilityVehicle', 'Cyclist', 'Bus', 'Trailer', 'Pedestrian']
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.3
# 4. 거리 제한 (이 거리 이내의 객체만 평가에 포함)
DISTANCE_THRESHOLD = 30.0
# ------------------------------------

def get_label_info(label_path, calib_path, frame_id):
    """
    OpenPCDet 데이터로딩 코드에서 발견된 '진짜' 변환 규칙을 적용합니다.
    """
    label_file = Path(label_path) / f'{frame_id}.txt'
    calib_file = Path(calib_path) / f'{frame_id}.txt'
    
    if not label_file.exists() or not calib_file.exists():
        return np.array([]), np.array([])
    
    calib = calibration_kitti.Calibration(calib_file)
    
    gt_boxes_lidar_final = []
    gt_names = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            class_name = parts[0]
            
            if class_name in CLASS_NAMES and class_name != 'DontCare':
                gt_names.append(class_name)

                # 1. 라벨에서 정보 파싱
                h, w, l = [float(x) for x in parts[8:11]]
                x_cam, y_cam, z_cam = [float(x) for x in parts[11:14]]
                ry_cam = float(parts[14])
                
                # 2. 위치 변환 (규칙 1)
                loc_cam_rect = np.array([[x_cam, y_cam, z_cam]])
                loc_lidar = calib.rect_to_lidar(loc_cam_rect)[0]
                
                # 3. 높이/위치 보정 (규칙 2)
                loc_lidar[1] -= h / 3.0
                
                # 4. 회전 변환 (규칙 3)
                ry_lidar_correct = -(ry_cam + np.pi / 2)
                
                # 5. 최종 박스 조합: [x, y, z, l, w, h, yaw]
                final_box = np.concatenate([loc_lidar, [l, w, h], [ry_lidar_correct]])
                gt_boxes_lidar_final.append(final_box)

    if not gt_boxes_lidar_final:
        return np.array([]), np.array([])

    return np.array(gt_boxes_lidar_final), np.array(gt_names)

def load_val_frames(file_path):
    """
    val.txt 파일에서 평가할 프레임 ID 목록을 읽어옵니다.
    """
    with open(file_path, 'r') as f:
        frame_ids = {line.strip() for line in f if line.strip()}
    return frame_ids

def main():
    with open(PRED_INFO_PATH, 'rb') as f:
        pred_infos = pickle.load(f)
    print(f"총 {len(pred_infos)}개의 프레임에 대한 예측 결과를 로드했습니다.")

    val_frame_ids = load_val_frames(VAL_FILE_PATH)
    print(f"평가 대상: {len(val_frame_ids)}개의 검증 프레임 ID를 로드했습니다. ")

    stats_by_class = {
        name: {'TP': 0, 'FP': 0, 'FN': 0} for name in CLASS_NAMES
    }

    filtered_preds = [info for info in pred_infos if info['frame_id'] in val_frame_ids]
    print(f"로드된 예측 결과 중 {len(filtered_preds)}개가 검증 세트에 포함됩니다.")

    for pred_info in tqdm(filtered_preds, desc="프레임 평가 중"):
        frame_id = pred_info['frame_id']
        
        gt_boxes, gt_names = get_label_info(LABEL_PATH, CALIB_PATH, frame_id)
        
        pred_boxes_all = pred_info['boxes_lidar']
        pred_scores_all = pred_info['score']
        pred_names_all = pred_info['name']

        if gt_boxes.shape[0] > 0:
            gt_distances = np.linalg.norm(gt_boxes[:, :2], axis=1) # x,y 좌표로 거리 계산
            gt_mask = gt_distances <= DISTANCE_THRESHOLD
            gt_boxes = gt_boxes[gt_mask]
            gt_names = gt_names[gt_mask]
        
        # Confidence Score 임계값 적용
        mask = pred_scores_all >= CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes_all[mask]
        pred_names = pred_names_all[mask]

        if pred_boxes.shape[0] > 0:
            pred_distances = np.linalg.norm(pred_boxes[:, :2], axis=1) # x,y 좌표로 거리 계산
            pred_mask = pred_distances <= DISTANCE_THRESHOLD
            pred_boxes = pred_boxes[pred_mask]
            pred_names = pred_names[pred_mask]

        # 클래스별로 TP, FP, FN 계산
        for class_name in CLASS_NAMES:
            gt_mask = (gt_names == class_name)
            pred_mask = (pred_names == class_name)
            
            class_gt_boxes = gt_boxes[gt_mask]
            class_pred_boxes = pred_boxes[pred_mask]

            num_gts = class_gt_boxes.shape[0]
            num_preds = class_pred_boxes.shape[0]

            if num_gts == 0:
                stats_by_class[class_name]['FP'] += num_preds
                continue
            
            if num_preds == 0:
                stats_by_class[class_name]['FN'] += num_gts
                continue

            # IoU 계산
            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                torch.from_numpy(class_pred_boxes).float().cuda(),
                torch.from_numpy(class_gt_boxes).float().cuda()
            ).cpu().numpy()

            # 매칭 및 TP, FP, FN 계산 (Greedy Matching)
            matched_gt_indices = np.zeros(num_gts, dtype=bool)
            tp_count_for_frame = 0

            for i in range(num_preds):
                # 각 예측에 대해 가장 높은 IoU를 가진 GT 찾기
                max_iou_for_pred = -1
                best_gt_idx = -1
                
                # 아직 매칭되지 않은 GT 중에서만 탐색
                for j in range(num_gts):
                    if not matched_gt_indices[j] and iou_matrix[i, j] > max_iou_for_pred:
                        max_iou_for_pred = iou_matrix[i, j]
                        best_gt_idx = j

                # IoU가 임계값을 넘으면 TP로 처리하고 해당 GT를 '매칭됨'으로 표시
                if max_iou_for_pred >= IOU_THRESHOLD:
                    tp_count_for_frame += 1
                    matched_gt_indices[best_gt_idx] = True
            
            stats_by_class[class_name]['TP'] += tp_count_for_frame
            stats_by_class[class_name]['FP'] += num_preds - tp_count_for_frame
            stats_by_class[class_name]['FN'] += num_gts - np.sum(matched_gt_indices)

    # 최종 결과 계산 및 출력
    print("\n" + "="*80)
    print("--- 최종 평가 결과 ---")
    print(f"IoU 임계값: {IOU_THRESHOLD}, Confidence 점수 임계값: {CONFIDENCE_THRESHOLD}\n")
    print(f"거리 제한: {DISTANCE_THRESHOLD}m 이내")
    print(f"{'클래스':<16} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-" * 80)

    for class_name, stats in stats_by_class.items():
        tp = stats['TP']
        fp = stats['FP']
        fn = stats['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{class_name:<16} | {(precision * 100):11.4f}% | {(recall * 100):11.2f}% | {f1_score:10.4f} | {tp:6} | {fp:6} | {fn:6}")
    
    print("="*80)

if __name__ == '__main__':
    main()