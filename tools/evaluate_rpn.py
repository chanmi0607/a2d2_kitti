import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# =================================================================
# ⭐️ 1. 설정 (사용자 환경에 맞게 수정)
# =================================================================

# 1. 예측 파일 (demo.py가 생성한 CSV)
PRED_FILE = 'data/a2d2/rpn_pred_all.csv' # ⭐️ pred_all.csv 경로 확인

# 2. 정답(GT) 라벨이 있는 폴더 (프레임별 .txt 파일이 있는 곳)
GT_LABEL_DIR = 'data/a2d2/new_label/label_new_val' # ⭐️ GT .txt 폴더 경로 확인

# 3. ⭐️ [수정됨] 클래스 이름 매핑 (pred_all.csv의 숫자 라벨을 문자열로 변환)
# (사용자가 제공한 CLASS_NAMES 리스트를 1-based 인덱스로 매핑)
# CLASS_MAPPING = {
#     1: 'Car',
#     2: 'Truck',
#     3: 'UtilityVehicle',
#     4: 'Cyclist',
#     5: 'Bus',
#     6: 'Trailer',
#     7: 'Pedestrian'
# }

# 4. ⭐️ [수정됨] 평가에 사용할 클래스 이름 목록
# (GT TXT 파일에 있는 이름과 정확히 일치해야 함)
CLASS_NAMES = ['Car','Truck','UtilityVehicle','Cyclist','Bus','Trailer','Pedestrian']

# 5. ⭐️ [수정됨] 클래스별 3D IoU 임계값
# (차량: 0.7, 보행자/자전거: 0.5로 가정)
IOU_THRESHOLDS = {
    'Car': 0.3,
    'Truck': 0.3,
    'UtilityVehicle': 0.3,
    'Cyclist': 0.3,
    'Bus': 0.3,
    'Trailer': 0.3,
    'Pedestrian': 0.3
}

# 6. 예측 박스로 인정할 최소 점수(confidence)
SCORE_THRESHOLD = 0.1

# 7. 박스 컬럼 이름 (CSV와 TXT에서 공통으로 사용)
BOX_COLS = ['x', 'y', 'z', 'l', 'w', 'h', 'yaw']

# 8. ⭐️ GT .txt 파일의 컬럼 인덱스 (가장 중요)
# "Truck 1.0000 25.7026 -0.8100 -0.9370 5.4100 3.0600 3.7800 -0.0008 ..."
# 위 형식을 기준으로 한 인덱스입니다. (000000168.txt 기준)
GT_COL_MAP = {
    'label': 0,
    'x': 2,
    'y': 3,
    'z': 4,
    'l': 5,
    'w': 6,
    'h': 7,
    'yaw': 8
}
# (만약 .txt 파일 형식이 다르면 이 인덱스를 수정해야 합니다)

# =================================================================

def calculate_metrics_for_frame(pred_boxes_tensor, gt_boxes_tensor, iou_threshold):
    """
    단일 프레임, 단일 클래스에 대해 TP, FP, FN을 계산합니다.
    """
    num_pred = pred_boxes_tensor.shape[0]
    num_gt = gt_boxes_tensor.shape[0]

    if num_pred == 0 and num_gt == 0:
        return 0, 0, 0
    if num_pred == 0:
        return 0, 0, num_gt # TP, FP, FN
    if num_gt == 0:
        return 0, num_pred, 0 # TP, FP, FN

    # (Num_Pred, Num_GT) 형태의 3D IoU 행렬 계산
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
        pred_boxes_tensor, 
        gt_boxes_tensor
    )
    
    # 1. 각 예측 박스에 대해 가장 IoU가 높은 GT 박스 찾기
    pred_to_gt_max_iou, _ = torch.max(iou_matrix, dim=1)
    
    # 2. 각 GT 박스에 대해 가장 IoU가 높은 예측 박스 찾기
    gt_to_pred_max_iou, _ = torch.max(iou_matrix, dim=0)

    # 3. TP (GT 박스 기준, IoU 임계값 통과)
    tp_mask = gt_to_pred_max_iou >= iou_threshold
    TP = torch.sum(tp_mask).item()
    
    # 4. FP (예측 박스 기준, IoU 임계값 미만)
    fp_mask = pred_to_gt_max_iou < iou_threshold
    FP = torch.sum(fp_mask).item()

    # 5. FN (TP가 아닌 GT 박스)
    FN = num_gt - TP

    return TP, FP, FN

def load_gt_boxes_from_txt(txt_file_path):
    """
    단일 .txt 파일에서 GT 박스를 로드하여 DataFrame으로 반환합니다.
    """
    gt_boxes = []
    if not txt_file_path.exists():
        return pd.DataFrame(columns=['label'] + BOX_COLS) # 빈 프레임

    with open(txt_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if len(parts) < max(GT_COL_MAP.values()) + 1:
                continue # 컬럼 수가 부족하면 스킵
            
            try:
                label = parts[GT_COL_MAP['label']]
                
                # ⭐️ 평가할 클래스 목록에 없으면 무시 (예: 'DontCare')
                if label not in CLASS_NAMES:
                    continue
                
                x = float(parts[GT_COL_MAP['x']])
                y = float(parts[GT_COL_MAP['y']])
                z = float(parts[GT_COL_MAP['z']])
                l = float(parts[GT_COL_MAP['l']])
                w = float(parts[GT_COL_MAP['w']])
                h = float(parts[GT_COL_MAP['h']])
                yaw = float(parts[GT_COL_MAP['yaw']])
                
                gt_boxes.append([label, x, y, z, l, w, h, yaw])
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in {txt_file_path.name}: {line}")
                continue
                
    return pd.DataFrame(gt_boxes, columns=['label'] + BOX_COLS)


def main():
    print(f"Loading predictions from {PRED_FILE}...")
    try:
        pred_df = pd.read_csv(PRED_FILE)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {PRED_FILE}")
        return

    gt_dir = Path(GT_LABEL_DIR)
    if not gt_dir.exists():
        print(f"Error: Ground truth directory not found at {GT_LABEL_DIR}")
        return

    # ⭐️ [중요] pred_all.csv의 숫자 라벨을 CLASS_MAPPING을 이용해 문자열로 변환
    # if pd.api.types.is_numeric_dtype(pred_df['label']):
    #     print(f"Converting numeric prediction labels ({pred_df['label'].unique()}) to string names...")
    #     pred_df['label'] = pred_df['label'].map(CLASS_MAPPING)
    #     pred_df.dropna(subset=['label'], inplace=True)
    #     print(f"Labels after mapping: {pred_df['label'].unique()}")
    print(f"Prediction labels are already strings (e.g., '{pred_df['label'].iloc[0]}'). Skipping mapping.")
    # ⭐️ [중요] pred_all.csv의 frame_id를 문자열로 변환 (예: 168 -> '000000168')
    # A2D2는 9자리 (000000168), KITTI는 6자리 (000168) 입니다.
    # GT .txt 파일 이름(예: 000000168.txt)의 자릿수와 동일하게 맞춰야 합니다.
    if pd.api.types.is_numeric_dtype(pred_df['frame_id']):
        print("Converting numeric frame_id to zero-padded strings (e.g., 168 -> '000000168')")
        
        # ⭐️ A2D2 (9자리) 기준
        pred_df['frame_id'] = pred_df['frame_id'].apply(lambda x: f'{x:09d}')
        
        # ⭐️ 만약 KITTI (6자리) 기준이라면 위 라인을 주석처리하고 아래 라인을 사용하세요.
        # pred_df['frame_id'] = pred_df['frame_id'].apply(lambda x: f'{x:06d}')


    # 평가할 전체 프레임 ID 목록 (예측 CSV에 있는 프레임 기준)
    all_frame_ids = pred_df['frame_id'].unique()
    all_frame_ids.sort()
    
    print(f"Found {len(all_frame_ids)} unique frames in Prediction CSV to evaluate.")

    # 클래스별로 TP, FP, FN 누적
    total_tp = {class_name: 0 for class_name in CLASS_NAMES}
    total_fp = {class_name: 0 for class_name in CLASS_NAMES}
    total_fn = {class_name: 0 for class_name in CLASS_NAMES}

    # 모든 예측 박스에 대해 신뢰도 필터링
    pred_df = pred_df[pred_df['score'] >= SCORE_THRESHOLD]
    
    print(f"\nStarting evaluation (Score Thresh >= {SCORE_THRESHOLD})...")

    # 모든 프레임 순회
    for frame_id in tqdm(all_frame_ids, desc="Evaluating frames", ncols=100):
        
        # 1. 현재 프레임의 예측 데이터 필터링
        preds_for_frame = pred_df[pred_df['frame_id'] == frame_id]
        
        # 2. ⭐️ 현재 프레임의 GT .txt 파일 로드
        gt_file_path = gt_dir / f"{frame_id}.txt"
        gts_for_frame = load_gt_boxes_from_txt(gt_file_path)

        # 클래스별로 순회하며 메트릭 계산
        for class_name in CLASS_NAMES:
            iou_thresh = IOU_THRESHOLDS.get(class_name, 0.5) 
            
            # 현재 클래스에 해당하는 박스들만 필터링
            class_preds = preds_for_frame[preds_for_frame['label'] == class_name]
            class_gts = gts_for_frame[gts_for_frame['label'] == class_name]

            # 박스 데이터를 GPU 텐서로 변환
            pred_boxes_tensor = torch.tensor(
                class_preds[BOX_COLS].values.astype(np.float32), 
            ).cuda()
            gt_boxes_tensor = torch.tensor(
                class_gts[BOX_COLS].values.astype(np.float32), 
            ).cuda()
            
            # 메트릭 계산
            tp, fp, fn = calculate_metrics_for_frame(
                pred_boxes_tensor, 
                gt_boxes_tensor, 
                iou_thresh
            )
            
            # 결과 누적
            total_tp[class_name] += tp
            total_fp[class_name] += fp
            total_fn[class_name] += fn

    print("\n\n========== Evaluation Results ==========")
    print(f"Confidence Threshold: {SCORE_THRESHOLD}")

    # 최종 Precision / Recall 계산 및 출력
    for class_name in CLASS_NAMES:
        tp = total_tp[class_name]
        fp = total_fp[class_name]
        fn = total_fn[class_name]
        
        # 0으로 나누기 방지
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"\n--- Class: {class_name} (IoU Thresh: {IOU_THRESHOLDS.get(class_name, 0.5)}) ---")
        print(f"Total GT Boxes: {tp + fn}")
        print(f"Total Pred Boxes (filtered): {tp + fp}")
        print(f"  True Positives (TP):   {tp}")
        print(f"  False Positives (FP):  {fp}")
        print(f"  False Negatives (FN):  {fn}")
        print(f"  Precision: {precision * 100:.2f} %")
        print(f"  Recall:    {recall * 100:.2f} %")
        
    print("==========================================")

if __name__ == '__main__':
    # PyTorch GPU가 사용 가능한지 확인
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
    else:
        print(f"Found CUDA device: {torch.cuda.get_device_name(0)}")
        main()