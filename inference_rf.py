import argparse
import numpy as np
import joblib

import time
import pandas as pd
import torch
from tqdm import tqdm # 진행률 표시
from pathlib import Path

from pcdet.datasets.a2d2.a2d2_dataset import A2D2Dataset

# OpenPCDet 임포트
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
# Sklearn 임포트 [추가]
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- [GLOBAL] 피처 컬럼 정의 ---
# 머신러닝에 사용할 피처 목록 (18개)
ML_FEATURE_COLUMNS = [
    "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
    "num_points", "width", "length", "height", "density",
    "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std"
]
# CSV에 저장될 전체 컬럼 목록 (피처 + 라벨/메타데이터)
CSV_COLUMNS = ML_FEATURE_COLUMNS + ["label", "max_iou", "frame_id"]
# RF Stage 1의 'Object' 라벨 (학습 시 1로 인코딩됨)
STAGE_1_OBJECT_LABEL = 1

# --- 설정 파싱 함수 (수정) ---
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # --- 1. 공통 인자 ---
    parser.add_argument('--mode', type=str, default='extract', 
                        choices=['extract', 'train', 'evaluate', 'inference'], # 'inference' 추가
                        help='Operation mode: extract, train, evaluate, or inference.')

    # --- 2. 'extract' & 'inference' 모드 인자 ---
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/a2d2_models/second.yaml', help='(extract/inference) pcdet config file')
    parser.add_argument('--split', type=str, default='val', help='(extract/inference) Data split to process')
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth', help='(extract/inference) pcdet checkpoint')
    parser.add_argument('--min_points_in_box', type=int, default=3, help='(extract/inference) Min points in box')
    parser.add_argument('--no_vis', action='store_true', help='(extract/inference) Disable visualization')
    parser.add_argument('--vis_frame_limit', type=int, default=5, help='(extract/inference) Limit visualization frames')
    
    # --- 3. 'extract' 전용 인자 ---
    parser.add_argument('--output_csv', type=str, default=None, help='(extract) Path to save the output CSV file.')
    parser.add_argument('--fg_thresh', type=float, default=0.2, help='(extract) IoU threshold for "Foreground"')
    parser.add_argument('--bg_thresh', type=float, default=0.5, help='(extract) IoU threshold for "Background"')

    # --- 4. 'train' 모드 인자 ---
    parser.add_argument('--train_file', type=str, default='data/a2d2/features_train.csv', help='(train) Path to the training features CSV')
    parser.add_argument('--model1_out', type=str, default='data/a2d2/rf_stage1_model.pkl', help='(train) Path to save the Stage 1 model')
    parser.add_argument('--model2_out', type=str, default='data/a2d2/rf_stage2_model.pkl', help='(train) Path to save the Stage 2 model')
    parser.add_argument('--le_out', type=str, default='data/a2d2/le_stage2.pkl', help='(train) Path to save the LabelEncoder') # [추가]

    # --- 5. 'evaluate' & 'inference' 모드 인자 ---
    parser.add_argument('--test_file', type=str, default='data/a2d2/features_val.csv', help='(evaluate) Path to the test features CSV')
    parser.add_argument('--model1_path', type=str, default='data/a2d2/rf_stage1_model.pkl', help='(evaluate/inference) Path to the Stage 1 model')
    parser.add_argument('--model2_path', type=str, default='data/a2d2/rf_stage2_model.pkl', help='(evaluate/inference) Path to the Stage 2 model')
    parser.add_argument('--le_path', type=str, default='data/a2d2/le_stage2.pkl', help='(evaluate/inference) Path to the LabelEncoder') # [추가]

    args = parser.parse_args()

    # 'extract' 또는 'inference' 모드일 때만 pcdet 설정 로드
    if args.mode in ['extract', 'inference']:
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
    
    if args.mode == 'extract' and args.output_csv is None:
        args.output_csv = f'data/a2d2/new_background_features_{args.split}.csv'
    
    return args, cfg

# 1. Data loading
    # dataset을 불러옴 (GT, lidar...)
def setup_dataset(cfg, args, logger):
    """
    args와 cfg를 기반으로 A2D2Dataset을 초기화하고 설정합니다.
    
    Args:
        cfg (obj): OpenPCDet 설정 객체
        args (obj): argparse로 파싱된 인자
        logger (obj): 로깅 객체

    Returns:
        dataset (A2D2Dataset): 성공적으로 로드된 데이터셋 객체
        None: 초기화 또는 경로 확인 실패 시
    """
    try:
        # args.split 값에 따라 training 플래그 결정
        is_training_split = (args.split == 'train')

        # 1. 데이터셋 객체 생성
        dataset = A2D2Dataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
            training=is_training_split, # <--- 'train'일 때만 True
            logger=logger
        )
        
        # 2. 스플릿 설정
        dataset.set_split(args.split)
        logger.info(f"Loaded {args.split} split with {len(dataset.sample_id_list)} frames (from ImageSets).")
        logger.info(f"Total {len(dataset)} frames in {args.split} split.")

        # 3. GT 라벨 경로 확인 (중요)
        gt_label_dir = dataset.root_path / f'new_label/label_new_{args.split}'
        if not gt_label_dir.exists():
            logger.error(f"GT label directory not found: {gt_label_dir}")
            logger.error("Please run 'create_new_label_files.py' first.")
            return None  # 실패 시 None 반환
        
        dataset.gt_label_dir = gt_label_dir
        # 4. 성공 시 데이터셋 반환
        return dataset

    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset initialization: {e}")
        return None # 예외 발생 시 None 반환
    
# --- (원본 함수) GT 박스 로더 ---
def load_gt_boxes(gt_txt_path):
    if not gt_txt_path.exists():
        return np.array([]), []
        
    gt_boxes, gt_names = [], []
    with open(gt_txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls_name = parts[0]
            try:
                # A2D2: x(2), y(3), z(4), l(5), w(6), h(7), yaw(8)
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                l, w, h, yaw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                gt_boxes.append([x, y, z, l, w, h, yaw])
                gt_names.append(cls_name)
            except ValueError:
                print(f"[경고] {gt_txt_path} 파일의 라인을 파싱할 수 없습니다: {line}")
                
    gt_boxes = np.array(gt_boxes, dtype=np.float32)

    # ⭐ 핵심: 하단(z_min) 기준 → 중심(z_center) 기준
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, 2] += gt_boxes[:, 5] / 2.0

    return gt_boxes, gt_names


# 2. RPN box 출력
    # data_dict가 Bounding box 인거임
    # NMS을 하는데, score 되게 작게해서 많은 박스를 최대한 살리거나 후처리 조금 추가
    # 후처리 알고리즘 구현 필요시 작성
def extract_point_features_cpu(points_np, boxes_np, min_points_in_box, logger):
    """
    CPU(Numpy)와 OpenPCDet 유틸리티(roiaware_pool3d)를 사용하여
    각 박스 내부 포인트로부터 특징 추출 (최적화 버전)
    (원본 스크립트의 함수와 동일)
    """
    num_boxes = boxes_np.shape[0]
    point_features_list = []
    valid_box_indices = [] # 특징 추출에 성공한 박스의 인덱스

    try:
        points_tensor = torch.from_numpy(points_np[:, 0:3]).float()
        boxes_tensor = torch.from_numpy(boxes_np).float()
        
        point_indices_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
            points_tensor, boxes_tensor
        ).numpy() # (num_boxes, num_points)
        
    except Exception as e:
        logger.error(f"  Fatal Error during roiaware_pool3d_utils.points_in_boxes_cpu: {e}")
        return np.array([]), np.array([], dtype=int)

    # 2. 각 박스를 순회하며 특징 계산 (NumPy)
    for i in range(num_boxes):
        try:
            mask = point_indices_mask[i]
            points_in_box = points_np[mask.astype(bool)]
        except Exception as e:
            continue

        if points_in_box.shape[0] < min_points_in_box:
            continue # 포인트 부족 시 건너뛰기

        valid_box_indices.append(i) # 유효한 박스 인덱스 추가

        try:
            num_points = points_in_box.shape[0]

            min_coords = np.min(points_in_box[:, :3], axis=0)
            max_coords = np.max(points_in_box[:, :3], axis=0)
            dims = max_coords - min_coords
            width = dims[0]       # 포인트 분포의 너비 (x)
            length = dims[1]      # 포인트 분포의 길이 (y)
            height = dims[2]      # 포인트 분포의 높이 (z)
            
            box_l, box_w, box_h = boxes_np[i, 3], boxes_np[i, 4], boxes_np[i, 5]
            box_volume = (box_l * box_w * box_h) + 1e-6 
            density = num_points / box_volume 

            aspect_ratio = width / (length + 1e-6) 
            mean_z = np.mean(points_in_box[:, 2])
            std_z = np.std(points_in_box[:, 2])
            mean_intensity = np.mean(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0
            std_intensity = np.std(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0

            features = [
                num_points, 
                width, length, height,
                density, aspect_ratio, 
                mean_z, std_z, mean_intensity, std_intensity
            ]
            point_features_list.append(features)
        except Exception as e:
            if i in valid_box_indices:
                 valid_box_indices.pop()

    if not point_features_list:
        return np.array([]), np.array([], dtype=int)

    return np.array(point_features_list, dtype=np.float32), np.array(valid_box_indices, dtype=int)

def match_rpn_to_gt_for_training(rpn_boxes, gt_boxes, gt_labels, fg_iou_thresh=0.2, bg_iou_thresh=0.4):
    """
    RPN box를 GT box와 IoU 매칭.
    - IoU >= fg_thresh: Foreground (e.g., "Car")
    - IoU < bg_thresh: Background
    - 그 외: Ignore
    """
    num_rpn_boxes = rpn_boxes.shape[0]
    
    if gt_boxes.shape[0] == 0:
        # GT가 없으면 모두 "Background"
        return ["Background"] * num_rpn_boxes, np.zeros(num_rpn_boxes, dtype=np.float32)

    ious = iou3d_nms_utils.boxes_iou3d_gpu(
        torch.from_numpy(rpn_boxes).cuda(),
        torch.from_numpy(gt_boxes).cuda()
    ).cpu().numpy()

    best_gt_indices = np.argmax(ious, axis=1)
    best_ious_np = ious[np.arange(num_rpn_boxes), best_gt_indices]

    matched_labels = []
    for i in range(num_rpn_boxes):
        best_iou = best_ious_np[i]
        if best_iou >= fg_iou_thresh:
            matched_labels.append(gt_labels[best_gt_indices[i]])
        elif best_iou < bg_iou_thresh:
            matched_labels.append("Background")
        else:
            # (예: 0.3 <= IoU < 0.5) 애매한 영역은 "Ignore"
            matched_labels.append("Ignore")
            
    return matched_labels, best_ious_np



def extract_and_save_features(dataset, model, args, logger):
    """
    데이터셋을 순회하며 피처를 추출하고 최종 DataFrame을 CSV 파일로 저장합니다.
    성공 시 저장된 파일 경로(str)를, 실패 시 None을 반환합니다.
    """
    gt_label_dir = dataset.gt_label_dir

    all_features_list = []
    logger.info(f"Starting feature extraction for {len(dataset)} frames...")

    # (원본 코드의 4. 모든 프레임 순회)
    for index in tqdm(range(len(dataset)), desc=f"Processing {args.split} split"):
        frame_id = "" # 오류 시 로깅을 위해 frame_id 미리 초기화
        try:
            # 4-1. 데이터 로드
            data_dict = dataset[index]
            frame_id = data_dict['frame_id']
            
            raw_points_np = dataset.get_lidar(frame_id)
            data_dict_batch = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict_batch)

            # 4-2. 모델 추론 (RPN + NMS)
            with torch.no_grad():
                 pred_dicts, _ = model(data_dict_batch)

            post_nms_boxes_tensor = pred_dicts[0]['pred_boxes']
            post_nms_scores_tensor = pred_dicts[0]['pred_scores']
            
            if post_nms_boxes_tensor.shape[0] == 0:
                continue

            rpn_boxes_np_filtered = post_nms_boxes_tensor.cpu().numpy()
            rpn_scores_np_filtered = post_nms_scores_tensor.cpu().numpy().reshape(-1, 1)

            # 4-3. 포인트 기반 피처 추출
            point_features_np, valid_indices_np = extract_point_features_cpu(
                raw_points_np, rpn_boxes_np_filtered, args.min_points_in_box, logger
            )
            
            if valid_indices_np.shape[0] == 0:
                continue
            
            # 4-4. 유효한 박스들만 필터링
            final_boxes_np = rpn_boxes_np_filtered[valid_indices_np]
            final_scores_np = rpn_scores_np_filtered[valid_indices_np]

            # 4-5. 최종 피처 벡터 결합
            final_rpn_features_np = np.concatenate((final_scores_np, final_boxes_np), axis=1)
            final_features_np = np.concatenate((final_rpn_features_np, point_features_np), axis=1)

            # 4-6. GT 매칭
            gt_txt_path = gt_label_dir / f"{frame_id}.txt"
            gt_boxes_np, gt_names = load_gt_boxes(gt_txt_path)
            
            matched_labels, matched_ious_np = match_rpn_to_gt_for_training(
                final_boxes_np, gt_boxes_np, gt_names, 
                args.fg_thresh, args.bg_thresh
            )

            # 4-7. DataFrame 생성
            df = pd.DataFrame(final_features_np, columns=ML_FEATURE_COLUMNS)
            df["label"] = matched_labels
            df["max_iou"] = matched_ious_np
            df["frame_id"] = frame_id
            # 4-8. "Ignore" 샘플 제거
            df_filtered = df[df['label'] != 'Ignore']
            
            if not df_filtered.empty:
                all_features_list.append(df_filtered)

            # 4-9. 시각화 (선택 사항)
            # ... (시각화 로직) ...

        except Exception as e:
            logger.error(f"\n[치명적 오류] Frame {index} ({frame_id}) 처리 중 실패: {e}")
            import traceback; traceback.print_exc()
            continue
    
    logger.info("Feature extraction loop finished.")

    # --- [신규] 피처 저장 (함수 내부에 포함) ---
    if not all_features_list:
        logger.warning(f"추출된 피처가 없습니다. {args.output_csv} 파일을 생성하지 않습니다.")
        return None

    try:
        logger.info(f"Combining {len(all_features_list)} DataFrames...")
        final_df = pd.concat(all_features_list, ignore_index=True)
        
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(final_df)} feature vectors to: {output_path}")
        return str(output_path) # 성공 시 파일 경로 반환

    except Exception as e:
        logger.error(f"피처 저장 중 오류 발생: {e}")
        import traceback; traceback.print_exc()
        return None # 실패 시 None 반환
def run_extraction(args, cfg, logger):
    """'extract' 모드 실행 함수"""
    logger.info('----------------- Mode: Feature Extraction -----------------')
    
    # 1. A2D2Dataset 준비
    dataset = setup_dataset(cfg, args, logger)
    if dataset is None:
        logger.error("Failed to set up ataset. Exiting.")
        return

    # 2. pcdet 모델 빌드 및 로드
    try:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda(); model.eval()
    except Exception as e:
        logger.error(f"Error building/loading model: {e}"); return
    
    logger.info("Model loaded successfully. Beginning inference...")

    # 3. RPN box 출력 및 피처 추출
    saved_csv_path = extract_and_save_features(
        dataset=dataset,
        model=model,
        args=args,
        logger=logger
    )
    
    if saved_csv_path:
        logger.info(f"Feature extraction complete. Data saved to: {saved_csv_path}")
    else:
        logger.error("Feature extraction failed or produced no data.")

# ======================================================================
# --- 2. 모델 학습 (train) 관련 함수들 ---
# ======================================================================

def run_training_stage1(df_train, logger):
    """1단계 모델 (Object / Background) 학습"""
    logger.info("--- 1단계 모델 학습 시작 ---")
    # 1. 라벨 생성
    df_train['s1_label'] = (df_train['label'] != 'Background').astype(int)

    # 2. Object와 Background 분리
    df_s1_obj = df_train[df_train['s1_label'] == STAGE_1_OBJECT_LABEL]
    df_s1_bg = df_train[df_train['s1_label'] != STAGE_1_OBJECT_LABEL]

    num_obj = len(df_s1_obj)

    if num_obj == 0:
        logger.error("오류: 1단계 학습을 위한 Object 샘플이 없습니다.")
        return None

    # --- [핵심 수정] ---
    # Background 샘플을 Object 샘플 수만큼 (혹은 2배수만큼) 랜덤 샘플링
    df_s1_bg_sampled = df_s1_bg.sample(n=num_obj * 2, random_state=42, replace=True if len(df_s1_bg) < num_obj*2 else False)

    # 데이터 다시 합치기
    df_s1_balanced = pd.concat([df_s1_obj, df_s1_bg_sampled])
    # -------------------

    y_s1 = df_s1_balanced['s1_label']
    X_s1 = df_s1_balanced[ML_FEATURE_COLUMNS]

    logger.info(f"1단계 *균형* 학습 샘플: {len(X_s1)}개")
    logger.info(f"Object(1) 샘플: {y_s1.sum()}개 / Background(0) 샘플: {(y_s1 == 0).sum()}개")
    
    # 2. 모델 정의 및 학습
    model_s1 = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        max_depth=15, min_samples_leaf=5,
        class_weight='balanced'
    )
    model_s1.fit(X_s1, y_s1)
    
    y_pred_s1 = model_s1.predict(X_s1)
    acc_s1 = accuracy_score(y_s1, y_pred_s1)
    logger.info(f"1단계 학습 정확도: {acc_s1 * 100:.2f}%")
    
    return model_s1

def run_training_stage2(df_train, logger):
    """[수정] 2단계 모델 학습 및 LabelEncoder 반환"""
    logger.info("\n--- 2단계 모델 학습 시작 ---")
    df_s2 = df_train[df_train['label'] != 'Background'].copy()
    if df_s2.empty:
        logger.error("오류: 2단계 학습을 위한 Object 샘플이 없습니다.")
        return None, None # 모델과 인코더 모두 None 반환

    le = LabelEncoder()
    y_s2_str = df_s2['label']
    y_s2_numeric = le.fit_transform(y_s2_str)
    X_s2 = df_s2[ML_FEATURE_COLUMNS]
    
    logger.info(f"2단계 학습 샘플: {len(X_s2)}개")
    logger.info(f"2단계 클래스: {le.classes_}")

    model_s2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=3)
    model_s2.fit(X_s2, y_s2_numeric)
    acc_s2 = accuracy_score(y_s2_numeric, model_s2.predict(X_s2))
    logger.info(f"2단계 학습 정확도: {acc_s2 * 100:.2f}%")
    
    return model_s2, le # ★ LabelEncoder 반환

def run_training(args, logger):
    """[수정] 'train' 모드 실행 (LabelEncoder 저장 추가)"""
    logger.info('----------------- Mode: Model Training -----------------')
    try:
        logger.info(f"학습 데이터 로드 중: {args.train_file}")
        df_train = pd.read_csv(args.train_file).fillna(0)
        logger.info(f" -> 총 {len(df_train)}개의 학습 샘플 발견.")
    except FileNotFoundError:
        logger.error(f"오류: 학습 파일을 찾을 수 없습니다. {args.train_file}"); return

    model_1 = run_training_stage1(df_train, logger)
    model_2, le_stage2 = run_training_stage2(df_train, logger) # ★ le 받기

    try:
        if model_1:
            joblib.dump(model_1, args.model1_out)
            logger.info(f"\n1단계 모델 저장 완료: {args.model1_out}")
        if model_2:
            joblib.dump(model_2, args.model2_out)
            logger.info(f"2단계 모델 저장 완료: {args.model2_out}")
        if le_stage2: # ★ le 저장
            joblib.dump(le_stage2, args.le_out)
            logger.info(f"LabelEncoder 저장 완료: {args.le_out}")
    except Exception as e:
        logger.error(f"모델 또는 LabelEncoder 저장 중 오류 발생: {e}")
        

# ======================================================================
# --- 3. 모델 평가 (evaluate) 관련 함수 ---
# ======================================================================

def run_evaluation(args, logger):
    """[수정] 'evaluate' 모드 실행 (LabelEncoder 파일에서 로드)"""
    logger.info('----------------- Mode: Model Evaluation -----------------')
    
    try:
        # --- 1. 모델 및 LabelEncoder 로드 ---
        logger.info(f"1단계 모델 로드 중: {args.model1_path}")
        model_1 = joblib.load(args.model1_path)
        logger.info(f"2단계 모델 로드 중: {args.model2_path}")
        model_2 = joblib.load(args.model2_path)
        logger.info(f"LabelEncoder 로드 중: {args.le_path}")
        le_stage2 = joblib.load(args.le_path) # ★ 파일에서 로드
        logger.info("모델 및 인코더 로드 완료.")
        logger.info(f" -> 2단계 클래스 (파일 기준): {le_stage2.classes_}")

        # --- 2. 테스트 데이터 로드 ---
        logger.info(f"테스트 데이터 로드 중: {args.test_file}")
        df_test = pd.read_csv(args.test_file).fillna(0)
        y_true = df_test['label']
        X_test = df_test.drop(columns=['label'], errors='ignore')
        logger.info(f" -> 총 {len(X_test)}개의 테스트 샘플 발견.")

        # ... (이하 평가 로직은 이전과 동일) ...
        features_s1 = model_1.feature_names_in_
        features_s2 = model_2.feature_names_in_
        final_predictions = []
        stage1_preds = []
        stage1_times, stage2_times = [], []

        logger.info("\n캐스케이드 평가 시작...")
        X_test_s1_features = X_test[features_s1]
        X_test_s2_features = X_test[features_s2]

        for i in tqdm(range(len(X_test)), desc="평가 진행 중"):
            sample_s1 = X_test_s1_features.values[i]
            start_s1 = time.time(); pred_1 = model_1.predict([sample_s1])[0]; end_s1 = time.time()
            stage1_times.append(end_s1 - start_s1); stage1_preds.append(pred_1)

            if pred_1 == STAGE_1_OBJECT_LABEL:
                sample_s2 = X_test_s2_features.values[i]
                start_s2 = time.time(); pred_2_numeric = model_2.predict([sample_s2])[0]; end_s2 = time.time()
                stage2_times.append(end_s2 - start_s2)
                try:
                    pred_2_string = le_stage2.inverse_transform([pred_2_numeric])[0]
                    final_predictions.append(pred_2_string)
                except ValueError: final_predictions.append('Background') 
            else: final_predictions.append('Background')

        # ... (통계 및 리포트 출력 로직) ...
        avg_s1_ms = np.mean(stage1_times) * 1000
        avg_s2_ms = np.mean(stage2_times) * 1000 if stage2_times else 0
        total_time_ms = (np.sum(stage1_times) + np.sum(stage2_times)) * 1000
        avg_total_per_sample_ms = total_time_ms / len(X_test)
        
        logger.info("\n" + "="*50); logger.info("--- [추론 시간 통계] ---")
        logger.info(f"Stage 1 평균 Inference Time : {avg_s1_ms:.3f} ms (per S1 sample)")
        logger.info(f"Stage 2 평균 Inference Time : {avg_s2_ms:.3f} ms (per S2 sample)")
        logger.info(f"총 평균 Inference Time (per *total* sample) : {avg_total_per_sample_ms:.3f} ms")
        logger.info(f"FPS (samples per second) ≈ {1000 / avg_total_per_sample_ms:.2f} FPS")
        logger.info("="*50); logger.info("--- CASCADE 시스템 최종 평가 결과 ---")
        accuracy = accuracy_score(y_true, final_predictions)
        report = classification_report(y_true, final_predictions, digits=4, zero_division=0)
        logger.info(f"전체 정확도 (Accuracy): {accuracy * 100:.4f}%")
        logger.info("\n[ 최종 분류 리포트 ]\n" + report); logger.info("="*50)

    except FileNotFoundError as e:
        logger.error(f"[오류] 파일 찾기 실패: {e.filename}")
    except Exception as e:
        logger.error(f"[오류] 평가 중 문제 발생: {e}"); import traceback; traceback.print_exc()


# ======================================================================
# --- 4. 실시간 추론 (inference) 관련 함수 ---
# ======================================================================

def run_inference(args, cfg, logger):
    """[신규] 'inference' 모드 실행 함수"""
    logger.info('----------------- Mode: Real-time Inference -----------------')

    try:
        # --- 1. 모든 모델 로드 (pcdet, RF S1, RF S2, LE) ---
        logger.info("Loading pcdet dataset...")
        dataset = setup_dataset(cfg, args, logger)
        if dataset is None: logger.error("Failed to set up dataset. Exiting."); return

        logger.info(f"Loading pcdet model: {args.ckpt}")
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda(); model.eval()
        
        logger.info(f"Loading Stage 1 RF model: {args.model1_path}")
        model_1 = joblib.load(args.model1_path)
        logger.info(f"Loading Stage 2 RF model: {args.model2_path}")
        model_2 = joblib.load(args.model2_path)
        logger.info(f"Loading LabelEncoder: {args.le_path}")
        le_stage2 = joblib.load(args.le_path)
        
        logger.info("All models loaded successfully.")
        logger.info(f" -> Inference classes: {le_stage2.classes_}")

        features_s1 = model_1.feature_names_in_
        features_s2 = model_2.feature_names_in_
        
        # 2단계 클래스(문자열)를 숫자 ID로 매핑 (시각화용)
        vis_class_to_id_map = {name: i+1 for i, name in enumerate(le_stage2.classes_)} # 1-based
        vis_class_to_id_map['Background'] = 0

    except Exception as e:
        logger.error(f"Error during model loading: {e}"); return
        
    
    # --- 2. 프레임 단위 추론 루프 ---
    logger.info(f"Starting inference on {len(dataset)} frames from '{args.split}' split...")
    
    for index in tqdm(range(len(dataset)), desc=f"Inferencing {args.split} split"):
        frame_id = ""
        try:
            # --- A. pcdet RPN + 피처 추출 (extract_and_save_features와 동일) ---
            data_dict = dataset[index]; frame_id = data_dict['frame_id']
            raw_points_np = dataset.get_lidar(frame_id)
            data_dict_batch = dataset.collate_batch([data_dict]); load_data_to_gpu(data_dict_batch)

            with torch.no_grad(): pred_dicts, _ = model(data_dict_batch)
            post_nms_boxes_tensor = pred_dicts[0]['pred_boxes']
            post_nms_scores_tensor = pred_dicts[0]['pred_scores']
            
            if post_nms_boxes_tensor.shape[0] == 0:
                logger.warning(f"[Frame {frame_id}] No RPN proposals found. Skipping."); continue
            
            rpn_boxes_np_filtered = post_nms_boxes_tensor.cpu().numpy()
            rpn_scores_np_filtered = post_nms_scores_tensor.cpu().numpy().reshape(-1, 1)

            point_features_np, valid_indices_np = extract_point_features_cpu(
                raw_points_np, rpn_boxes_np_filtered, args.min_points_in_box, logger
            )
            if valid_indices_np.shape[0] == 0:
                logger.warning(f"[Frame {frame_id}] No valid point features extracted. Skipping."); continue
            
            # 피처 추출에 성공한 박스/점수/피처만 필터링
            final_boxes_np = rpn_boxes_np_filtered[valid_indices_np]
            final_scores_np = rpn_scores_np_filtered[valid_indices_np]
            final_rpn_features_np = np.concatenate((final_scores_np, final_boxes_np), axis=1)
            final_features_np = np.concatenate((final_rpn_features_np, point_features_np), axis=1)

            # --- B. RF 캐스케이드 추론 (evaluate와 동일) ---
            df_features = pd.DataFrame(final_features_np, columns=ML_FEATURE_COLUMNS)
            X_s1 = df_features[features_s1]
            X_s2 = df_features[features_s2]

            final_pred_labels = []    # 최종 예측 라벨 (문자열)
            final_pred_scores = []    # 최종 예측 신뢰도
            final_pred_vis_ids = []   # 최종 예측 ID (시각화용)

            for j in range(len(df_features)):
                sample_s1 = X_s1.values[j]
                pred_1 = model_1.predict([sample_s1])[0]

                if pred_1 == STAGE_1_OBJECT_LABEL:
                    sample_s2 = X_s2.values[j]
                    pred_2_numeric = model_2.predict([sample_s2])[0]
                    pred_2_proba = model_2.predict_proba([sample_s2])[0]
                    pred_2_score = np.max(pred_2_proba) # S2의 최대 신뢰도
                    pred_2_string = le_stage2.inverse_transform([pred_2_numeric])[0]
                    
                    final_pred_labels.append(pred_2_string)
                    final_pred_scores.append(pred_2_score)
                    final_pred_vis_ids.append(vis_class_to_id_map.get(pred_2_string, 0))
                
                # (참고: Background로 분류된 박스는 시각화/로깅에서 제외하므로
                #  else 문에서 리스트에 추가할 필요 없음)

            # --- C. 결과 로깅 및 시각화 ---
            num_objects = len(final_pred_labels)
            logger.info(f"\n[Frame {frame_id}] Inference complete. Found {num_objects} objects (BG 제외).")
            
            log_boxes_np = []       # 로깅/시각화를 위한 박스
            log_scores_np = []      # RF 신뢰도 점수
            log_labels_numeric = [] # RF 라벨 ID (시각화용)

            for i in range(num_objects):
                label = final_pred_labels[i]
                score = final_pred_scores[i]
                box = final_boxes_np[i]
                
                # (여기서 score 임계값을 설정할 수 있음, 예: if score > 0.3:)
                logger.info(f"  > {label} (Score: {score:.4f}) @ [x:{box[0]:.1f}, y:{box[1]:.1f}, z:{box[2]:.1f}]")
                
                log_boxes_np.append(box)
                log_scores_np.append(score)
                log_labels_numeric.append(final_pred_vis_ids[i])

            # (선택) 시각화
            if not args.no_vis and index < args.vis_frame_limit:
                logger.info(f"Visualizing frame {frame_id}...")
                try:
                    # (V.draw_scenes를 사용하려면 visual_utils 임포트 필요)
                    pass
                    # V.draw_scenes(
                    #     points=raw_points_np[:, :3],
                    #     ref_boxes=np.array(log_boxes_np),
                    #     ref_scores=np.array(log_scores_np),
                    #     ref_labels=np.array(log_labels_numeric)
                    # )
                    # if not OPEN3D_FLAG:
                    #     from mayavi import mlab
                    #     mlab.show(stop=True)
                        
                except ImportError:
                    logger.warning("Visual_utils or Mayavi/Open3D not found. Skipping visualization.")
                except Exception as e:
                    logger.error(f"Visualization failed: {e}")

        except Exception as e:
            logger.error(f"\n[치명적 오류] Frame {index} ({frame_id}) 추론 중 실패: {e}")
            import traceback; traceback.print_exc()
            continue
            
    logger.info("Inference finished.")


# ======================================================================
# --- [수정] 메인 함수 (라우터) ---
# ======================================================================

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    if args.mode == 'extract':
        run_extraction(args, cfg, logger)
        
    elif args.mode == 'train':
        run_training(args, logger)
        
    elif args.mode == 'evaluate':
        run_evaluation(args, logger)
        
    elif args.mode == 'inference': # [추가]
        run_inference(args, cfg, logger)
        
    else:
        logger.error(f"알 수 없는 모드입니다: {args.mode}")

if __name__ == '__main__':
    main()