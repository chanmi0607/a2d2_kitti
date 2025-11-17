
import joblib
from tqdm import tqdm # 진행률 표시
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, accuracy_score
# pcdet 관련
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from randomforest.config import STAGE_1_OBJECT_LABEL , ML_FEATURE_COLUMNS
from randomforest.dataset.dataset_loader import setup_dataset
from randomforest.dataset.gt_parser import load_gt_boxes
from randomforest.features.point_ops import extract_point_features_cpu
from randomforest.features.matcher import match_rpn_to_gt_for_training

# ======================================================================
# --- 3. 모델 평가 (evaluate) 관련 함수 ---
# ======================================================================
def calculate_iou_cpu(boxes1, boxes2):
    """
    CPU 기반 IoU 계산 (간단한 2D BEV IoU 또는 3D IoU)
    여기서는 pcdet의 GPU utils를 활용하거나, 단순화를 위해 BEV IoU를 씁니다.
    정확도를 위해 pcdet의 GPU 함수를 호출하여 CPU로 변환하는 방식을 추천합니다.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # pcdet의 GPU 함수 활용 (이미 cuda context가 있으므로)
    boxes1_tensor = torch.from_numpy(boxes1).cuda().float()
    boxes2_tensor = torch.from_numpy(boxes2).cuda().float()
    ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes1_tensor, boxes2_tensor)
    return ious.cpu().numpy()

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


def run_evaluation_gt(args, cfg, logger):
    logger.info('----------------- Mode: Full Evaluation (Classification + Detection) -----------------')

    # ------------------------------------------------------------------
    # 1. 초기화
    # ------------------------------------------------------------------
    dataset = setup_dataset(cfg, args, logger)
    if dataset is None: return

    pcdet_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    pcdet_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    pcdet_model.cuda().eval()

    model_1 = joblib.load(args.model1_path)
    model_2 = joblib.load(args.model2_path)
    le_stage2 = joblib.load(args.le_path)

    # --- 평가를 위한 저장소 ---
    # 1. 분류 성능용 (Classifier Metric)
    all_match_y_true = []
    all_match_y_pred = []

    # 2. 검출 성능용 (Detection Metric)
    # 클래스별 {tp, fp, fn} 카운트
    det_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0})
    iou_threshold = 0.1  # 검출 성공 기준 IoU

    logger.info(f"Starting evaluation on {len(dataset)} frames...")
    
    for index in tqdm(range(len(dataset)), desc="Evaluating"):
        frame_id = ""
        try:
            # --- A. 데이터 로드 및 RPN ---
            data_dict = dataset[index]; frame_id = data_dict['frame_id']
            data_dict_batch = dataset.collate_batch([data_dict]); load_data_to_gpu(data_dict_batch)
            raw_points_np = dataset.get_lidar(frame_id)

            with torch.no_grad():
                pred_dicts, _ = pcdet_model(data_dict_batch)

            # RPN 결과
            post_nms_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            post_nms_scores = pred_dicts[0]['pred_scores'].cpu().numpy().reshape(-1, 1)

            if post_nms_boxes.shape[0] == 0:
                # RPN이 아무것도 못 찾은 경우 -> 이 프레임의 모든 GT는 FN 처리됨
                gt_txt_path = dataset.gt_label_dir / f"{frame_id}.txt"
                gt_boxes_all, gt_names_all = load_gt_boxes(gt_txt_path)
                for g_name in gt_names_all:
                    det_stats[g_name]['fn'] += 1
                    det_stats[g_name]['gt_count'] += 1
                continue

            # --- B. 피처 추출 및 RF 예측 ---
            point_features_np, valid_indices = extract_point_features_cpu(
                raw_points_np, post_nms_boxes, args.min_points_in_box, logger
            )
            
            # 유효한 박스 필터링
            if len(valid_indices) > 0:
                final_boxes = post_nms_boxes[valid_indices]
                final_scores = post_nms_scores[valid_indices]
                
                # 피처 조립
                final_rpn_features = np.concatenate((final_scores, final_boxes), axis=1)
                final_features = np.concatenate((final_rpn_features, point_features_np), axis=1)
                
                # RF 예측 수행
                df_features = pd.DataFrame(final_features, columns=ML_FEATURE_COLUMNS)
                X_s1 = df_features[model_1.feature_names_in_]
                X_s2 = df_features[model_2.feature_names_in_]
                
                current_pred_labels = [] # 각 박스의 예측 라벨 (Background 포함)

                for j in range(len(df_features)):
                    pred_1 = model_1.predict([X_s1.iloc[j]])[0]
                    final_label = "Background"
                    if pred_1 == STAGE_1_OBJECT_LABEL:
                        pred_2_num = model_2.predict([X_s2.iloc[j]])[0]
                        try:
                            final_label = le_stage2.inverse_transform([pred_2_num])[0]
                        except: pass
                    current_pred_labels.append(final_label)
            else:
                final_boxes = np.array([])
                current_pred_labels = []

            # --- C. Ground Truth 로드 ---
            gt_txt_path = dataset.gt_label_dir / f"{frame_id}.txt"
            gt_boxes_all, gt_names_all = load_gt_boxes(gt_txt_path) # (N, 7), List

            # =========================================================
            # [평가 1] 분류 성능 (Classifier Performance)
            # RPN이 제안한 박스가 GT와 매칭되었을 때, RF가 맞췄는가?
            # =========================================================
            if len(final_boxes) > 0:
                matched_labels_for_cls, _ = match_rpn_to_gt_for_training(
                    final_boxes, gt_boxes_all, gt_names_all, 
                    fg_iou_thresh=args.fg_thresh, bg_iou_thresh=args.bg_thresh
                )
                
                for true_lbl, pred_lbl in zip(matched_labels_for_cls, current_pred_labels):
                    if true_lbl != 'Ignore':
                        all_match_y_true.append(true_lbl)
                        all_match_y_pred.append(pred_lbl)

            # =========================================================
            # [평가 2] 객체 검출 성능 (Detection Performance)
            # 실제 GT를 시스템이 얼마나 놓치지 않고 정확히 찾았는가? (TP, FP, FN)
            # =========================================================
            
            # 2-1. 최종 예측 중 'Background'가 아닌 것만 필터링 (검출된 객체)
            final_pred_mask = np.array(current_pred_labels) != 'Background'
            if np.any(final_pred_mask):
                det_pred_boxes = final_boxes[final_pred_mask]
                det_pred_labels = np.array(current_pred_labels)[final_pred_mask]
            else:
                det_pred_boxes = np.array([])
                det_pred_labels = np.array([])

            # 2-2. 클래스별 IoU 매칭 및 TP/FP/FN 계산
            unique_classes = set(gt_names_all) | set(det_pred_labels)
            
            for cls_name in unique_classes:
                if cls_name == 'Background': continue

                # 해당 클래스의 GT 박스들
                cls_gt_indices = [i for i, name in enumerate(gt_names_all) if name == cls_name]
                cls_gt_boxes = gt_boxes_all[cls_gt_indices] if len(cls_gt_indices) > 0 else np.array([])
                
                # 해당 클래스의 예측 박스들
                cls_pred_indices = [i for i, name in enumerate(det_pred_labels) if name == cls_name]
                cls_pred_boxes = det_pred_boxes[cls_pred_indices] if len(cls_pred_indices) > 0 else np.array([])

                num_gt = len(cls_gt_boxes)
                num_pred = len(cls_pred_boxes)
                
                det_stats[cls_name]['gt_count'] += num_gt

                if num_pred == 0:
                    det_stats[cls_name]['fn'] += num_gt
                    continue
                
                if num_gt == 0:
                    det_stats[cls_name]['fp'] += num_pred
                    continue

                # IoU 매칭 (Greedy Matching)
                ious = calculate_iou_cpu(cls_pred_boxes, cls_gt_boxes) # (num_pred, num_gt)
                
                # 매칭된 GT를 추적하기 위한 마스크
                gt_matched = np.zeros(num_gt, dtype=bool)
                pred_matched = np.zeros(num_pred, dtype=bool)
                
                # IoU가 높은 순서대로 매칭
                if ious.size > 0:
                    # (pred_idx, gt_idx) 형태의 인덱스 쌍을 iou 내림차순으로 정렬
                    sorted_indices = np.argsort(-ious.flatten())
                    
                    for idx in sorted_indices:
                        p_idx, g_idx = np.unravel_index(idx, ious.shape)
                        
                        if pred_matched[p_idx] or gt_matched[g_idx]:
                            continue # 이미 매칭된 박스는 패스
                        
                        if ious[p_idx, g_idx] >= iou_threshold:
                            det_stats[cls_name]['tp'] += 1
                            pred_matched[p_idx] = True
                            gt_matched[g_idx] = True
                
                # 매칭되지 않은 Pred는 FP
                det_stats[cls_name]['fp'] += np.sum(~pred_matched)
                # 매칭되지 않은 GT는 FN
                det_stats[cls_name]['fn'] += np.sum(~gt_matched)

        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            continue

    # ------------------------------------------------------------------
    # 4. 결과 리포트 출력
    # ------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("  Evaluation Report 1: Classification Accuracy (RPN Matched)")
    logger.info("  (RPN이 찾은 박스를 RF가 얼마나 잘 분류했는지)")
    logger.info("="*60)
    if all_match_y_true:
        acc = accuracy_score(all_match_y_true, all_match_y_pred)
        cls_report = classification_report(all_match_y_true, all_match_y_pred, digits=4, zero_division=0)
        logger.info(f"Classification Accuracy: {acc*100:.2f}%")
        logger.info("\n" + cls_report)
    else:
        logger.warning("No matched samples for classification report.")

    logger.info("\n" + "="*60)
    logger.info(f"  Evaluation Report 2: Object Detection Performance (IoU={iou_threshold})")
    logger.info("  (실제 GT 대비 검출 성공률 - Precision / Recall / F1)")
    logger.info("="*60)
    
    logger.info(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'GT Count':<10}")
    logger.info("-" * 60)
    
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls_name, stats in det_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        gt_cnt = stats['gt_count']
        
        total_tp += tp; total_fp += fp; total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{cls_name:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}     {gt_cnt:<10}")
    
    logger.info("-" * 60)
    # 전체 평균 (Micro Average)
    m_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    m_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    m_f1 = 2 * (m_prec * m_rec) / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0
    logger.info(f"{'Micro Avg':<15} {m_prec:.4f}     {m_rec:.4f}     {m_f1:.4f}     {total_tp+total_fn:<10}")
    logger.info("="*60)