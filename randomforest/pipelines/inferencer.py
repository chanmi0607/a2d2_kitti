import joblib
from tqdm import tqdm # 진행률 표시
import pandas as pd
import numpy as np
import torch

from pcdet.models import build_network, load_data_to_gpu
from randomforest.features.point_ops import extract_point_features_cpu
from randomforest.dataset.dataset_loader import setup_dataset
from randomforest.config import STAGE_1_OBJECT_LABEL, ML_FEATURE_COLUMNS

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

