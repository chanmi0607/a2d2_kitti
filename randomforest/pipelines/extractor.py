import torch
import numpy as np
from tqdm import tqdm # 진행률 표시
import pandas as pd
from pathlib import Path

from pcdet.models import load_data_to_gpu
#from randomforest.dataset.gt_parser import load_gt_boxes
from randomforest.features.point_ops import extract_point_features_cpu
from randomforest.features.matcher import match_rpn_to_gt_for_training
from randomforest.config import ML_FEATURE_COLUMNS
from randomforest.dataset.dataset_loader import setup_dataset
from pcdet.models import build_network
# (시각화 유틸리티 임포트는 원본과 동일)
try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    try:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V
        OPEN3D_FLAG = False
    except ImportError: V = None; OPEN3D_FLAG = False


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
            
            raw_points_np = data_dict['points']

            gt_boxes_np = data_dict['gt_boxes']
            gt_names = data_dict.get('gt_names', [])

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

            # ============================================================
            # [수정됨] 시각화 보정 파트
            # ============================================================
            if not args.no_vis and index < args.vis_frame_limit:
                logger.info(f"Visualizing frame {frame_id}...")
                try:
                    # 1. RPN 박스 복사 및 회전축 반전 (필수)
                    vis_rpn_boxes = rpn_boxes_np_filtered.copy() 
                    vis_gt_boxes = gt_boxes_np.copy()

                    # 시각화 실행
                    V.draw_scenes(
                        points=raw_points_np[:, :3], 
                        gt_boxes=vis_gt_boxes,       # 수정된 GT 사용
                        ref_boxes=vis_rpn_boxes,     # 수정된 RPN 사용
                        ref_scores=rpn_scores_np_filtered.flatten() 
                    )
                except ImportError:
                    logger.warning("Visual_utils or Mayavi/Open3D not found. Skipping visualization.")
                except Exception as e:
                    logger.error(f"Visualization failed: {e}")

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
