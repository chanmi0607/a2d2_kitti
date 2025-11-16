# RPN 예측 박스들을 뽑고 GT와 IoU 매칭하여 Background만 남김 (30m 이내만)

import argparse
import numpy as np
import torch
from pathlib import Path
import warnings
import time
import pandas as pd
from tqdm import tqdm # 진행률 표시

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# OpenPCDet 임포트
from pcdet.config import cfg, cfg_from_yaml_file
# --- [수정] A2D2Dataset 임포트 ---
from pcdet.datasets.a2d2.a2d2_dataset import A2D2Dataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# (시각화 유틸리티 임포트는 원본과 동일)
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    try:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V
        OPEN3D_FLAG = False
    except ImportError: V = None; OPEN3D_FLAG = False


# --- 설정 파싱 함수 (수정) ---
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default='tools/cfgs/a2d2_models/second.yaml',
                        help='specify the config file')
    
    
    parser.add_argument('--split', type=str, default='train',  # split train or val
                        help='Which data split to process: train or val')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save the final combined CSV file. (default: rpn_features_[split].csv)')
    # ---
    
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth',
                        help='specify the pretrained model checkpoint (.pth)')
    parser.add_argument('--min_points_in_box', type=int, default=3,
                        help='Minimum number of points required inside a box to extract features')
    
    # --- [추가] Background 라벨링 임계값 ---
    parser.add_argument('--fg_thresh', type=float, default=0.2,
                        help='IoU threshold for a proposal to be "Foreground"')
    parser.add_argument('--bg_thresh', type=float, default=0.5,
                        help='IoU threshold for a proposal to be "Background" (below this value)')
    # ---
    
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    parser.add_argument('--vis_frame_limit', type=int, default=5, help='Limit visualization to first N frames')


    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # (선택) 데이터 증강 비활성화 (피처 추출 시 권장)
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
    
    # (선택) NMS 설정 오버라이드 (더 많은 Proposal을 보려면)
    # cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.1
    # cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH = 0.7
    
    if args.output_csv is None:
        args.output_csv = f'data/a2d2/new_background_features_{args.split}.csv'
    
    return args, cfg

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


# --- [수정] "Ignore" 라벨을 포함하는 GT 매칭 함수 ---
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

# --- (원본 함수) 피처 추출 CPU 함수 ---
# (extract_point_features_cpu 함수는 원본과 동일하므로 생략)
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
# --- CPU 함수 끝 ---


# --- 메인 실행 함수 (대폭 수정) ---
def main():
    main_start_time = time.time()
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Generate RPN Features for Training -------------------------')
    
    # 1. A2D2Dataset 준비 (전체 스플릿)
    try:
        # [수정] args.split 값에 따라 training 플래그 결정
        is_training_split = (args.split == 'train')

        dataset = A2D2Dataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
            training=is_training_split, # <--- 'train'일 때만 True
            logger=logger
        )
        dataset.set_split(args.split)
        logger.info(f"Loaded {args.split} split with {len(dataset.sample_id_list)} frames (from ImageSets).")

        logger.info(f"Loaded {args.split} split with {len(dataset)} frames.")
        
        # GT 라벨 경로 확인
        gt_label_dir = dataset.root_path / f'new_label/label_new_{args.split}'
        if not gt_label_dir.exists():
            logger.error(f"GT label directory not found: {gt_label_dir}")
            logger.error("Please run 'create_new_label_files.py' first.")
            return

    except Exception as e:
        logger.error(f"Error initializing dataset: {e}"); return

    # 2. 모델 빌드 및 로드 (원본과 동일)
    try:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda(); model.eval()
    except Exception as e:
        logger.error(f"Error building/loading model: {e}"); return

    # 3. CSV 컬럼 이름 정의
    # (RF 학습에 사용할 18개 피처)
    columns = [
        "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
        "num_points", "width", "length", "height", "density",
        "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std", 
        "label", "max_iou", "frame_id"
    ]
    rpn_feature_names = columns[1:8] # x, y, z, l, w, h, yaw
    point_feature_names = columns[8:18] # num_points ... intensity_std

    # 4. 모든 프레임을 순회하며 피처 추출
    all_features_list = [] # 모든 프레임의 DataFrame을 저장할 리스트

    logger.info(f"Starting feature extraction for {len(dataset)} frames...")
    
    for index in tqdm(range(len(dataset)), desc=f"Processing {args.split} split"):
        try:
            # 4-1. 데이터 로드
            data_dict = dataset[index]
            frame_id = data_dict['frame_id']
            #raw_points_np = data_dict['raw_points'] # 원본 포인트
            try:
                raw_points_np = dataset.get_lidar(frame_id)
            except Exception as e:
                logger.warning(f"Frame {frame_id}: 원본 LiDAR 로드 실패: {e}. 건너뜁니다.")
                continue # 이 프레임 스킵
            
            data_dict = dataset.collate_batch([data_dict]) # 배치 생성
            load_data_to_gpu(data_dict)

            # 4-2. 모델 추론 (RPN + NMS)
            with torch.no_grad():
                 # 모델 forward -> NMS까지 모두 수행된 결과
                 pred_dicts, _ = model(data_dict)

            # --- [NMS 적용] ---
            # NMS를 통과한 최종 Proposal 박스/점수/라벨(DL)을 가져옴
            post_nms_boxes_tensor = pred_dicts[0]['pred_boxes']
            post_nms_scores_tensor = pred_dicts[0]['pred_scores']
            # post_nms_labels_tensor = pred_dicts[0]['pred_labels'] # (참고: 이것은 DL 헤드의 예측)
            
            if post_nms_boxes_tensor.shape[0] == 0:
                # NMS 통과한 박스가 없으면 스킵
                continue

            rpn_boxes_np_filtered = post_nms_boxes_tensor.cpu().numpy()
            rpn_scores_np_filtered = post_nms_scores_tensor.cpu().numpy().reshape(-1, 1)
            # --------------------

            # 4-3. 포인트 기반 피처 추출 (CPU)
            point_features_np, valid_indices_np = extract_point_features_cpu(
                raw_points_np, rpn_boxes_np_filtered, args.min_points_in_box, logger
            )
            
            if valid_indices_np.shape[0] == 0:
                # 피처 추출에 성공한 박스가 없으면 스킵
                continue
            
            # 4-4. 유효한 박스들만 필터링
            final_boxes_np = rpn_boxes_np_filtered[valid_indices_np]
            final_scores_np = rpn_scores_np_filtered[valid_indices_np]
            # final_dl_labels_np = post_nms_labels_tensor.cpu().numpy()[valid_indices_np]

            # 4-5. 최종 피처 벡터 결합
            # [N, 1] (Score) + [N, 7] (Box) = [N, 8]
            final_rpn_features_np = np.concatenate((final_scores_np, final_boxes_np), axis=1)
            # [N, 8] (RPN) + [N, 10] (Point) = [N, 18]
            final_features_np = np.concatenate((final_rpn_features_np, point_features_np), axis=1)

            # 4-6. GT 매칭을 통한 라벨(Background/Car) 부여
            gt_txt_path = gt_label_dir / f"{frame_id}.txt"
            gt_boxes_np, gt_names = load_gt_boxes(gt_txt_path)
            
            matched_labels, matched_ious_np = match_rpn_to_gt_for_training(
                final_boxes_np, gt_boxes_np, gt_names, 
                args.fg_thresh, args.bg_thresh
            )

            # 4-7. DataFrame 생성
            df = pd.DataFrame(final_features_np, columns=columns[:-3]) # label, max_iou, frame_id 제외
            df["label"] = matched_labels
            df["max_iou"] = matched_ious_np
            df["frame_id"] = frame_id
            
            # 4-8. "Ignore" 샘플 제거 (요청사항 2)
            df_filtered = df[df['label'] != 'Ignore']
            
            if not df_filtered.empty:
                all_features_list.append(df_filtered)

            # 4-9. 시각화 (선택 사항)
            if not args.no_vis and index < args.vis_frame_limit:
                logger.info(f"\nVisualizing frame {frame_id} (first {args.vis_frame_limit} frames)...")
                # GT 매칭된 라벨을 숫자 ID로 변환
                class_to_id_map = {name: i + 1 for i, name in enumerate(cfg.CLASS_NAMES)}
                class_to_id_map['Background'] = 0
                gt_matched_numeric_labels_np = np.array([
                    class_to_id_map.get(name, 0) for name in matched_labels 
                ])
                
                V.draw_scenes(
                    points=raw_points_np[:, :3],
                    ref_boxes=final_boxes_np,
                    ref_scores=final_scores_np.squeeze(),
                    ref_labels=gt_matched_numeric_labels_np,
                    gt_boxes=gt_boxes_np
                )
                if not OPEN3D_FLAG: mlab.show(stop=True)

        except Exception as e:
            logger.error(f"\n[치명적 오류] Frame {index} ({frame_id}) 처리 중 실패: {e}")
            import traceback; traceback.print_exc()
            logger.warning("이 프레임을 건너뛰고 계속합니다.")
            continue # 다음 프레임으로

    # 5. 모든 피처 결합 및 CSV 저장
    if not all_features_list:
        logger.warning("추출된 피처가 없습니다. CSV 파일이 생성되지 않습니다.")
        return

    logger.info("\nCombining features from all frames...")
    final_df = pd.concat(all_features_list, ignore_index=True)
    
    # 저장
    # 1. 원하는 최종 CSV 컬럼 순서 정의
    #    (사용자 f-string 순서 + label, max_iou, frame_id)
    #    ※ 주의: 스크립트의 컬럼 이름과 매칭 ('actual_width' -> 'length' 등)
    desired_column_order = [
        'label',           # 라벨 (추가됨)
        'RPN_MaxScore',    # f-string: rpn_score
        'x',               # f-string: x
        'y',               # f-string: y
        'z',               # f-string: z
        'l',               # f-string: l_lidar (RPN 박스 길이)
        'w',               # f-string: w_lidar (RPN 박스 너비)
        'h',               # f-string: h_lidar (RPN 박스 높이)
        'yaw',             # f-string: yaw
        'num_points',      # f-string: num_points
        'length',          # f-string: actual_width (포인트 분포 Y 길이) <- 이름 주의!
        'width',           # f-string: actual_length (포인트 분포 X 너비) <- 이름 주의!
        'height',          # f-string: actual_height (포인트 분포 Z 높이) <- 이름 주의!
        'density',         # f-string: density
        'aspect_ratio',    # f-string: aspect_ratio (AABB 기반 비율)
        'mean_z',          # f-string: mean_z
        'std_z',           # f-string: std_z
        'intensity_mean',  # f-string: intensity_mean
        'intensity_std',   # f-string: intensity_std
        'max_iou',         # 최대 IoU (추가됨)
        'frame_id'         # 프레임 ID (추가됨)
    ]

    # 2. DataFrame의 컬럼 순서를 위 리스트대로 재정렬
    try:
        final_df_reordered = final_df[desired_column_order]
    except KeyError as e:
        logger.error(f"컬럼 이름 오류: {e}. 'desired_column_order' 리스트를 확인하세요.")
        logger.error(f"사용 가능한 컬럼: {final_df.columns.tolist()}")
        return # 오류 발생 시 중단

    # 3. 'label'이 'Background'인 행만 선택
    #final_df_background_only = final_df_reordered[final_df_reordered['label'] == 'Background'].copy()
    final_df_fg_bg = final_df_reordered[final_df_reordered['label'] != 'Ignore'].copy()
    logger.info(f"Filtered to keep only FG/BG (Ignore removed): {len(final_df_reordered)} -> {len(final_df_fg_bg)}")



    # --- ▼▼▼▼▼ 거리 필터링 추가 ▼▼▼▼▼ ---
    # 4. 거리 계산 (XY 평면 기준)
    distances = np.sqrt(final_df_fg_bg['x']**2 + final_df_fg_bg['y']**2)

    # 5. 거리 30m 이내인 샘플만 선택
    distance_threshold = 30.0
    final_df_fg_bg_nearby = final_df_fg_bg[distances <= distance_threshold].copy()
    logger.info(f"Filtered by distance <= {distance_threshold}m: {len(final_df_fg_bg)} -> {len(final_df_fg_bg_nearby)}")
    # --- ▲▲▲▲▲ 거리 필터링 추가 끝 ▲▲▲▲▲ ---

    # 저장
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # final_df.to_csv(output_path, index=False) # <--- 기존 코드
    #final_df_reordered.to_csv(output_path, index=False) # <--- 재정렬된 DataFrame 저장
    final_df_fg_bg_nearby['label'] = final_df_fg_bg_nearby['label'].apply(
        lambda x: 'Object' if x != 'Background' else 'Background'
    )

    final_df_fg_bg_nearby.to_csv(output_path, index=False)

    # --- ▲▲▲▲▲ 수정된 부분 끝 ▲▲▲▲▲ ---
    
    logger.info(f"✅ 최종 피처 파일 저장 완료: {output_path}")
    logger.info(f"총 {len(final_df_fg_bg_nearby)}개의 샘플 (Proposal)이 저장되었습니다.")
    
    # --- [요청사항 4] 백그라운드 구분이 잘 되었는지 확인 ---
    logger.info("\n--- 최종 라벨 분포 (RF 학습에 사용될 샘플) ---")
    print(final_df_fg_bg['label'].value_counts())
    logger.info("--------------------------------------------------")

    total_time = time.time() - main_start_time
    logger.info(f"총 실행 시간: {total_time:.2f} 초")


if __name__ == '__main__':
    main()