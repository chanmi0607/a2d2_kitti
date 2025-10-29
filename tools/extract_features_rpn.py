# tools/extract_rpn_features_single_frame.py

import argparse
import numpy as np
import torch
from pathlib import Path
import warnings
import time

# 경고 메시지 무시 (선택 사항)
warnings.filterwarnings("ignore", category=FutureWarning)

# OpenPCDet 임포트
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

try:
    import open3d
    from visual_utils import open3d_vis_utils as V # Open3D 사용
    OPEN3D_FLAG = True
    print("Using Open3D for visualization.")
except ImportError:
    try:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V # Mayavi 사용
        OPEN3D_FLAG = False
        print("Open3D not found, using Mayavi for visualization.")
    except ImportError:
        print("Neither Open3D nor Mayavi found for visualization. Skipping visualization.")
        V = None # 시각화 비활성화
        OPEN3D_FLAG = False

# --- DemoDataset 정의 (단일 파일 처리용) ---
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path) if root_path is not None else None
        self.ext = ext
        # 단일 파일 경로만 리스트에 저장
        self.sample_file_list = [self.root_path] if self.root_path and self.root_path.is_file() else []
        if not self.sample_file_list: logger.warning(f"File not found or not specified: {root_path}")
    def __len__(self): return len(self.sample_file_list)
    def __getitem__(self, index):
        if not self.sample_file_list: raise IndexError("No file to process")
        current_file = self.sample_file_list[0] # 항상 첫 번째(유일한) 파일
        points = self.get_lidar(current_file)
        input_dict = {'points': points, 'frame_id': current_file.stem, 'raw_points': points.copy()}
        # --- [수정] 끝 ---
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    def get_lidar(self, file_path):
        if not file_path.exists(): raise FileNotFoundError(f"LiDAR file not found: {file_path}")
        points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
        return points
# --- DemoDataset 끝 ---


# --- 설정 파싱 함수 ---
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # --- ▼▼▼▼▼ 수정 ▼▼▼▼▼ ---
    parser.add_argument('--cfg_file', type=str,
                        default='tools/cfgs/a2d2_models/second.yaml', # 👈 본인 모델 설정 파일 경로
                        help='specify the config file')
    parser.add_argument('--data_path', type=str,
                        default='data/a2d2/training/velodyne/000000074.bin', # 👈 테스트할 .bin 파일 *하나*의 경로
                        help='specify the single point cloud data file')
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth', # 체크포인트 필수
                        help='specify the pretrained model checkpoint (.pth)')
    parser.add_argument('--vis_thresh', type=float, default=0.2,
                        help='Score threshold for visualizing RPN proposals (0.0 to show all - may lag)')
    parser.add_argument('--min_points_in_box', type=int, default=5,
                        help='Minimum number of points required inside a box to extract features')
    # --- ▲▲▲▲▲ 수정 ▲▲▲▲▲ ---
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    # 데이터 증강 비활성화 (선택 사항, 필요 시)
    # cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
    return args, cfg

def load_gt_boxes(gt_txt_path):
    gt_boxes, gt_names = [], []
    with open(gt_txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls_name = parts[0]
            # A2D2: x(2), y(3), z(4), l(5), w(6), h(7), yaw(8)
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            l, w, h, yaw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            gt_boxes.append([x, y, z, l, w, h, yaw])
            gt_names.append(cls_name)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)

    # ⭐ 핵심 수정: 하단(z_min) 기준 → 중심(z_center) 기준
    gt_boxes[:, 2] += gt_boxes[:, 5] / 2.0

    return gt_boxes, gt_names


from pcdet.ops.iou3d_nms import iou3d_nms_utils

def match_rpn_to_gt(rpn_boxes, gt_boxes, gt_labels, iou_thresh=0.5):
            """RPN box를 GT box와 IoU 매칭하여 라벨과 최대 IoU 부여"""
            num_rpn_boxes = rpn_boxes.shape[0]
            
            if gt_boxes.shape[0] == 0:
                # GT가 없으면 라벨은 'Background', IoU는 0.0
                return ["Background"] * num_rpn_boxes, np.zeros(num_rpn_boxes, dtype=np.float32)

            # IoU 계산 (N_rpn, N_gt)
            ious = iou3d_nms_utils.boxes_iou3d_gpu(
                torch.from_numpy(rpn_boxes).cuda(),
                torch.from_numpy(gt_boxes).cuda()
            ).cpu().numpy()

            # 각 RPN 박스마다 가장 IoU가 높은 GT 박스의 인덱스와 IoU 값 찾기
            best_gt_indices = np.argmax(ious, axis=1) # (N_rpn,)
            best_ious_np = ious[np.arange(num_rpn_boxes), best_gt_indices] # (N_rpn,)

            matched_labels = []
            for i in range(num_rpn_boxes):
                best_iou = best_ious_np[i]
                if best_iou >= iou_thresh:
                    # 임계값 이상이면 해당 GT 라벨
                    matched_labels.append(gt_labels[best_gt_indices[i]])
                else:
                    # 임계값 미만이면 'Background'
                    matched_labels.append("Background")
            
            # [수정] 라벨 리스트와 최대 IoU 점수 배열을 함께 반환
            return matched_labels, best_ious_np

# --- CPU 기반 포인트 특징 추출 함수 ---
def points_in_rotated_box_cpu(points, box):
    """ CPU에서 주어진 박스 내부에 있는 포인트들의 인덱스를 반환 """
    center = box[:3]
    dims = box[3:6] # l, w, h
    yaw = box[6]
    points_translated = points[:, :3] - center
    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw, 0], [sin_yaw,  cos_yaw, 0], [0, 0, 1]
    ])
    points_rotated = points_translated @ rot_matrix.T
    half_dims = dims / 2.0
    mask = (np.abs(points_rotated[:, 0]) <= half_dims[0]) & \
           (np.abs(points_rotated[:, 1]) <= half_dims[1]) & \
           (np.abs(points_rotated[:, 2]) <= half_dims[2])
    return np.where(mask)[0]

# --- CPU 함수 (최적화 버전) ---
# --- CPU 함수 (최적화 버전) ---
def extract_point_features_cpu(points_np, boxes_np, min_points_in_box, logger):
    """
    CPU(Numpy)와 OpenPCDet 유틸리티(roiaware_pool3d)를 사용하여
    각 박스 내부 포인트로부터 특징 추출 (최적화 버전)
    """
    num_boxes = boxes_np.shape[0]
    point_features_list = []
    valid_box_indices = [] # 특징 추출에 성공한 박스의 인덱스

    # 1. ... (roiaware_pool3d_utils.points_in_boxes_cpu 부분은 동일) ...
    try:
        points_tensor = torch.from_numpy(points_np[:, 0:3]).float()
        boxes_tensor = torch.from_numpy(boxes_np).float()
        
        point_indices_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
            points_tensor, boxes_tensor
        ).numpy() # (num_boxes, num_points)
        
    except Exception as e:
        logger.error(f"  Fatal Error during roiaware_pool3d_utils.points_in_boxes_cpu: {e}")
        logger.error("  Falling back to slower NumPy implementation (if available) or failing.")
        return np.array([]), np.array([], dtype=int)


    # 2. 각 박스를 순회하며 특징 계산 (NumPy)
    for i in range(num_boxes):
        try:
            mask = point_indices_mask[i]
            points_in_box = points_np[mask.astype(bool)]
        except Exception as e:
            logger.warning(f"  Box {i}: Error finding points inside box (masking step): {e}. Skipping.")
            continue

        if points_in_box.shape[0] < min_points_in_box:
            continue # 포인트 부족 시 건너뛰기

        valid_box_indices.append(i) # 유효한 박스 인덱스 추가

        try:
            # --- ▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼ ---
            
            num_points = points_in_box.shape[0]

            # 특징 1: 포인트들의 실제 분포 범위 (AABB)
            min_coords = np.min(points_in_box[:, :3], axis=0)
            max_coords = np.max(points_in_box[:, :3], axis=0)
            dims = max_coords - min_coords
            width = dims[0]       # 포인트 분포의 너비
            length = dims[1]      # 포인트 분포의 길이
            height = dims[2]      # 포인트 분포의 높이
            
            # 특징 2: Proposal Box의 실제 부피 (l, w, h)
            # boxes_np[i]는 [x, y, z, l, w, h, yaw]
            box_l = boxes_np[i, 3]
            box_w = boxes_np[i, 4]
            box_h = boxes_np[i, 5]
            
            # [수정] 밀도 계산을 위한 부피: Proposal Box의 부피 사용
            box_volume = (box_l * box_w * box_h) + 1e-6 
            
            # [수정] 밀도(Density) = (포인트 수) / (Proposal Box의 부피)
            density = num_points / box_volume 

            # (기존 코드) AABB 기반 부피: 
            # volume = (width * length * height) + 1e-6
            # (기존 코드) 밀도: 
            # density = num_points / volume 
            # --- ▲▲▲▲▲ 수정 끝 ▲▲▲▲▲ ---

            # 나머지 특징 계산
            aspect_ratio = width / (length + 1e-6) # AABB 기반 비율 유지
            mean_z = np.mean(points_in_box[:, 2])
            std_z = np.std(points_in_box[:, 2])
            mean_intensity = np.mean(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0
            std_intensity = np.std(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0

            features = [
                num_points, 
                width,        # AABB 기반 너비 (포인트 분포)
                length,       # AABB 기반 길이 (포인트 분포)
                height,       # AABB 기반 높이 (포인트 분포)
                density,      # [수정됨] Box 부피 기반 밀도
                aspect_ratio, # AABB 기반 비율
                mean_z, std_z, mean_intensity, std_intensity
            ]
            point_features_list.append(features)
        except Exception as e:
            logger.warning(f"  Box {i}: Error calculating features for points in box: {e}. Skipping box.")
            if i == valid_box_indices[-1]:
                 valid_box_indices.pop()

    if not point_features_list:
        return np.array([]), np.array([], dtype=int)

    return np.array(point_features_list, dtype=np.float32), np.array(valid_box_indices, dtype=int)
# --- CPU 함수 끝 ---


# --- 메인 실행 함수 ---
def main():
    main_start_time = time.time()
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Extract RPN Features for a Single Frame -------------------------')

    # 1. DemoDataset 준비 (단일 파일)
    try:
        data_path = Path(args.data_path)
        if not data_path.is_file(): raise FileNotFoundError(f"Data file not found or is not a file: {data_path}")
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=data_path, logger=logger
        )
        if len(demo_dataset) == 0: raise ValueError("Dataset could not be initialized.")
        logger.info(f"Processing file: {data_path}")
    except Exception as e:
        logger.error(f"Error initializing dataset: {e}"); return

    # 2. 전체 모델 빌드 및 체크포인트 로드
    try:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda(); model.eval()
    except Exception as e:
        logger.error(f"Error building/loading model: {e}"); return

    # 3. 데이터 로드 및 전처리
    start_preprocess = time.time()
    try:
        data_dict = demo_dataset[0] # 단일 데이터 가져오기
        raw_points_np = data_dict['raw_points'] # 원본 포인트 클라우드 (Numpy)
        data_dict = demo_dataset.collate_batch([data_dict]) # 배치 생성
        load_data_to_gpu(data_dict)
        frame_id = data_dict['frame_id'][0]
    except Exception as e:
         logger.error(f"Error loading data: {e}"); return
    preprocess_time = time.time() - start_preprocess
    logger.info(f"===> Data loading & preprocessing took: {preprocess_time:.4f} seconds.")

    # 4. 모델 추론 실행
    torch.cuda.synchronize() # GPU 연산 동기화 (시작 전)
    start_inference = time.time()
    try:
         with torch.no_grad():
             # 모델 forward 실행 -> RPN 결과가 batch_dict에 저장됨
             pred_dicts, recall_dicts = model(data_dict)
         logger.info(f'Frame {frame_id}: Model inference done.')
    except Exception as e:
         logger.error(f"Frame {frame_id}: Error during inference: {e}"); return
    # --- [시간 측정 추가] ---
    torch.cuda.synchronize() # GPU 연산 동기화 (완료 대기)
    inference_time = time.time() - start_inference
    logger.info(f'Frame {frame_id}: Model inference done.')
    logger.info(f"===> GPU Inference Time (model forward): {inference_time:.4f} seconds.")
    # -------------------------

    # --- ▼▼▼▼▼ 특징 추출 시작 ▼▼▼▼▼ ---
    # 5. RPN 결과 추출 (NMS 전 단계)
    # batch_cls_preds: [Batch=1, N_anchors, Num_Classes_or_1] (로짓)
    # batch_box_preds: [Batch=1, N_anchors, BoxEncodingSize=7] (디코딩된 박스)
    rpn_cls_preds = data_dict.get('batch_cls_preds', None) # 로짓
    rpn_box_preds_tensor = data_dict.get('batch_box_preds', None) # 디코딩된 박스

    if rpn_cls_preds is None or rpn_box_preds_tensor is None:
         logger.warning(f"RPN predictions not found."); return

    rpn_cls_preds = rpn_cls_preds[0]
    rpn_box_preds_tensor = rpn_box_preds_tensor[0]

    num_proposals = rpn_box_preds_tensor.shape[0]
    if num_proposals == 0: logger.info(f"RPN generated 0 proposals."); return
    logger.info(f"RPN generated {num_proposals} proposals (before NMS).")

    # 6. RPN 기반 특징 생성 (점수 + 박스 정보, GPU Tensor)
    rpn_scores = torch.sigmoid(rpn_cls_preds)
    if rpn_scores.shape[-1] > 1:
         rpn_max_scores, pred_labels_tensor = torch.max(rpn_scores[:, 1:], dim=1)
         pred_labels_tensor += 1
    else:
         rpn_max_scores = rpn_scores[:, 0]
         pred_labels_tensor = torch.ones_like(rpn_max_scores, dtype=torch.long)

    # RPN 특징: [N_anchors, 8]
    rpn_feature_tensor = torch.cat(
        (rpn_max_scores.unsqueeze(-1), rpn_box_preds_tensor), dim=1
    )
    # CPU 계산을 위해 Numpy 배열로 변환
    rpn_features_np = rpn_feature_tensor.cpu().numpy()
    rpn_boxes_np = rpn_box_preds_tensor.cpu().numpy()
    pred_labels_np = pred_labels_tensor.cpu().numpy() # 시각화용 라벨

    # --- ▼▼▼▼▼ [수정] 7번 단계 이전에 스코어 필터링 ▼▼▼▼▼ ---
    
    # 7-1. RPN 스코어를 기준으로 먼저 필터링 (GPU에서 CPU로 넘어온 직후)
    rpn_max_scores_np = rpn_features_np[:, 0] # (RPN_MaxScore)
    score_mask = rpn_max_scores_np >= args.vis_thresh
    
    num_before_filter = rpn_boxes_np.shape[0]
    num_after_filter = np.sum(score_mask)

    if num_after_filter == 0:
        logger.info(f"No proposals with score >= {args.vis_thresh}. Skipping feature extraction.")
        logger.info('----------------- Feature Extraction Done -------------------------')
        return # 스크립트 종료

    logger.info(f"Filtered proposals by score >= {args.vis_thresh}: {num_before_filter} -> {num_after_filter}")

    # 필터링된 유효한 박스/특징만 남김
    rpn_boxes_np_filtered = rpn_boxes_np[score_mask]
    rpn_features_np_filtered = rpn_features_np[score_mask]
    pred_labels_np_filtered = pred_labels_np[score_mask]
    # --------------------------------------------------------

    # 7. 포인트 기반 특징 추출 (CPU 사용)
    start_time = time.time()
    logger.info("Extracting point features using CPU...")
    # CPU 함수 호출
    point_features_np, valid_indices_np = extract_point_features_cpu(
        raw_points_np, rpn_boxes_np_filtered, args.min_points_in_box, logger
    )
    extraction_time = time.time() - start_time
    logger.info(f"Point feature extraction took {extraction_time:.2f} seconds.")


    # 8. 최종 특징 벡터 결합 (Numpy 사용)
    if valid_indices_np.shape[0] > 0:
        # 유효한 인덱스에 해당하는 RPN 특징, 포인트 특징, 라벨, 박스 필터링
        final_rpn_features_np = rpn_features_np_filtered[valid_indices_np]
        final_point_features_np = point_features_np # 이미 유효한 것만 계산됨
        final_labels_np = pred_labels_np_filtered[valid_indices_np]
        final_boxes_np = rpn_boxes_np_filtered[valid_indices_np] # 시각화용
        final_scores_np = final_rpn_features_np[:, 0] # 시각화용 점수 (RPN MaxScore)

        # mask_vis = final_scores_np >= args.vis_thresh
        # if np.sum(mask_vis) == 0:
        #     logger.info(f"No proposals with score >= {args.vis_thresh}. Skipping feature logging.")
        #     return
        # final_rpn_features_np = final_rpn_features_np[mask_vis]
        # final_point_features_np = final_point_features_np[mask_vis]
        # final_labels_np = final_labels_np[mask_vis]
        # final_boxes_np = final_boxes_np[mask_vis]
        # final_scores_np = final_scores_np[mask_vis]

        # 최종 특징 벡터 결합: [N_valid, 8 + 10] = [N_valid, 18]
        final_features_np = np.concatenate((final_rpn_features_np, final_point_features_np), axis=1)
        # --- ▼▼▼▼▼ GT 라벨링 및 저장 추가 ▼▼▼▼▼ ---
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        import pandas as pd

        def load_gt_boxes(gt_txt_path):
            gt_boxes, gt_names = [], []
            with open(gt_txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_name = parts[0]
                    # A2D2 형식 기준: x(2), y(3), z(4), l(5), w(6), h(7), yaw(8)
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    l, w, h, yaw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                    gt_boxes.append([x, y, z, l, w, h, yaw])
                    gt_names.append(cls_name)
            return np.array(gt_boxes, dtype=np.float32), gt_names


        # --- GT 파일 경로 자동 지정 ---
        gt_dir = Path("/home/a/OpenPCDet/data/a2d2/new_label/label_new_train")
        gt_txt_path = gt_dir / f"{frame_id}.txt"

        if not gt_txt_path.exists():
            logger.warning(f"[GT Missing] {gt_txt_path} not found. Skipping GT matching.")
            matched_labels = ["Unknown"] * final_features_np.shape[0]
        else:
            logger.info(f"Using GT file: {gt_txt_path}")
            gt_boxes_np, gt_names = load_gt_boxes(gt_txt_path)
            matched_labels, matched_ious_np = match_rpn_to_gt(final_boxes_np, gt_boxes_np, gt_names, iou_thresh=0.1)

        # --- feature + label 결합 및 저장 ---
        columns = [
            "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
            "num_points", "width", "length", "height", "density",
            "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std", "label", "max_iou"
        ]

        df = pd.DataFrame(final_features_np, columns=columns[:-2])
        df["label"] = matched_labels
        df["max_iou"] = matched_ious_np

        save_dir = Path("output/features")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{frame_id}_rpn_features.csv"

        df.to_csv(save_path, index=False)
        logger.info(f"✅ Saved RPN features with GT labels to: {save_path}")
        logger.info(f"Sample labels: {df['label'].value_counts().to_dict()}")
        # --- ▲▲▲▲▲ GT 라벨링 및 저장 추가 ▲▲▲▲▲ ---

        # --- ▼▼▼▼▼ [신규 추가] ▼▼▼▼▼ ---
        # 시각화를 위해 GT 매칭된 라벨(string)을 숫자 ID로 변환합니다.
        # (V.draw_scenes는 숫자 라벨을 받습니다)
        
        # 1. 클래스 이름 -> 숫자 ID 맵 생성 (e.g., 'Car': 1, 'Cyclist': 2)
        #    OpenPCDet은 1-based index를 사용합니다.
        class_to_id_map = {name: i + 1 for i, name in enumerate(cfg.CLASS_NAMES)}
        # 2. 'Background'와 'Unknown'은 0으로 매핑
        class_to_id_map['Background'] = 0
        class_to_id_map['Unknown'] = 0 # GT 파일이 없는 경우 대비

        # 3. 'matched_labels' (['Car', 'Background', ...]) 리스트를 숫자 numpy 배열로 변환
        #    (이 배열은 final_boxes_np와 길이가 같습니다)
        gt_matched_numeric_labels_np = np.array([
            class_to_id_map.get(name, 0) for name in matched_labels 
        ])
        # --- ▲▲▲▲▲ [신규 추가 끝] ▲▲▲▲▲ ---


        

        # 결과 출력
        logger.info(f"\n--- Combined Features for Frame {frame_id} (CPU Extracted) ---")
        logger.info(f"Shape of final feature vectors (X): {final_features_np.shape}")
        logger.info("Feature columns: [RPN_MaxScore, x, y, z, l, w, h, yaw, num_points, width, length, height, density, aspect_ratio, mean_z, std_z, intensity_mean, intensity_std]")
        logger.info("\nSample Final Feature Vectors (first 5):")
        for i in range(min(100, final_features_np.shape[0])):
            feature_str = ", ".join([f"{val:.4f}" for val in final_features_np[i]])
            label_name = matched_labels[i]
            iou_score = matched_ious_np[i] # [수정] 매칭된 IoU 값
            
            # [수정] 로그에 IoU 값 추가
            if label_name == "Background":
                # Background일 경우, 최대 IoU 값을 함께 표시
                logger.info(f"  Valid Proposal {i}: [{feature_str}] (Label: {label_name}, Max_IoU: {iou_score:.4f})")
            else:
                # Background가 아닐 경우 (GT와 매칭됨)
                logger.info(f"  Valid Proposal {i}: [{feature_str}] (Label: {label_name}, IoU: {iou_score:.4f})")

        # 9. 시각화
        if V is not None:
            logger.info(f"\nVisualizing RPN proposals with combined features...")
            try:
                score_thresh_vis = args.vis_thresh
                mask_vis = final_scores_np >= score_thresh_vis
                num_boxes_to_draw = np.sum(mask_vis)

                if num_boxes_to_draw == 0:
                    logger.info(f"  No proposals above score {score_thresh_vis}. Showing points only.")
                    V.draw_scenes(points=raw_points_np[:, :3])
                else:
                    logger.info(f"  Visualizing {num_boxes_to_draw} proposals (Labels based on GT IoU) ...")
                    V.draw_scenes(
                        points=raw_points_np[:, :3],
                        ref_boxes=final_boxes_np[mask_vis],
                        ref_scores=final_scores_np[mask_vis],
                        # ref_labels=final_labels_np[mask_vis]  # <--- (기존) RPN 예측 라벨
                        ref_labels=gt_matched_numeric_labels_np[mask_vis], # <--- (수정) GT 매칭 라벨
                        gt_boxes=gt_boxes_np
                    )
                if not OPEN3D_FLAG: mlab.show(stop=True)
            except Exception as e:
                logger.error(f"  Error during visualization: {e}")
                import traceback; traceback.print_exc()
        else: logger.warning("Visualization utility (V) is not available.")

    else: # 유효한 proposal이 하나도 없는 경우
        logger.info("No proposals with enough points were found to extract features.")

    logger.info('----------------- Feature Extraction Done -------------------------')
    # --- [시간 측정 추가] ---
    total_time = time.time() - main_start_time
    logger.info(f"\nTotal script execution time: {total_time:.4f} seconds.")
    # --- ▲▲▲▲▲ 특징 추출 수정 ▲▲▲▲▲ ---

# --- 스크립트 직접 실행 시 main() 호출 ---
if __name__ == '__main__':
    main()

