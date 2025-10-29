# import argparse
# import glob
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn # extract_point_features_cpu의 의존성
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# # OpenPCDet 임포트 (demo.py 기반)
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils
# # 3D 시각화 유틸리티
# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except ImportError:
#     print('Open3D not installed. Visualization disabled.')
#     V = None
#     OPEN3D_FLAG = False

# # --- [!] RPN -> RF 피처 추출에 필요한 의존성 ---
# from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# # --- [!] 님이 제공한 10개 피처 추출 함수 ---
# def extract_point_features_cpu(points_np, boxes_np, min_points_in_box, logger):
#     """
#     CPU(Numpy)와 OpenPCDet 유틸리티(roiaware_pool3d)를 사용하여
#     각 박스 내부 포인트로부터 특징 추출 (최적화 버전)
#     (제공해주신 함수와 동일)
#     """
#     num_boxes = boxes_np.shape[0]
#     point_features_list = []
#     valid_box_indices = [] # 특징 추출에 성공한 박스의 인덱스

#     try:
#         # GPU 사용이 가능하면 GPU를 쓰도록 수정 (속도 향상)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         points_tensor = torch.from_numpy(points_np[:, 0:3]).float().to(device)
#         boxes_tensor = torch.from_numpy(boxes_np).float().to(device)
        
#         # points_in_boxes_cpu가 실제로는 GPU도 지원함
#         point_indices_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
#             points_tensor, boxes_tensor
#         ).cpu().numpy() # (num_boxes, num_points)
        
#     except Exception as e:
#         logger.error(f"  Fatal Error during roiaware_pool3d_utils.points_in_boxes_cpu: {e}")
#         return np.array([]), np.array([], dtype=int)

#     # 2. 각 박스를 순회하며 특징 계산 (NumPy)
#     for i in range(num_boxes):
#         try:
#             mask = point_indices_mask[i]
#             points_in_box = points_np[mask.astype(bool)]
#         except Exception as e:
#             continue

#         if points_in_box.shape[0] < min_points_in_box:
#             continue # 포인트 부족 시 건너뛰기

#         valid_box_indices.append(i) # 유효한 박스 인덱스 추가

#         try:
#             num_points = points_in_box.shape[0]

#             min_coords = np.min(points_in_box[:, :3], axis=0)
#             max_coords = np.max(points_in_box[:, :3], axis=0)
#             dims = max_coords - min_coords
#             width = dims[0]       # 포인트 분포의 너비 (x)
#             length = dims[1]      # 포인트 분포의 길이 (y)
#             height = dims[2]      # 포인트 분포의 높이 (z)
            
#             box_l, box_w, box_h = boxes_np[i, 3], boxes_np[i, 4], boxes_np[i, 5]
#             box_volume = (box_l * box_w * box_h) + 1e-6 
#             density = num_points / box_volume 

#             aspect_ratio = width / (length + 1e-6) 
#             mean_z = np.mean(points_in_box[:, 2])
#             std_z = np.std(points_in_box[:, 2])
#             mean_intensity = np.mean(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0
#             std_intensity = np.std(points_in_box[:, 3]) if points_np.shape[1] > 3 else 0

#             features = [
#                 num_points, 
#                 width, length, height,
#                 density, aspect_ratio, 
#                 mean_z, std_z, mean_intensity, std_intensity
#             ]
#             point_features_list.append(features)
#         except Exception as e:
#             if i in valid_box_indices:
#                  valid_box_indices.pop()

#     if not point_features_list:
#         return np.array([]), np.array([], dtype=int)

#     return np.array(point_features_list, dtype=np.float32), np.array(valid_box_indices, dtype=int)

# # --- [!] 님이 제공한 18개 피처 순서 (글로벌 변수) ---
# # (RPN 8개 + 위 함수 10개)
# RF_FEATURE_COLUMNS = [
#     "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
#     "num_points", "width", "length", "height", "density",
#     "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std"
# ]

# # --- demo.py의 DemoDataset 클래스 (원본과 동일) ---
# class DemoDataset(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin'):
#         super().__init__(
#             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
#         )
#         self.root_path = root_path
#         self.ext = ext
#         data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
#         data_file_list.sort()
#         self.sample_file_list = data_file_list

#     def __len__(self):
#         return len(self.sample_file_list)

#     def __getitem__(self, index):
#         if self.ext == '.bin':
#             points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
#         elif self.ext == '.npy':
#             points = np.load(self.sample_file_list[index])
#         else:
#             raise NotImplementedError

#         input_dict = {
#             'points': points,
#             'frame_id': index,
#             # 'raw_points': points.copy() # (참고: prepare_data가 points를 수정함)
#         }

#         data_dict = self.prepare_data(data_dict=input_dict)
#         return data_dict

# # --- demo.py의 parse_config (RF 모델 경로 추가) ---
# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default='tools/cfgs/a2d2_models/second.yaml',
#                         help='specify the config for RPN')
#     parser.add_argument('--data_path', type=str, default='data/a2d2/training/velodyne',
#                         help='specify the point cloud data file or directory')
#     parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth', help='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth')
#     parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    
#     # --- [!] RF 모델 경로 인자 추가 ---
#     parser.add_argument('--model1_path', type=str, 
#                         default='data/a2d2/rf_stage1_model.pkl', 
#                         help='Path to rf_stage1_model.pkl')
#     parser.add_argument('--model2_path', type=str, 
#                         default='data/a2d2/rf_stage2_model.pkl', 
#                         help='Path to rf_stage2_model.pkl')
#     parser.add_argument('--encoder_path', type=str, 
#                         default='data/a2d2/rf_stage2_encoder.pkl', 
#                         help='Path to rf_stage2_encoder.pkl')
#     parser.add_argument('--min_points_in_box', type=int, default=3,
#                         help='Minimum points to extract features for RF')
    
#     args = parser.parse_args()
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     return args, cfg

# # --- [!] "실시간" 메인 함수 ---
# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger()
#     logger.info('----------------- RPN + RF Cascade Demo -----------------')

#     # --- 1. 모든 모델 로드 (시작 시 1회) ---
#     logger.info("모든 모델을 메모리에 로드합니다...")

#     # --- 2. 데모 데이터 로더 준비 ---
#     demo_dataset = DemoDataset(
#         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#         root_path=Path(args.data_path), ext=args.ext, logger=logger
#     )
#     logger.info(f"데이터셋에서 총 {len(demo_dataset)}개의 프레임 발견.")

#     rpn_model = build_network(
#         model_cfg=cfg.MODEL,
#         num_class=len(cfg.CLASS_NAMES),
#         dataset=demo_dataset # <-- 생성된 데이터셋 객체를 여기에 전달
#     )
#     rpn_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
#     rpn_model.cuda()
#     rpn_model.eval()

#     # 1b. RF (RandomForest) 모델 로드
#     try:
#         model_1 = joblib.load(args.model1_path)
#         model_2 = joblib.load(args.model2_path)
#         le_stage2 = joblib.load(args.encoder_path) # 2단계 번역기
#     except FileNotFoundError as e:
#         logger.error(f"[오류] RF 모델 로드 실패: {e.filename}")
#         return
        
#     # 1c. RF 모델 피처 순서 확인
#     features_s1 = model_1.feature_names_in_
#     features_s2 = model_2.feature_names_in_
#     logger.info("모든 모델 로드 완료.")

    

#     # --- 3. 실시간 처리 루프 ---
#     with torch.no_grad():
#         for i in range(len(demo_dataset)):
#             #logger.info(f"--- 프레임 {i} ({demo_dataset.sample_file_list[i].name}) 처리 시작 ---")
#             logger.info(f"--- 프레임 {i} ({Path(demo_dataset.sample_file_list[i]).name}) 처리 시작 ---")
            
#             # 3-1. 데이터 로드 및 RPN 추론
#             data_dict = demo_dataset[i]
            
#             # [!] 원본 포인트는 피처 추출을 위해 따로 저장 (Voxelize 이전)
#             # DemoDataset의 prepare_data가 points를 수정하므로, get_lidar로 원본을 다시 로드
#             try:
#                 if args.ext == '.bin':
#                     raw_points_np = np.fromfile(demo_dataset.sample_file_list[i], dtype=np.float32).reshape(-1, 4)
#                 elif args.ext == '.npy':
#                     raw_points_np = np.load(demo_dataset.sample_file_list[i])
#                 else: 
#                     raise NotImplementedError
#             except Exception as e:
#                 logger.error(f"원본 포인트 로드 실패: {e}")
#                 continue
            
#             data_dict = demo_dataset.collate_batch([data_dict])
#             load_data_to_gpu(data_dict)
            
#             # --- [Stage 0] RPN 실행 ---
#             pred_dicts, _ = rpn_model(data_dict)

#             # RPN 후보 박스(Proposals) 추출
#             proposals_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
#             proposals_scores = pred_dicts[0]['pred_scores'].cpu().numpy().reshape(-1, 1) # (N, 1)

#             if len(proposals_boxes) == 0:
#                 logger.warning(" -> RPN이 제안한 후보가 없습니다.")
#                 continue

#             # --- [Stage 0.5] 실시간 피처 추출 (CPU/GPU) ---
#             # RPN 박스와 원본 포인트를 이용해 10개 피처(density 등) 계산
#             point_features, valid_indices = extract_point_features_cpu(
#                 raw_points_np, proposals_boxes, args.min_points_in_box, logger
#             )

#             if len(valid_indices) == 0:
#                 logger.warning(" -> 유효한 피처를 가진 후보가 없습니다.")
#                 continue
                
#             # 유효한(포인트 3개 이상) 후보들만 필터링
#             valid_boxes = proposals_boxes[valid_indices]
#             valid_scores = proposals_scores[valid_indices]
            
#             # --- [Stage 1] RF 1차 분류 ---
#             # (18개 피처 합치기)
#             X_features_np = np.concatenate([valid_scores, valid_boxes, point_features], axis=1)
            
#             # (DataFrame 변환: 훈련 때와 컬럼 순서 일치)
#             df_features = pd.DataFrame(X_features_np, columns=RF_FEATURE_COLUMNS)
            
#             # 1단계 모델에 필요한 피처만 선택
#             df_for_s1 = df_features[features_s1]
            
#             # 1단계 예측: (0='Background', 1='Object')
#             pred_1_list = model_1.predict(df_for_s1)

#             # --- [Stage 2] RF 2차 분류 ---
#             # 2단계 모델에 필요한 피처만 선택
#             df_for_s2 = df_features[features_s2]

#             final_predictions = [] # 최종 클래스 이름
#             final_boxes_s2 = []    # 2단계가 예측한 최종 박스
            
#             for j, pred_1 in enumerate(pred_1_list):
#                 if pred_1 == 1: # 1 == 'Object' (1단계 통과)
#                     sample_s2 = df_for_s2.iloc[j]
                    
#                     # 2단계 예측: (예: 2)
#                     pred_2_numeric = model_2.predict([sample_s2])[0]
                    
#                     # 번역: (2 -> 'Car')
#                     pred_2_string = le_stage2.inverse_transform([pred_2_numeric])[0]
                    
#                     final_predictions.append(pred_2_string)
#                     final_boxes_s2.append(valid_boxes[j])
#                 else:
#                     # 1단계가 'Background'로 분류 (탈락)
#                     # (선택: Background도 시각화하려면)
#                     # final_predictions.append('Background')
#                     # final_boxes_s2.append(valid_boxes[j])
#                     pass

#             logger.info(f" -> 최종 예측 결과: {pd.Series(final_predictions).value_counts().to_dict()}")

#             # --- [Stage 3] 시각화 ---
#             if OPEN3D_FLAG:
#                 # 2단계가 최종 예측한 박스들 (numpy array)
#                 final_boxes_np = np.array(final_boxes_s2)
                
#                 # 라벨을 숫자로 변환 (시각화 라이브러리용)
#                 # {'Car': 1, 'Cyclist': 2, ...}
#                 s2_class_to_id = {name: idx + 1 for idx, name in enumerate(le_stage2.classes_)}
#                 final_labels_numeric = [s2_class_to_id.get(name, 0) for name in final_predictions]

#                 V.draw_scenes(
#                     points=data_dict['points'][:, 1:].cpu().numpy(), # Voxelize된 포인트
#                     ref_boxes=final_boxes_np,
#                     ref_labels=final_labels_numeric,
#                     ref_scores=None # RF는 점수를 안 줌
#                 )
            
#             logger.info(f"--- 프레임 {i} 처리 완료 ---")

# if __name__ == '__main__':
#     main()

import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn # GPU 피처 추출 의존성
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- OpenPCDet 임포트 ---
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# --- Optional visualization ---
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    V = None
    OPEN3D_FLAG = False
    print('⚠️ Open3D not installed. Visualization disabled.')

# =========================================================
# 1. GPU 기반 포인트 피처 추출 함수 (제공해주신 내용)
# =========================================================
def extract_point_features_gpu(points_np, boxes_np, min_points_in_box, logger):
    """
    GPU 버전: 각 박스 내부의 포인트들로부터 통계 피처 계산
    """
    if boxes_np.shape[0] == 0:
        return np.array([]), np.array([], dtype=int)

    try:
        # [수정] yaw 값 정규화 및 z좌표 바닥 기준으로 변경 (GPU 함수 내부에서 처리)
        boxes_for_gpu = boxes_np.copy()
        boxes_for_gpu[:, 6] = np.mod(boxes_for_gpu[:, 6], 2 * np.pi) # Yaw 정규화 (0 ~ 2pi)
        boxes_for_gpu[:, 2] -= boxes_for_gpu[:, 5] / 2.0  # z_bottom = z_center - h/2

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        points_tensor = torch.from_numpy(points_np[:, :4]).float().to(device)  # [N,4] (x,y,z,intensity)
        boxes_tensor = torch.from_numpy(boxes_for_gpu).float().to(device)  # [M,7] (z는 bottom 기준)

        # GPU 전용 함수 사용
        # points_in_boxes_gpu는 (N, 3) 포인트와 (M, 7) 박스를 기대함
        mask_gpu = roiaware_pool3d_utils.points_in_boxes_gpu(
            points_tensor[:, :3].unsqueeze(0), boxes_tensor.unsqueeze(0)
        ).squeeze(0)  # [M, N]

        mask_np = mask_gpu.cpu().numpy().astype(bool) # 각 박스에 어떤 포인트가 속하는지 bool 마스크

    except Exception as e:
        logger.error(f"[GPU Extract] Error in points_in_boxes_gpu: {e}")
        import traceback; traceback.print_exc() # 상세 에러 출력
        return np.array([]), np.array([], dtype=int)

    point_features_list, valid_indices = [], []
    # 각 박스(i)에 대해 반복
    for i in range(boxes_np.shape[0]):
        # mask_np[i]가 True인 포인트들만 선택 (박스 내부 포인트)
        pts_in_box = points_np[mask_np[i]]
        num_pts_found = pts_in_box.shape[0]
        if num_pts_found < min_points_in_box:
            # 실패하기 전에 몇 개였는지 출력
            print(f"[DEBUG] Box {i}: Found only {num_pts_found} points (min required: {min_points_in_box}). Skipping.")
            continue

        # 박스 안 포인트가 min_points_in_box보다 적으면 건너뜀
        # if pts_in_box.shape[0] < min_points_in_box:
        #     continue

        try:
            # 피처 계산 (이전 CPU 함수와 동일 로직)
            num_points = pts_in_box.shape[0]
            if num_points > 1:
                min_xyz, max_xyz = np.min(pts_in_box[:, :3], axis=0), np.max(pts_in_box[:, :3], axis=0)
                dims = max_xyz - min_xyz
                width, length, height = float(dims[0]), float(dims[1]), float(dims[2]) # 명시적 float 변환
                mean_z, std_z = np.mean(pts_in_box[:, 2]), np.std(pts_in_box[:, 2])
                mean_i = np.mean(pts_in_box[:, 3]) if pts_in_box.shape[1] > 3 else 0.0
                std_i = np.std(pts_in_box[:, 3]) if pts_in_box.shape[1] > 3 else 0.0
            else: # num_points == 1
                # 포인트가 1개일 경우 분포 관련 피처는 모두 0으로 설정
                width, length, height = 0.0, 0.0, 0.0
                # 평균은 해당 포인트 값, 표준편차는 0
                mean_z, std_z = pts_in_box[0, 2], 0.0
                mean_i = pts_in_box[0, 3] if pts_in_box.shape[1] > 3 else 0.0
                std_i = 0.0

            l, w, h = boxes_np[i, 3:6] # RPN 박스의 l, w, h
            volume = (l * w * h) + 1e-6
            density = num_points / volume
            aspect_ratio = width / (length + 1e-6) # 포인트가 1개면 width=0, length=0 -> aspect_ratio=0

            # 최종 피처 리스트 (모든 요소가 float인지 확인)
            features = [
                float(num_points), float(width), float(length), float(height), float(density),
                float(aspect_ratio), float(mean_z), float(std_z), float(mean_i), float(std_i)
            ]
            point_features_list.append(features)
            valid_indices.append(i) # 피처 추출 성공한 박스의 원래 인덱스 저장
        except Exception as e:
            # 개별 박스 피처 계산 오류 시 해당 박스만 건너뜀
            logger.warning(f"[GPU Extract] Feature calculation failed for box {i}: {e}")
            continue

    if not point_features_list:
        return np.array([]), np.array([], dtype=int)
    
    # --- ▼▼▼ [디버그 추가] ▼▼▼ ---
    # 반환하기 직전에 point_features_list의 내용과 각 내부 리스트의 길이를 확인
    try:
        # 모든 내부 리스트의 길이를 확인
        list_lengths = [len(f) for f in point_features_list]
        # 길이가 10이 아닌 리스트가 있는지 확인
        if not all(l == 10 for l in list_lengths):
            logger.error("[DEBUG] Inhomogeneous shape detected in point_features_list!")
            logger.error(f"[DEBUG] Lengths of inner lists: {list_lengths}")
            # 문제가 되는 리스트의 내용 일부 출력 (예시)
            for idx, length in enumerate(list_lengths):
                if length != 10:
                    logger.error(f"[DEBUG] Problematic list at index {idx} (len={length}): {point_features_list[idx]}")
        # else:
        #    logger.debug("[DEBUG] point_features_list seems homogeneous.")

        # 배열 변환 시도 (오류 발생 지점 재현)
        np_array_attempt = np.array(point_features_list, np.float32)

    except ValueError as ve:
       # ValueError 발생 시 리스트 내용을 더 자세히 로깅
       logger.error(f"[DEBUG] ValueError during np.array conversion: {ve}")
       logger.error(f"[DEBUG] Contents of point_features_list that caused error:")
       for idx, sublist in enumerate(point_features_list):
            logger.error(f"[DEBUG] Index {idx} (len={len(sublist)}): {sublist}")
       # 오류를 다시 발생시켜 프로그램 중단 (디버깅 목적)
       raise ve 
    except Exception as e:
       logger.error(f"[DEBUG] Unexpected error during final check: {e}")
    # --- ▲▲▲ [디버그 추가] ▲▲▲ ---

    # 최종 반환: [성공한 박스 개수, 10] 피처 배열, [성공한 박스들의 원래 인덱스] 배열
    return np.array(point_features_list, np.float32), np.array(valid_indices, int)

# =========================================================
# 2. RF 피처 컬럼 정의 (제공해주신 내용)
# =========================================================
RF_FEATURE_COLUMNS = [
    "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
    "num_points", "width", "length", "height", "density",
    "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std"
]

# =========================================================
# 3. DemoDataset (제공해주신 내용과 동일)
# =========================================================
class DemoDataset(DatasetTemplate):
    # ... (이전 코드와 동일) ...
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self): return len(self.sample_file_list)

    def __getitem__(self, index):
        try: # 파일 로드 오류 방지
            if self.ext == '.bin':
                pts = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            elif self.ext == '.npy':
                pts = np.load(self.sample_file_list[index])
            else:
                raise NotImplementedError
            # [수정] 원본 포인트도 함께 반환하도록 prepare_data 수정 필요 (DatasetTemplate 수정)
            # 임시방편: raw_points 키 추가 (prepare_data 내부에서 수정되지 않도록)
            input_dict = {'points': pts.copy(), 'frame_id': index, 'raw_points': pts}
            data_dict = self.prepare_data(data_dict=input_dict)
            # prepare_data 후 원본 포인트 복원 (Voxelize 등으로 변경된 points 대신)
            data_dict['raw_points'] = pts
            return data_dict
        except Exception as e:
            self.logger.error(f"Error loading/processing file {self.sample_file_list[index]}: {e}")
            # 빈 데이터 반환 또는 예외 재발생 등 처리 필요
            return None # 오류 발생 시 None 반환

# =========================================================
# 4. Config Parser (TopK 추가)
# =========================================================
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/a2d2_models/second.yaml')
    parser.add_argument('--data_path', type=str, default='data/a2d2/training/velodyne')
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth')
    parser.add_argument('--ext', type=str, default='.bin')
    parser.add_argument('--model1_path', type=str, default='data/a2d2/rf_stage1_model.pkl')
    parser.add_argument('--model2_path', type=str, default='data/a2d2/rf_stage2_model.pkl')
    parser.add_argument('--encoder_path', type=str, default='data/a2d2/rf_stage2_encoder.pkl')
    parser.add_argument('--min_points_in_box', type=int, default=1)
    # --- ▼▼▼ [TopK 인자 추가] ▼▼▼ ---
    parser.add_argument('--topk', type=int, default=200, help='RPN 상위 K개 proposal만 사용 (0이면 모두 사용)')
    # --- ▲▲▲ [TopK 인자 추가] ▲▲▲ ---
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

# =========================================================
# 5. Main (GPU 피처 추출 + TopK + Debug 적용)
# =========================================================
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('---------------- RPN + RF Cascade Demo (GPU ver.) ----------------')

    # --- 모델 로드 ---
    demo_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path(args.data_path), logger=logger, ext=args.ext)
    if len(demo_dataset) == 0:
        logger.error(f"No data found in {args.data_path}")
        return

    rpn_model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    rpn_model.load_params_from_file(args.ckpt, logger=logger)
    rpn_model.cuda().eval()

    try:
        model_1 = joblib.load(args.model1_path)
        model_2 = joblib.load(args.model2_path)
        le_stage2 = joblib.load(args.encoder_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load RF model or encoder: {e.filename}")
        return
    features_s1, features_s2 = model_1.feature_names_in_, model_2.feature_names_in_
    logger.info("✅ RF & RPN models loaded successfully")

    STAGE_1_OBJECT_LABEL = 1 # 1단계 모델에서 'Object'에 해당하는 숫자 라벨

    with torch.no_grad():
        for i in range(len(demo_dataset)):
            fname = Path(demo_dataset.sample_file_list[i]).name
            logger.info(f"--- Frame {i} ({fname}) ---")

            # --- 데이터 로드 ---
            data_dict = demo_dataset[i]
            if data_dict is None: # 데이터 로딩 실패 시 건너뜀
                 logger.warning(f"Skipping frame {i} due to loading error.")
                 continue

            # [수정] __getitem__에서 raw_points를 받도록 수정됨
            raw_points_np = data_dict.get('raw_points', None)
            if raw_points_np is None:
                 logger.error("raw_points not found in data_dict. Check DemoDataset.")
                 continue

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # --- Stage 0: RPN ---
            pred_dicts, _ = rpn_model(data_dict)
            # RPN 결과는 항상 CPU로 복사 후 처리
            boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            scores = pred_dicts[0]['pred_scores'].cpu().numpy().reshape(-1, 1)

            if len(boxes) == 0:
                logger.warning(" -> No RPN proposals")
                continue

            # --- [DEBUG] 좌표계 일치 여부 확인 ---
            if boxes.shape[0] > 0 and raw_points_np.shape[0] > 0:
                # np.set_printoptions(precision=3, suppress=True) # 소수점 자리수 조절
                print(f"[DEBUG] first raw point (x,y,z) = {raw_points_np[0, :3]}")
                print(f"[DEBUG] first pred box center   = {boxes[0, :3]}")
                print(f"[DEBUG] box z={boxes[0,2]:.2f}, point z range=({raw_points_np[:,2].min():.1f},{raw_points_np[:,2].max():.1f})")

            # --- [추가] 상위 K개 proposal만 필터링 ---
            if args.topk > 0:
                topk = min(args.topk, len(scores))
                idxs = np.argsort(-scores.squeeze())[:topk] # 점수 높은 순 K개 인덱스
                boxes, scores = boxes[idxs], scores[idxs]
                logger.info(f" -> Using top {topk} RPN proposals.")

            # --- Stage 0.5: GPU Feature Extraction ---
            # [수정] GPU 함수 호출
            point_features, valid_idx = extract_point_features_gpu(
                raw_points_np, boxes, args.min_points_in_box, logger
            )

            # --- [DEBUG] 피처 추출 실패 시 상세 정보 출력 ---
            if len(valid_idx) == 0:
                print(f"[DEBUG] No valid features extracted after GPU function call.")
                print(f"[DEBUG] Input boxes shape to GPU func: {boxes.shape}")
                print(f"[DEBUG] Input points shape to GPU func: {raw_points_np.shape}")
                if boxes.shape[0] > 0: # 박스가 있었는지 확인
                    print(f"[DEBUG] Example box input to GPU func: {boxes[0]}")
                print(f"[DEBUG] Points range X({raw_points_np[:,0].min():.1f},{raw_points_np[:,0].max():.1f}) "
                      f"Y({raw_points_np[:,1].min():.1f},{raw_points_np[:,1].max():.1f}) "
                      f"Z({raw_points_np[:,2].min():.1f},{raw_points_np[:,2].max():.1f})")
                logger.warning(" -> No valid features extracted.")
                continue # 다음 프레임으로

            # 피처 추출에 성공한 박스와 점수만 사용
            boxes, scores = boxes[valid_idx], scores[valid_idx]

            # --- Stage 1: RF Binary Classification ---
            # 18개 피처 합치기: RPN Score(1) + Box(7) + PointFeatures(10)
            X = np.concatenate([scores, boxes, point_features], axis=1)
            # DataFrame 변환 및 훈련 시 사용한 컬럼 순서로 정렬
            df = pd.DataFrame(X, columns=RF_FEATURE_COLUMNS)
            df_s1 = df[features_s1] # model_1 훈련 시 사용된 피처만 선택/정렬
            pred_stage1 = model_1.predict(df_s1) # 0 또는 1 예측

            # --- Stage 2: RF Multi-Class Classification ---
            df_s2 = df[features_s2] # model_2 훈련 시 사용된 피처만 선택/정렬
            final_boxes, final_preds = [], []
            for j, p1 in enumerate(pred_stage1):
                if p1 == STAGE_1_OBJECT_LABEL: # 1단계 통과 (Object)
                    # 2단계 예측 (숫자)
                    p2_num = model_2.predict([df_s2.iloc[j]])[0]
                    try:
                        # 번역 (숫자 -> 문자)
                        p2_str = le_stage2.inverse_transform([p2_num])[0]
                        final_preds.append(p2_str)
                        final_boxes.append(boxes[j]) # 최종 예측에 성공한 박스만 저장
                    except ValueError:
                         logger.warning(f"Stage 2 prediction '{p2_num}' not in encoder classes. Skipping.")
                         pass # 번역 실패 시 해당 예측은 버림
                # else: # 1단계 탈락 (Background) - 아무것도 안 함 (시각화 X)
                #     pass

            if not final_preds:
                logger.info(" -> No final predictions after Stage 2.")
                continue

            logger.info(f" -> Final Results: {pd.Series(final_preds).value_counts().to_dict()}")

            # --- Visualization ---
            if OPEN3D_FLAG and V is not None:
                final_boxes_np = np.array(final_boxes)
                # 라벨을 숫자로 변환 (Open3D 컬러맵용)
                class_to_id = {n: k + 1 for k, n in enumerate(le_stage2.classes_)}
                label_ids = [class_to_id.get(n, 0) for n in final_preds]

                # Open3D 실행 전 박스 유효성 검사 (크기가 0이거나 NaN이 있는지)
                valid_boxes_mask = np.all(final_boxes_np[:, 3:6] > 0, axis=1) & \
                                   ~np.any(np.isnan(final_boxes_np), axis=1)
                if not np.all(valid_boxes_mask):
                    logger.warning(f"Found {np.sum(~valid_boxes_mask)} invalid boxes. Filtering for visualization.")
                    final_boxes_np = final_boxes_np[valid_boxes_mask]
                    label_ids = np.array(label_ids)[valid_boxes_mask]
                    if final_boxes_np.shape[0] == 0:
                        logger.warning("No valid boxes left for visualization.")
                        continue

                try:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:].cpu().numpy(), # Voxelize된 포인트 사용
                        ref_boxes=final_boxes_np, # 최종 예측된 박스만 표시
                        ref_labels=np.array(label_ids),
                        ref_scores=None # RF는 점수 출력 안 함
                    )
                except Exception as e:
                    logger.error(f"Error during Open3D visualization: {e}")

            logger.info(f"--- Frame {i} done ---")

if __name__ == '__main__':
    main()