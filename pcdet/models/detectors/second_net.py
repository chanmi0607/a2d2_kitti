import torch
import torch.nn.functional as F
import numpy as np
import joblib  # ⭐️ .pkl 파일을 로드하기 위해 추가
import os
from ...ops.roiaware_pool3d import roiaware_pool3d_utils # ⭐️ 포인트-박스 매칭용
from ...ops.iou3d_nms import iou3d_nms_utils
try:
    from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max # ⭐️ 그룹 통계용
except ImportError:
    print("Warning: torch_scatter not found. Point-based RF features will fail.")
    
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        self.module_list = self.build_networks()

        # ⭐️ [추가] Lidar 범위(point_cloud_range)를 클래스 변수로 저장
        # dataset.point_cloud_range는 [xmin, ymin, zmin, xmax, ymax, zmax]
        self.point_cloud_range = dataset.point_cloud_range 
        print(f"[INFO] Loaded Point Cloud Range: {self.point_cloud_range}")

        # =================================================================
        #  1. 설정값 읽기 및 RF 모델 로드
        # =================================================================
        post_process_cfg = self.model_cfg.POST_PROCESSING
        self.use_cascade_rf = post_process_cfg.get('USE_CASCADE_RF', False)

        # =================================================================
        #  데이터 수집 모드 설정
        # =================================================================

        self.create_rf_dataset = self.model_cfg.get('CREATE_RF_DATASET', False)
        self.use_cascade_rf = self.model_cfg.POST_PROCESSING.get('USE_CASCADE_RF', False)

        if self.create_rf_dataset:
            #  데이터 수집 모드
            self.rf_dataset_save_path = self.model_cfg.get('RF_DATASET_SAVE_PATH', 'temp_rf_data.csv')
            os.makedirs(os.path.dirname(self.rf_dataset_save_path), exist_ok=True)
            #  새 .csv 파일 열고, 헤더(컬럼명) 작성
            # (RF 훈련 스크립트에서 drop했던 'RPN_MaxScore', 'yaw'는 제외하고 16개 + 1개)
            header = [
                'x', 'y', 'z', 'l', 'w', 'h', 
                'num_points', 'length', 'width', 'height', 
                'density', 'aspect_ratio', 
                'mean_z', 'std_z', 'intensity_mean', 'intensity_std', 
                'label' #  정답 라벨
            ]
            with open(self.rf_dataset_save_path, 'w') as f:
                f.write(','.join(header) + '\n')
            
            #  [중요] 데이터 수집 중에는 RF 모델 로드 안 함
            print(f"[INFO] CREATE_RF_DATASET mode is ON. Saving features to {self.rf_dataset_save_path}")

        elif self.use_cascade_rf:
            print("Cascade RandomForest post-processing (2-Stage) is ENABLED.")
            
            # ⭐️ .yaml에 정의된 경로에서 PKL 파일 로드
            try:
                self.rf_stage1 = joblib.load(post_process_cfg.RF_STAGE1_PATH)
                self.rf_stage2 = joblib.load(post_process_cfg.RF_STAGE2_PATH)
                self.rf_stage2_encoder = joblib.load(post_process_cfg.RF_STAGE2_ENCODER_PATH)
            except Exception as e:
                raise FileNotFoundError(f"Failed to load RF models. Check paths in YAML. Error: {e}")

            # ⭐️ RF 설정값 로드
            self.rf_obj_thresh = post_process_cfg.get('RF_OBJECT_THRESH', 0.9)
            
            # ⭐️ RF 클래스 순서와 OpenPCDet 클래스 순서 매핑
            self.rf_to_openpcdet_class_map = self._map_rf_classes_to_dataset(
                dataset.class_names, self.rf_stage2_encoder.classes_
            )
            print(f"RF class map (RF Index -> OpenPCDet Index): {self.rf_to_openpcdet_class_map}")

        else:
            print("Using default (SECOND Head) post-processing.")
            
    def _map_rf_classes_to_dataset(self, dataset_classes, rf_encoder_classes):
        """RF 출력 순서(인코더)를 OpenPCDet의 클래스 인덱스로 매핑하는 딕셔너리 생성"""
        # (OpenPCDet는 0을 배경으로 사용하므로 1-based index)
        dataset_map = {name: i + 1 for i, name in enumerate(dataset_classes)}
        
        rf_map = {} # {rf_idx: openpcdet_idx}
        for rf_idx, class_name in enumerate(rf_encoder_classes):
            if class_name in dataset_map:
                openpcdet_idx = dataset_map[class_name]
                rf_map[rf_idx] = openpcdet_idx
            else:
                print(f"Warning: RF class '{class_name}' not in OpenPCDet dataset classes.")
        
        # 결과 예: {0: 1, 1: 2} (RF 출력 0번('Car')은 OpenPCDet 1번('Car'), ...)
        return rf_map
    
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # ⭐️ demo.py 실행 시 이 부분이 호출됨
            # ⭐️ self.post_processing은 우리가 덮어쓴 RF 버전 함수를 호출
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
        
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    # =================================================================
    # ⭐️ 2. [새로 추가] RF 피처 생성 헬퍼 함수
    # =================================================================
    def create_features_for_rf(self, batch_dict, index, box_preds, cls_preds_rpn):
        """
        RF 모델에 입력으로 사용할 18개 피처 벡터를 생성합니다.
        [RPN_MaxScore, x, y, z, l, w, h, yaw, num_points, 
         width, length, height, density, aspect_ratio, 
         mean_z, std_z, intensity_mean, intensity_std]
        """
        num_boxes = box_preds.shape[0]
        if num_boxes == 0:
            return torch.empty((0, 18), device=box_preds.device)

        # 1. RPN 예측 기반 피처
        RPN_MaxScore, _ = cls_preds_rpn.max(dim=-1) # (M,)
        x, y, z = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2]
        dx, dy, dz = box_preds[:, 3], box_preds[:, 4], box_preds[:, 5] # l, w, h
        yaw = box_preds[:, 6]
        
        volume = dx * dy * dz
        aspect_ratio = dx / (dy + 1e-6) # 0으로 나누기 방지

        # 2. 포인트 통계 기반 피처
        point_batch_mask = batch_dict['points'][:, 0] == index
        cur_points = batch_dict['points'][point_batch_mask] # (N_scene, 5+) [batch_idx, x, y, z, intensity, ...]
        
        if cur_points.shape[0] == 0:
            num_points = torch.zeros(num_boxes, device=box_preds.device)
            mean_z = torch.zeros(num_boxes, device=box_preds.device)
            std_z = torch.zeros(num_boxes, device=box_preds.device)
            intensity_mean = torch.zeros(num_boxes, device=box_preds.device)
            intensity_std = torch.zeros(num_boxes, device=box_preds.device)
            length = torch.zeros(num_boxes, device=box_preds.device) # ⭐️ (length)
            width = torch.zeros(num_boxes, device=box_preds.device)  # ⭐️ (width)
            height = torch.zeros(num_boxes, device=box_preds.device) # ⭐️ (height)
        else:
            points_batch = cur_points[:, 1:4].unsqueeze(0)
            # ⭐️ [수정] (K, 7) -> (1, K, 7)으로 가짜 배치 차원 추가
            boxes_batch = box_preds[:, 0:7].unsqueeze(0)
            
            box_indices_of_points_batch = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_batch, boxes_batch
            ) # 결과는 (1, N_scene) 형태가 됨
            
            # ⭐️ [수정] (1, N_scene) -> (N_scene,)으로 가짜 배치 차원 제거
            box_indices_of_points = box_indices_of_points_batch.squeeze(0)
            
            point_mask = box_indices_of_points >= 0
            points_in_boxes = cur_points[point_mask] # (N_fg, 5+)
            box_indices_fg = box_indices_of_points[point_mask].to(torch.int64) # (N_fg,)

            if box_indices_fg.shape[0] == 0:
                num_points = torch.zeros(num_boxes, device=box_preds.device)
                mean_z = torch.zeros(num_boxes, device=box_preds.device)
                std_z = torch.zeros(num_boxes, device=box_preds.device)
                intensity_mean = torch.zeros(num_boxes, device=box_preds.device)
                intensity_std = torch.zeros(num_boxes, device=box_preds.device)
                length = torch.zeros(num_boxes, device=box_preds.device) # ⭐️ (length)
                width = torch.zeros(num_boxes, device=box_preds.device)  # ⭐️ (width)
                height = torch.zeros(num_boxes, device=box_preds.device) # ⭐️ (height)
            else:
                num_points = torch.bincount(box_indices_fg, minlength=num_boxes).float()[0:num_boxes]
                
                point_x = points_in_boxes[:, 1] # (N_fg,)
                point_y = points_in_boxes[:, 2] # (N_fg,)
                point_z = points_in_boxes[:, 3] # (N_fg,)
                point_intensity = points_in_boxes[:, 4] # (N_fg,)
                
                # ⭐️ [추가] 박스 내부 점군의 실제 x, y, z min/max 계산
                min_x_per_box = scatter_min(point_x, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('inf'), device=box_preds.device))[0]
                max_x_per_box = scatter_max(point_x, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('-inf'), device=box_preds.device))[0]
                min_y_per_box = scatter_min(point_y, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('inf'), device=box_preds.device))[0]
                max_y_per_box = scatter_max(point_y, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('-inf'), device=box_preds.device))[0]
                min_z_per_box = scatter_min(point_z, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('inf'), device=box_preds.device))[0]
                max_z_per_box = scatter_max(point_z, box_indices_fg, dim=0, out=torch.full((num_boxes,), float('-inf'), device=box_preds.device))[0]

                # ⭐️ [추가] 실제 점군 분포 크기 (length, width, height) 계산
                # (점이 없는 박스는 inf - (-inf) = inf가 될 수 있으므로, 0으로 마스킹)
                length = max_x_per_box - min_x_per_box
                width = max_y_per_box - min_y_per_box
                height = max_z_per_box - min_z_per_box
                
                no_points_mask = (num_points == 0)
                length[no_points_mask] = 0
                width[no_points_mask] = 0
                height[no_points_mask] = 0
                
                # (기존) z축 및 intensity 통계
                mean_z = scatter_mean(point_z, box_indices_fg, dim=0, out=torch.zeros(num_boxes, device=box_preds.device))
                std_z = scatter_std(point_z, box_indices_fg, dim=0, out=torch.zeros(num_boxes, device=box_preds.device))
                intensity_mean = scatter_mean(point_intensity, box_indices_fg, dim=0, out=torch.zeros(num_boxes, device=box_preds.device))
                intensity_std = scatter_std(point_intensity, box_indices_fg, dim=0, out=torch.zeros(num_boxes, device=box_preds.device))
        density = num_points / (volume + 1e-6)

        # 3. 모든 피처를 (M, 18) 텐서로 결합
        features = torch.stack([
            #RPN_MaxScore,     # 1
            x,                # 2
            y,                # 3
            z,                # 4
            dx,               # 5 (l)
            dy,               # 6 (w)
            dz,               # 7 (h)
            #yaw,              # 8
            num_points,       # 9
            length,               # 10 (width) - l, w, h와 중복이지만 리스트 순서대로
            width,               # 11 (length)
            height,               # 12 (height)
            density,          # 13
            aspect_ratio,     # 14
            mean_z,           # 15
            std_z,            # 16
            intensity_mean,   # 17
            intensity_std     # 18
        ], dim=1)

        return features

    # =================================================================
    # ⭐️ 3. [새로 추가/오버라이딩] post_processing 함수
    # =================================================================
    def post_processing(self, batch_dict):
        # 3-1. RF 사용 안 함: 부모(원본)의 NMS 로직 실행
        if not self.use_cascade_rf:
            return super().post_processing(batch_dict)

        # 3-2. RF 사용: Cascade 분류기 실행
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        new_final_cls_preds_list = [] # 최종 RF 점수를 저장할 리스트
        new_final_box_preds_list = [] # ⭐️ [추가] 필터링된 박스를 저장할 리스트

        for index in range(batch_size):
            # 3-3. RPN 결과(박스, 점수) 가져오기
            if batch_dict.get('batch_index', None) is not None:
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask] # (M, 7)
            cls_preds_rpn = batch_dict['batch_cls_preds'][batch_mask] # (M, num_class)

            if not batch_dict['cls_preds_normalized']:
                cls_preds_rpn = torch.sigmoid(cls_preds_rpn)

            pcr_tensor = torch.tensor(
                self.point_cloud_range, dtype=torch.float32, device=box_preds.device
            )
            # 2. 박스 중심 좌표 (M, 3)
            box_centers = box_preds[:, 0:3]

            # 3. 범위 마스크 생성
            # .all(dim=1)을 사용하여 x,y,z가 *모두* 범위 내에 있는지 확인
            mask_min = (box_centers >= pcr_tensor[0:3]).all(dim=1)
            mask_max = (box_centers <= pcr_tensor[3:6]).all(dim=1)
            
            # ⭐️ [수정] 필터링 전 박스 수 저장
            total_boxes_before_filter = box_preds.shape[0]

            range_mask = mask_min & mask_max
            
            # 5. [핵심] 범위 내의 박스만 필터링
            box_preds = box_preds[range_mask]
            cls_preds_rpn = cls_preds_rpn[range_mask]

            

            # ⭐️ [수정] print 문 수정
            print(f"[DEBUG] 'Lidar 범위 밖' 필터로 {total_boxes_before_filter - box_preds.shape[0]} 개의 박스가 제거됨.")
            forward_limit = 30.0
            mask_forward = box_preds[:, 0] <= forward_limit   # LiDAR x축 기준
            mask_backward = box_preds[:, 0] >= 0.0            # 후방 박스 제거
            mask_fov = mask_forward & mask_backward

            num_before = box_preds.shape[0]
            box_preds = box_preds[mask_fov]
            cls_preds_rpn = cls_preds_rpn[mask_fov]
            print(f"[DEBUG] '전방 0~30m' 필터로 {num_before - box_preds.shape[0]}개 박스 제거됨. (남은 {box_preds.shape[0]})")

            # 3-4. RF 피처 생성 헬퍼 함수 호출
            features_for_rf = self.create_features_for_rf(
                batch_dict, index, box_preds, cls_preds_rpn
            )
            
            # 3-5. CPU로 이동 (RF 모델 입력을 위해)
            features_np = features_for_rf.detach().cpu().numpy()

            if features_np.shape[0] == 0:
                final_scores = torch.zeros((0, self.num_class), device=box_preds.device)
                new_final_cls_preds_list.append(final_scores)
                continue

            # 3-6. [RF STAGE 1] Object vs Background 예측
            prob_obj_np = self.rf_stage1.predict_proba(features_np)[:, 1] # (M,)

            # ==========================================================
            # ⭐️ 점군(point)이 0개인 박스 강제 0점 처리
            # ==========================================================
            # 1. 16개 피처 중 7번째(인덱스 6)가 num_points입니다.
            # (순서: [x, y, z, dx, dy, dz, num_points, ...])
            num_points_np = features_np[:, 6] 
            
            # 2. num_points가 0인 박스 마스크 생성
            pointless_box_mask_np = (num_points_np <= 1)
            
            # 3. [핵심] 해당 박스들의 Stage 1 점수(P(Object))를 0.0으로 강제 할당
            prob_obj_np[pointless_box_mask_np] = 0.0
            
            # 4. (디버깅) 얼마나 많은 박스가 0점 처리되었는지 확인
            print(f"[DEBUG] '점군 0개' 필터로 {pointless_box_mask_np.sum()} 개의 박스가 0점 처리됨.")

            print(f"\n[DEBUG] LiDAR 범위 내 RPN이 제안한 총 박스 수: {features_np.shape[0]} 개")
            print(f"[DEBUG] RF Stage 1 최고점수 (Top 5): {np.sort(prob_obj_np)[-5:]}")

            # 3-7. [RF STAGE 2] Specific Class (Car, Ped, Cyc) 예측
            # 3-7. [RF STAGE 1] Specific Class (Car, Ped, Cyc) 예측
            object_mask_np = prob_obj_np > self.rf_obj_thresh  # (M,)

            # ==========================================================
            # ⭐ Stage 1 통과 박스만 남기기 (나머지 완전 제거)
            # ==========================================================
            keep_mask = torch.from_numpy(object_mask_np).to(box_preds.device)

            box_preds = box_preds[keep_mask]
            cls_preds_rpn = cls_preds_rpn[keep_mask]
            features_np = features_np[object_mask_np]
            prob_obj_np = prob_obj_np[object_mask_np]
            new_final_box_preds_list.append(box_preds)

            print(f"[DEBUG] Stage 1 통과 (>{self.rf_obj_thresh}) 박스 수: {object_mask_np.sum()} 개\n")

            final_scores = torch.zeros((features_np.shape[0], self.num_class), device=box_preds.device)

            if features_np.shape[0] > 0:
                # ⭐ Stage 2는 Stage 1 통과 박스만 대상으로 실행
                print(f"[DEBUG] RF Stage 2 실행. 입력 박스 수: {features_np.shape[0]} 개")

                prob_specific_class_np = self.rf_stage2.predict_proba(features_np)  # (K, num_rf_classes)
                print(f"[DEBUG] Stage 2 예측 확률 (Max per class): {np.max(prob_specific_class_np, axis=0)}")

                prob_specific_class = torch.from_numpy(prob_specific_class_np).to(box_preds.device)

                prob_obj = torch.from_numpy(prob_obj_np).to(box_preds.device).unsqueeze(-1)
                final_obj_scores_rf = prob_specific_class * prob_obj  # (K, num_rf_classes)

                print(f"[DEBUG] 최종 결합 점수 (Max per class): {torch.max(final_obj_scores_rf, dim=0)[0].cpu().numpy()}")

                final_obj_scores_openpcdet = torch.zeros(
                    (final_obj_scores_rf.shape[0], self.num_class),
                    device=box_preds.device
                )

                # 3-8. RF 클래스 인덱스를 OpenPCDet 인덱스로 매핑
                for rf_idx, openpcdet_idx in self.rf_to_openpcdet_class_map.items():
                    # openpcdet_idx는 1-based, 텐서 인덱스는 0-based
                    final_obj_scores_openpcdet[:, openpcdet_idx - 1] = final_obj_scores_rf[:, rf_idx]

                final_scores = final_obj_scores_openpcdet

            new_final_cls_preds_list.append(final_scores)

        # 3-9. [핵심] 원본 RPN 점수를 RF가 만든 최종 점수로 교체
        if batch_dict.get('batch_index', None) is not None:
            # ⭐️ [추가] 박스 목록 덮어쓰기
            batch_dict['batch_box_preds'] = torch.cat(new_final_box_preds_list, dim=0)
            batch_dict['batch_cls_preds'] = torch.cat(new_final_cls_preds_list, dim=0)
            # ⭐️ [추가] 배치 인덱스도 덮어써야 함 (매우 중요)
            new_batch_index_list = []
            for i, boxes in enumerate(new_final_box_preds_list):
                new_batch_index_list.append(
                    torch.full((boxes.shape[0],), i, device=boxes.device, dtype=torch.int64)
                )
            batch_dict['batch_index'] = torch.cat(new_batch_index_list, dim=0)
        else:
            # ⭐️ [추가] 박스 목록 덮어쓰기
            batch_dict['batch_box_preds'] = torch.stack(new_final_box_preds_list, dim=0)
            batch_dict['batch_cls_preds'] = torch.stack(new_final_cls_preds_list, dim=0)
            
        # 3-10. [재사용] 부모의 원본 NMS 로직을 "새로운 점수"로 실행
        return super().post_processing(batch_dict)
