import numpy as np
import pickle
from pathlib import Path
import os

# -----------------------------------------------------------------------------
# a2d2_dataset.py에서 필요한 클래스와 유틸리티 가져오기
# (경로는 실제 프로젝트 구조에 맞게 조정해야 할 수 있습니다)
# -----------------------------------------------------------------------------
try:
    # pcdet 프로젝트 내에서 실행하는 경우
    from pcdet.datasets.a2d2.a2d2_dataset import A2D2Dataset
    from pcdet.utils import box_utils, common_utils
except ImportError:
    # 스크립트를 pcdet/datasets/a2d2/ 폴더에서 직접 실행하는 경우
    from a2d2_dataset import A2D2Dataset
    from ...utils import box_utils, common_utils


def create_new_label_files(dataset_cfg_path, data_root_path, split='train'):
    """
    A2D2Dataset의 로직을 사용해 새로운 포맷의 라벨 파일을 생성합니다.
    """
    
    # 1. A2D2Dataset 인스턴스 생성 (파일 경로, calib 등 접근용)
    # (dataset_cfg, class_names 등은 실제 설정에 맞게 로드해야 함)
    # 여기서는 간단히 클래스 이름을 하드코딩합니다.
    
    import yaml
    from easydict import EasyDict
    
    with open(dataset_cfg_path, 'r') as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))
        
    class_names = ['Car','Truck','UtilityVehicle','Cyclist','Bus','Trailer','Pedestrian'] # 예시
    dataset = A2D2Dataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=Path(data_root_path),
        training=False, # True/False는 무관
        logger=None
    )
    
    # 2. 처리할 데이터 스플릿(train/val) 설정
    dataset.set_split(split)
    
    # 3. 새로운 라벨을 저장할 폴더 생성
    new_label_dir = dataset.root_path / f'label_new_{split}'
    new_label_dir.mkdir(parents=True, exist_ok=True)
    print(f"새로운 라벨 저장 위치: {new_label_dir}")

    # 4. 모든 샘플 ID에 대해 루프 실행
    sample_id_list = dataset.sample_id_list
    
    for sample_idx in sample_id_list:
        print(f"Processing: {sample_idx}")
        
        # --- `process_single_scene` 로직 재현 ---
        
        # 5. 필수 데이터 로드
        try:
            calib = dataset.get_calib(sample_idx)
            points = dataset.get_lidar(sample_idx)
            obj_list = dataset.get_label(sample_idx)
        except AssertionError:
            print(f"[경고] {sample_idx} 파일 로드 실패. 건너뜁니다.")
            continue
            
        if len(obj_list) == 0:
            # 객체가 없는 경우 빈 파일 생성
            (new_label_dir / f'{sample_idx}.txt').touch()
            continue

        # 6. Annotations 파싱 (a2d2_dataset.py의 로직과 동일)
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list]) # (l, h, w - camera)
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0) # (camera)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        
        # DontCare 필터링
        annotations = common_utils.drop_info_with_name(annotations, name='DontCare')
        
        if len(annotations['name']) == 0:
            # DontCare만 있었던 경우 빈 파일 생성
            (new_label_dir / f'{sample_idx}.txt').touch()
            continue
            
        num_objects = len(annotations['name'])

        # 7. LiDAR 좌표계 변환
        loc = annotations['location']
        dims = annotations['dimensions'] # (l, h, w - camera)
        rots = annotations['rotation_y']
        
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 1] -= h[:, 0] / 3
        
        # (x, y, z, l, w, h, yaw)
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

        # 8. 원하는 특징 계산
        output_lines = []
        for k in range(num_objects):
            # (1) RPN_MaxScore (GT이므로 1.0)
            rpn_score = 1.0
            
            # (2) x, y, z, l, w, h, yaw (LiDAR 좌표계)
            x, y, z = gt_boxes_lidar[k, 0], gt_boxes_lidar[k, 1], gt_boxes_lidar[k, 2]
            l_lidar = gt_boxes_lidar[k, 3] # length (이름이 헷갈릴 수 있으니 명확히)
            w_lidar = gt_boxes_lidar[k, 4] # width
            h_lidar = gt_boxes_lidar[k, 5] # height
            yaw = gt_boxes_lidar[k, 6]
            
            # (3) 포인트 통계
            flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
            points_in_box = points[flag]
            
            num_points = points_in_box.shape[0]
            
            if num_points > 0:
                volume = l_lidar * w_lidar * h_lidar
                density = num_points / volume if volume > 0 else 0.0
                mean_z = np.mean(points_in_box[:, 2])
                std_z = np.std(points_in_box[:, 2])
                intensity_mean = np.mean(points_in_box[:, 3])
                intensity_std = np.std(points_in_box[:, 3])

                min_coords = np.min(points_in_box[:, :3], axis=0)
                max_coords = np.max(points_in_box[:, :3], axis=0)
                extent = max_coords - min_coords
                
                # LiDAR 좌표계 기준: x(length), y(width), z(height)
                actual_length = extent[0] 
                actual_width = extent[1]
                actual_height = extent[2]
            else:
                density = 0.0
                mean_z = 0.0
                std_z = 0.0
                intensity_mean = 0.0
                intensity_std = 0.0
                actual_length = 0.0
                actual_width = 0.0
                actual_height = 0.0
                
            # (4) 2D Aspect Ratio (width / height)
            bbox_2d = annotations['bbox'][k]
            w_2d = bbox_2d[2] - bbox_2d[0]
            h_2d = bbox_2d[3] - bbox_2d[1]
            aspect_ratio = w_2d / (h_2d + 1e-6) # 0으로 나누기 방지
            
            # (5) width, length, height (이름 중복?)
            # l, w, h가 이미 LiDAR 기준 l, w, h이므로 
            # width = w_lidar, length = l_lidar, height = h_lidar
            
            # 9. 최종 문자열 포맷팅
            # RPN_MaxScore, x, y, z, l, w, h, yaw, 
            # num_points, width, length, height, density, 
            # aspect_ratio, mean_z, std_z, intensity_mean, intensity_std
            line = (
                f"{annotations['name'][k]} " 
                f"{rpn_score:.4f} {x:.4f} {y:.4f} {z:.4f} {l_lidar:.4f} {w_lidar:.4f} {h_lidar:.4f} {yaw:.4f} "
                f"{num_points:.4f} {actual_width:.4f} {actual_length:.4f} {actual_height:.4f} {density:.4f} "
                f"{aspect_ratio:.4f} {mean_z:.4f} {std_z:.4f} {intensity_mean:.4f} {intensity_std:.4f}"
            )
            output_lines.append(line)

        # 10. 새로운 라벨 파일(.txt) 저장
        output_file_path = new_label_dir / f'{sample_idx}.txt'
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(output_lines))
            
    print(f"--- {split} 스플릿 완료 ---")


if __name__ == '__main__':
    # ---------------------------------------------------
    # [!] 사용자가 직접 경로를 설정해야 하는 부분
    # ---------------------------------------------------
    
    # 1. a2d2_dataset.yaml 파일 경로
    CFG_PATH = 'tools/cfgs/dataset_configs/a2d2_dataset.yaml'
    
    # 2. A2D2 데이터셋 루트 경로 (data/a2d2)
    DATA_ROOT = 'data/a2d2'
    
    # ---------------------------------------------------
    
    # 'train' 스플릿에 대해 실행
    create_new_label_files(CFG_PATH, DATA_ROOT, split='train')
    
    # 'val' 스플릿에 대해 실행
    create_new_label_files(CFG_PATH, DATA_ROOT, split='val')
    
    print("모든 작업 완료.")