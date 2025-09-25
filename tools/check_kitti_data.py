import numpy as np
import argparse
from pathlib import Path
import sys

# OpenPCDet 경로를 sys.path에 추가
# 이 스크립트를 tools 폴더에서 실행하는 것을 가정합니다.
# 만약 다른 곳에서 실행한다면 경로를 맞게 수정해주세요.
pcdet_path = Path(__file__).resolve().parent.parent
sys.path.append(str(pcdet_path))

from pcdet.utils import calibration_kitti, object3d_custom
from pcdet.datasets.a2d2.a2d2_dataset import A2D2Dataset
from pcdet.utils import box_utils as box
from visual_utils import open3d_vis_utils as V

def load_lidar_file(filepath):
    """ .bin 파일에서 포인트 클라우드를 로드합니다. """
    points = np.fromfile(str(filepath), dtype=np.float32).reshape(-1, 4)
    return points

def load_label_file(filepath):
    """ label_2/*.txt 파일에서 객체 정보를 로드합니다. """
    # pcdet/utils/object3d_kitti.py 를 사용합니다.
    # 만약 object3d_custom.py를 사용해야 한다면 아래 라인을 수정하세요.
    from pcdet.utils import object3d_custom 
    objects = object3d_custom.get_objects_from_label(filepath)
    return objects

def main():
    parser = argparse.ArgumentParser(description='Check KITTI-format data')
    parser.add_argument('--data_path', type=str, default='../data/a2d2/', help='Path to your converted KITTI-style dataset root')
    parser.add_argument('--frame_id', type=str, required=True, help='Frame ID to visualize (e.g., 20180807_145028_000000)')
    args = parser.parse_args()

    root_path = Path(args.data_path)
    frame_id = args.frame_id
    
    # 1. 파일 경로 정의
    lidar_file = root_path / 'training' / 'velodyne' / f'{frame_id}.bin'
    label_file = root_path / 'training' / 'label_2' / f'{frame_id}.txt'
    calib_file = root_path / 'training' / 'calib' / f'{frame_id}.txt'

    print(f"Lidar: {lidar_file.exists()}, Label: {label_file.exists()}, Calib: {calib_file.exists()}")

    # 2. 데이터 로드
    points = load_lidar_file(lidar_file)
    calib = calibration_kitti.Calibration(calib_file)
    objects = load_label_file(label_file)

    # 3. 카메라 좌표계 GT Box를 LiDAR 좌표계로 변환 (A2D2Dataset.__getitem__ 로직과 동일)
    gt_boxes_camera = []
    gt_names = []
    for obj in objects:
        if obj.cls_type == 'DontCare':
            continue
        # h,w,l 순서의 dims
        dims = np.array([obj.l, obj.h, obj.w]) 
        box_camera = np.concatenate([obj.loc, dims, [obj.ry]]).astype(np.float32)
        gt_boxes_camera.append(box_camera)
        gt_names.append(obj.cls_type)

    gt_boxes_camera = np.array(gt_boxes_camera)
    
    # KittiDataset 클래스에 포함된 변환 함수를 사용하여 LiDAR 박스로 변환
    gt_boxes_lidar = box.boxes3d_a2d2_camera_to_lidar(gt_boxes_camera, calib)

    print(f"Loaded {len(points)} points and {len(gt_boxes_lidar)} ground truth boxes.")

    # 4. 시각화
    V.draw_scenes(
        points=points[:, :3], 
        gt_boxes=gt_boxes_lidar
    )

if __name__ == '__main__':
    main()