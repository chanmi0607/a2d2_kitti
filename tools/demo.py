import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# kros 캡쳐용
import cv2
from clusterRF import bev_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        # DemoDataset.__init__ 내부 수정
        split_file = Path(root_path).parent / "ImageSets/val.txt"
        if logger:
            logger.info(f"======================[ DEBUG ]======================")
            logger.info(f"Root path received: {root_path}")
            logger.info(f"Checking for split file at (absolute): {split_file.resolve()}")
            logger.info(f"File exists? -> {split_file.exists()}")
            logger.info(f"=====================================================")

        if split_file.exists():
            with open(split_file, 'r') as f:
                valid_ids = [Path(line.strip()).stem for line in f.readlines()]
            data_file_list = [p for p in data_file_list if Path(p).stem in valid_ids]
            print(f"[INFO] Loaded split file: {split_file}, total {len(data_file_list)} samples selected.")
        else:
            print(f"[WARN] Split file not found: {split_file}, using all files in {root_path}")


        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

# 2D BEV 이미지에 LiDAR 박스 그리기
def draw_boxes_on_bev(
    bev_image: np.ndarray,
    boxes: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    resolution: float,
    color: tuple = (0, 0, 255), # Red (BGR)
    thickness: int = 2
) -> np.ndarray:
    """
    (N, 7) [x, y, z, l, w, h, yaw] 형식의 LiDAR 박스를 BEV 이미지에 그립니다.
    """
    # 1. 1채널(Grayscale) 이미지를 3채널(Color) 이미지로 변환
    if bev_image.ndim == 2:
        color_bev = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
    else:
        color_bev = bev_image.copy()

    # 2. 박스 순회
    for box in boxes:
        # (x, y) = LiDAR 좌표계에서의 박스 중심
        center_x_lidar = box[0]
        center_y_lidar = box[1]
        
        # (l, w) = LiDAR 좌표계에서의 박스 크기 (l=x방향, w=y방향)
        l_lidar = box[3]
        w_lidar = box[4]
        
        # (yaw) = LiDAR 좌표계에서의 회전 (Radian)
        yaw_lidar = box[6]

        # 3. LiDAR 좌표 -> 이미지 픽셀 좌표로 변환 (bev_utils.py와 동일한 로직)
        
        # 중심점 변환
        pixel_center_y = int((x_range[1] - center_x_lidar) / resolution)
        pixel_center_x = int((y_range[1] - center_y_lidar) / resolution)

        # 크기 변환 (LiDAR l -> 이미지 height, LiDAR w -> 이미지 width)
        pixel_height = int(l_lidar / resolution)
        pixel_width = int(w_lidar / resolution)

        # 각도 변환 (Radian -> Degree, OpenCV는 CCW가 +이므로 LiDAR Yaw에 -를 붙임)
        angle_degrees = -np.rad2deg(yaw_lidar)

        # 4. OpenCV의 RotatedRect 생성
        # (center(x,y), (width, height), angle_in_degrees)
        rect = ((pixel_center_x, pixel_center_y), (pixel_width, pixel_height), angle_degrees)
        
        # 5. RotatedRect의 4개 코너 좌표 계산
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points) # 정수형으로 변환

        # 6. BEV 이미지에 사각형 그리기
        cv2.drawContours(color_bev, [box_points], 0, color, thickness)
        
    return color_bev


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    class_names = demo_dataset.class_names
    logger.info(f"Class names mapping: { {i+1: name for i, name in enumerate(class_names)} }")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    bev_save_dir = Path("data/a2d2/bev_images")
    bev_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"BEV images will be saved to: {bev_save_dir.resolve()}")
    with torch.no_grad():
        # ✅ (1) 전체 결과를 담을 리스트 생성
        all_records = []

        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="[RUNNING inference]", ncols=100)):

            frame_id = Path(demo_dataset.sample_file_list[idx]).stem
            # 📍 1. 원본 포인트(Numpy)를 BEV 생성을 위해 미리 저장 (위치 유지)
            #raw_points_numpy = data_dict['points']
            raw_points_numpy = np.fromfile(demo_dataset.sample_file_list[idx], dtype=np.float32).reshape(-1, 4)

            
            # 📍 2. None으로 초기화 (위치 유지)
            filtered_boxes = None
            filtered_scores = None
            filtered_labels = None
            
            # --- (이전 코드의 BEV 생성 로직은 여기서 삭제) ---

            # 📍 3. 모델 추론 및 필터링 (먼저 수행)
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # frame_id는 위에서 이미 정의했으므로 중복 제거
            # frame_id = Path(demo_dataset.sample_file_list[idx]).stem 

            logger.info(f"==> Processing Frame: {frame_id} (Index: {idx})")

            for pred in pred_dicts:
                pred_boxes = pred['pred_boxes'].cpu().numpy()
                pred_scores = pred['pred_scores'].cpu().numpy()
                pred_labels = pred['pred_labels'].cpu().numpy()

                # (x, y) 좌표를 기준으로 2D 유클리드 거리 계산
                dists = np.sqrt(pred_boxes[:, 0]**2 + pred_boxes[:, 1]**2)
                
                # 30m 이하인 객체만 선택하는 마스크 생성
                mask = (dists <= 30.0)
                
                # 마스크를 적용하여 최종 결과 필터링
                filtered_boxes = pred_boxes[mask]
                filtered_scores = pred_scores[mask]
                filtered_labels = pred_labels[mask]

                for b, s, l in zip(filtered_boxes, filtered_scores, filtered_labels):
                    label_name = class_names[l - 1]
                    all_records.append([frame_id, *b.tolist(), s, label_name])

            if not OPEN3D_FLAG:
                mlab.show(stop=True)


            # ✅ (선택) 시각화 (3D)
            # if filtered_boxes is not None: # 필터링된 결과가 있는 경우에만
            #     V.draw_scenes(
            #         points=data_dict['points'][:, 1:],
            #         ref_boxes=filtered_boxes,     
            #         ref_scores=filtered_scores,   
            #         ref_labels=filtered_labels    
            #     )
            # else: # 필터링된 결과가 없으면 포인트 클라우드만 표시
            #     V.draw_scenes(
            #         points=data_dict['points'][:, 1:]
            #     )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            # 📍 4. BEV 생성 및 박스 저장 로직 (이 위치로 이동)
            # 이제 filtered_boxes에 실제 값이 들어있습니다.

            pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
            x_range = (pc_range[0], pc_range[3])
            y_range = (pc_range[1], pc_range[4])
            z_range = (pc_range[2], pc_range[5])
            resolution = 0.1
            
            # 1. 기본 BEV 이미지 생성 (원본 포인트 사용)
            base_bev_image = bev_utils.pointcloud_to_bev(
                points=raw_points_numpy, # 루프 시작 시 저장해 둔 numpy 사용
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                resolution=resolution
            )

            # 2. 포인트가 없는 경우 빈 캔버스 생성
            if base_bev_image is None:
                height = int((x_range[1] - x_range[0]) / resolution)
                width = int((y_range[1] - y_range[0]) / resolution)
                base_bev_image = np.zeros((height, width), dtype=np.uint8)
            
            # 3. 박스 그리기 (이제 filtered_boxes에 값이 있음)
            if filtered_boxes is not None and len(filtered_boxes) > 0:
                bev_with_boxes = draw_boxes_on_bev(
                    base_bev_image, 
                    filtered_boxes, 
                    x_range, 
                    y_range, 
                    resolution
                )
            else:
                # 박스가 없으면 그냥 3채널 컬러로 변환
                bev_with_boxes = cv2.cvtColor(base_bev_image, cv2.COLOR_GRAY2BGR)

            # 4. 최종 이미지 저장
            save_path = bev_save_dir / f"{frame_id}_bev_boxes.png"
            cv2.imwrite(str(save_path), bev_with_boxes)


            # ✅ (선택) 시각화
            # if filtered_boxes is not None: # 필터링된 결과가 있는 경우에만
            #     V.draw_scenes(
            #         points=data_dict['points'][:, 1:],
            #         ref_boxes=filtered_boxes,     # pred_dicts[0]['pred_boxes'] -> filtered_boxes
            #         ref_scores=filtered_scores,   # pred_dicts[0]['pred_scores'] -> filtered_scores
            #         ref_labels=filtered_labels    # pred_dicts[0]['pred_labels'] -> filtered_labels
            #     )
            # else: # 필터링된 결과가 없으면 포인트 클라우드만 표시
            #     V.draw_scenes(
            #         points=data_dict['points'][:, 1:]
            #     )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

        # ✅ (3) 모든 프레임의 결과를 하나의 DataFrame으로 저장
        # save_path = Path("data/a2d2/pred_all.csv")
        # save_path.parent.mkdir(parents=True, exist_ok=True)

        # cols = ["frame_id", "x", "y", "z", "l", "w", "h", "yaw", "score", "label"]
        # df = pd.DataFrame(all_records, columns=cols)
        # df.to_csv(save_path, index=False)
        # print(f"\n[SAVE] All predictions saved to {save_path}\n")



if __name__ == '__main__':
    main()
