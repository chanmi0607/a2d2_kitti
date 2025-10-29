# import argparse
# import numpy as np
# import torch
# from pathlib import Path
# import open3d as o3d
# import warnings
# import time

# warnings.filterwarnings("ignore", category=FutureWarning)

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network
# from pcdet.utils import common_utils, box_utils


# # ===============================
# # DemoDataset
# # ===============================
# class DemoDataset(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
#         super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
#                          root_path=root_path, logger=logger)
#         self.root_path = Path(root_path)
#         self.ext = ext
#         if self.root_path.is_dir():
#             self.sample_file_list = sorted(list(self.root_path.glob(f'*{ext}')))
#         elif self.root_path.is_file():
#             self.sample_file_list = [self.root_path]
#         else:
#             raise FileNotFoundError(f"Invalid data path: {self.root_path}")

#         if len(self.sample_file_list) == 0:
#             raise FileNotFoundError(f"No {ext} files found in {self.root_path}")

#     def __len__(self):
#         return len(self.sample_file_list)

#     def __getitem__(self, index):
#         current_file = self.sample_file_list[index]
#         points = np.fromfile(current_file, dtype=np.float32).reshape(-1, 4)
#         input_dict = {'points': points, 'frame_id': current_file.stem}
#         return self.prepare_data(data_dict=input_dict)

#     def get_lidar(self, file_path):
#         points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
#         return points


# # ===============================
# # GPU 데이터 로드
# # ===============================
# def load_data_to_gpu(batch_dict):
#     for key, val in batch_dict.items():
#         if not isinstance(val, np.ndarray):
#             continue
#         elif key in ['frame_id', 'metadata', 'calib']:
#             continue
#         elif key in ['images']:
#             pass
#         elif key in ['image_shape']:
#             batch_dict[key] = torch.from_numpy(val).int().cuda()
#         else:
#             batch_dict[key] = torch.from_numpy(val).float().cuda()


# # ===============================
# # 설정 파싱
# # ===============================
# def parse_config():
#     parser = argparse.ArgumentParser(description='RPN Test')
#     parser.add_argument('--cfg_file', type=str,
#                         default='/home/a/OpenPCDet/tools/cfgs/a2d2_models/second.yaml')
#     parser.add_argument('--data_path', type=str,
#                         default='data/a2d2/training/velodyne')
#     parser.add_argument('--ckpt', type=str,
#                         default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth')
#     parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
#     parser.add_argument('--vis_score_thresh', type=float, default=0.1)
#     parser.add_argument('--frame_idx', type=int, default=91,
#                         help='Specify which frame index to visualize (-1 = all frames)')
#     args = parser.parse_args()
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     return args, cfg


# # ===============================
# # 메인 실행
# # ===============================
# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger()
#     logger.info('----------------- RPN Visualization -------------------------')

#     data_path = Path(args.data_path)
#     demo_dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, root_path=data_path, logger=logger)
#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
#     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
#     model.cuda()
#     model.eval()

#     total_frames = len(demo_dataset)
#     logger.info(f"Total frames found: {total_frames}")

#     # 단일 프레임 모드일 경우 인덱스 범위 확인
#     frame_indices = [args.frame_idx] if args.frame_idx >= 0 else range(total_frames)
#     if args.frame_idx >= total_frames:
#         logger.error(f"Invalid frame index {args.frame_idx}. Dataset has only {total_frames} frames.")
#         return

#     for i in frame_indices:
#         data_dict_raw = demo_dataset[i]
#         frame_id = demo_dataset.sample_file_list[i].stem
#         logger.info(f"Processing frame {i} ({frame_id})")

#         data_dict = demo_dataset.collate_batch([data_dict_raw])
#         load_data_to_gpu(data_dict)

#         with torch.no_grad():
#             model(data_dict)

#         # --- RPN 출력 (before NMS) ---
#         box_preds = data_dict.get('batch_box_preds', None)
#         cls_preds = data_dict.get('batch_cls_preds', None)

#         if box_preds is None or cls_preds is None:
#             logger.warning(f"No RPN outputs for frame {frame_id}")
#             continue

#         pred_boxes_all = box_preds[0].cpu().numpy()
#         pred_scores = torch.sigmoid(cls_preds[0]).cpu().numpy()
#         max_scores = np.max(pred_scores, axis=1)
#         mask = max_scores >= args.vis_score_thresh
#         pred_boxes = pred_boxes_all[mask]

#         logger.info(f"  - Showing {len(pred_boxes)} RPN boxes (score ≥ {args.vis_score_thresh})")

#         # --- 시각화 ---
#         if not args.no_vis:
#             points = demo_dataset.get_lidar(demo_dataset.sample_file_list[i])

#             vis = o3d.visualization.Visualizer()
#             vis.create_window(window_name=f"RPN - Frame {frame_id}", width=1280, height=720)
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#             vis.add_geometry(pcd)

#             for box in pred_boxes:
#                 corners = box_utils.boxes_to_corners_3d(box[np.newaxis, :])[0]
#                 lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
#                 colors = [[1, 0, 0] for _ in range(len(lines))]
#                 line_set = o3d.geometry.LineSet(
#                     points=o3d.utility.Vector3dVector(corners),
#                     lines=o3d.utility.Vector2iVector(lines)
#                 )
#                 line_set.colors = o3d.utility.Vector3dVector(colors)
#                 vis.add_geometry(line_set)

#             opt = vis.get_render_option()
#             opt.background_color = np.asarray([0, 0, 0])
#             vis.run()
#             vis.destroy_window()

#             # 프레임 지정 모드가 아닐 경우, 다음 프레임 이동 제어
#             if args.frame_idx == -1:
#                 user_in = input("Press [Enter] for next frame or [q] to quit: ").strip().lower()
#                 if user_in == 'q':
#                     break

#     logger.info('----------------- End of RPN Visualization -------------------------')


# if __name__ == '__main__':
#     main()



# 쭉 보기
import argparse
import numpy as np
import torch
from pathlib import Path
import open3d as o3d
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root_path, logger=logger)
        self.root_path = Path(root_path)
        self.ext = ext
        self.sample_file_list = sorted(list(self.root_path.glob(f'*{ext}')))
    def __len__(self):
        return len(self.sample_file_list)
    def __getitem__(self, index):
        points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        return self.prepare_data({'points': points, 'frame_id': index})
    def get_lidar(self, file_path):
        if not file_path.exists():
             raise FileNotFoundError(f"LiDAR file not found: {file_path}")
        points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
        return points


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/a2d2_models/second.yaml')
    parser.add_argument('--data_path', type=str, default='data/a2d2/training/velodyne')
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_config()
    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, False, Path(args.data_path), logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    vis = o3d.visualization.Visualizer()
    vis.create_window("RPN Visualization", width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    for i, data_dict in enumerate(dataset):
        frame_id = dataset.sample_file_list[i].stem
        logger.info(f"Frame {i+1}/{len(dataset)}: {frame_id}")
        data_dict = dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            model.forward(data_dict)

        # === RPN 결과 가져오기 ===
        box_preds = data_dict['batch_box_preds'][0].cpu().numpy()
        cls_preds = torch.sigmoid(data_dict['batch_cls_preds'][0]).cpu().numpy()
        scores = np.max(cls_preds, axis=1)
        mask = scores > args.score_thresh
        box_preds = box_preds[mask]

        # === 포인트 클라우드 ===
        points = dataset.get_lidar(dataset.sample_file_list[i])

        # === 시각화 ===
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        vis.add_geometry(pcd)

        for box in box_preds:
            corners = box_utils.boxes_to_corners_3d(box[np.newaxis, :])[0]
            lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            color = [[1, 0, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners),
                                            lines=o3d.utility.Vector2iVector(lines))
            line_set.colors = o3d.utility.Vector3dVector(color)
            vis.add_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()
        print(f"[Frame {frame_id}] showing {len(box_preds)} boxes (score>{args.score_thresh})")
        user_input = input("Press [Enter] for next, [q] to quit: ").strip().lower()
        if user_input == 'q':
            break

    vis.destroy_window()
    logger.info("RPN visualization finished.")


if __name__ == '__main__':
    main()