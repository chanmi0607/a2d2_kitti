import argparse
import glob
from pathlib import Path

import open3d as o3d
import torch
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.isaacsim.isaacsim_dataset import IsaacSimDataset 

# [핵심 수정 1] GPU 로딩 함수 직접 정의 (ImportError 방지)
def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        # numpy 배열을 torch tensor로 변환 후 GPU로 이동
        # 이미 tensor인 경우에도 처리하도록 안전장치 추가
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/isaacsim_models/second.yaml', help='dataset config path')
    parser.add_argument('--ckpt', type=str, default='output/isaacsim_models/second/default/ckpt/checkpoint_epoch_10.pth', help='pretrained checkpoint')
    parser.add_argument('--data_path', type=str, default='data/isaacsim/testing/velodyne', help='folder containing .bin files')
    parser.add_argument('--save_path', type=str, default='output/vis_results', help='folder to save images')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='score threshold for visualization')
    args = parser.parse_args()
    return args

def get_geometry_from_box(box, color=[1, 0, 0]):
    # OpenPCDet Box(x, y, z, dx, dy, dz, heading) -> Open3D LineSet
    x, y, z, dx, dy, dz, heading = box
    center = np.array([x, y, z])
    l, w, h = dx, dy, dz
    c = np.cos(heading)
    s = np.sin(heading)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = np.dot(rotation_matrix, corners) + center.reshape(3, 1)
    corners = corners.transpose() 

    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    
    return line_set

def main():
    args = parse_config()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    # 1. IsaacSimDataset 초기화
    # root_path 경고는 무시해도 됩니다 (실제 로딩은 아래 glob에서 수행)
    demo_dataset = IsaacSimDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        logger=logger
    )

    # 2. 모델 빌드
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # 3. 데이터 파일 리스트 스캔
    data_file_list = glob.glob(str(Path(args.data_path) / '*.bin'))
    data_file_list.sort()
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total files to process: {len(data_file_list)}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=768)
    
    with torch.no_grad():
        for idx, data_path in enumerate(data_file_list):
            if idx % 100 == 0:
                print(f"Processing [{idx+1}/{len(data_file_list)}]: {data_path}")
            
            # ------------------------------------------------
            # [Data Loading] 3채널 -> 4채널 자동 패딩
            # ------------------------------------------------
            points = np.fromfile(data_path, dtype=np.float32)
            
            if points.shape[0] % 4 == 0:
                points = points.reshape(-1, 4)
            elif points.shape[0] % 3 == 0:
                points = points.reshape(-1, 3)
                zeros = np.zeros((points.shape[0], 1), dtype=points.dtype)
                points = np.hstack((points, zeros))
            else:
                print(f"Skipping corrupted file: {data_path}")
                continue

            # ------------------------------------------------
            # [Preprocessing] Voxelization (prepare_data 사용)
            # ------------------------------------------------
            input_dict = {
                'points': points,
                'frame_id': idx,
            }
            
            # Points -> Voxels 변환
            data_dict = demo_dataset.prepare_data(data_dict=input_dict)
            
            # Batch 구성
            data_dict = demo_dataset.collate_batch([data_dict])
            
            # [핵심 수정 2] 직접 정의한 함수 사용
            load_data_to_gpu(data_dict)

            # ------------------------------------------------
            # [Inference]
            # ------------------------------------------------
            pred_dicts, _ = model.forward(data_dict)
            
            # ------------------------------------------------
            # [Visualization]
            # ------------------------------------------------
            vis.clear_geometries()
            
            # Point Cloud
            pcd = o3d.geometry.PointCloud()
            # 
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.paint_uniform_color([0.7, 0.7, 0.7]) 
            vis.add_geometry(pcd)
            
            # Boxes
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            
            found = False
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score > args.score_thresh:
                    found = True
                    # 
                    color = [1, 0, 0] if label == 1 else [0, 1, 0]
                    line_set = get_geometry_from_box(box, color=color)
                    vis.add_geometry(line_set)
            
            vis.poll_events()
            vis.update_renderer()
            
            if idx == 0:
                vis.reset_view_point(True)

            file_name = Path(data_path).stem
            save_file = save_dir / f"{file_name}_pred.png"
            vis.capture_screen_image(str(save_file), do_render=True)
            
    vis.destroy_window()
    print(f"Done! Results saved to {args.save_path}")

if __name__ == '__main__':
    main()