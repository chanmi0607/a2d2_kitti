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
    with torch.no_grad():
        # ✅ (1) 전체 결과를 담을 리스트 생성
        all_records = []

        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="[RUNNING inference]", ncols=100)):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            frame_id = Path(demo_dataset.sample_file_list[idx]).stem

            for pred in pred_dicts:
                pred_boxes = pred['pred_boxes'].cpu().numpy()
                pred_scores = pred['pred_scores'].cpu().numpy()
                pred_labels = pred['pred_labels'].cpu().numpy()

                for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
                    label_name = class_names[l - 1]
                    all_records.append([frame_id, *b.tolist(), s, label_name])

            if not OPEN3D_FLAG:
                mlab.show(stop=True)


            # ✅ (선택) 시각화
            V.draw_scenes(
                points=data_dict['points'][:, 1:],
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=pred_dicts[0]['pred_labels']
            )
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

        # ✅ (3) 모든 프레임의 결과를 하나의 DataFrame으로 저장
        save_path = Path("data/a2d2/pred_all.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cols = ["frame_id", "x", "y", "z", "l", "w", "h", "yaw", "score", "label"]
        df = pd.DataFrame(all_records, columns=cols)
        df.to_csv(save_path, index=False)
        print(f"\n[SAVE] All predictions saved to {save_path}\n")



if __name__ == '__main__':
    main()
