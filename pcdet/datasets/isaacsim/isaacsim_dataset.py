import copy
import pickle
import numpy as np
from pathlib import Path
import concurrent.futures as futures

from . import isaacsim_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
import random # [추가] 맨 위에 임포트 필요

class IsaacSimDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        
        # 1. 초기 모드 설정
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = None
        self.sample_id_list = []
        
        # 2. 경로 및 ID 리스트 초기화
        # __init__에서도 이 로직이 돌아가야 초기 상태가 잡힘
        self.refresh_data_path()

        if logger is not None:
             logger.info(f'Total samples for IsaacSim dataset ({self.split}): {len(self.sample_id_list)}')
        
        self.isaac_infos = []
        self.include_isaac_data(self.mode)

    def refresh_data_path(self):
        """
        self.split 값에 따라 root_split_path와 sample_id_list를 갱신하는 함수
        """
        # 1. 폴더 선택 로직
        if self.split == 'test':
            self.root_split_path = self.root_path / 'testing'
        else:
            self.root_split_path = self.root_path / 'training'

        # 2. LiDAR 데이터 경로 확인
        lidar_dir = self.root_split_path / 'velodyne_points' / 'data'
        
        if not lidar_dir.exists():
            print(f"Warning: LiDAR directory not found at {lidar_dir}")
            self.sample_id_list = []
            return

        # 3. 파일 스캔
        file_list = sorted(list(lidar_dir.glob('*.bin')))
        all_ids = [file.stem for file in file_list]

        # [핵심 수정] 4. 데이터 셔플 (Shuffle)
        # Training 폴더를 Train/Val로 나눌 때 섞어서 나눕니다.
        if self.split != 'test':
            # 시드를 고정해야 언제 실행하든 똑같은 Train/Val 세트가 만들어집니다.
            random.seed(666) 
            random.shuffle(all_ids)

        # 5. 데이터 분할 로직 (8:2)
        if self.split == 'test':
            self.sample_id_list = all_ids
        else:
            total_len = len(all_ids)
            split_idx = int(total_len * 0.8)

            if self.split == 'train':
                self.sample_id_list = all_ids[:split_idx]
            elif self.split == 'val':
                self.sample_id_list = all_ids[split_idx:]
            else:
                self.sample_id_list = all_ids # trainval
        
        # [디버깅] 셔플 결과 확인용
        # print(f"[{self.split}] Sample count: {len(self.sample_id_list)} (First: {self.sample_id_list[0]})")

    def __len__(self):
            if self._merge_all_iters_to_one_epoch:
                return len(self.isaac_infos) * self.total_epochs

            return len(self.isaac_infos)

    def __getitem__(self, index):
        # 1. Index 처리 (Epoch 병합 옵션 대응)
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.isaac_infos)

        # 2. 정보 가져오기
        info = copy.deepcopy(self.isaac_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 3. 포인트 클라우드 로드
        points = self.get_lidar(sample_idx)
        
        # 4. Input Dict 구성
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        # 5. 라벨(Annotation) 처리
        if 'annos' in info:
            annos = info['annos']
            # DontCare 제거 (필요 시)
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            
            # [중요] gt_boxes_lidar 사용
            # create_isaacsim_infos 과정에서 이미 LiDAR 좌표계 박스를 저장했으므로 그대로 사용
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            else:
                # 만약 없다면 예외처리 혹은 빈 박스
                gt_boxes_lidar = np.zeros((0, 7))

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar
            })

        # 6. 데이터 증강 (Augmentation) 및 전처리 (Voxelization 등)
        # DatasetTemplate의 prepare_data가 config에 정의된 프로세서를 순서대로 실행함
        data_dict = self.prepare_data(data_dict=input_dict)

        # 7. 이미지 Shape 더미 처리 (모델 구조상 요구할 경우)
        data_dict['image_shape'] = [0, 0] 

        return data_dict
    def set_split(self, split):
        """
        외부에서 데이터셋의 모드를 변경할 때 호출됨 (create_isaacsim_infos 등)
        """
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        
        # [핵심] 모드가 바뀌었으니 경로와 파일 리스트를 다시 계산
        self.refresh_data_path()

    def include_isaac_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Isaac SIM dataset')
        isaac_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                isaac_infos.extend(infos)

        self.isaac_infos.extend(isaac_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Isaac dataset: %d' % (len(isaac_infos)))

    def get_lidar(self, idx):
        # 갱신된 root_split_path 사용
        lidar_file = self.root_split_path / 'velodyne_points' / 'data' / ('%s.bin' % idx)
        assert lidar_file.exists(), f"File not found: {lidar_file}"
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    
    def get_label(self, idx):
        # 갱신된 root_split_path 사용
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        # Test 셋에는 라벨이 없을 수 있으므로 체크
        if not label_file.exists():
            return []
        return object3d_kitti.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            # 진행상황 표시
            # print('%s sample_idx: %s' % (self.split, sample_idx)) 
            info = {}
            
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # points 로드 (count_inside_pts 계산용)
            if count_inside_pts:
                points = self.get_lidar(sample_idx)

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                
                # 라벨이 없는 경우 (빈 리스트 처리)
                if len(obj_list) == 0:
                    annotations['name'] = np.array([])
                    annotations['score'] = np.array([])
                    annotations['difficulty'] = np.array([], dtype=np.int32)
                    annotations['index'] = np.array([], dtype=np.int32)
                    annotations['bbox'] = np.zeros((0, 4))
                    annotations['truncated'] = np.zeros(0)
                    annotations['occluded'] = np.zeros(0)
                    annotations['alpha'] = np.zeros(0)
                    annotations['gt_boxes_lidar'] = np.zeros((0, 7))
                    annotations['location'] = np.zeros((0, 3))
                    annotations['dimensions'] = np.zeros((0, 3))
                    annotations['rotation_y'] = np.zeros(0)
                    annotations['num_points_in_gt'] = np.zeros(0, dtype=np.int32)
                else:
                    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                    annotations['score'] = np.array([obj.score for obj in obj_list])
                    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                    
                    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                    num_gt = len(annotations['name'])
                    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                    annotations['index'] = np.array(index, dtype=np.int32)

                    annotations['bbox'] = np.zeros((num_gt, 4))
                    annotations['truncated'] = np.zeros(num_gt)
                    annotations['occluded'] = np.zeros(num_gt)
                    annotations['alpha'] = np.zeros(num_gt)

                    if num_objects > 0:
                        loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)[:num_objects]
                        dims = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])[:num_objects]
                        rots = np.array([obj.ry for obj in obj_list])[:num_objects]

                        # GT Boxes LiDAR: [x, y, z, dx, dy, dz, heading]
                        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                    else:
                        gt_boxes_lidar = np.zeros((0, 7))
                        loc = np.zeros((0, 3))
                        dims = np.zeros((0, 3))
                        rots = np.zeros(0)

                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                    annotations['location'] = loc
                    annotations['dimensions'] = dims
                    annotations['rotation_y'] = rots

                    if count_inside_pts and num_objects > 0:
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                        for k in range(num_objects):
                            flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt

                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = list(executor.map(process_single_scene, sample_id_list))
        return infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('isaac_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        # [디버깅 코드 시작] ----------------------------
        unique_classes = set()
        for info in infos:
            if 'annos' in info:
                unique_classes.update(info['annos']['name'])
        
        print(f"DEBUG: Loaded pkl file contains classes: {unique_classes}")
        
        if 'Pedestrian' not in unique_classes:
            print("!!! WARNING: 'Pedestrian' class NOT found in pkl file. Check get_infos() or Label files.")
        # [디버깅 코드 끝] ------------------------------

        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                # GT Box 중심으로 포인트 이동 (Crop)
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

# ---------------------------------------------------------------------------- #
# 데이터 생성 실행 함수
# ---------------------------------------------------------------------------- #

def create_isaacsim_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    # 1. 초기 데이터셋 인스턴스 생성
    dataset = IsaacSimDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    
    train_split = 'train'
    val_split = 'val'
    test_split = 'test'

    train_filename = save_path / ('isaac_infos_%s.pkl' % train_split)
    val_filename = save_path / ('isaac_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'isaac_infos_trainval.pkl'
    test_filename = save_path / 'isaac_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    # 2. Train Infos 생성
    # set_split('train') -> 내부적으로 root/training 폴더를 바라보고, 리스트의 앞 80%를 가져옴
    print(f"Generating {train_split} infos...")
    dataset.set_split(train_split)
    isaac_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(isaac_infos_train, f)
    print('isaac info train file is saved to %s' % train_filename)

    # 3. Val Infos 생성
    # set_split('val') -> 내부적으로 root/training 폴더를 바라보고, 리스트의 뒤 20%를 가져옴
    print(f"Generating {val_split} infos...")
    dataset.set_split(val_split)
    isaac_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(isaac_infos_val, f)
    print('isaac info val file is saved to %s' % val_filename)

    # 4. TrainVal 통합 Info 저장
    with open(trainval_filename, 'wb') as f:
        pickle.dump(isaac_infos_train + isaac_infos_val, f)
    print('isaac info trainval file is saved to %s' % trainval_filename)

    # 5. Test Infos 생성
    # set_split('test') -> 내부적으로 root/testing 폴더를 바라보고, 전체 리스트를 가져옴
    print(f"Generating {test_split} infos...")
    dataset.set_split(test_split)
    # Test셋은 보통 Label이 없다고 가정 (has_label=False). 만약 라벨이 있다면 True로 변경.
    isaac_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(isaac_infos_test, f)
    print('isaac info test file is saved to %s' % test_filename)

    # 6. GT Database 생성 (Data Augmentation용)
    # Train 셋을 기준으로 생성
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split) 
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_isaacsim_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        
        # 실제 데이터 경로에 맞게 data_path 수정 필요
        create_isaacsim_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Truck','Pedestrian'],
            data_path=ROOT_DIR / 'data' / 'isaacsim',
            save_path=ROOT_DIR / 'data' / 'isaacsim'
        )