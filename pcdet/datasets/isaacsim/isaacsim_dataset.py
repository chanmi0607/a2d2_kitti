import copy
import pickle

import numpy as np
from skimage import io

from . import isaacsim_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

import concurrent.futures as futures

class IsaacSimDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        
        # 1. 모드 확인 ('train', 'val', 'test')
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        
        # 2. [수정됨] 폴더 선택 로직 (지역변수 split_dir -> 멤버변수 self.root_split_path)
        if self.split == 'test':
            self.root_split_path = self.root_path / 'testing'
        else:
            self.root_split_path = self.root_path / 'training'
        # 3. 실제 LiDAR 데이터 경로 설정
        # 구조: /training/velodyne_points/data/*.bin
        lidar_dir = self.root_split_path / 'velodyne_points' / 'data'
        

        if not lidar_dir.exists():
            print(f"Warning: LiDAR directory not found at {lidar_dir}")
            self.sample_id_list = []
        else:
            # 4. 파일 스캔 (ID 리스트 생성)
            # .bin 파일들을 읽어서 정렬 (순서가 섞이지 않도록 sorted 필수)
            file_list = sorted(list(lidar_dir.glob('*.bin')))
            all_ids = [file.stem for file in file_list] # 확장자 제외한 ID만 추출

            # 5. Train / Validation 분할 로직
            # 'test' 모드일 때는 testing 폴더의 모든 파일을 사용
            if self.split == 'test':
                self.sample_id_list = all_ids
            else:
                # 'training' 폴더 안에 있는 데이터를 train용과 val용으로 나눔
                # 예: 전체의 80%는 학습용, 뒤쪽 20%는 검증용
                # (데이터가 섞여있지 않고 순서대로라고 가정)
                total_len = len(all_ids)
                split_idx = int(total_len * 0.8) # 8:2 비율 (조절 가능)

                if self.split == 'train':
                    self.sample_id_list = all_ids[:split_idx]
                elif self.split == 'val':
                    self.sample_id_list = all_ids[split_idx:]
                else:
                    # 'trainval' 등의 옵션이 있을 경우 전체 사용
                    self.sample_id_list = all_ids

        # 로그 출력
        if logger is not None:
             logger.info(f'Total samples for IsaacSim dataset ({self.split}): {len(self.sample_id_list)}')
        
        # [중요] include_isaac_data는 이제 리스트를 만드는 역할이 아니라, 
        # 만들어진 self.sample_id_list를 정보를 로딩하는 역할만 해야 함.
        # (만약 include_isaac_data 함수 안에서 또 파일을 읽는다면 그 부분은 지워야 합니다)
        self.isaac_infos = []
        self.include_isaac_data(self.mode)

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

    def set_split(self, split):
        # 부모 클래스 초기화 (기본 설정 유지)
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        
        # 1. 루트 경로 설정 (Training vs Testing)
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        # 2. LiDAR 데이터 경로 설정
        # 구조: /training/velodyne_points/data/*.bin
        lidar_dir = self.root_split_path / 'velodyne_points' / 'data'

        # 3. 파일 스캔 및 리스트 생성 (기존 txt 파일 읽는 코드 삭제)
        if not lidar_dir.exists():
            print(f"Warning: LiDAR directory not found at {lidar_dir}")
            self.sample_id_list = []
        else:
            # 파일 읽기 (.bin)
            file_list = sorted(list(lidar_dir.glob('*.bin')))
            all_ids = [file.stem for file in file_list]
            
            # 4. Train / Val 데이터 분할 (8:2 비율)
            if self.split == 'test':
                self.sample_id_list = all_ids
            else:
                total_len = len(all_ids)
                split_idx = int(total_len * 0.8) # 80% 지점 계산

                if self.split == 'train':
                    self.sample_id_list = all_ids[:split_idx]
                elif self.split == 'val':
                    self.sample_id_list = all_ids[split_idx:]
                else:
                    self.sample_id_list = all_ids

    def get_lidar(self, idx):
        # [수정] 경로를 velodyne -> velodyne_points/data 로 변경
        lidar_file = self.root_split_path / 'velodyne_points' / 'data' / ('%s.bin' % idx)
        assert lidar_file.exists(), f"File not found: {lidar_file}"
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    
    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            
            # [변경 1] 포인트 클라우드 정보 생성
            # PCDet 등에서는 num_features가 중요하므로 유지
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # [추가] count_inside_pts를 위해 포인트 클라우드를 실제로 읽어와야 함
            # self.get_lidar() 함수가 구현되어 있다고 가정
            if count_inside_pts:
                points = self.get_lidar(sample_idx) 

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                
                # 1. 기본 정보 (유지)
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                # difficulty는 평가 시 필요할 수 있으므로 유지 (없으면 기본값 0)
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32) 
                
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                
                # index 생성 (유지)
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                # 2. 이미지 관련 불필요한 속성 (더미 값 처리)
                # 이미지가 없으므로 2D bbox는 모두 0으로 처리
                annotations['bbox'] = np.zeros((num_gt, 4)) 
                # 잘림, 가려짐 등은 라이다에서 큰 의미 없으나 포맷 유지를 위해 0 처리
                annotations['truncated'] = np.zeros(num_gt)
                annotations['occluded'] = np.zeros(num_gt)
                annotations['alpha'] = np.zeros(num_gt)

                # 3. 3D 좌표 및 박스 생성 (핵심 변경 구간)
                # 가정: obj_list의 데이터가 이미 'LiDAR 좌표계' 기준이라고 가정합니다.
                # obj.loc = [x, y, z], obj.l/w/h = dx, dy, dz, obj.ry = heading
                
                if num_objects > 0:
                    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)[:num_objects]
                    dims = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])[:num_objects] # 순서 주의 (dx, dy, dz)
                    rots = np.array([obj.ry for obj in obj_list])[:num_objects]

                    # [변경] calib.rect_to_lidar 삭제 및 좌표 직접 할당
                    # 만약 라벨의 z좌표가 바닥면 기준이라면 중심점으로 올리는 작업 필요 (상황에 따라 주석 처리)
                    # loc[:, 2] += dims[:, 2] / 2  

                    # gt_boxes_lidar 포맷: [x, y, z, dx, dy, dz, heading]
                    # 원래 KITTI 코드는 회전축 변환(-(np.pi/2 + rot))을 하지만, 
                    # 데이터가 순수 라이다 기준이면 rots 그대로 사용 (rots[..., np.newaxis])
                    gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                else:
                    gt_boxes_lidar = np.zeros((0, 7))

                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                
                # [삭제] info['annos']에 넣기 전 위치 변환(camera coordinates) 관련 필드는
                # 훈련 코드에서 gt_boxes_lidar만 쓴다면 location, dimensions, rotation_y는 
                # 더미로 넣거나 위에서 계산한 값 그대로 넣어주면 됩니다.
                annotations['location'] = loc if num_objects > 0 else np.zeros((0, 3))
                annotations['dimensions'] = dims if num_objects > 0 else np.zeros((0, 3))
                annotations['rotation_y'] = rots if num_objects > 0 else np.zeros(0)

                info['annos'] = annotations

                # 4. 박스 내부 포인트 개수 계산 (유지하되 points 변수 사용)
                if count_inside_pts and num_objects > 0:
                    # points 변수는 위에서 로드함
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        # pts_fov 대신 실제 로드한 points 사용
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('isaac_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
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
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
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

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            
            # 템플릿 생성
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # [삭제] calib, image_shape 로딩 및 카메라 좌표계 변환 로직 제거
            # 대신 LiDAR 좌표계를 그대로 사용합니다.

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            
            # [변경] Alpha: 카메라 관측 각도이므로 LiDAR에서는 의미 없음 -> -10 (KITTI 'unknown' 관례)
            pred_dict['alpha'] = -np.ones(len(pred_boxes)) * 10 
            
            # [변경] 2D Bbox: 이미지가 없으므로 [0, 0, 0, 0] 처리
            pred_dict['bbox'] = np.zeros((len(pred_boxes), 4))
            
            # [변경] 3D 정보: 변환 없이 LiDAR 예측값 그대로 사용
            # pred_boxes: [x, y, z, dx, dy, dz, heading]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]  # dx, dy, dz
            pred_dict['location'] = pred_boxes[:, 0:3]    # x, y, z
            pred_dict['rotation_y'] = pred_boxes[:, 6]    # heading
            
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # 현재 순서: dx, dy, dz
                    
                    for idx in range(len(bbox)):
                        # KITTI 포맷 순서: 
                        # type, truncated, occluded, alpha, bbox(4), h, w, l, x, y, z, ry, score
                        
                        # [주의] dimensions 출력 순서 변경
                        # 기존 코드: l,h,w 순서에서 h,w,l 로 출력했음
                        # 현재 코드: dx(l), dy(w), dz(h) 순서임
                        # 따라서 높이(h)=dims[2], 너비(w)=dims[1], 길이(l)=dims[0] 순으로 출력해야 함
                        
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (single_pred_dict['name'][idx], 
                                single_pred_dict['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                dims[idx][2], dims[idx][1], dims[idx][0], # h, w, l (dz, dy, dx)
                                loc[idx][0], loc[idx][1], loc[idx][2],    # x, y, z
                                single_pred_dict['rotation_y'][idx],
                                single_pred_dict['score'][idx]), file=f)

        return annos
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.isaac_infos[0].keys():
            return None, {}

        from .isaacsim_object_eval_python import eval as isaac_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.isaac_infos]
        ap_result_str, ap_dict = isaac_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.isaac_infos) * self.total_epochs

        return len(self.isaac_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.isaac_infos)

        info = copy.deepcopy(self.isaac_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # [삭제/변경] 이미지 형식이 없으므로 더미 값 혹은 삭제
        # img_shape = info['image']['image_shape'] 
        
        # [삭제] calib 파일 로딩 제거
        # calib = self.get_calib(sample_idx) 

        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            # 'calib': calib, # calib가 파이프라인에서 필수라면 None 혹은 더미 객체 전달 필요
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            
            # [핵심 변경] 카메라 좌표계 변환 로직 제거 -> 저장된 라이다 박스 바로 사용
            # 이전 단계(get_infos)에서 'gt_boxes_lidar' 키로 저장했다고 가정
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            else:
                # 만약 get_infos를 수정 안했다면 여기서 변환해야 하지만, 
                # 수정했다면 아래 로직은 필요 없습니다.
                # gt_boxes_camera = ... (삭제)
                # gt_boxes_lidar = ... (삭제)
                pass

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar
            })
            
            # gt_boxes2d는 이미지상 박스이므로 필요 없으면 제거하거나 더미 처리
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos.get("bbox", np.zeros((len(annos['name']), 4)))

            # road_plane도 카메라 기반으로 생성되는 경우가 많아 라이다 단독이면 보통 제거
            # if road_plane is not None: ... (삭제 권장)

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            # [삭제] FOV_POINTS_ONLY 제거
            # 카메라 화각 밖의 포인트를 자르는 로직인데, 360도 라이다만 쓴다면 불필요함
            # if self.dataset_cfg.FOV_POINTS_ONLY: ...
            
            input_dict['points'] = points

        # [삭제] 이미지 및 뎁스맵 로딩 부분 전체 삭제
        # if "images" in get_item_list: ...
        # if "depth_maps" in get_item_list: ...
        # if "calib_matricies" in get_item_list: ...

        # input_dict['calib'] = calib # 필요 시 주석 처리 혹은 더미

        # 데이터 증강(Augmentation) 수행
        data_dict = self.prepare_data(data_dict=input_dict)

        # 이미지 shape 더미 처리 (모델 구조상 필요할 경우)
        data_dict['image_shape'] = [0, 0] 
        
        return data_dict


def create_isaacsim_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = IsaacSimDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('isaac_infos_%s.pkl' % train_split)
    val_filename = save_path / ('isaac_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'isaac_infos_trainval.pkl'
    test_filename = save_path / 'isaac_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    isaac_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(isaac_infos_train, f)
    print('isaac info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    isaac_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(isaac_infos_val, f)
    print('isaac info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(isaac_infos_train + isaac_infos_val, f)
    print('isaac info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    isaac_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(isaac_infos_test, f)
    print('isaac info test file is saved to %s' % test_filename)

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
        create_isaacsim_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Truck'],
            data_path=ROOT_DIR / 'data' / 'isaacsim',
            save_path=ROOT_DIR / 'data' / 'isaacsim'
        )
