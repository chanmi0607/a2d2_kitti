from pathlib import Path
from pcdet.datasets.a2d2.a2d2_dataset import A2D2Dataset

# 1. Data loading
    # dataset을 불러옴 (GT, lidar...)
def setup_dataset(cfg, args, logger):
    """
    args와 cfg를 기반으로 A2D2Dataset을 초기화하고 설정합니다.
    
    Args:
        cfg (obj): OpenPCDet 설정 객체
        args (obj): argparse로 파싱된 인자
        logger (obj): 로깅 객체

    Returns:
        dataset (A2D2Dataset): 성공적으로 로드된 데이터셋 객체
        None: 초기화 또는 경로 확인 실패 시
    """
    try:
        # args.split 값에 따라 training 플래그 결정
        is_training_split = (args.split == 'train')

        # 1. 데이터셋 객체 생성
        dataset = A2D2Dataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
            training=is_training_split, # <--- 'train'일 때만 True
            logger=logger
        )
        
        # 2. 스플릿 설정
        dataset.set_split(args.split)
        logger.info(f"Loaded {args.split} split with {len(dataset.sample_id_list)} frames (from ImageSets).")
        logger.info(f"Total {len(dataset)} frames in {args.split} split.")

        # 3. GT 라벨 경로 확인 (중요)
        gt_label_dir = dataset.root_path / f'new_label/label_new_{args.split}'
        if not gt_label_dir.exists():
            logger.error(f"GT label directory not found: {gt_label_dir}")
            logger.error("Please run 'create_new_label_files.py' first.")
            return None  # 실패 시 None 반환
        
        dataset.gt_label_dir = gt_label_dir
        # 4. 성공 시 데이터셋 반환
        return dataset

    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset initialization: {e}")
        return None # 예외 발생 시 None 반환
    