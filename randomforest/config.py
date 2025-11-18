import argparse

# OpenPCDet 임포트
from pcdet.config import cfg, cfg_from_yaml_file

# --- [GLOBAL] 피처 컬럼 정의 ---
# 머신러닝에 사용할 피처 목록 (18개)
ML_FEATURE_COLUMNS = [
    "RPN_MaxScore", "x", "y", "z", "l", "w", "h", "yaw",
    "num_points", "width", "length", "height", "density",
    "aspect_ratio", "mean_z", "std_z", "intensity_mean", "intensity_std"
]
# CSV에 저장될 전체 컬럼 목록 (피처 + 라벨/메타데이터)
CSV_COLUMNS = ML_FEATURE_COLUMNS + ["label", "max_iou", "frame_id"]
# RF Stage 1의 'Object' 라벨 (학습 시 1로 인코딩됨)
STAGE_1_OBJECT_LABEL = 1
##
##cm test
##

# --- 설정 파싱 함수 (수정) ---
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # --- 1. 공통 인자 ---
    parser.add_argument('--mode', type=str, default='extract', 
                        choices=['extract', 'train', 'evaluate', 'inference', 'evaluate gt'], # 'inference' 추가
                        help='Operation mode: extract, train, evaluate, or inference.')

    # --- 2. 'extract' & 'inference' 모드 인자 ---
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/a2d2_models/second.yaml', help='(extract/inference) pcdet config file')
    parser.add_argument('--split', type=str, default='val', help='(extract/inference) Data split to process')
    parser.add_argument('--ckpt', type=str, default='output/a2d2_models/second/a2d2_cyclist_best/ckpt/checkpoint_epoch_200.pth', help='(extract/inference) pcdet checkpoint')
    parser.add_argument('--min_points_in_box', type=int, default=3, help='(extract/inference) Min points in box')
    parser.add_argument('--no_vis', action='store_true', help='(extract/inference) Disable visualization')
    parser.add_argument('--vis_frame_limit', type=int, default=5, help='(extract/inference) Limit visualization frames')
    
    # --- 3. 'extract' 전용 인자 ---
    parser.add_argument('--output_csv', type=str, default=None, help='(extract) Path to save the output CSV file.')
    parser.add_argument('--fg_thresh', type=float, default=0.3, help='(extract) IoU threshold for "Foreground"')
    parser.add_argument('--bg_thresh', type=float, default=0.1, help='(extract) IoU threshold for "Background"')

    # --- 4. 'train' 모드 인자 ---
    parser.add_argument('--train_file', type=str, default='data/a2d2/features_train.csv', help='(train) Path to the training features CSV')
    parser.add_argument('--model1_out', type=str, default='data/a2d2/rf_stage1_model.pkl', help='(train) Path to save the Stage 1 model')
    parser.add_argument('--model2_out', type=str, default='data/a2d2/rf_stage2_model.pkl', help='(train) Path to save the Stage 2 model')
    parser.add_argument('--le_out', type=str, default='data/a2d2/le_stage2.pkl', help='(train) Path to save the LabelEncoder') # [추가]

    # --- 5. 'evaluate' & 'inference' 모드 인자 ---
    parser.add_argument('--test_file', type=str, default='data/a2d2/features_val.csv', help='(evaluate) Path to the test features CSV')
    parser.add_argument('--model1_path', type=str, default='data/a2d2/rf_stage1_model.pkl', help='(evaluate/inference) Path to the Stage 1 model')
    parser.add_argument('--model2_path', type=str, default='data/a2d2/rf_stage2_model.pkl', help='(evaluate/inference) Path to the Stage 2 model')
    parser.add_argument('--le_path', type=str, default='data/a2d2/le_stage2.pkl', help='(evaluate/inference) Path to the LabelEncoder') # [추가]

    args = parser.parse_args()

    # 'extract' 또는 'inference' 모드일 때만 pcdet 설정 로드
    if args.mode in ['extract', 'inference', 'evaluate gt']:
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
    
    if args.mode == 'extract' and args.output_csv is None:
        args.output_csv = f'data/a2d2/new_background_features_{args.split}.csv'
    
    return args, cfg