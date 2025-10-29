import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 경고 무시
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- 1. 경로 및 설정 ---

# [!] (필수) 훈련용 GT 라벨 폴더 경로
GT_TRAIN_DIR = Path('data/a2d2/new_label/label_new_train') 

# [!] (필수) 테스트(검증)용 GT 라벨 폴더 경로
GT_TEST_DIR = Path('data/a2d2/new_label/label_new_val') # 혹은 label_new_test

# [!] (필수) CSV 파일을 생성합니다.
SAVE_FEATURES_TO_CSV = True
CSV_OUTPUT_DIR = Path('data/a2d2/new_label/gt_features') # CSV 저장 위치

# --- ▼▼▼▼▼ 컬럼 정의 (요청하신 21개) ▼▼▼▼▼ ---

# .txt 파일에서 읽어올 21개 컬럼 이름
# --- ▼▼▼▼▼ [수정된 부분] 컬럼 정의 (19개) ▼▼▼▼▼ ---

# .txt 파일에서 읽어올 19개 컬럼 이름
# (stage1_gt_train_features.csv 기준)
COLUMN_NAMES = [
    'label', 'RPN_MaxScore', 'x', 'y', 'z', 'l', 'w', 'h', 'yaw',
    'num_points', 'length', 'width', 'height', 'density',
    'aspect_ratio', 'mean_z', 'std_z', 'intensity_mean', 'intensity_std'
]

LABEL_COLUMN = 'label'

# --- ▲▲▲▲▲ 컬럼 정의 끝 ▲▲▲▲▲ ---
LABEL_COLUMN = 'label'

# --- ▲▲▲▲▲ 컬럼 정의 끝 ▲▲▲▲▲ ---


def load_gt_features_from_directory(directory_path: Path) -> pd.DataFrame:
    """
    지정된 디렉터리에서 모든 GT .txt 파일을 읽어
    하나의 Pandas DataFrame으로 결합합니다.
    """
    all_data_frames = []
    gt_files = list(directory_path.glob('*.txt'))
    
    if not gt_files:
        raise FileNotFoundError(f"'{directory_path}'에서 .txt 파일을 찾을 수 없습니다.")
        
    print(f"'{directory_path}'에서 총 {len(gt_files)}개의 파일 로드 중...")
    
    for file_path in gt_files:
        try:
            # on_bad_lines='skip' : 컬럼 수가 21개가 아닌 줄은 무시
            df = pd.read_csv(
                file_path, sep=' ', header=None, 
                names=COLUMN_NAMES, on_bad_lines='skip' 
            )
            if not df.empty:
                all_data_frames.append(df)
        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"[경고] {file_path} 로드 오류: {e}")
            
    if not all_data_frames:
        raise ValueError(f"'{directory_path}'에서 로드된 데이터가 없습니다.")
        
    full_dataset = pd.concat(all_data_frames, ignore_index=True)
    return full_dataset

def main():
    
    # --- 1. 훈련(Train) 데이터 로드 ---
    try:
        train_dataset_full = load_gt_features_from_directory(GT_TRAIN_DIR)
        print(f"✅ 훈련 데이터 로드 완료 (필터링 전): 총 {len(train_dataset_full)}개 샘플")
    except Exception as e:
        print(f"훈련 데이터 로드 실패: {e}"); return

    # --- 2. 테스트(Test) 데이터 로드 ---
    try:
        test_dataset_full = load_gt_features_from_directory(GT_TEST_DIR)
        print(f"✅ 테스트 데이터 로드 완료 (필터링 전): 총 {len(test_dataset_full)}개 샘플")
    except Exception as e:
        print(f"테스트 데이터 로드 실패: {e}"); return
    
    # --- ▼▼▼▼▼ 거리 필터링 추가 ▼▼▼▼▼ ---
    distance_threshold = 30.0
    print(f"\n거리 <= {distance_threshold}m 기준으로 데이터 필터링 중...")

    # 훈련 데이터 필터링
    train_distances = np.sqrt(train_dataset_full['x']**2 + train_dataset_full['y']**2)
    train_dataset = train_dataset_full[train_distances <= distance_threshold].copy()
    print(f"  - 훈련 데이터: {len(train_dataset_full)} -> {len(train_dataset)} 샘플")
    print("  --- 필터링 후 훈련셋 클래스 분포 ---")
    print(train_dataset[LABEL_COLUMN].value_counts())
    print("-" * 30)

    # 테스트 데이터 필터링
    test_distances = np.sqrt(test_dataset_full['x']**2 + test_dataset_full['y']**2)
    test_dataset = test_dataset_full[test_distances <= distance_threshold].copy()
    print(f"  - 테스트 데이터: {len(test_dataset_full)} -> {len(test_dataset)} 샘플")
    print("  --- 필터링 후 테스트셋 클래스 분포 ---")
    print(test_dataset[LABEL_COLUMN].value_counts())
    print("-" * 30)

    if train_dataset.empty or test_dataset.empty:
        print("[오류] 필터링 후 남은 데이터가 없습니다. 거리 임계값을 확인하세요.")
        return
    # --- ▲▲▲▲▲ 거리 필터링 추가 끝 ▲▲▲▲▲ ---

    # --- 3. 피처를 CSV로 저장 ---
    if SAVE_FEATURES_TO_CSV:
        print("피처를 CSV 파일로 저장합니다...")
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        train_csv_path = CSV_OUTPUT_DIR / 'gt_train_features.csv'
        test_csv_path = CSV_OUTPUT_DIR / 'gt_test_features.csv'
        
        # 21개 컬럼 모두 CSV로 저장
        train_dataset.to_csv(train_csv_path, index=False)
        test_dataset.to_csv(test_csv_path, index=False)
        print(f"  - 훈련 피처 저장: {train_csv_path}")
        print(f"  - 테스트 피처 저장: {test_csv_path}")
        print("-" * 30)
    
    print("\n✅ CSV 파일 생성이 완료되었습니다.")
    print("생성된 'gt_train_features.csv'와 'gt_test_features.csv' 파일을 업로드해주세요.")


if __name__ == '__main__':
    main()