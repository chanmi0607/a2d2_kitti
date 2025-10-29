import pandas as pd
from pathlib import Path

# --- [!] 1. 검증용 파일 경로 2개 ---
# (이름은 님의 실제 파일명으로 수정하세요)

# 2단계 훈련 때 쓴 검증용 파일 ('Car', 'Cyclist' 등이 있는 파일)
FOREGROUND_VAL_FILE = 'data/a2d2/new_label/gt_features/gt_test_features.csv'

# 1단계 훈련 때 쓴 검증용 파일 ('Background'만 있는 파일)
# (주의: 1단계 훈련 파일(gt_test_features.csv)에 'Object'가 섞여있었다면,
#  그 파일에서 'Background'만 따로 추출해야 합니다.)
#  가장 쉬운 방법은 pcdet 스크립트로 생성한 background_features_val.csv를 쓰는 것입니다.
BACKGROUND_VAL_FILE = 'data/a2d2/stage1_background_val_data.csv' 

# --- 2. 최종 저장될 파일 이름 ---
FINAL_TEST_FILE = 'data/a2d2/final_cascade_test_set.csv'

try:
    print(f"포그라운드 파일 로드 중: {FOREGROUND_VAL_FILE}")
    df_fg = pd.read_csv(FOREGROUND_VAL_FILE)
    print(f" -> {len(df_fg)}개 샘플 로드됨 (예: Car, Cyclist...)")
    
    print(f"백그라운드 파일 로드 중: {BACKGROUND_VAL_FILE}")
    df_bg = pd.read_csv(BACKGROUND_VAL_FILE)
    # (pcdet 스크립트 결과물은 이미 'Background' 라벨을 갖고 있음)
    print(f" -> {len(df_bg)}개 샘플 로드됨 (Background)")

    # --- 3. 두 데이터프레임 합치기 ---
    # (컬럼 순서나 개수가 달라도, 공통된 컬럼 기준으로 합쳐짐)
    final_df = pd.concat([df_fg, df_bg], ignore_index=True)

    # --- 4. 최종 정답지 저장 ---
    final_df.to_csv(FINAL_TEST_FILE, index=False)
    
    print("\n" + "="*50)
    print(f"✅ '종합 정답지' 생성 완료: {FINAL_TEST_FILE}")
    print(f"  - 총 샘플 수: {len(final_df)}개")
    print("  - 최종 라벨 분포:")
    print(final_df['label'].value_counts())
    print("="*50)

except FileNotFoundError as e:
    print(f"[오류] 파일을 찾을 수 없습니다: {e.filename}")
    print("1, 2번 파일 경로를 정확히 확인해주세요.")
except Exception as e:
    print(f"[오류] 병합 중 문제 발생: {e}")

# import pandas as pd

# # 두 CSV 로드
# df_obj = pd.read_csv("data/a2d2/stage1_gt_val_features.csv")
# df_bg = pd.read_csv("data/a2d2/stage1_background_val_data.csv")

# # 라벨 통일
# df_obj["label"] = "Object"
# df_bg["label"] = "Background"

# # ✅ background 컬럼 순서를 기준으로 정렬
# bg_cols = df_bg.columns.tolist()
# common_cols = [col for col in bg_cols if col in df_obj.columns]

# # df_obj를 background 컬럼 순서에 맞게 정렬
# df_obj = df_obj[common_cols]
# df_bg = df_bg[common_cols]

# # 합치기
# df_combined = pd.concat([df_obj, df_bg], ignore_index=True)

# # 저장
# output_path = "data/a2d2/val_features_combined.csv"
# df_combined.to_csv(output_path, index=False)

# print(f"✅ Combined CSV saved! Total samples: {len(df_combined)}")
# print(f"📊 Column order preserved from background: {list(df_combined.columns)}")
# print(df_combined['label'].value_counts())


# import pandas as pd
# from pathlib import Path

# # --- 1. 설정: 사용할 파일 경로 ---

# # 원본 피처 파일 (방금 업로드해주신 파일)
# FEATURES_FILE = 'data/a2d2/background_features_train.csv'

# # 프레임 ID 목록 파일
# TRAIN_SPLIT_FILE = 'data/a2d2/ImageSets/train.txt'
# VAL_SPLIT_FILE = 'data/a2d2/ImageSets/val.txt'

# # 최종 저장될 파일 이름
# OUTPUT_TRAIN_CSV = 'data/a2d2/background_train_data.csv'
# OUTPUT_VAL_CSV = 'data/a2d2/background_val_data.csv'

# # 제거할 컬럼 목록
# COLUMNS_TO_DROP = ['max_iou', 'frame_id']

# # --- 2. 헬퍼 함수: .txt 파일 읽기 ---
# def read_split_file(file_path: Path) -> set:
#     """ .txt 파일에서 프레임 ID 목록을 읽어 set으로 반환합니다. """
#     if not file_path.exists():
#         raise FileNotFoundError(f"필수 파일 없음: {file_path}")
    
#     # .read_text()로 읽고, .splitlines()로 줄바꿈 기준 리스트 생성
#     frame_ids = set(file_path.read_text().splitlines())
#     print(f"✅ '{file_path}'에서 {len(frame_ids)}개의 고유 프레임 ID 로드됨.")
#     return frame_ids

# # --- 3. 메인 로직 ---
# def main():
#     try:
#         # 1. 분할 기준이 될 train/val 프레임 ID 로드
#         train_frame_ids = read_split_file(Path(TRAIN_SPLIT_FILE))
#         val_frame_ids = read_split_file(Path(VAL_SPLIT_FILE))

#         print(f"\n원본 CSV 파일 로드 중: '{FEATURES_FILE}'...")
        
#         # 2. 원본 피처 CSV 로드
#         # [중요] 'frame_id'를 문자로 로드(dtype=str)해야 
#         # '000123' 같은 ID가 123으로 변환되지 않고 정확히 매칭됨
#         df = pd.read_csv(FEATURES_FILE, dtype={'frame_id': str})
#         print(f"✅ 원본 CSV 로드 완료. (총 {len(df)}개 샘플)")

#         # 3. 'frame_id'를 기준으로 훈련(train) 데이터 필터링
#         print(f"\n'train.txt' 목록 기준으로 필터링 중...")
#         train_df = df[df['frame_id'].isin(train_frame_ids)].copy()
#         print(f"  -> 훈련용 샘플 {len(train_df)}개 추출됨.")

#         # 4. 'frame_id'를 기준으로 검증(val) 데이터 필터링
#         print(f"\n'val.txt' 목록 기준으로 필터링 중...")
#         val_df = df[df['frame_id'].isin(val_frame_ids)].copy()
#         print(f"  -> 검증용 샘플 {len(val_df)}개 추출됨.")

#         # 5. 불필요한 컬럼 제거
#         print(f"\n제거할 컬럼: {COLUMNS_TO_DROP}")
        
#         # errors='ignore' : 혹시 컬럼이 없더라도 오류 없이 통과
#         final_train_df = train_df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
#         final_val_df = val_df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

#         # 6. 최종 CSV 파일로 저장
#         final_train_df.to_csv(OUTPUT_TRAIN_CSV, index=False)
#         print(f"\n✅ 훈련용 피처 저장 완료: {OUTPUT_TRAIN_CSV} (형태: {final_train_df.shape})")
        
#         final_val_df.to_csv(OUTPUT_VAL_CSV, index=False)
#         print(f"✅ 검증용 피처 저장 완료: {OUTPUT_VAL_CSV} (형태: {final_val_df.shape})")
        
#         print("\n--- 남은 컬럼 목록 (예시) ---")
#         print(final_train_df.columns.tolist())

#     except FileNotFoundError as e:
#         print(f"\n[오류] 파일 찾기 실패: {e}")
#         print("스크립트와 같은 위치에 3개의 파일이 모두 있는지 확인하세요.")
#     except Exception as e:
#         print(f"\n[오류] 예기치 못한 문제 발생: {e}")

# if __name__ == "__main__":
#     main()

# import pandas as pd

# # 파일 경로
# obj_path = "data/a2d2/gt_train_features.csv"
# bg_path = "data/a2d2/background_features_train.csv"

# # 로드
# df_obj = pd.read_csv(obj_path)
# df_bg = pd.read_csv(bg_path)

# # ✅ GT(Object) 컬럼명 → Background 기준으로 통일
# rename_map = {
#     "actual_width": "length",
#     "actual_length": "width",
#     "actual_height": "height",
#     "aspect_ratio_2d": "aspect_ratio"
# }
# df_obj.rename(columns=rename_map, inplace=True)

# # ✅ Background에 있는 컬럼 순서 기준으로 재정렬
# common_cols = [col for col in df_bg.columns if col in df_obj.columns]
# df_obj_aligned = df_obj[common_cols]

# # 누락된 컬럼 채우기 (없으면 NaN → 0)
# for col in df_bg.columns:
#     if col not in df_obj_aligned.columns:
#         df_obj_aligned[col] = 0

# # 최종 순서 bg 기준으로 정렬
# df_obj_aligned = df_obj_aligned[df_bg.columns]

# # 저장
# df_obj_aligned.to_csv("data/a2d2/gt_train_features_aligned.csv", index=False)
# print("✅ gt_train_features_aligned.csv 생성 완료!")
# print(f"총 {len(df_obj_aligned)}행, {len(df_obj_aligned.columns)}열")

# import pandas as pd

# # 파일 경로
# bg_path = "data/a2d2/background_features_train.csv"
# gt_path = "data/a2d2/gt_train_features_aligned.csv"  # 혹은 gt_train_features_aligned.csv

# # CSV 로드
# df_bg = pd.read_csv(bg_path)
# df_gt = pd.read_csv(gt_path)

# # 삭제할 컬럼
# drop_cols = ["max_iou", "frame_id"]

# # 존재하면 삭제
# for df in [df_bg, df_gt]:
#     for col in drop_cols:
#         if col in df.columns:
#             df.drop(columns=col, inplace=True)

# # 저장
# df_bg.to_csv("data/a2d2/background_features_train_clean.csv", index=False)
# df_gt.to_csv("data/a2d2/gt_train_features_clean.csv", index=False)

# print("✅ 완료! 두 파일에서 'max_iou'와 'frame_id' 컬럼 제거됨.")
# print(f"Background shape: {df_bg.shape}")
# print(f"GT(Object) shape: {df_gt.shape}")
