import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# --- 1. 파일 경로 설정 ---
MODEL_1_PATH = 'data/a2d2/rf_stage1_model.pkl'
MODEL_2_PATH = 'data/a2d2/rf_stage2_model.pkl'
TEST_FILE_PATH = 'data/a2d2/final_cascade_test_set.csv'
TARGET_COLUMN = 'label'
STAGE_1_OBJECT_LABEL = 1 


try:
    # --- 2. 모델 로드 ---
    print(f"1단계 모델 로드 중: {MODEL_1_PATH}")
    model_1 = joblib.load(MODEL_1_PATH)
    
    print(f"2단계 모델 로드 중: {MODEL_2_PATH}")
    model_2 = joblib.load(MODEL_2_PATH)
    print("모델 로드 완료.")

    # --- 3. 테스트 데이터 로드 ---
    print(f"테스트 데이터 로드 중: {TEST_FILE_PATH}")
    df_test = pd.read_csv(TEST_FILE_PATH).fillna(0)
    
    y_true = df_test[TARGET_COLUMN]
    X_test = df_test.drop(columns=[TARGET_COLUMN])
    print(f" -> 총 {len(X_test)}개의 테스트 샘플 발견.")

    # --- 4. 피처 목록 ---
    features_s1 = model_1.feature_names_in_
    features_s2 = model_2.feature_names_in_

    # --- 5. 2단계 LabelEncoder 복원 ---
    print("2단계 모델(LabelEncoder) 복원 중...")
    le_stage2 = LabelEncoder()
    s2_labels = np.unique(y_true[y_true != 'Background'])
    le_stage2.fit(s2_labels)
    print(f" -> 2단계 클래스: {le_stage2.classes_}")

    # --- 6. 캐스케이드 예측 ---
    final_predictions = []
    stage1_preds = []  # Stage 1 결과도 저장
    print("\n캐스케이드 평가 시작...")

    for i in tqdm(range(len(X_test)), desc="평가 진행 중"):
        sample = X_test.iloc[i]
        sample_s1 = sample[features_s1]

        # Stage 1 예측
        pred_1 = model_1.predict([sample_s1])[0]
        stage1_preds.append(pred_1)

        if pred_1 == STAGE_1_OBJECT_LABEL:
            # Stage 2 예측
            sample_s2 = sample[features_s2]
            pred_2_numeric = model_2.predict([sample_s2])[0]
            try:
                pred_2_string = le_stage2.inverse_transform([pred_2_numeric])[0]
                final_predictions.append(pred_2_string)
            except ValueError:
                final_predictions.append('Background')
        else:
            final_predictions.append('Background')

    # --- 7. 전체 평가 ---
    print("\n... 전체 평가 완료.")
    print("="*50)
    print("--- CASCADE 시스템 최종 평가 결과 ---")
    accuracy = accuracy_score(y_true, final_predictions)
    report = classification_report(y_true, final_predictions, digits=4, zero_division=0)
    print(f"전체 정확도 (Accuracy): {accuracy * 100:.4f}%")
    print("\n[ 최종 분류 리포트 ]")
    print(report)
    print("="*50)

    # --- 8. ✅ Stage 1에서 Object로 예측된 샘플만 따로 평가 ---
    print("\n--- [추가] Stage 1에서 Object로 분류된 샘플만 평가 ---")
    object_indices = [i for i, p in enumerate(stage1_preds) if p == STAGE_1_OBJECT_LABEL]

    if len(object_indices) == 0:
        print("⚠️ Stage 1에서 Object로 분류된 샘플이 없습니다.")
    else:
        y_true_obj = y_true.iloc[object_indices].reset_index(drop=True)
        y_pred_obj = pd.Series(final_predictions).iloc[object_indices].reset_index(drop=True)

        # ✅ Background 제외 (Stage 2 전용 평가)
        mask = y_true_obj != "Background"
        y_true_obj = y_true_obj[mask].reset_index(drop=True)
        y_pred_obj = y_pred_obj[mask].reset_index(drop=True)

        acc_obj = accuracy_score(y_true_obj, y_pred_obj)
        report_obj = classification_report(y_true_obj, y_pred_obj, digits=4, zero_division=0)

        print(f"Object 샘플 개수: {len(y_true_obj)}")
        print(f"Object 전용 정확도 (Stage 2 성능): {acc_obj * 100:.4f}%")
        print("\n[ Stage 2 (Object 전용) Classification Report ]")
        print(report_obj)
        print("="*50)

except FileNotFoundError as e:
    print(f"[오류] 파일 찾기 실패: {e.filename}")
except Exception as e:
    print(f"[오류] 평가 중 문제 발생: {e}")

# import pandas as pd
# import joblib
# from sklearn.metrics import classification_report, accuracy_score
# import numpy as np
# from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder
# import time
# import torch  # ⚠️ GPU 사용 시 정확한 시간 측정용 (CPU만 사용 시 없어도 됨)

# # --- 1. 파일 경로 설정 ---
# MODEL_1_PATH = 'data/a2d2/rf_stage1_model.pkl'
# MODEL_2_PATH = 'data/a2d2/rf_stage2_model.pkl'
# TEST_FILE_PATH = 'data/a2d2/final_cascade_test_set.csv' 
# TARGET_COLUMN = 'label'
# STAGE_1_OBJECT_LABEL = 1 
# VIS_SAVE_PATH = 'data/a2d2/visualization_results.csv'

# try:
#     # --- 2. 모델 로드 ---
#     print(f"1단계 모델 로드 중: {MODEL_1_PATH}")
#     model_1 = joblib.load(MODEL_1_PATH)
#     print(f"2단계 모델 로드 중: {MODEL_2_PATH}")
#     model_2 = joblib.load(MODEL_2_PATH)
#     print("모델 로드 완료.")

#     # --- 3. 테스트 데이터 로드 ---
#     print(f"테스트 데이터 로드 중: {TEST_FILE_PATH}")
#     df_test = pd.read_csv(TEST_FILE_PATH).fillna(0)
    
#     y_true = df_test[TARGET_COLUMN]
#     X_test = df_test.drop(columns=[TARGET_COLUMN])
#     print(f" -> 총 {len(X_test)}개의 테스트 샘플 발견.")

#     # --- 4. 피처 목록 ---
#     features_s1 = model_1.feature_names_in_
#     features_s2 = model_2.feature_names_in_

#     # --- 5. 2단계 LabelEncoder 복원 ---
#     print("2단계 모델(LabelEncoder) 복원 중...")
#     le_stage2 = LabelEncoder()
#     s2_labels = np.unique(y_true[y_true != 'Background'])
#     le_stage2.fit(s2_labels)
#     print(f" -> 2단계 클래스: {le_stage2.classes_}")

#     # --- 6. 캐스케이드 예측 + Inference Time 측정 ---
#     final_predictions = []
#     stage1_preds = []
#     stage1_times, stage2_times = [], []  # ✅ 시간 기록용 리스트

#     print("\n캐스케이드 평가 시작...")

#     for i in tqdm(range(len(X_test)), desc="평가 진행 중"):
#         sample = X_test.iloc[i]
#         sample_s1 = sample[features_s1]

#         # --- Stage 1 Inference ---
#         start_s1 = time.time()
#         pred_1 = model_1.predict([sample_s1])[0]
#         end_s1 = time.time()
#         stage1_times.append(end_s1 - start_s1)

#         stage1_preds.append(pred_1)

#         # --- Stage 2 Inference (if Object) ---
#         if pred_1 == STAGE_1_OBJECT_LABEL:
#             sample_s2 = sample[features_s2]

#             start_s2 = time.time()
#             pred_2_numeric = model_2.predict([sample_s2])[0]
#             end_s2 = time.time()
#             stage2_times.append(end_s2 - start_s2)

#             try:
#                 pred_2_string = le_stage2.inverse_transform([pred_2_numeric])[0]
#                 final_predictions.append(pred_2_string)
#             except ValueError:
#                 final_predictions.append('Background')
#         else:
#             final_predictions.append('Background')

#     # --- 7. 추론 시간 통계 출력 ---
#     avg_s1_ms = np.mean(stage1_times) * 1000
#     avg_s2_ms = np.mean(stage2_times) * 1000 if stage2_times else 0
#     avg_total_ms = (np.mean(stage1_times) + (np.mean(stage2_times) if stage2_times else 0)) * 1000

#     print("\n--- [추론 시간 통계] ---")
#     print(f"Stage 1 평균 Inference Time : {avg_s1_ms:.3f} ms")
#     print(f"Stage 2 평균 Inference Time : {avg_s2_ms:.3f} ms")
#     print(f"총 평균 Inference Time (Stage1+2) : {avg_total_ms:.3f} ms")
#     print(f"FPS (frames per second) ≈ {1000 / avg_total_ms:.2f} FPS")
#     print("="*50)

#     # --- 8. 전체 평가 ---
#     print("--- CASCADE 시스템 최종 평가 결과 ---")
#     accuracy = accuracy_score(y_true, final_predictions)
#     report = classification_report(y_true, final_predictions, digits=4, zero_division=0)
#     print(f"전체 정확도 (Accuracy): {accuracy * 100:.4f}%")
#     print("\n[ 최종 분류 리포트 ]")
#     print(report)
#     print("="*50)

#     # --- 9. Stage 1에서 Object로 예측된 샘플만 따로 평가 ---
#     print("\n--- [추가] Stage 1에서 Object로 분류된 샘플만 평가 ---")
#     object_indices = [i for i, p in enumerate(stage1_preds) if p == STAGE_1_OBJECT_LABEL]

#     if len(object_indices) == 0:
#         print("⚠️ Stage 1에서 Object로 분류된 샘플이 없습니다.")
#     else:
#         y_true_obj = y_true.iloc[object_indices].reset_index(drop=True)
#         y_pred_obj = pd.Series(final_predictions).iloc[object_indices].reset_index(drop=True)

#         mask = y_true_obj != "Background"
#         y_true_obj = y_true_obj[mask].reset_index(drop=True)
#         y_pred_obj = y_pred_obj[mask].reset_index(drop=True)

#         acc_obj = accuracy_score(y_true_obj, y_pred_obj)
#         report_obj = classification_report(y_true_obj, y_pred_obj, digits=4, zero_division=0)

#         print(f"Object 샘플 개수: {len(y_true_obj)}")
#         print(f"Object 전용 정확도 (Stage 2 성능): {acc_obj * 100:.4f}%")
#         print("\n[ Stage 2 (Object 전용) Classification Report ]")
#         print(report_obj)
#         print("="*50)

# except FileNotFoundError as e:
#     print(f"[오류] 파일 찾기 실패: {e.filename}")
# except Exception as e:
#     print(f"[오류] 평가 중 문제 발생: {e}")
