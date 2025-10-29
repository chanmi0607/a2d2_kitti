import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib  # <-- [추가] 모델 저장을 위해 임포트

# --- 1. 파일 이름 설정 ---
#TRAIN_FILE = 'data/a2d2/train_features_combined.csv'
#VAL_FILE = 'data/a2d2/val_features_combined.csv'

TRAIN_FILE = 'data/a2d2/new_label/gt_features/gt_train_features.csv'
VAL_FILE = 'data/a2d2/new_label/gt_features/gt_test_features.csv'

# 분류할 목표(Target) 컬럼 이름
TARGET_COLUMN = 'label'

# --- ▼▼▼ [수정된 부분] ▼▼▼ ---
# 훈련에서 제외할 컬럼 목록 (정답 자체 + 누수 피처)
COLUMNS_TO_DROP = [TARGET_COLUMN, 'RPN_MaxScore', 'yaw']
# --- ▲▲▲ [수정된 부분] ▲▲▲ ---

try:
    # --- 2. 데이터 로드 ---
    print(f"훈련 데이터 로드 중: {TRAIN_FILE}")
    train_df = pd.read_csv(TRAIN_FILE)
    
    print(f"검증 데이터 로드 중: {VAL_FILE}")
    val_df = pd.read_csv(VAL_FILE)

    print("\n--- 훈련 데이터 정보 ---")
    train_df.info()
    print("\n--- 훈련 데이터 샘플 (head) ---")
    print(train_df.head())

    # --- 3. X (특징), y (라벨) 분리 ---
    
    # 'label' 컬럼이 있는지 확인
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in val_df.columns:
        print(f"\n[오류] '{TARGET_COLUMN}' 컬럼을 찾을 수 없습니다.")
        print("TARGET_COLUMN 변수 값을 실제 라벨 컬럼 이름으로 변경해주세요.")
    else:
        print(f"\n목표 컬럼(y)을 '{TARGET_COLUMN}'로 설정합니다.")
        print(f"데이터 누수 방지를 위해 다음 컬럼을 훈련에서 제외합니다: {COLUMNS_TO_DROP}")
        
        # 훈련 데이터 분리
        # --- ▼▼▼ [수정된 부분] ▼▼▼ ---
        X_train = train_df.drop(columns=COLUMNS_TO_DROP)
        y_train_raw = train_df[TARGET_COLUMN]
        
        # 검증 데이터 분리
        X_val = val_df.drop(columns=COLUMNS_TO_DROP)
        # --- ▲▲▲ [수정된 부분] ▲▲▲ ---
        y_val_raw = val_df[TARGET_COLUMN]
        
        # (중요) 훈련 데이터와 검증 데이터의 컬럼 순서/개수가 일치하는지 확인
        if list(X_train.columns) != list(X_val.columns):
            print("[경고] 훈련 데이터와 검증 데이터의 컬럼 구성이 다릅니다.")
            # 컬럼 순서를 훈련 데이터 기준으로 강제 일치
            X_val = X_val[X_train.columns]

        # --- 4. 결측치 처리 (fillna) ---
        # 훈련 스크립트에서 fillna(0)을 사용했으므로 동일하게 적용
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        # --- 5. 라벨 인코딩 ---
        # (예: 'Background', 'Car' -> 0, 1)
        le = LabelEncoder()
        
        # [중요] 훈련(train) 데이터 기준으로 인코더를 '학습(fit)'시킵니다.
        y_train = le.fit_transform(y_train_raw)
        
        # [중요] 검증(val) 데이터는 학습된 인코더로 '변환(transform)'만 합니다.
        y_val = le.transform(y_val_raw)
        
        print("\n--- 라벨 인코딩 정보 (클래스) ---")
        print(f"클래스 목록: {le.classes_}")
        print(f"훈련 라벨 분포: {pd.Series(y_train_raw).value_counts().to_dict()}")
        print(f"검증 라벨 분포: {pd.Series(y_val_raw).value_counts().to_dict()}")

        # --- 6. Random Forest 모델 훈련 ---
        print("\nRandom Forest 모델 훈련 시작...")
        print(f"(훈련 샘플 {len(X_train)}개, 검증 샘플 {len(X_val)}개)")
        print(f"사용된 피처 개수: {len(X_train.columns)}개")

        rf_classifier = RandomForestClassifier(
            n_estimators=100,  # 100개의 트리 사용
            random_state=42,   # 결과 재현을 위한 시드 고정
            n_jobs=-1          # 모든 CPU 코어 사용
        )
        
        # 훈련 데이터로 모델을 학습시킵니다.
        rf_classifier.fit(X_train, y_train)
        print("모델 훈련 완료.")

        # --- 7. 모델 성능 평가 (검증 데이터 사용) ---
        print(f"\n--- 검증 세트 ({VAL_FILE}) 성능 평가 ---")
        
        # 학습된 모델로 검증 데이터의 라벨을 예측합니다.
        y_pred = rf_classifier.predict(X_val)
        
        # 실제 라벨(y_val)과 예측 라벨(y_pred)을 비교합니다.
        accuracy = accuracy_score(y_val, y_pred)
        #report = classification_report(y_val, y_pred, target_names=le.classes_)
        report = classification_report(y_val, y_pred, target_names=le.classes_, digits=4)

        print(f"전체 정확도 (Accuracy): {accuracy * 100:.4f}%")
        print("\n[ Classification Report ]")
        print(report)

        # --- 8. 피처 중요도 확인 ---
        print("\n--- 상위 10개 피처 중요도 ---")
        importances = pd.Series(
            rf_classifier.feature_importances_, 
            index=X_train.columns
        )
        print(importances.sort_values(ascending=False).head(10))

        # --- 9. (추가) 학습된 모델을 .pkl 파일로 저장 ---
        MODEL_SAVE_PATH = 'data/a2d2/rf_stage2_model.pkl' # 1단계 모델 저장 경로
        ENCODER_SAVE_PATH = 'data/a2d2/rf_stage2_encoder.pkl'
        print(f"\n--- 모델 저장 ---")
        try:
            joblib.dump(rf_classifier, MODEL_SAVE_PATH)
            print(f"✅ 훈련된 모델을 {MODEL_SAVE_PATH} 에 성공적으로 저장했습니다.")
            joblib.dump(le, ENCODER_SAVE_PATH) 
            print(f"✅ 모델에 사용된 '번역기'를 {ENCODER_SAVE_PATH} 에 성공적으로 저장했습니다.")
        except Exception as e:
            print(f"[오류] 모델 저장 중 문제가 발생했습니다: {e}")

except FileNotFoundError as e:
    print(f"[오류] 파일을 찾을 수 없습니다: {e.filename}")
    print(f"'{TRAIN_FILE}'와 '{VAL_FILE}' 파일이 코드와 같은 위치에 있는지 확인하세요.")
except KeyError as e:
    print(f"[오류] '{e}' 컬럼을 찾을 수 없습니다.")
    print(f"'{VAL_FILE}' 파일에 '{y_train_raw.unique()}' 클래스에 없는 새로운 라벨이 포함되어 있을 수 있습니다.")
except Exception as e:
    print(f"[오류] 예기치 못한 문제 발생: {e}")