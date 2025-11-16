import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# --- 설정 ---
# 1. 원본 데이터 파일 경로
TRAINING_DATA_PATH = 'data/a2d2/rf_dataset_all.csv'

# 2. 학습/검증용 프레임 ID 목록 파일 경로
TRAIN_FILE_PATH = 'data/a2d2/ImageSets/train.txt'  # 👈 train.txt 파일이 있는 경로로 수정!
VAL_FILE_PATH = 'data/a2d2/ImageSets/val.txt'      # 👈 val.txt 파일이 있는 경로로 수정!

# 3. CSV 파일에서 프레임 ID를 식별하는 컬럼 이름
FRAME_ID_COLUMN = 'frame_id'

# 4. 학습된 모델을 저장할 경로
MODEL_OUTPUT_PATH = 'random_forest_model_from_csv.joblib'


# --- [추가] 헬퍼 함수: txt 파일에서 프레임 ID 로드 ---
def load_frame_ids(file_path):
    """
    .txt 파일에서 프레임 ID 목록을 읽어와 set으로 반환합니다.
    프레임 ID는 000057177과 같은 형식을 유지하기 위해 문자열로 처리합니다.
    """
    try:
        with open(file_path, 'r') as f:
            # 양쪽 공백을 제거하고, 비어있지 않은 라인만 set에 추가
            frame_ids = {line.strip() for line in f if line.strip()}
        return frame_ids
    except FileNotFoundError:
        print(f"❗️ Error: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit()
# --- [추가] ---


# --- 1. 데이터 로드 및 전처리 ---
print(f"'{TRAINING_DATA_PATH}' 파일에서 전체 데이터를 로드합니다...")
try:
    # [수정] FRAME_ID_COLUMN을 문자열(str) 타입으로 읽어오도록 강제합니다.
    # 이렇게 하면 '000057177' 같은 ID가 숫자로 변환되지 않습니다.
    df = pd.read_csv(TRAINING_DATA_PATH, dtype={FRAME_ID_COLUMN: str})
except FileNotFoundError:
    print(f"❗️ Error: '{TRAINING_DATA_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()
except ValueError as e:
    print(f"❗️ Error: '{FRAME_ID_COLUMN}' 컬럼을 찾을 수 없거나 dtype이 맞지 않습니다. ({e})")
    print(f"'{FRAME_ID_COLUMN}' 변수가 CSV의 실제 프레임 ID 컬럼명과 일치하는지 확인하세요.")
    exit()

# 누락된 값이 있는 행이 있다면 제거합니다.
df.dropna(inplace=True)

# [수정] 'frame_id' 컬럼이 있는지 확인
if FRAME_ID_COLUMN not in df.columns:
    print(f"❗️ Error: CSV 파일에 '{FRAME_ID_COLUMN}' 컬럼이 없습니다.")
    print(f"'{FRAME_ID_COLUMN}' 변수를 실제 프레임 ID가 포함된 컬럼 이름으로 수정해주세요.")
    exit()

# [수정] 라벨 인코더는 *전체* 데이터셋(df)을 기준으로 fit합니다.
# (그래야 train/test에 특정 클래스가 없더라도 모든 클래스를 인식할 수 있습니다)
label_encoder = LabelEncoder()
label_encoder.fit(df['class'])

class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("✅ 전체 데이터 로드 및 라벨 인코딩 완료.")
print("\n클래스 매핑 정보 (숫자 -> 이름):")
print(class_mapping)


# --- 2. 데이터셋 분리 (train_test_split 대체) ---
print(f"\n'{TRAIN_FILE_PATH}'와 '{VAL_FILE_PATH}'를 기준으로 데이터를 분리합니다...")
train_ids = load_frame_ids(TRAIN_FILE_PATH)
val_ids = load_frame_ids(VAL_FILE_PATH)
print(f"학습용 ID {len(train_ids)}개, 테스트용 ID {len(val_ids)}개를 로드했습니다.")

# [수정] train.txt와 val.txt의 ID를 기준으로 DataFrame 필터링
train_df = df[df[FRAME_ID_COLUMN].isin(train_ids)]
test_df = df[df[FRAME_ID_COLUMN].isin(val_ids)]

if len(train_df) == 0:
    print(f"❗️ Error: '{TRAIN_FILE_PATH}'의 ID와 일치하는 데이터가 CSV에 없습니다.")
    exit()
if len(test_df) == 0:
    print(f"❗️ Error: '{VAL_FILE_PATH}'의 ID와 일치하는 데이터가 CSV에 없습니다.")
    exit()

# [수정] X, y 분리
# 'class'와 'frame_id' 컬럼을 제외한 모든 열을 특징으로 사용합니다.
feature_columns = [col for col in df.columns if col not in ['class', FRAME_ID_COLUMN]]

X_train = train_df[feature_columns]
y_train_str = train_df['class']

X_test = test_df[feature_columns]
y_test_str = test_df['class']

# 위에서 fit한 라벨 인코더를 사용하여 문자열 라벨을 숫자로 변환
y_train = label_encoder.transform(y_train_str)
y_test = label_encoder.transform(y_test_str)

print(f"\n총 {len(train_df)}개의 데이터를 학습에, {len(test_df)}개의 데이터를 테스트에 사용합니다.")
print("\n학습 데이터 클래스 분포:")
print(sorted(Counter(y_train).items()))
print("\n테스트 데이터 클래스 분포:")
print(sorted(Counter(y_test).items()))


# --- 3. 랜덤 포레스트 모델 학습 ---
print("\nRandomForest 모델 학습을 시작합니다...")
# n_jobs=-1: 컴퓨터의 모든 CPU 코어를 사용하여 학습 속도를 높입니다.
# verbose=1: 학습 과정을 간략하게 출력합니다.
model = RandomForestClassifier(
    n_estimators=100, # 100개의 의사결정 나무를 사용
    random_state=42, # 난수를 고정해 결과를 재현 가능하게 만듦
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)
print("✅ 모델 학습 완료.")


# --- 4. 모델 성능 평가 ---
print(f"\n테스트 데이터({VAL_FILE_PATH} 기준)로 모델 성능을 평가합니다...")
y_pred = model.predict(X_test)

# --- ▼▼▼▼▼ 수정 시작 ▼▼▼▼▼ ---
# 성능 평가 리포트를 딕셔너리로 받음
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

print("\n" + "="*65) # 구분선 길이 조정
print("📊 모델 성능 평가 리포트 (퍼센트 형식)")
# 헤더 출력 (정렬 맞춤)
print(f"{'Class':<16} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>7}")
print("-" * 65) # 구분선 길이 조정

# 각 클래스별 결과 포맷팅 및 출력
for class_name, metrics in report_dict.items():
    # 클래스 이름인 경우에만 처리 (accuracy, macro avg 등 제외)
    if class_name in label_encoder.classes_:
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        support = metrics['support']

        # 퍼센트 형식으로 포맷팅 (소수점 둘째 자리까지)
        precision_str = f"{precision * 100:.2f}%"
        recall_str = f"{recall * 100:.2f}%"
        f1_score_str = f"{f1_score:.4f}" # F1은 보통 비율로 표시

        print(f"{class_name:<16} | {precision_str:>10} | {recall_str:>10} | {f1_score_str:>10} | {support:>7}")

# 요약 정보(accuracy, macro avg, weighted avg) 출력 (선택 사항)
print("-" * 65)
if 'accuracy' in report_dict:
     accuracy_str = f"{report_dict['accuracy'] * 100:.2f}%"
     # accuracy 행 포맷팅 (Precision/Recall 자리는 비워둠)
     total_support = report_dict['weighted avg']['support'] # 전체 샘플 수
     print(f"{'accuracy':<16} | {'':>10} | {'':>10} | {accuracy_str:>10} | {total_support:>7}")

if 'macro avg' in report_dict:
    metrics = report_dict['macro avg']
    precision_str = f"{metrics['precision'] * 100:.2f}%"
    recall_str = f"{metrics['recall'] * 100:.2f}%"
    f1_score_str = f"{metrics['f1-score']:.4f}"
    support = metrics['support']
    print(f"{'macro avg':<16} | {precision_str:>10} | {recall_str:>10} | {f1_score_str:>10} | {support:>7}")

if 'weighted avg' in report_dict:
    metrics = report_dict['weighted avg']
    precision_str = f"{metrics['precision'] * 100:.2f}%"
    recall_str = f"{metrics['recall'] * 100:.2f}%"
    f1_score_str = f"{metrics['f1-score']:.4f}"
    support = metrics['support']
    print(f"{'weighted avg':<16} | {precision_str:>10} | {recall_str:>10} | {f1_score_str:>10} | {support:>7}")

print("="*65)
# --- ▲▲▲▲▲ 수정 끝 ▲▲▲▲▲ ---

# # 성능 평가 리포트 출력 (정확도, 정밀도, 재현율 등)
# print("\n" + "="*50)
# print("📊 모델 성능 평가 리포트")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# print("="*50)

# Confusion Matrix 시각화
print("\n 정규화된 혼동 행렬(Confusion Matrix)을 시각화합니다. (행 기준, Recall)")
cm = confusion_matrix(y_test, y_pred, normalize='pred')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix (Test Set: {VAL_FILE_PATH})')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()


# --- 5. 학습된 모델 저장 ---
print(f"\n학습된 모델을 '{MODEL_OUTPUT_PATH}' 파일로 저장합니다...")
joblib.dump(model, MODEL_OUTPUT_PATH)
joblib.dump(label_encoder, 'label_encoder.joblib') # 라벨 인코더도 함께 저장해야 나중에 예측 결과를 해석할 수 있습니다.
print("✅ 모델 저장이 완료되었습니다. 이제 이 파일을 사용하여 새로운 데이터의 클래스를 예측할 수 있습니다.")