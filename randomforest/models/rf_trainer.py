
from randomforest.config import ML_FEATURE_COLUMNS, STAGE_1_OBJECT_LABEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np 

# ======================================================================
# --- 2. 모델 학습 (train) 관련 함수들 ---
# ======================================================================

def run_training_stage1(df_train, logger):
    """1단계 모델 (Object / Background) 학습"""
    logger.info("--- 1단계 모델 학습 시작 ---")
    # 1. 라벨 생성
    df_train['s1_label'] = (df_train['label'] != 'Background').astype(int)

    # 2. Object와 Background 분리
    df_s1_obj = df_train[df_train['s1_label'] == STAGE_1_OBJECT_LABEL]
    df_s1_bg = df_train[df_train['s1_label'] != STAGE_1_OBJECT_LABEL]

    num_obj = len(df_s1_obj)

    if num_obj == 0:
        logger.error("오류: 1단계 학습을 위한 Object 샘플이 없습니다.")
        return None

    # --- [핵심 수정] ---
    # Background 샘플을 Object 샘플 수만큼 (혹은 2배수만큼) 랜덤 샘플링
    df_s1_bg_sampled = df_s1_bg.sample(n=num_obj * 2, random_state=42, replace=True if len(df_s1_bg) < num_obj*2 else False)

    # 데이터 다시 합치기
    df_s1_balanced = pd.concat([df_s1_obj, df_s1_bg_sampled])
    # -------------------

    y_s1 = df_s1_balanced['s1_label']
    X_s1 = df_s1_balanced[ML_FEATURE_COLUMNS]

    logger.info(f"1단계 *균형* 학습 샘플: {len(X_s1)}개")
    logger.info(f"Object(1) 샘플: {y_s1.sum()}개 / Background(0) 샘플: {(y_s1 == 0).sum()}개")
    
    # 2. 모델 정의 및 학습
    model_s1 = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1,
        max_depth=30, min_samples_leaf=2,
        class_weight='balanced_subsample'
    )
    model_s1.fit(X_s1, y_s1)
    
    y_pred_s1 = model_s1.predict(X_s1)
    acc_s1 = accuracy_score(y_s1, y_pred_s1)
    logger.info(f"1단계 학습 정확도: {acc_s1 * 100:.2f}%")
    
    return model_s1

def run_training_stage2(df_train, logger):
    """[수정] 2단계 모델 학습 및 LabelEncoder 반환"""
    logger.info("\n--- 2단계 모델 학습 시작 ---")
    df_s2 = df_train[df_train['label'] != 'Background'].copy()
    if df_s2.empty:
        logger.error("오류: 2단계 학습을 위한 Object 샘플이 없습니다.")
        return None, None # 모델과 인코더 모두 None 반환

    le = LabelEncoder()
    y_s2_str = df_s2['label']
    y_s2_numeric = le.fit_transform(y_s2_str)
    X_s2 = df_s2[ML_FEATURE_COLUMNS]
    
    logger.info(f"2단계 학습 샘플: {len(X_s2)}개")
    logger.info(f"2단계 클래스: {le.classes_}")

    model_s2 = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, max_depth=40, min_samples_leaf=2, class_weight='balanced_subsample')
    model_s2.fit(X_s2, y_s2_numeric)
    acc_s2 = accuracy_score(y_s2_numeric, model_s2.predict(X_s2))
    logger.info(f"2단계 학습 정확도: {acc_s2 * 100:.2f}%")
    
    return model_s2, le # ★ LabelEncoder 반환
