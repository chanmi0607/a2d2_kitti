# classifier_utils.py (4-feature 버전)

import numpy as np
import joblib
from typing import List, Tuple, Dict

def extract_features(clusters: List[Tuple[int, int, int, int]], resolution: float) -> np.ndarray:
    """
    클러스터(픽셀 단위)로부터 4개의 기본 특징 벡터(미터 단위)를 추출합니다.
    (train_model.py와 특징 구성을 정확히 일치시킴)
    """
    if not clusters:
        return np.array([]).reshape(0, 4)

    features = []
    for (x, y, w_pixels, h_pixels) in clusters:
        # 픽셀 단위를 미터 단위로 변환
        bev_w = w_pixels * resolution  # 너비 (m)
        bev_l = h_pixels * resolution  # 길이 (m)
        
        # train_model.py와 동일한 4개 특징 계산
        feature_vector = [
            bev_w,                            # 너비
            bev_l,                            # 길이
            bev_w * bev_l,                    # 면적
            bev_w / bev_l if bev_l > 0 else 0 # 너비/길이 비율
        ]
        features.append(feature_vector)
    
    return np.array(features)

def load_model(model_path: str):
    """저장된 머신러닝 모델을 불러옵니다."""
    try:
        model = joblib.load(model_path)
        print(f"✅ 모델 로드 성공: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❗️ Error: 모델 파일을 찾을 수 없습니다: {model_path}")
        return None