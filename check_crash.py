import pickle
import numpy as np
from pathlib import Path

# ==========================================
# 1. 파일 경로 설정 (본인 환경에 맞게 수정하세요)
# 보통 output 디렉토리 안에 있습니다.
# 예: output/a2d2_models/second/default/eval/epoch_3/val/result.pkl
RESULT_PKL_PATH = "output/a2d2_models/second/default/eval/eval_epoch_4/result.pkl" 
# ==========================================

def check_data():
    path = Path(RESULT_PKL_PATH)
    if not path.exists():
        print(f"파일을 찾을 수 없습니다: {path}")
        return

    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        det_annos = pickle.load(f)

    print(f"Total {len(det_annos)} samples loaded.")
    
    nan_count = 0
    inf_count = 0
    weird_names = set()
    
    # 예측된 클래스 이름들을 담을 집합
    valid_classes = {'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Van', 'Person_sitting', 'Tram', 'Misc'} # KITTI/A2D2 기준 (설정에 따라 다름)

    for idx, anno in enumerate(det_annos):
        # 1. Box 값 검사 (NaN / Inf)
        if 'boxes_3d' in anno:
            boxes = anno['boxes_3d']
            if np.isnan(boxes).any():
                print(f"[CRITICAL] Sample {idx}: NaN found in boxes_3d!")
                nan_count += 1
            if np.isinf(boxes).any():
                print(f"[CRITICAL] Sample {idx}: Inf found in boxes_3d!")
                inf_count += 1
        
        # 2. Score 값 검사
        if 'score' in anno:
            scores = anno['score']
            if np.isnan(scores).any():
                print(f"[CRITICAL] Sample {idx}: NaN found in score!")
                nan_count += 1
        
        # 3. Class Name 검사 (가장 의심됨)
        if 'name' in anno:
            names = anno['name']
            for n in names:
                if n not in valid_classes:
                    weird_names.add(n)
                # 혹시 이름이 빈 문자열이거나 None인지 확인
                if not n or n == "":
                    print(f"[CRITICAL] Sample {idx}: Empty class name found!")

    print("-" * 30)
    print("검사 결과 리포트:")
    print(f"NaN 발견: {nan_count} 건")
    print(f"Inf 발견: {inf_count} 건")
    print(f"알 수 없는 클래스 이름: {weird_names}")

    if nan_count > 0 or inf_count > 0:
        print("\n[진단] 학습이 발산(Exploding Gradient)했습니다. Learning Rate를 낮추거나 데이터를 확인하세요.")
    elif len(weird_names) > 0:
        print("\n[진단] 평가 코드가 모르는 클래스 이름이 있습니다. C++ 평가 코드는 정해진 이름 외의 값이 들어오면 인덱스 에러로 터질 수 있습니다.")
    else:
        print("\n[진단] 데이터 값은 정상으로 보입니다. C++ 라이브러리 호환성 문제일 수 있습니다.")

if __name__ == "__main__":
    check_data()