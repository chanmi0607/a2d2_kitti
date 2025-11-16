# import pandas as pd

# # 분석할 CSV 파일 이름
# CSV_FILE = "data/a2d2/rf_training_data.csv"

# def analyze_class_features(file_path):
#     """
#     CSV 파일을 읽어 각 클래스별로 특징들의 평균과 분산을 계산합니다.

#     Args:
#         file_path (str): 분석할 CSV 파일의 경로
#     """
#     try:
#         # 1. CSV 파일을 Pandas DataFrame으로 로드합니다.
#         df = pd.read_csv(file_path)
#         print(f"'{file_path}' 파일 로드 완료. 총 {len(df)}개의 데이터 확인.")
#         print("-" * 50)

#     except FileNotFoundError:
#         print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요.")
#         return
#     except pd.errors.EmptyDataError:
#         print(f"오류: '{file_path}' 파일이 비어있습니다.")
#         return

#     # 2. 'class' 열을 기준으로 데이터를 그룹화합니다.
#     grouped = df.groupby('class')

#     # 3. 각 그룹(클래스)별로 평균을 계산하고 출력합니다.
#     print("각 클래스별 특징 평균 (Mean):")
#     # 소수점 3자리까지만 표시하여 가독성을 높입니다.
#     print(round(grouped.mean(), 3))
#     print("-" * 50)

#     # 4. 각 그룹(클래스)별로 분산을 계산하고 출력합니다.
#     # 분산은 데이터가 평균으로부터 얼마나 흩어져 있는지를 나타냅니다.
#     print("각 클래스별 특징 분산 (Variance):")
#     print(round(grouped.var(), 3))
#     print("-" * 50)


# if __name__ == "__main__":
#     analyze_class_features(CSV_FILE)

import pandas as pd
import numpy as np

CSV_FILE = "data/a2d2/rf_training_data.csv"

def analyze_class_features(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"'{file_path}' 파일 로드 완료. 총 {len(df)}개의 데이터 확인.")
        print("-" * 50)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return
    except pd.errors.EmptyDataError:
        print(f"오류: '{file_path}' 파일이 비어있습니다.")
        return

    grouped = df.groupby('class')

    # 🔹 1. 원본 데이터 기준 평균/분산
    print("각 클래스별 특징 평균 (Mean):")
    print(round(grouped.mean(), 3))
    print("-" * 50)

    print("각 클래스별 특징 분산 (Variance):")
    print(round(grouped.var(), 3))
    print("-" * 50)

    # 🔹 2. 로그 스케일링한 값 기준 추가 통계 (참고용)
    df_log = df.copy()
    for col in ['num_points', 'density']:
        df_log[col] = np.log1p(df_log[col])

    grouped_log = df_log.groupby('class')

    print("로그 스케일링 후 평균 (Mean, log1p):")
    print(round(grouped_log.mean(), 3))
    print("-" * 50)

if __name__ == "__main__":
    analyze_class_features(CSV_FILE)
