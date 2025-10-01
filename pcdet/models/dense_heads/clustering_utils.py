# clustering_utils.py
import cv2
import numpy as np
from typing import List, Tuple

def cluster_bev_image(
    bev_image: np.ndarray, 
    min_area_threshold: int = 5
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    OpenCV의 연결 요소 레이블링을 사용해 BEV 이미지 내 객체들을 클러스터링합니다.

    Args:
        bev_image (np.ndarray): 흑백 BEV 이미지 (uint8).
        min_area_threshold (int): 클러스터로 인정할 최소 픽셀 면적. 
                                   노이즈를 제거하는 데 사용됩니다.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
            - clusters: 감지된 클러스터들의 바운딩 박스 리스트 [(x, y, w, h), ...].
            - visual_image: 바운딩 박스가 그려진 컬러 시각화 이미지.
    """
    # 1. 연결 요소 분석 수행 (라벨, 통계, 중심점 정보 반환)
    # connectivity=8은 대각선을 포함하여 연결된 것으로 판단
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bev_image, connectivity=8
    )

    # 2. 시각화를 위해 흑백 이미지를 컬러(BGR) 이미지로 변환
    visual_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
    
    clusters = []
    # 라벨 0은 배경이므로, 1부터 순회
    for i in range(1, num_labels):
        # 3. 각 클러스터의 통계 정보 추출
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 4. 노이즈 제거: 면적이 임계값보다 작은 클러스터는 무시
        if area >= min_area_threshold:
            clusters.append((x, y, w, h))
            
            # 5. 시각화 이미지에 바운딩 박스 그리기 (초록색)
            cv2.rectangle(visual_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 클러스터 번호 표시 (파란색)
            cv2.putText(visual_image, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1)

    return clusters, visual_image