# bev_utils.py
import numpy as np
from typing import Tuple, Optional

def pointcloud_to_bev(
    points: np.ndarray,
    x_range: Tuple[float, float] = (0, 70.4),
    y_range: Tuple[float, float] = (-40, 40),
    z_range: Tuple[float, float] = (-2.0, 1.0),
    resolution: float = 0.1
) -> Optional[np.ndarray]:
    """
    3D 포인트 클라우드를 2D BEV(Bird's-Eye View) 이미지로 변환합니다.

    Args:
        points (np.ndarray): (N, 3+) 크기의 포인트 클라우드 배열 (x, y, z, ...).
        x_range (Tuple[float, float]): BEV 이미지에 포함할 X축 범위 (전방/후방).
        y_range (Tuple[float, float]): BEV 이미지에 포함할 Y축 범위 (좌/우).
        z_range (Tuple[float, float]): 필터링할 Z축 높이 범위.
        resolution (float): 이미지 픽셀 하나가 나타내는 실제 미터(m) 크기.

    Returns:
        Optional[np.ndarray]: 생성된 2D BEV 이미지 (uint8). 포인트가 없으면 None을 반환.
    """
    # 1. 지정된 범위 내의 포인트만 선택
    mask = np.where(
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    points = points[mask]

    if points.shape[0] == 0:
        return None

    # 2. 3D 좌표를 2D 픽셀 좌표로 변환
    # 이미지의 Y축 (세로) -> 포인트의 X축 (전방)
    # 이미지의 X축 (가로) -> 포인트의 Y축 (좌우)
    x_points = points[:, 0]
    y_points = points[:, 1]

    # X좌표 -> 이미지 y좌표 (위쪽이 멀어지도록 y_range[1]에서 뺌)
    pixel_y = ((x_range[1] - x_points) / resolution).astype(np.int32)
    # Y좌표 -> 이미지 x좌표 (왼쪽이 +y가 되도록 y_range[1]에서 뺌)
    pixel_x = ((y_range[1] - y_points) / resolution).astype(np.int32)

    # 3. BEV 이미지 크기 계산
    height = int((x_range[1] - x_range[0]) / resolution)
    width = int((y_range[1] - y_range[0]) / resolution)

    # 4. BEV 이미지 생성
    # 0으로 채워진 빈 캔버스(검은색 이미지)를 만듭니다.
    bev_image = np.zeros((height, width), dtype=np.uint8)
    
    # 5. 계산된 픽셀 좌표에 점 찍기
    # 해당하는 픽셀 값을 255(흰색)으로 설정합니다.
    bev_image[pixel_y, pixel_x] = 255
    
    return bev_image