import numpy as np
import open3d as o3d
from typing import Tuple, List

def read_kitti_bin(bin_path: str) -> o3d.geometry.PointCloud:
    """
    KITTI .bin 파일을 Open3D PointCloud로 변환합니다.

    Args:
        bin_path (str): KITTI .bin 파일 경로

    Returns:
        o3d.geometry.PointCloud: Open3D 포인트 클라우드 객체
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N, 4): x, y, z, intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

def remove_ground_by_z_axis(pcd: o3d.geometry.PointCloud, height_threshold: float) -> o3d.geometry.PointCloud:
    """
    Z축 높이 임계값을 기준으로 지면 포인트를 제거합니다.
    임계값보다 '높은' 포인트만 남깁니다.

    Args:
        pcd (o3d.geometry.PointCloud): 원본 포인트 클라우드
        height_threshold (float): 지면으로 판단할 Z축 높이 임계값

    Returns:
        o3d.geometry.PointCloud: 지면이 제거된 포인트 클라우드
    """
    points = np.asarray(pcd.points)
    
    # Z축 값이 height_threshold보다 큰 포인트들의 인덱스를 찾습니다.
    non_ground_indices = np.where(points[:, 2] > height_threshold)[0]
    
    # 해당 인덱스의 포인트들만 선택하여 새로운 포인트 클라우드를 생성합니다.
    object_pcd = pcd.select_by_index(non_ground_indices)
    
    return object_pcd

def visualize_z_axis_filtering(original_pcd: o3d.geometry.PointCloud, 
                               filtered_pcd: o3d.geometry.PointCloud,
                               height_threshold: float) -> None:
    """
    Z축 필터링 전/후를 비교하여 시각화합니다.
    - 원본 포인트: 회색
    - 지면 제거 후 남은 포인트: 빨간색

    Args:
        original_pcd (o3d.geometry.PointCloud): 원본 포인트 클라우드
        filtered_pcd (o3d.geometry.PointCloud): 필터링된 포인트 클라우드
        height_threshold (float): 시각화에 표시할 높이 임계값
    """
    # 시각화를 위해 원본과 필터링된 pcd를 복사하여 사용
    original_copy = original_pcd.__copy__()
    filtered_copy = filtered_pcd.__copy__()
    
    # 원본은 회색, 필터링된 객체는 빨간색으로 칠합니다.
    original_copy.paint_uniform_color([0.7, 0.7, 0.7]) # Gray
    filtered_copy.paint_uniform_color([1, 0, 0])     # Red
    
    # 좌표축을 추가합니다.
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # 시각화 창에 표시
    o3d.visualization.draw_geometries(
        [original_copy, filtered_copy, coordinate_frame],
        window_name=f"Z-axis Ground Removal (Threshold: {height_threshold:.2f}m)",
        width=1280,
        height=720
    )


# #### 메인 실행 블록 ####
if __name__ == "__main__":
    import os

    # --- 🔧 사용자가 수정할 파라미터 ---
    # KITTI 데이터셋의 센서는 지면으로부터 약 1.73m 높이에 있습니다.
    # 따라서 -1.73m 보다 약간 높은 값을 임계값으로 설정하는 것이 일반적입니다.
    # 데이터셋에 맞춰 이 값을 조정해야 합니다.
    GROUND_HEIGHT_THRESHOLD = -2.7  # Z축 지면 제거 임계값 (단위: 미터)
    
    bin_dir = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    # ---------------------------------

    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    
    for bin_name in bin_files:
        bin_path = os.path.join(bin_dir, bin_name)
        print(f"\n=== Processing: {bin_name} ===")
    
        # 1. 데이터 로드
        pcd_original = read_kitti_bin(bin_path)
        print(f"원본 포인트 수: {len(pcd_original.points)}")
        
        # 2. Z축 기준으로 지면 제거
        pcd_filtered = remove_ground_by_z_axis(pcd_original, GROUND_HEIGHT_THRESHOLD)
        print(f"지면 제거 후 포인트 수: {len(pcd_filtered.points)}")
        
        # 3. 결과 시각화
        visualize_z_axis_filtering(pcd_original, pcd_filtered, GROUND_HEIGHT_THRESHOLD)