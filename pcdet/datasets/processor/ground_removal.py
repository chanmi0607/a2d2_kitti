import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional
from sklearn.cluster import DBSCAN

#1. ROI ...완료
#2. RANSAC ...진행중
#3. Inlnier Removal
def read_kitti_bin(bin_path):
    """
    KITTI .bin 파일을 Open3D PointCloud로 변환

    Args:
        bin_path (str): KITTI .bin 파일 경로

    Returns:
        o3d.geometry.PointCloud: Open3D 포인트 클라우드 객체
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N, 4): x, y, z, intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

##### 첫번째 단계 : ROI ####
def custom_extract_roi(pcd: o3d.geometry.PointCloud, 
                      x_range: Tuple[float, float] = (-20, 20),
                      y_range: Tuple[float, float] = (-20, 20),
                      z_range: Tuple[float, float] = (-3, 3)) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """ROI(Region of Interest) 영역 추출
    
    Args:
        pcd: 입력 포인트 클라우드
        x_range: X축 범위 (min, max)
        y_range: Y축 범위 (min, max)
        z_range: Z축 범위 (min, max)
    
    Returns:
        roi_pcd: ROI 영역 내의 포인트 클라우드
        roi_outliers_pcd: ROI 영역 외의 포인트 클라우드
    """
    # 포인트 클라우드를 numpy 배열로 변환
    points = np.array(pcd.points)
    
    # ROI 영역 내의 포인트들만 필터링
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])  # X축 조건 - TODO: x_range 값을 사용하여 필터링
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])  # Y축 조건 - TODO: y_range 값을 사용하여 필터링
    z_mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])  # Z축 조건 - TODO: z_range 값을 사용하여 필터링
    
    # 모든 조건을 만족하는 포인트들의 인덱스
    roi_mask = x_mask & y_mask & z_mask # TODO: 각 mask 영역을 AND 연산하여 roi_mask 생성
    roi_points = points[roi_mask] # TODO: ROI 영역 내의 포인트들 추출
    roi_outliers = points[~roi_mask] # TODO: ROI 영역 외의 포인트들 추출
    
    # 새로운 포인트 클라우드 생성
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    
    # 색상 정보가 있다면 함께 복사
    if len(pcd.colors) > 0:
        colors = np.array(pcd.colors)
        roi_colors = colors[roi_mask]
        roi_pcd.colors = o3d.utility.Vector3dVector(roi_colors)
    
    # 새로운 포인트 클라우드 생성
    roi_outliers_pcd = o3d.geometry.PointCloud()
    roi_outliers_pcd.points = o3d.utility.Vector3dVector(roi_outliers)
    
    # 색상 정보가 있다면 함께 복사
    if len(pcd.colors) > 0:
        colors = np.array(pcd.colors)
        roi_outliers_colors = colors[~roi_mask]
        roi_outliers_pcd.colors = o3d.utility.Vector3dVector(roi_outliers_colors)
    
    return roi_pcd, roi_outliers_pcd

#### 두번째 단계 : RANSAC ####
# RANSAC 알고리즘 단계별 구현 #
def custom_ransac_sample_points(pcd: o3d.geometry.PointCloud, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC을 위한 랜덤 포인트 샘플링
    
    Args:
        pcd: 포인트 클라우드
        n_samples: 샘플링할 포인트 개수
    
    Returns:
        sampled_points: 샘플링된 포인트들
        sampled_indices: 샘플링된 포인트들의 인덱스
    """
    points = np.array(pcd.points) #포인트 클라우드의 점들을 가져와서 넘파이 배열로 변환해서 points변수에 저장 (N,3)
    n_points = len(points) #points의 개수
    
    # 중복 없이 랜덤 인덱스 선택
    sampled_indices = np.random.choice(n_points, n_samples, replace=False) # TODO: np.random.choice를 사용하여 중복 없이(replace=False) 랜덤 인덱스 선택
    sampled_points = points[sampled_indices] # TODO: sampled_indices를 사용하여 points에서 해당 점들을 추출
    
    return sampled_points, sampled_indices

def custom_find_inliers_outliers(distances: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """거리 기반으로 인라이어와 아웃라이어 분류
    
    Args:
        distances: 각 포인트의 거리
        threshold: 인라이어 판정 임계값
    
    Returns:
        inlier_indices: 인라이어 포인트들의 인덱스
        outlier_indices: 아웃라이어 포인트들의 인덱스
    """
    inlier_indices = np.where(distances < threshold)[0]
    outlier_indices = np.where(distances >= threshold)[0]
    
    return inlier_indices, outlier_indices

def custom_fit_plane_from_samples(sampled_points: np.ndarray) -> np.ndarray:
    """샘플링된 포인트들로부터 평면 피팅 (역행렬 사용)
    
    평면 방정식: ax + by + cz + d = 0
    
    Args:
        sampled_points: 샘플링된 포인트들 (최소 3개)
    
    Returns:
        plane_params: 평면 파라미터 [a, b, c, d] (정규화됨)
    """
    if len(sampled_points) < 3:
        raise ValueError("평면 피팅을 위해서는 최소 3개의 포인트가 필요합니다.")
    
    # 포인트들을 homogeneous 좌표로 변환: [x, y, z, 1]
    n_points = len(sampled_points)
    H = np.column_stack([sampled_points, np.ones(n_points)])  # (n, 4) - TODO: H 행렬 완성

    # 최소자승법을 위한 정규방정식: H^T * H * x = 0
    # SVD를 사용하여 해를 구함 (H^T * H의 고유벡터 중 가장 작은 고유값에 해당하는 벡터)
    try:
        # SVD 분해: H = U * S * V^T
        U, S, Vt = np.linalg.svd(H)

        # 가장 작은 특이값에 해당하는 오른쪽 특이벡터가 해
        plane_params = Vt[-1, :]  # TODO: 마지막 행 (가장 작은 특이값) 을 기입
        
        # 법선 벡터 정규화 (a^2 + b^2 + c^2 = 1)
        normal_length = np.linalg.norm(plane_params[:3])
        if normal_length < 1e-8:
            raise ValueError("평면의 법선 벡터를 계산할 수 없습니다.")
        
        plane_params = plane_params / normal_length
        
    except np.linalg.LinAlgError:
        # SVD가 실패한 경우 외적 방법 사용
        p1, p2, p3 = sampled_points[:3]
        
        # 두 벡터 계산
        v1 = p2 - p1  # 첫 번째 벡터
        v2 = p3 - p1  # 두 번째 벡터
        
        # 외적으로 법선 벡터 계산
        normal = np.cross(v1, v2)
        
        # 법선 벡터 정규화
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-8:
            raise ValueError("선택된 포인트들이 한 직선 위에 있어 평면을 정의할 수 없습니다.")
        
        normal = normal / normal_length
        
        # 평면 방정식 계수 계산: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, p1)  # d = -(ax1 + by1 + cz1)
        
        plane_params = np.array([a, b, c, d])
    
    return plane_params

def custom_compute_plane_distances(pcd: o3d.geometry.PointCloud, plane_params: np.ndarray) -> np.ndarray:
    """모든 포인트와 평면 사이의 거리 계산
    
    평면 방정식: ax + by + cz + d = 0
    점-평면 거리: |ax + by + cz + d| / √(a² + b² + c²)
    
    Args:
        pcd: 포인트 클라우드
        plane_params: 평면 파라미터 [a, b, c, d]
    
    Returns:
        distances: 각 포인트와 평면 사이의 거리
    """
    points = np.array(pcd.points)
    a, b, c, d = plane_params
    
    # 점-평면 거리 공식
    numerator = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) # TODO: 각 포인트에 대해 |ax + by + cz + d| 계산
    denominator = np.sqrt(a**2 + b**2 + c**2) # TODO: √(a² + b² + c²) 계산
    
    distances = numerator / denominator
    
    return distances


def custom_ransac_plane_fitting(pcd: o3d.geometry.PointCloud, 
                               max_iterations: int = 1000, 
                               distance_threshold: float = 0.25, 
                               min_samples: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC을 사용한 평면 피팅
    
    Args:
        pcd: 포인트 클라우드
        max_iterations: 최대 반복 횟수
        distance_threshold: 인라이어 판정 거리 임계값
        min_samples: 모델 추정에 필요한 최소 샘플 수
    
    Returns:
        best_plane_params: 최적 평면 파라미터 [a, b, c, d]
        best_inlier_indices: 최적 인라이어 인덱스들
        best_outlier_indices: 최적 아웃라이어 인덱스들
    """
    # RANSAC 샘플링 함수 import
    #import custom_ransac_sample_points, custom_find_inliers_outliers
    
    best_plane_params = None
    best_inlier_indices = None
    best_outlier_indices = None
    max_inliers = 0
    
    for iteration in range(max_iterations):
        try:
            # 1. 랜덤 샘플링 (3개 포인트)
            sampled_points, _ = custom_ransac_sample_points(pcd, min_samples)
            
            # 2. 평면 피팅
            plane_params = custom_fit_plane_from_samples(sampled_points) # TODO: 평면 파라미터 계산
            
            # 3. 모든 포인트에 대한 거리 계산
            distances = custom_compute_plane_distances(pcd, plane_params) # TODO: 각 포인트와 평면 사이의 거리 계산
            
            # 4. 인라이어/아웃라이어 분류
            inlier_indices, outlier_indices = custom_find_inliers_outliers(distances, distance_threshold)
            
            # 5. 최적 모델 업데이트
            if len(inlier_indices) > max_inliers:  # TODO: inlier_indices 개수와 최대 인라이어 수 비교
                max_inliers = len(inlier_indices)             # TODO: 최대 인라이어 수 업데이트
                best_plane_params = plane_params.copy()     # TODO: 최적 평면 파라미터 업데이트
                best_inlier_indices = inlier_indices.copy()   # TODO: 최적 인라이어 인덱스 업데이트
                best_outlier_indices = outlier_indices.copy()  # TODO: 최적 아웃라이어 인덱스 업데이트
                
        except (np.linalg.LinAlgError, ValueError):
            # 수치적 불안정성이나 부족한 샘플로 인한 에러 무시
            continue
    
    return best_plane_params, best_inlier_indices, best_outlier_indices

def custom_remove_ground_points(pcd: o3d.geometry.PointCloud, 
                               ground_indices: np.ndarray) -> o3d.geometry.PointCloud:
    """지면 포인트들을 제거하여 객체 포인트만 남김
    
    Args:
        pcd: 원본 포인트 클라우드
        ground_indices: 지면으로 분류된 포인트들의 인덱스
    
    Returns:
        object_pcd: 지면이 제거된 포인트 클라우드
    """
    # 전체 포인트 인덱스
    all_indices = np.arange(len(pcd.points))
    
    # 지면이 아닌 포인트들의 인덱스 (객체 포인트들)
    object_indices = np.setdiff1d(all_indices, ground_indices) # TODO: 전체 인덱스에서 지면 인덱스를 제외한 객체 인덱스 계산 (np.setdiff1d 사용)
    
    # 객체 포인트들만 추출
    points = np.array(pcd.points)
    object_points = points[object_indices] # TODO: 객체 포인트들 추출 (object_indices 사용)
    
    # 새로운 포인트 클라우드 생성
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    
    # 색상 정보가 있다면 함께 복사
    if len(pcd.colors) > 0:
        colors = np.array(pcd.colors)
        object_colors = colors[object_indices]
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    
    return object_pcd

### 모듈 정리 ####
def remove_ground_open3d(points: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Step 1. ROI 추출
    roi_z_range = (-2.73, -1.53)    # Z축 범위
    roi_x_range = (-70.4, 80.4)  # X축 범위
    roi_y_range = (-40, 40)  # Y축 범위 
    roi_pcd, _ = custom_extract_roi(pcd, roi_x_range, roi_y_range, roi_z_range)
    
    # Step 2. RANSAC
    plane_coeffs, _, _ = custom_ransac_plane_fitting(roi_pcd)
    if plane_coeffs is None:
        return points  # 평면 찾기 실패 시 전체 반환

    # Step 3. 전체에서 inlier 계산
    distances = custom_compute_plane_distances(pcd, plane_coeffs)
    inlier_indices, outlier_indices = custom_find_inliers_outliers(distances, 0.25)

    return points[outlier_indices]

### DBSCAN ###
def cluster_objects_dbscan(pcd: o3d.geometry.PointCloud, eps: float = 0.5, min_points: int = 10) -> List[np.ndarray]:
    """
    DBSCAN을 사용하여 포인트 클라우드를 객체별로 클러스터링
    
    Args:
        pcd: 지면이 제거된 포인트 클라우드
        eps: 클러스터의 최대 반경
        min_points: 클러스터에 포함될 최소 포인트 수
        
    Returns:
        List[np.ndarray]: 각 클러스터에 속하는 포인트들의 배열 리스트
    """
    points = np.asarray(pcd.points)
    
    # DBSCAN 모델 생성 및 학습
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    
    # 각 클러스터의 인덱스 추출
    unique_labels = set(labels)
    clusters = []
    
    for label in unique_labels:
        # 노이즈(-1) 제외
        if label == -1:
            continue
        
        cluster_indices = np.where(labels == label)[0]
        clusters.append(points[cluster_indices])
        
    return clusters

#### 시각화 함수 ####
def custom_visualize_roi_extraction(original_pcd: o3d.geometry.PointCloud, 
                                   roi_pcd: o3d.geometry.PointCloud,
                                   x_range: Tuple[float, float],
                                   y_range: Tuple[float, float],
                                   z_range: Tuple[float, float]) -> None:
    """ROI 추출 결과 시각화
    
    Args:
        original_pcd: 원본 포인트 클라우드
        roi_pcd: ROI 영역 포인트 클라우드
        x_range: X축 범위
        y_range: Y축 범위
        z_range: Z축 범위
    """
    geometries = []
    
    # 원본 포인트 클라우드 (회색)
    original_copy = original_pcd.__copy__()
    original_copy.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(original_copy)
    
    # ROI 영역 포인트 클라우드 (빨간색)
    roi_copy = roi_pcd.__copy__()
    roi_copy.paint_uniform_color([1, 0, 0])
    geometries.append(roi_copy)
    
    # 좌표계 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(coordinate_frame)
    
    o3d.visualization.draw_geometries(geometries, window_name="ROI Extraction Result")

def custom_visualize_ground_removal(pcd: o3d.geometry.PointCloud, 
                                   plane_params: np.ndarray,
                                   ground_indices: np.ndarray,
                                   object_indices: np.ndarray) -> None:
    """지면 제거 결과 시각화 (투명한 mesh 평면 포함)
    
    Args:
        pcd: 원본 포인트 클라우드
        plane_params: 평면 파라미터 [a, b, c, d]
        ground_indices: 지면 포인트 인덱스들
        object_indices: 객체 포인트 인덱스들
    """
    geometries = []
    
    # 포인트 클라우드 색상 설정
    pcd_copy = pcd.__copy__()
    points = np.array(pcd_copy.points)
    colors = np.zeros((len(points), 3))
    
    # 지면 포인트는 초록색, 객체 포인트는 빨간색
    colors[ground_indices] = [0, 1, 0]  # 지면: 초록색
    colors[object_indices] = [1, 0, 0]  # 객체: 빨간색
    
    pcd_copy.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd_copy)
    
    # 평면 시각화 (투명한 mesh)
    a, b, c, d = plane_params
    
    # 포인트 클라우드 범위 계산
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    
    # 평면 mesh 생성
    if abs(c) > 1e-8:  # c가 0이 아닌 경우
        # 4개 모서리 점 계산
        corners = [
            [x_min, y_min, -(a * x_min + b * y_min + d) / c],
            [x_max, y_min, -(a * x_max + b * y_min + d) / c],
            [x_max, y_max, -(a * x_max + b * y_max + d) / c],
            [x_min, y_max, -(a * x_min + b * y_max + d) / c]
        ]
        
        # 삼각형 mesh 생성
        vertices = np.array(corners)
        # 4 개의 삼각형으로 사각형 구성
        triangles = []
        for i in range(4):
            triangles.append([i, (i + 1) % 4, (i + 2) % 4])
            triangles.append([i, (i + 2) % 4, (i + 3) % 4])
        
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 법선 벡터 계산
        plane_mesh.compute_vertex_normals()

        plane_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(plane_mesh)
        
        # 파란색으로 설정
        plane_wireframe.paint_uniform_color([0, 0, 1])
        
        geometries.append(plane_wireframe)
    
    # 좌표계 추가
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(coordinate_frame)
    
    o3d.visualization.draw_geometries(geometries, window_name="Ground Removal Result")

def save_geometry_as_image(geometry_list, save_path, width=1280, height=720):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=width, height=height)  # 창을 띄움

    for g in geometry_list:
        vis.add_geometry(g)

    for _ in range(20):  # 렌더링 안정화 시간 확보
        vis.poll_events()
        vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f"✅ 저장됨: {save_path}")

# #### 임시 메인 ####
if __name__ == "__main__":
    #bin_path = "/home/a/OpenPCDet/data/kitti/training/velodyne/001010.bin"
    #pcd = read_kitti_bin(bin_path)
    #o3d.visualization.draw_geometries([pcd])
    import os
    from tqdm import tqdm
    bin_dir = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    for bin_name in bin_files:
        bin_path = os.path.join(bin_dir, bin_name)
        print(f"\n=== Processing {bin_name} ===")
    
        pcd = read_kitti_bin(bin_path)
        points = np.asarray(pcd.points)
        # ROI 파라미터 설정
        sensor_height = 1.73       # 센서 높이
        roi_z_range = (-3.73, -1.30)    # Z축 범위
        roi_x_range = (-70.4, 80.4)  # X축 범위
        roi_y_range = (-40, 40)  # Y축 범위 

        print("=== ROI 추출 ===")
        print(f"원본 포인트 수: {len(pcd.points)}")

        # ROI 영역 추출
        roi_pcd, _ = custom_extract_roi(pcd, roi_x_range, roi_y_range, roi_z_range) # TODO: custom_extract_roi 함수 완성
        print(f"ROI 포인트 수: {len(roi_pcd.points)}")

        # ROI 추출 결과 시각화
        custom_visualize_roi_extraction(pcd, roi_pcd, roi_x_range, roi_y_range, roi_z_range)

        # RANSAC 평면 피팅 파라미터 설정
        distance_threshold = 0.20  # 평면과 포인트 사이의 거리 임계값
        max_iterations = 1000     # RANSAC 반복 횟수
        min_samples = 3          # 평면 피팅에 필요한 최소 샘플 수

        print("=== RANSAC 평면 피팅 ===")
        print(f"거리 임계값: {distance_threshold}m")
        print(f"최대 반복 횟수: {max_iterations}")
        print(f"최소 샘플 수: {min_samples}")

        # RANSAC으로 지면 평면 찾기
        plane_coeffs, ground_indices, object_indices = custom_ransac_plane_fitting(
            roi_pcd, 
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_samples=min_samples
        ) # TODO: custom_ransac_plane_fitting 함수 완성
        if plane_coeffs is not None:
            a, b, c, d = plane_coeffs
            print(f"지면 평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
            print(f"ROI 내 지면 inlier 개수: {len(ground_indices)}")
            print(f"ROI 내 객체 포인트 수: {len(object_indices)}")

            custom_visualize_ground_removal(roi_pcd, plane_coeffs, ground_indices, object_indices)
            # ROI 인덱스 (원본 pcd -> ROI 추출 시 사용된 mask 기반)
            roi_mask = (
                (points[:, 0] >= roi_x_range[0]) & (points[:, 0] <= roi_x_range[1]) &
                (points[:, 1] >= roi_y_range[0]) & (points[:, 1] <= roi_y_range[1]) &
                (points[:, 2] >= roi_z_range[0]) & (points[:, 2] <= roi_z_range[1])
            )
            roi_indices = np.where(roi_mask)[0]

            # ROI 내에서 지면으로 잡힌 인덱스 (전체 인덱스로 변환)
            roi_ground_indices = roi_indices[ground_indices]

            # 전체 포인트 중 지면 제거
            all_indices = np.arange(len(points))
            object_indices = np.setdiff1d(all_indices, roi_ground_indices)

            non_ground_points = points[object_indices]

            print(f"지면 제거 후 객체 포인트 수: {len(non_ground_points)}")

            # 시각화
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(non_ground_points[:, :3])
            o3d.visualization.draw_geometries([pcd_filtered], window_name="ROI 기반 지면 제거 결과")


        # if plane_coeffs is not None:
        #     a, b, c, d = plane_coeffs
        #     print(f"지면 평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        #     print(f"지면 inlier 개수: {len(ground_indices)}")
        #     print(f"객체 포인트 수: {len(object_indices)}")
            
        #     # 지면 제거 결과 시각화
        #     custom_visualize_ground_removal(roi_pcd, plane_coeffs, ground_indices, object_indices)
            
        #     # ROI 내 포인트 클라우드에서 추출한 지면 방정식을 전체 포인트 클라우드에서도 적용하여 지면 방정식의 inlier 포인트들 재선정
        #     distances = custom_compute_plane_distances(pcd, plane_coeffs)  # TODO: 전체 포인트 클라우드에서 지면 방정식에 대한 거리 계산
        #     inlier_indices, outlier_indices = custom_find_inliers_outliers(distances, distance_threshold)  # TODO: 전체 포인트 클라우드에서 inlier/ outlier 인덱스 추출

        #     # 지면이 제거된 객체 포인트 클라우드 생성 (inlier_indices)
        #     non_ground_pcd = custom_remove_ground_points(pcd, inlier_indices) # TODO: custom_remove_ground_points 함수 완성
        #     print(f"지면 제거 후 객체 포인트 수: {len(non_ground_pcd.points)}")
        # else:
        #     print("지면 평면을 찾을 수 없습니다!")

        # # 시각화 - 원본
        # pcd_all = o3d.geometry.PointCloud()
        # pcd_all.points = o3d.utility.Vector3dVector(points[:, :3])
        # o3d.visualization.draw_geometries([pcd_all], window_name="원본 포인트 클라우드")

        # # ground 제거된 포인트 얻기
        # filtered_points = remove_ground_open3d(points)

        # # 시각화용 포인트클라우드 생성
        # pcd_filtered = o3d.geometry.PointCloud()
        # pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points[:, :3])

        # # 시각화
        # o3d.visualization.draw_geometries([pcd_filtered], window_name="지면 제거된 포인트 클라우드")


# if __name__ == "__main__":
#     import os
#     from tqdm import tqdm
#     bin_dir = "/home/a/OpenPCDet/data/a2d2/training/velodyne"
#     bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])

#     # 저장 폴더 준비
#     output_dirs = {
#         "original": "/media/a/새 볼륨/a2d2_ground_removal_output/original",
#         "roi": "/media/a/새 볼륨/a2d2_ground_removal_output/roi",
#         "ransac": "/media/a/새 볼륨/a2d2_ground_removal_output/ransac",
#         "filtered(final_input)": "/media/a/새 볼륨/a2d2_ground_removal_output/filtered(final_input)"
#     }
#     os.makedirs("a2d2_ground_removal_output", exist_ok=True)
#     for path in output_dirs.values():
#         os.makedirs(path, exist_ok=True)

#     for bin_name in tqdm(bin_files, desc="Processing BIN files"):
#         bin_path = os.path.join(bin_dir, bin_name)
#         base_name = os.path.splitext(bin_name)[0]
#         print(f"\n=== Processing {bin_name} ===")

#         # 원본 로드
#         pcd = read_kitti_bin(bin_path)
#         points = np.asarray(pcd.points)

#         # 시각화용 원본 저장
#         pcd_all = o3d.geometry.PointCloud()
#         pcd_all.points = o3d.utility.Vector3dVector(points[:, :3])
#         o3d.io.write_point_cloud(f"{output_dirs['original']}/{base_name}.ply", pcd_all, write_ascii=True)

#         # ROI 추출
#         roi_z_range = (-2.73, -1.53)
#         roi_x_range = (-70.4, 80.4)
#         roi_y_range = (-40, 40)
#         roi_pcd, _ = custom_extract_roi(pcd, roi_x_range, roi_y_range, roi_z_range)
#         o3d.io.write_point_cloud(f"{output_dirs['roi']}/{base_name}.ply", roi_pcd, write_ascii=True)

#         # RANSAC
#         distance_threshold = 0.20
#         plane_coeffs, ground_indices, object_indices = custom_ransac_plane_fitting(
#             roi_pcd, max_iterations=1000, distance_threshold=distance_threshold, min_samples=3
#         )

#         if plane_coeffs is not None:
#             # 시각화용 색상 부여
#             colors = np.zeros_like(np.asarray(roi_pcd.points))
#             colors[ground_indices] = [0, 1, 0]  # green
#             colors[object_indices] = [1, 0, 0]  # red

#             vis_pcd = o3d.geometry.PointCloud()
#             vis_pcd.points = roi_pcd.points
#             vis_pcd.colors = o3d.utility.Vector3dVector(colors)
#             o3d.io.write_point_cloud(f"{output_dirs['ransac']}/{base_name}.ply", vis_pcd, write_ascii=True)

#         # 전체 포인트에서 지면 제거
#         filtered_points = remove_ground_open3d(points)

#         pcd_filtered = o3d.geometry.PointCloud()
#         pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points[:, :3])
#         o3d.io.write_point_cloud(f"{output_dirs['filtered(final_input)']}/{base_name}.ply", pcd_filtered, write_ascii=True)

