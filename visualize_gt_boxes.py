import pickle
import numpy as np
from pathlib import Path

# OpenPCDet의 시각화 유틸리티를 가져옵니다.
# 이 스크립트를 실행하려면 PYTHONPATH에 OpenPCDet 경로가 포함되어 있어야 합니다.
# 보통 OpenPCDet/tools 폴더에서 실행하면 잘 동작합니다.
try:
    from tools.visual_utils import open3d_vis_utils as V
except ImportError:
    print("="*50)
    print("ERROR: OpenPCDet의 visual_utils를 찾을 수 없습니다.")
    print("OpenPCDet/tools 폴더에서 이 스크립트를 실행하거나,")
    print("PYTHONPATH 환경 변수에 OpenPCDet 경로를 추가해주세요.")
    print("예시: export PYTHONPATH=$PYTHONPATH:/path/to/OpenPCDet")
    print("="*50)
    exit()

# --- ⚙️ 사용자 설정 부분 ---
# 본인의 OpenPCDet 프로젝트 경로에 맞게 수정해주세요.
# 보통 'OpenPCDet/data/a2d2'를 가리킵니다.
ROOT_PATH = Path('/home/a/OpenPCDet/data/a2d2')

# 확인하고 싶은 pkl 파일의 전체 경로
PKL_FILE_PATH = ROOT_PATH / 'a2d2_infos_train.pkl'

# pkl 파일 안에서 시각화하고 싶은 샘플의 인덱스 (0은 첫 번째 샘플)
SAMPLE_INDEX_TO_VISUALIZE = 0
# -------------------------

def get_lidar(lidar_file_path):
    """지정된 경로의 .bin 파일에서 포인트 클라우드를 로드합니다."""
    if not lidar_file_path.exists():
        print(f"LiDAR 파일이 존재하지 않습니다: {lidar_file_path}")
        return None
    # A2D2 데이터는 4개의 feature (x, y, z, intensity)를 가집니다.
    points = np.fromfile(str(lidar_file_path), dtype=np.float32).reshape(-1, 4)
    return points

def main():
    print(f"'{PKL_FILE_PATH}' 파일 로딩 중...")
    if not PKL_FILE_PATH.exists():
        print(f"ERROR: PKL 파일을 찾을 수 없습니다! 경로를 확인해주세요.")
        return

    with open(PKL_FILE_PATH, 'rb') as f:
        infos = pickle.load(f)

    print(f"총 {len(infos)}개의 샘플 정보를 로드했습니다.")
    if SAMPLE_INDEX_TO_VISUALIZE >= len(infos):
        print(f"ERROR: 요청한 인덱스({SAMPLE_INDEX_TO_VISUALIZE})가 샘플 개수({len(infos)})를 벗어납니다.")
        return

    # 지정된 인덱스의 샘플 정보 가져오기
    sample_info = infos[SAMPLE_INDEX_TO_VISUALIZE]
    
    print(f"\n--- 샘플 인덱스 #{SAMPLE_INDEX_TO_VISUALIZE} 시각화 ---")

    # 1. GT 박스 정보 추출
    try:
        gt_boxes = sample_info['annos']['gt_boxes_lidar']
        print(f"GT 박스 {gt_boxes.shape[0]}개를 찾았습니다.")
    except KeyError:
        print("이 샘플에는 GT 박스 정보가 없습니다.")
        gt_boxes = np.array([]) # 빈 배열로 초기화

    # 2. 포인트 클라우드 파일 경로 찾기 및 로드
    # pkl 파일이 'train'용인지 'val'용인지에 따라 폴더가 다를 수 있습니다.
    split = 'training' if 'train' in PKL_FILE_PATH.name else 'testing' # A2D2는 val도 training 폴더에 있음
    
    lidar_idx = sample_info['point_cloud']['lidar_idx']
    lidar_file = ROOT_PATH / split / 'velodyne' / f'{lidar_idx}.bin'
    
    print(f"포인트 클라우드 파일: {lidar_file}")
    points = get_lidar(lidar_file)

    if points is None:
        return
        
    print(f"{points.shape[0]}개의 포인트를 로드했습니다.")

    # 3. 시각화 함수 호출
    print("\nOpen3D 시각화 창을 엽니다...")
    V.draw_scenes(
        points=points[:, :3],  # x, y, z 좌표만 사용
        gt_boxes=gt_boxes
    )
    print("시각화 완료.")


if __name__ == '__main__':
    main()