import numpy as np
import pickle
from pathlib import Path
import open3d as o3d

# OpenPCDet 유틸리티 임포트
from pcdet.utils import box_utils, calibration_kitti

# --- 설정 ---
# 1. 확인하고 싶은 프레임 ID
FRAME_ID = '000045185'  # <-- 확인하고 싶은 파일 이름으로 변경 (확장자 제외)

# 2. 데이터 경로 설정 (본인 환경에 맞게 수정)
PKL_PATH = 'output/a2d2_models/second/a2d2_cyclist_best/eval/eval_epoch_73/result.pkl'
LABEL_PATH = 'data/a2d2/training/label_2'
CALIB_PATH = 'data/a2d2/training/calib'
LIDAR_PATH = 'data/a2d2/training/velodyne' # <-- 포인트 클라우드(.bin) 파일 경로
# --------------------
def get_label_info_for_vis(label_path, calib_path, frame_id):
    """
    OpenPCDet 데이터로딩 코드에서 발견된 '진짜' 변환 규칙을 적용합니다.
    """
    label_file = Path(label_path) / f'{frame_id}.txt'
    calib_file = Path(calib_path) / f'{frame_id}.txt'
    if not label_file.exists() or not calib_file.exists():
        return np.array([])
    calib = calibration_kitti.Calibration(calib_file)
    
    gt_boxes_lidar_final = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            class_name = parts[0]
            
            if class_name != 'DontCare':
                # 1. 라벨에서 정보 파싱
                h, w, l = [float(x) for x in parts[8:11]]
                x_cam, y_cam, z_cam = [float(x) for x in parts[11:14]]
                ry_cam = float(parts[14])
                
                # 2. 위치 변환 (규칙 1)
                loc_cam_rect = np.array([[x_cam, y_cam, z_cam]])
                loc_lidar = calib.rect_to_lidar(loc_cam_rect)[0]
                
                # 3. 높이/위치 보정 (규칙 2)
                loc_lidar[1] -= h / 3.0
                
                # 4. 회전 변환 (규칙 3)
                ry_lidar_correct = -(ry_cam + np.pi / 2)
                
                # 5. 최종 박스 조합: [x, y, z, l, w, h, yaw]
                final_box = np.concatenate([loc_lidar, [l, w, h], [ry_lidar_correct]])
                gt_boxes_lidar_final.append(final_box)

    if not gt_boxes_lidar_final:
        return np.array([])
        
    return np.array(gt_boxes_lidar_final)

def main():
    # 1. 포인트 클라우드 로드
    pc_file = Path(LIDAR_PATH) / f'{FRAME_ID}.bin'
    if not pc_file.exists():
        print(f"오류: 포인트 클라우드 파일을 찾을 수 없습니다: {pc_file}")
        return
    point_cloud = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 4)

    # 2. 예측 결과 로드
    with open(PKL_PATH, 'rb') as f:
        all_preds = pickle.load(f)
    pred_info_for_frame = next((item for item in all_preds if item["frame_id"] == FRAME_ID), None)
    
    if pred_info_for_frame is None:
        print(f"정보: pkl 파일에서 프레임 {FRAME_ID}의 예측을 찾을 수 없습니다. GT만 시각화합니다.")
        pred_boxes = np.array([])
        pred_scores = np.array([])
    else:
        pred_boxes = pred_info_for_frame['boxes_lidar']
        pred_scores = pred_info_for_frame['score']
    
    # 3. Ground Truth 라벨 로드 및 변환
    gt_boxes = get_label_info_for_vis(LABEL_PATH, CALIB_PATH, FRAME_ID)

    print(f"프레임 {FRAME_ID} 시각화:")
    print(f" - 예측된 박스 수: {len(pred_boxes)}")
    print(f" - 정답 박스 수: {len(gt_boxes)}")

    # Open3D 시각화 준비
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    vis.add_geometry(pcd)

    # 정답 박스(초록색) 추가
    for box in gt_boxes:
        corners = box_utils.boxes_to_corners_3d(box[np.newaxis, :])[0]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[0, 1, 0] for _ in range(len(lines))] # Green
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners), lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    # 예측 박스(빨간색) 추가 (Confidence 0.3 이상만)
    for i, box in enumerate(pred_boxes):
        if pred_scores[i] < 0.3: continue
        corners = box_utils.boxes_to_corners_3d(box[np.newaxis, :])[0]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for _ in range(len(lines))] # Red
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners), lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    # 시각화 실행
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()