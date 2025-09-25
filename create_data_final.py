import os
import re
import json
import math
import numpy as np
from pathlib import Path
# 스크립트 상단에 추가
from scipy.spatial.transform import Rotation as R

import tools.visual_utils.open3d_vis_utils as V
# --- ❗️ NEW: Add required imports for visualization ---
try:
    import open3d
    import torch
    VISUALIZATION_ENABLED = True
except ImportError:
    print("[WARN] open3d or torch is not installed. Visualization will be disabled.")
    print("       To enable, run: pip install open3d torch")
    VISUALIZATION_ENABLED = False
# ----------------------------------------------------

# --- ❗️ 실행 전 이 경로들을 자신의 환경에 맞게 수정해주세요 ---
# A2D2 원본 데이터셋의 루트 경로
A2D2_ROOT = Path("/home/a/OpenPCDet/data/a2d2/camera_lidar_semantic_bboxes")
# 변환된 KITTI 데이터가 저장될 루트 경로
KITTI_ROOT = Path("/home/a/OpenPCDet/data/a2d2")
# ---------------------------------------------------------

# --- BIN 변환 함수 --- (A2D2 차량 중심 좌표계 -> KITTI LiDAR 좌표계)
def transform_and_save_npz_to_bin(npz_path, bin_path, calib_data, lidar_name='front_center'):
    try:
        data = np.load(npz_path) # .npz 파일 로드
        points_vehicle = data['points'] # A2D2 차량 좌표계의 포인트 클라우드
        reflectance = data['reflectance'].reshape(-1, 1) # 반사 강도

        # --- ▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼ ---
        # The A2D2 vehicle frame and KITTI velodyne frame share the same orientation.
        # The only required transformation is to shift the origin from the vehicle
        # reference point to the LiDAR sensor's location.

        # 1. Get the LiDAR's position in the A2D2 vehicle frame.
        lidar_view = calib_data['lidars'][lidar_name]['view']
        origin = np.array(lidar_view['origin']) # This is the translation vector.

        # 2. Translate the points.
        # points_in_lidar_frame = points_in_vehicle_frame - lidar_origin_in_vehicle_frame
        points_lidar = points_vehicle - origin

        # 변환된 XYZ와 기존 반사율을 합쳐 .bin 파일로 저장
        combined_data = np.hstack([points_lidar, reflectance]).astype(np.float32)
        combined_data.tofile(bin_path)
        return True
    except Exception as e:
        print(f"    [ERR] BIN 변환 실패: {npz_path.name} ({e})")
        return False
    
# --- LABEL 변환 함수 --- (A2D2 차량 중심 좌표계 -> KITTI 카메라 좌표계)
_CLASS_MAP = {
    "car": "Car", "caravan": "Car", "suv": "Car", "van": "Car", "caravantransporter": "Car",
    "emergencyvehicle": "Car", "vansuv": "Car", "truck": "Truck", "trailer": "Trailer",
    "pedestrian": "Pedestrian", "cyclist":"Cyclist", "bicycle": "Bicycle", "bike": "Bicycle", "motorbiker": "MotorBiker",
    "motorcycle": "MotorBiker", "motorbike": "MotorBiker", "bus": "Bus", "utilityvehicle": "UtilityVehicle", "utility_vehicle": "UtilityVehicle", "animals": "DontCare"
}
def _norm_key(name: str) -> str: return name.lower().replace(" ", "").replace("-", "_").replace("\\", "").replace("/", "")
def map_class(a2d2_raw: str): return _CLASS_MAP.get(_norm_key(a2d2_raw), "DontCare")
def normalize_angle(angle_rad): return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# --- LABEL 변환 함수 --- (C_swap 제거 및 Yaw 계산 수정 버전)
def convert_a2d2_to_kitti_label(label_file, calib_data, output_file, camera_name='front_center'):
    try:
        with open(label_file, 'r') as f:
            labels = json.load(f)
        
        cam_view = calib_data['cameras'][camera_name]['view']
        origin = np.array(cam_view['origin'])
        x_axis = np.array(cam_view['x-axis'])
        y_axis = np.array(cam_view['y-axis'])
        z_axis = np.cross(x_axis, y_axis)

        R_cam_to_vehicle = np.array([x_axis, y_axis, z_axis]).T
        T_cam_to_vehicle = np.eye(4)
        T_cam_to_vehicle[:3, :3] = R_cam_to_vehicle
        T_cam_to_vehicle[:3, 3] = origin

        # 차량 -> 카메라 좌표계 변환 (역행렬)
        T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)

        # --- ▼▼▼▼▼ C_swap 제거 ▼▼▼▼▼ ---
        # A2D2 카메라와 KITTI 카메라의 좌표계 정의 (X-right, Y-down, Z-fwd)가 동일하므로
        # 별도의 축 교환(C_swap)이 필요 없습니다.
        T_final = T_vehicle_to_cam
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        kitti_labels = []
        dontcare_classes_in_file = set()

        for box_id, data in labels.items():
            obj_type = map_class(data['class'])
            if obj_type == "DontCare":
                original_class = data['class']
                dontcare_classes_in_file.add(original_class)
                # print(f"    [INFO] '{original_class}' -> DontCare로 매핑됨 (파일: {label_file.name})")

            truncated = float(data.get('truncation', 0.0))
            occluded = int(data.get('occlusion', 0))
            bbox = data['2d_bbox']
            h, w, l = data['size'][2], data['size'][1], data['size'][0]

            center_a2d2 = np.append(np.array(data['center']), 1)
            center_kitti = T_final @ center_a2d2
            location = center_kitti[:3]
            location[1] += h / 2 # 박스 중심을 바닥면으로 이동

            # --- ▼▼▼▼▼ Yaw 계산 로직 수정 ▼▼▼▼▼ ---
            # A2D2 객체의 Yaw (차량 Z축 기준)
            obj_yaw = data['rot_angle']
            if data['axis'][2] < 0:
                obj_yaw = -obj_yaw

            # A2D2 차량 좌표계의 Z축(상)과 KITTI 카메라 좌표계의 Y축(하)은 반대 방향.
            # 따라서 차량 좌표계에서의 Yaw(obj_yaw)는 KITTI 카메라 Y축 기준 회전각으로 변환할 때
            # 90도(pi/2)의 위상차를 가집니다.
            rotation_y = -obj_yaw - (np.pi / 2)
            rotation_y = normalize_angle(rotation_y)
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

            alpha = rotation_y - math.atan2(location[0], location[2])
            alpha = normalize_angle(alpha)

            kitti_line = (f"{obj_type} {truncated:.2f} {occluded} {alpha:.2f} "
                          f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
                          f"{h:.2f} {w:.2f} {l:.2f} "
                          f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} "
                          f"{rotation_y:.2f}")
            kitti_labels.append(kitti_line)

        with open(output_file, 'w') as f:
            f.write("\n".join(kitti_labels))
        return True , dontcare_classes_in_file
    except Exception as e:
        print(f"    [ERR] LABEL 변환 실패: {label_file.name} ({e})")
        return False, set()
    
def visualize_a2d2_frame(npz_file, label_file):
    """
    Loads and visualizes a single A2D2 frame in its original vehicle coordinate system.
    """
    print(f"\n--- Visualizing Frame ---")
    print(f"  Points: {npz_file.name}")
    print(f"  Labels: {label_file.name}")

    # 1. Load point cloud data
    try:
        data = np.load(npz_file)
        points = data['points']
    except Exception as e:
        print(f"  [ERR] Failed to load point cloud: {e}")
        return

    # 2. Load and parse 3D label data
    gt_boxes = []
    try:
        with open(label_file, 'r') as f:
            labels = json.load(f)

        for box_id, data in labels.items():
            center = data['center']  # [x, y, z] in vehicle coords
            size = data['size']      # [l, w, h]
            rot_angle = data['rot_angle'] # yaw in radians

            # Correct yaw based on axis
            if data['axis'][2] < 0:
                rot_angle = -rot_angle

            # Format for draw_box: [x, y, z, l, w, h, yaw]
            gt_box = [
                center[0], center[1], center[2],
                size[0], size[1], size[2],
                rot_angle
            ]
            gt_boxes.append(gt_box)
    except FileNotFoundError:
        print(f"  [WARN] Label file not found, visualizing points only.")
    except Exception as e:
        print(f"  [ERR] Failed to load labels: {e}")

    # 3. Call the visualization function
    V.draw_scenes(
        points=points,
        gt_boxes=np.array(gt_boxes) if gt_boxes else None
    )

# --- CALIB 파일 생성 함수 ---
def build_transform_matrix(view_dict):
    origin = np.array(view_dict['origin'])
    x_axis, y_axis = np.array(view_dict['x-axis']), np.array(view_dict['y-axis'])
    z_axis = np.cross(x_axis, y_axis)
    R = np.array([x_axis, y_axis, z_axis]).T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = origin
    return T

# --- CALIB 파일 생성 함수 --- (C_swap 제거 버전)
def create_kitti_calib_file(calib_data, output_path, cam_name='front_center', lidar_name='front_center'):
    try:
        cam_info = calib_data['cameras'][cam_name]
        lidar_info = calib_data['lidars'][lidar_name]

        cam_matrix = np.array(cam_info['CamMatrix'])
        P2 = np.hstack([cam_matrix, np.zeros((3, 1))])

        cam_view = cam_info['view']
        x_axis = np.array(cam_view['x-axis'])
        y_axis = np.array(cam_view['y-axis'])
        z_axis = np.cross(x_axis, y_axis)
        R0_rect = np.array([x_axis, y_axis, z_axis]).T

        T_cam_to_vehicle = build_transform_matrix(cam_info['view'])
        T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)

        lidar_view = lidar_info['view']
        lidar_origin = np.array(lidar_view['origin'])
        T_lidar_to_vehicle = np.eye(4)
        T_lidar_to_vehicle[:3, 3] = lidar_origin

        # --- ▼▼▼▼▼ C_swap 제거 ▼▼▼▼▼ ---
        # A2D2 LiDAR -> A2D2 차량 -> A2D2 카메라 (KITTI 카메라와 동일)
        Tr_velo_to_cam_4x4 = T_vehicle_to_cam @ T_lidar_to_vehicle
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        Tr_velo_to_cam_3x4 = Tr_velo_to_cam_4x4[:3, :]

        with open(output_path, 'w') as f:
            f.write(f"P0: {' '.join(map(str, P2.flatten()))}\n")
            f.write(f"P1: {' '.join(map(str, P2.flatten()))}\n")
            f.write(f"P2: {' '.join(map(str, P2.flatten()))}\n")
            f.write(f"P3: {' '.join(map(str, P2.flatten()))}\n")
            f.write(f"R0_rect: {' '.join(map(str, R0_rect.flatten()))}\n")
            f.write(f"Tr_velo_to_cam: {' '.join(map(str, Tr_velo_to_cam_3x4.flatten()))}\n")
            f.write(f"Tr_imu_to_velo: {' '.join(map(str, np.eye(3, 4).flatten()))}\n")
        return True
    except Exception as e:
        print(f"    [ERR] CALIB 생성 실패: {output_path.name} ({e})")
        return False
    
# --- 메인 실행 함수 ---
def process_all_sequences(a2d2_root, kitti_root):
    print("A2D2 to KITTI 데이터 변환을 시작합니다...")
    
    # KITTI 출력 폴더 생성
    out_velo_path = kitti_root / "training" / "velodyne"
    out_label_path = kitti_root / "training" / "label_2"
    out_calib_path = kitti_root / "training" / "calib"
    out_velo_path.mkdir(parents=True, exist_ok=True)
    out_label_path.mkdir(parents=True, exist_ok=True)
    out_calib_path.mkdir(parents=True, exist_ok=True)

    # A2D2 마스터 보정 파일 로드
    master_calib_path = a2d2_root / "camera_lidar.json"
    if not master_calib_path.exists():
        print(f"[FATAL] 마스터 보정 파일을 찾을 수 없습니다: {master_calib_path}")
        return
    with open(master_calib_path, 'r') as f:
        calib_data = json.load(f)

    total_count = 0
    master_dontcare_set = set() # 모든 DontCare 클래스를 취합할 마스터 set
    # --- ❗️ MODIFIED: Added a flag to control visualization ---
    visualized_once = not VISUALIZATION_ENABLED
    # A2D2의 모든 시퀀스 폴더 순회
    for seq_path in sorted(a2d2_root.iterdir()):
        if not seq_path.is_dir(): continue

        print(f"\nProcessing sequence: {seq_path.name}")
        lidar_path = seq_path / "lidar" / "cam_front_center"
        label_path = seq_path / "label3D" / "cam_front_center"

        if not lidar_path.exists():
            print(f"  [WARN] LiDAR 폴더 없음, 건너뜀: {lidar_path}")
            continue

        # 해당 시퀀스의 모든 LiDAR(.npz) 파일 순회
        for npz_file in sorted(lidar_path.glob("*.npz")):
            frame_id = npz_file.stem.split('_')[-1]
            
            # --- 경로 정의 ---
            label_file = label_path / f"{npz_file.stem.replace('lidar', 'label3D')}.json"
            bin_out_file = out_velo_path / f"{frame_id}.bin"
            label_out_file = out_label_path / f"{frame_id}.txt"
            calib_out_file = out_calib_path / f"{frame_id}.txt"

            # print(f"  - Frame {frame_id}:")

            # --- 변환 실행 ---
            # 1. BIN 변환
            bin_success = transform_and_save_npz_to_bin(npz_file, bin_out_file, calib_data)

            # 2. LABEL 변환
            if not label_file.exists():
                print(f"    [WARN] 라벨 파일 없음, 건너뜀: {label_file.name}")
                label_success = False
            else:
                label_success, found_dontcares = convert_a2d2_to_kitti_label(label_file, calib_data, label_out_file)
                if label_success:
                    master_dontcare_set.update(found_dontcares) # 마스터 set에 취합            
            # 3. CALIB 생성
            calib_success = create_kitti_calib_file(calib_data, calib_out_file)

            if bin_success and label_success and calib_success:
                total_count += 1
                # print(f"    [OK] Frame {frame_id} 변환 완료.")
                # --- ❗️ MODIFIED: Call visualization for the first successful frame ---
                if not visualized_once:
                    visualize_a2d2_frame(npz_file, label_file)
                    visualized_once = True # Set flag to true to prevent further visualizations
            else:
                print(f"    [FAIL] Frame {frame_id} 변환 중 일부 실패.")

    print(f"\n✅ 총 {total_count} 개의 프레임 변환 완료!")

     # --- 모든 변환 완료 후 DontCare 목록 최종 출력 ---
    if master_dontcare_set:
        print("\n--- 'DontCare'로 처리된 클래스 목록 ---")
        for cls_name in sorted(list(master_dontcare_set)):
            print(f"- {cls_name}")
        print("-----------------------------------------")
    else:
        print("\n'DontCare'로 처리된 클래스가 없습니다.")



# ==============================
# 실행
# ==============================
if __name__ == "__main__":
    process_all_sequences(A2D2_ROOT, KITTI_ROOT)