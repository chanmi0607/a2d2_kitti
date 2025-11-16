# visualize_rf_predictions.py
import os
import numpy as np
import pandas as pd
import pickle
import joblib
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm # 📍 진행률 표시를 위해 tqdm 임포트

# 📍 1. [수정] predict_with_rf.py와 동일한 함수 임포트
try:
    from ground_removal_onlyz import read_kitti_bin, remove_ground_by_z_axis
    from pcdet.models.detectors.bev_utils import pointcloud_to_bev
    from propose_main import filter_points_by_z_spread 
except ImportError as e:
    print(f"❗️ Error: 필요한 헬퍼 모듈을 찾을 수 없습니다. ({e})")
    print("predict_with_rf.py와 동일한 환경, 동일한 위치에서 실행해야 합니다.")
    print("('ground_removal_onlyz.py', 'propose_main.py' 등이 필요합니다)")
    exit()

# --- 설정 (predict_with_rf.py와 반드시 동일해야 함) ---
PREDICTION_PKL_PATH = 'output/rf_prediction_results.pkl'
BASE_DIR = "data/a2d2/training"
LIDAR_DIR = os.path.join(BASE_DIR, "velodyne")
BEV_X_RANGE = (0, 30)
BEV_Y_RANGE = (-40, 40)
BEV_RESOLUTION = 0.1
GROUND_HEIGHT_THRESHOLD = -2.7
Z_SPREAD_THRESHOLD = 13.0
# ----------------------------------------------------


# 📍 demo.py에서 가져온 바운딩 박스 그리기 함수
def draw_boxes_on_bev(
    bev_image: np.ndarray,
    boxes: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    resolution: float,
    color: tuple = (0, 0, 255), # 🔴 빨간색 (BGR)
    thickness: int = 2 # 👈 두께 2로 수정 (잘 보이게)
) -> np.ndarray:
    """ (N, 7) [x, y, z, l, w, h, yaw] 형식의 LiDAR 박스를 BEV 이미지에 그립니다. """
    if bev_image.ndim == 2:
        color_bev = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2BGR)
    else:
        color_bev = bev_image.copy()

    for box in boxes:
        center_x_lidar, center_y_lidar = box[0], box[1]
        l_lidar, w_lidar, yaw_lidar = box[3], box[4], box[6]

        pixel_center_y = int((x_range[1] - center_x_lidar) / resolution)
        pixel_center_x = int((y_range[1] - center_y_lidar) / resolution)
        pixel_height = int(l_lidar / resolution)
        pixel_width = int(w_lidar / resolution)
        angle_degrees = -np.rad2deg(yaw_lidar)

        rect = ((pixel_center_x, pixel_center_y), (pixel_width, pixel_height), angle_degrees)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        cv2.drawContours(color_bev, [box_points], 0, color, thickness)
        
    return color_bev

def main(args):
    # 1. 예측 결과(.pkl) 로드
    print(f"'{PREDICTION_PKL_PATH}'에서 예측 결과를 로드합니다...")
    try:
        with open(PREDICTION_PKL_PATH, 'rb') as f:
            all_predictions = pickle.load(f)
    except FileNotFoundError:
        print(f"❗️ Error: '{PREDICTION_PKL_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 `predict_with_rf.py`를 실행하여 .pkl 파일을 생성해야 합니다.")
        return

    # 📍 --- 1. [추가] 시각화 이미지 저장 디렉터리 ---
    BEV_SAVE_DIR = Path("a2d2/rf_bev_visualized")
    BEV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_only:
        print(f"Visualized BEV images will be saved to: {BEV_SAVE_DIR.resolve()}")
    # 📍 --- --- --- --- --- --- --- --- --- ---

    if args.frame_id:
        print(f"--frame_id '{args.frame_id}'만 처리합니다.")
        target_frame_data = None
        for item in all_predictions:
            if item['frame_id'] == args.frame_id:
                target_frame_data = item
                break
        
        if target_frame_data:
            frames_to_process = [target_frame_data]
        else:
            print(f"❗️ Error: '{args.frame_id}'에 대한 예측 결과가 .pkl 파일에 없습니다.")
            return
    else:
        print(f"모든 프레임을 처리합니다. (총 {len(all_predictions)}개)")
        frames_to_process = all_predictions 

    SCORE_THRESHOLD = 0.8 
    
    print(f"\n[INFO] 신뢰도 점수 > {SCORE_THRESHOLD:.2f} 인 박스만 표시합니다.")
    if not args.save_only:
        print("시각화 창이 떴습니다. 'q' 키를 누르면 종료, 그 외 아무 키나 누르면 다음 프레임으로 넘어갑니다.")
    
    # 📍 --- 2. [수정] --save_only일 때 tqdm으로 진행률 표시 ---
    frame_iterable = frames_to_process
    # --save_only 모드이고, 전체 프레임을 처리할 때만 tqdm 적용
    if args.save_only and not args.frame_id:
        frame_iterable = tqdm(frames_to_process, desc="Saving BEV images", ncols=100)
    
    for frame_data in frame_iterable:
        frame_id = frame_data['frame_id']

        # 3. 원본 .bin 파일 로드
        bin_path = os.path.join(LIDAR_DIR, f"{frame_id}.bin")
        if not os.path.exists(bin_path):
            if not args.save_only: # 저장 모드일 땐 너무 많이 출력되므로 생략
                print(f"❗️ [Skip] 원본 .bin 파일 '{bin_path}'를 찾을 수 없습니다.")
            continue 
            
        pcd_original = read_kitti_bin(bin_path)
        
        # 4. 전처리 파이프라인
        pcd_non_ground = remove_ground_by_z_axis(pcd_original, GROUND_HEIGHT_THRESHOLD)
        non_ground_points = np.asarray(pcd_non_ground.points)
        if non_ground_points.shape[0] == 0:
            if not args.save_only:
                print(f"❗️ [Skip] {frame_id}: 지면 제거 후 포인트 없음.")
            continue

        filtered_points = filter_points_by_z_spread(
            points=non_ground_points, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION, z_spread_threshold=Z_SPREAD_THRESHOLD
        )
        if filtered_points.shape[0] == 0:
            if not args.save_only:
                print(f"❗️ [Skip] {frame_id}: Z-Spread 필터링 후 포인트 없음.")
            continue
        
        base_bev_image = pointcloud_to_bev(
            points=filtered_points, 
            x_range=BEV_X_RANGE,
            y_range=BEV_Y_RANGE,
            resolution=BEV_RESOLUTION
        )

        if base_bev_image is None:
            height = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RESOLUTION)
            width = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RESOLUTION)
            base_bev_image = np.zeros((height, width), dtype=np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        processed_bev_image = cv2.morphologyEx(base_bev_image, cv2.MORPH_CLOSE, kernel)

        # 5. 예측 박스 가져오기 및 필터링
        pred_boxes_all = frame_data['boxes_lidar']
        pred_scores_all = frame_data['score']
        pred_names_all = frame_data['name']

        score_threshold = SCORE_THRESHOLD 
        
        if score_threshold > 0.0:
            mask = (pred_scores_all >= score_threshold)
            pred_boxes = pred_boxes_all[mask]
            pred_scores = pred_scores_all[mask]
            pred_names = pred_names_all[mask]
        else:
            pred_boxes = pred_boxes_all
            pred_scores = pred_scores_all
            pred_names = pred_names_all
        
        # 6. BEV 이미지에 박스 그리기
        bev_with_boxes = draw_boxes_on_bev(
            processed_bev_image, 
            pred_boxes, 
            BEV_X_RANGE,
            BEV_Y_RANGE,
            BEV_RESOLUTION
        )
        
        # 7. 텍스트 표시
        for box, score, name in zip(pred_boxes, pred_scores, pred_names):
            pixel_center_y = int((BEV_X_RANGE[1] - box[0]) / BEV_RESOLUTION)
            pixel_center_x = int((BEV_Y_RANGE[1] - box[1]) / BEV_RESOLUTION)
            
            text = f"{name} {score:.2f}"
            cv2.putText(
                bev_with_boxes, text, (pixel_center_x + 5, pixel_center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )
            
        # 📍 --- 3. [추가] 이미지 파일로 저장 ---
        save_path = BEV_SAVE_DIR / f"{frame_id}_pred_bev.png"
        cv2.imwrite(str(save_path), bev_with_boxes)
        # 📍 --- --- --- --- --- --- --- ---

        # 📍 --- 4. [수정] --save_only 플래그에 따라 표시 여부 결정 ---
        if not args.save_only:
            # 8. 결과 이미지 보여주기 (기존 로직)
            window_title = f"RF Prediction BEV | Frame: {frame_id} | Score: >={score_threshold:.2f} | 'q': quit, OTHER: next"
            cv2.imshow(window_title, bev_with_boxes)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'): 
                break
        # 📍 --- --- --- --- --- --- --- ---
            
    cv2.destroyAllWindows()
    if args.save_only:
        print(f"\n✅ 모든 시각화 이미지 저장이 완료되었습니다. -> {BEV_SAVE_DIR.resolve()}")
    else:
        print("시각화를 종료합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Forest BEV Prediction Visualizer')
    
    parser.add_argument(
        '--frame_id', 
        type=str, 
        default=None,
        help='(선택) 시각화할 특정 프레임 ID. 지정하지 않으면 모든 프레임을 순차적으로 재생합니다.'
    )
    
    # 📍 --- 5. [추가] --save_only 인자 ---
    parser.add_argument(
        '--save_only',
        action='store_true', # 이 플래그가 있으면 True가 됨
        help='(선택) 이미지를 화면에 표시하지 않고 "output/rf_bev_visualized" 폴더에 파일로만 "쭉" 저장합니다.'
    )
    # 📍 --- --- --- --- --- --- --- ---
    
    args = parser.parse_args()
    main(args)