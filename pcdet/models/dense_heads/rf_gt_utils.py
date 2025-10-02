
import os
import numpy as np

class A2D2Calibration:
    """A2D2 보정 파일을 읽고 좌표 변환을 수행하는 클래스"""
    def __init__(self, calib_path):
        self.data = self._load_calib_file(calib_path)
        self.R0_rect = self._get_matrix('R0_rect')
        self.Tr_velo_to_cam = self._get_matrix('Tr_velo_to_cam')

    def _load_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError: pass
        return data

    def _get_matrix(self, key):
        matrix = self.data[key].reshape(3, -1)
        if matrix.shape[1] == 3:
            homo_matrix = np.eye(4)
            homo_matrix[:3, :3] = matrix
            return homo_matrix
        elif matrix.shape[1] == 4:
            homo_matrix = np.vstack([matrix, [0, 0, 0, 1]])
            return homo_matrix
        return matrix

    def cam_to_lidar(self, points_cam):
        """카메라 좌표계 포인트를 LiDAR 좌표계로 변환"""
        points_cam_homo = np.hstack((points_cam, np.ones((len(points_cam), 1))))
        transform_matrix = np.linalg.inv(self.R0_rect @ self.Tr_velo_to_cam)
        points_lidar = points_cam_homo @ transform_matrix.T
        return points_lidar[:, :3]

def get_a2d2_gt_boxes(label_path, calib_path, bev_x_range, bev_y_range, resolution):
    """
    A2D2 라벨과 보정 파일을 읽어, 회전된 BEV 2D 박스의 네 꼭짓점을 반환합니다.
    """
    if not os.path.exists(label_path) or not os.path.exists(calib_path):
        return []
        
    calibration = A2D2Calibration(calib_path)
    gt_boxes = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if line[0] in ['DontCare', 'Misc']: continue

            class_name = line[0]
            h, w, l = float(line[8]), float(line[9]), float(line[10])
            cx_cam, cy_cam, cz_cam = float(line[11]), float(line[12]), float(line[13])
            ry_cam = float(line[14])

            center_cam = np.array([[cx_cam, cy_cam - h / 2, cz_cam]])
            center_lidar = calibration.cam_to_lidar(center_cam)[0]
            ry_lidar = -ry_cam - np.pi / 2

            corners_bev = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
            rotation_matrix = np.array([[np.cos(ry_lidar), -np.sin(ry_lidar)], [np.sin(ry_lidar), np.cos(ry_lidar)]])
            rotated_corners = corners_bev @ rotation_matrix.T + center_lidar[:2]

            pixel_corners = []
            for corner in rotated_corners:
                px, py = corner[0], corner[1]
                pixel_y = int((bev_x_range[1] - px) / resolution)
                pixel_x = int((bev_y_range[1] - py) / resolution)
                pixel_corners.append([pixel_x, pixel_y])

            gt_boxes.append({'corners': np.array(pixel_corners), 'class': class_name})
            
    return gt_boxes