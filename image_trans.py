import os
import shutil
import glob

# 경로 설정
base_path = os.path.expanduser('~/OpenPCDet/data/a2d2/camera_lidar_semantic_bboxes')
target_path = os.path.expanduser('~/OpenPCDet/data/a2d2/image_2')

# 저장할 폴더 생성 (없으면 생성)
if not os.path.exists(target_path):
    os.makedirs(target_path)
    print(f"폴더 생성됨: {target_path}")

# 모든 하위 폴더에서 png 파일 찾기
search_pattern = os.path.join(base_path, '*', 'camera', 'cam_front_center', '*.png')
files = glob.glob(search_pattern)

print(f"총 {len(files)}개의 파일을 발견했습니다. 복사를 시작합니다...")

count = 0
for src_file in files:
    # 파일명 파싱
    # 예: 20180807145028_camera_frontcenter_000000091.png
    filename = os.path.basename(src_file)
    
    # '_'로 분리 후 마지막 부분(숫자ID.png) 가져오기 -> "000000091.png"
    new_filename = filename.split('_')[-1]
    
    dst_file = os.path.join(target_path, new_filename)
    
    # 파일 복사 (메타데이터 유지)
    shutil.copy2(src_file, dst_file)
    
    count += 1
    if count % 100 == 0:
        print(f"{count}개 처리 완료... (현재 파일: {new_filename})")

print("-------------")
print("작업 완료!")
print(f"총 {count}개의 파일이 {target_path} 로 복사되었습니다.")