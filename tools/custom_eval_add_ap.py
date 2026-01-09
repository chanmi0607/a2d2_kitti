import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import calibration_kitti


# --- 설정값 (사용자 환경에 맞게 수정) ---
PRED_INFO_PATH = 'output/a2d2_models/pointpillar/pointpillar_dataconfig/eval/eval_epoch_200/result.pkl'
LABEL_PATH = 'data/a2d2/training/label_2'
CALIB_PATH = 'data/a2d2/training/calib'
VAL_FILE_PATH = 'data/a2d2/ImageSets/val.txt'

CLASS_NAMES = ['Car', 'Pedestrian', 'Truck']
IOU_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 30.0

# mAP 계산 시 너무 낮은 score까지 다 쓰면 느려질 수 있어서 컷(원하면 0.0으로)
MAP_MIN_SCORE = 0.0

PLOT_PR_CURVES = True
PR_SAVE_DIR = "/home/a/Pictures/pr_curves"

ANALYZE_CLASS = 'Pedestrian'     # TP/FP score 분석할 클래스
ANALYSIS_DIR = 'ped_score_analysis'
HIST_BINS = 50
# ------------------------------------


def get_label_info(label_path, calib_path, frame_id):
    """
    OpenPCDet 데이터로딩 코드에서 발견된 '진짜' 변환 규칙을 적용합니다.
    """
    label_file = Path(label_path) / f'{frame_id}.txt'
    calib_file = Path(calib_path) / f'{frame_id}.txt'

    if not label_file.exists() or not calib_file.exists():
        return np.array([]), np.array([])

    calib = calibration_kitti.Calibration(calib_file)

    gt_boxes_lidar_final = []
    gt_names = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            class_name = parts[0]

            if class_name in CLASS_NAMES and class_name != 'DontCare':
                gt_names.append(class_name)

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
        return np.array([]), np.array([])

    return np.array(gt_boxes_lidar_final), np.array(gt_names)


def load_val_frames(file_path):
    with open(file_path, 'r') as f:
        frame_ids = {line.strip() for line in f if line.strip()}
    return frame_ids


def compute_ap(recalls, precisions):
    """
    VOC-style AP (precision envelope + area under PR curve)
    recalls, precisions: 1D numpy arrays (monotonic recall expected after cumulative)
    """
    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # area under curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다. (iou3d_nms_utils.boxes_iou3d_gpu 사용)")

    device = torch.device('cuda')

    with open(PRED_INFO_PATH, 'rb') as f:
        pred_infos = pickle.load(f)
    print(f"총 {len(pred_infos)}개의 프레임에 대한 예측 결과를 로드했습니다.")

    val_frame_ids = load_val_frames(VAL_FILE_PATH)
    print(f"평가 대상: {len(val_frame_ids)}개의 검증 프레임 ID를 로드했습니다.")

    filtered_preds = [info for info in pred_infos if info['frame_id'] in val_frame_ids]
    print(f"로드된 예측 결과 중 {len(filtered_preds)}개가 검증 세트에 포함됩니다.")

    # ------------------------------------------------------------
    # 1) GT/Pred를 프레임 단위로 미리 로딩 (중복 IO 방지)
    # ------------------------------------------------------------
    gt_by_frame = {}    # frame_id -> (gt_boxes, gt_names) (distance filtered)
    pred_by_frame = {}  # frame_id -> (pred_boxes, pred_scores, pred_names) (distance filtered)

    for pred_info in tqdm(filtered_preds, desc="GT/Pred 로딩"):
        frame_id = pred_info['frame_id']

        gt_boxes, gt_names = get_label_info(LABEL_PATH, CALIB_PATH, frame_id)
        if gt_boxes.shape[0] > 0:
            gt_dist = np.linalg.norm(gt_boxes[:, :2], axis=1)
            m = gt_dist <= DISTANCE_THRESHOLD
            gt_boxes, gt_names = gt_boxes[m], gt_names[m]
        gt_by_frame[frame_id] = (gt_boxes, gt_names)

        pb = pred_info['boxes_lidar']
        ps = pred_info['score']
        pn = pred_info['name']

        if pb.shape[0] > 0:
            pd = np.linalg.norm(pb[:, :2], axis=1)
            m = pd <= DISTANCE_THRESHOLD
            pb, ps, pn = pb[m], ps[m], pn[m]
        pred_by_frame[frame_id] = (pb, ps, pn)

    # ------------------------------------------------------------
    # 2) 기존 고정 threshold 성능(Precision/Recall/F1) 계산
    # ------------------------------------------------------------
    stats_by_class = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in CLASS_NAMES}

    for frame_id in tqdm(gt_by_frame.keys(), desc="고정 Threshold 평가"):
        gt_boxes, gt_names = gt_by_frame[frame_id]
        pred_boxes_all, pred_scores_all, pred_names_all = pred_by_frame[frame_id]

        # confidence threshold 적용 (고정 지표용)
        mask = pred_scores_all >= CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes_all[mask]
        pred_names = pred_names_all[mask]

        for class_name in CLASS_NAMES:
            gt_mask = (gt_names == class_name)
            pred_mask = (pred_names == class_name)

            class_gt_boxes = gt_boxes[gt_mask]
            class_pred_boxes = pred_boxes[pred_mask]

            num_gts = class_gt_boxes.shape[0]
            num_preds = class_pred_boxes.shape[0]

            if num_gts == 0:
                stats_by_class[class_name]['FP'] += num_preds
                continue

            if num_preds == 0:
                stats_by_class[class_name]['FN'] += num_gts
                continue

            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                torch.from_numpy(class_pred_boxes).float().to(device),
                torch.from_numpy(class_gt_boxes).float().to(device)
            ).cpu().numpy()

            matched_gt = np.zeros(num_gts, dtype=bool)
            tp_cnt = 0

            for i in range(num_preds):
                max_iou = -1.0
                best_j = -1
                for j in range(num_gts):
                    if (not matched_gt[j]) and (iou_matrix[i, j] > max_iou):
                        max_iou = iou_matrix[i, j]
                        best_j = j
                if max_iou >= IOU_THRESHOLD:
                    tp_cnt += 1
                    matched_gt[best_j] = True

            stats_by_class[class_name]['TP'] += tp_cnt
            stats_by_class[class_name]['FP'] += (num_preds - tp_cnt)
            stats_by_class[class_name]['FN'] += (num_gts - np.sum(matched_gt))

    # ------------------------------------------------------------
    # 3) mAP(AP) 계산: score 내림차순 정렬 기반 (표준)
    # ------------------------------------------------------------
    dets_by_class = {c: [] for c in CLASS_NAMES}  # list of (score, frame_id, box)
    gt_boxes_by_frame_class = {fid: {c: np.zeros((0, 7), dtype=np.float32) for c in CLASS_NAMES}
                               for fid in gt_by_frame.keys()}

    # GT 정리
    for fid, (gboxes, gnames) in gt_by_frame.items():
        for c in CLASS_NAMES:
            m = (gnames == c)
            gt_boxes_by_frame_class[fid][c] = gboxes[m].astype(np.float32)

    # Pred 정리 (mAP용: confidence sweep 대신 score sorting)
    for fid, (pboxes, pscores, pnames) in pred_by_frame.items():
        if pboxes.shape[0] == 0:
            continue
        keep = pscores >= MAP_MIN_SCORE
        pboxes, pscores, pnames = pboxes[keep], pscores[keep], pnames[keep]
        for c in CLASS_NAMES:
            m = (pnames == c)
            if np.any(m):
                for box, sc in zip(pboxes[m], pscores[m]):
                    dets_by_class[c].append((float(sc), fid, box.astype(np.float32)))

    # 매칭 플래그 (frame_id, class) -> bool array
    matched = {fid: {c: np.zeros(gt_boxes_by_frame_class[fid][c].shape[0], dtype=bool)
                     for c in CLASS_NAMES}
               for fid in gt_by_frame.keys()}
    
    pr_by_class = {}
    ap_by_class = {}
    op_point_by_class = {}
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    analysis = {
        'tp_scores': [],
        'fp_scores': [],
        'fp_iou_max': [],   # FP인 pred가 GT와 얼마나 근접했는지(max IoU)
        'tp_iou': []        # (선택) TP들의 IoU 분포도 같이 보고 싶으면
    }

    for c in CLASS_NAMES:
        dets = dets_by_class[c]
        # GT 총 개수
        npos = int(sum(gt_boxes_by_frame_class[fid][c].shape[0] for fid in gt_by_frame.keys()))
        if npos == 0:
            ap_by_class[c] = 0.0
            continue

        # score 내림차순 정렬
        dets.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, (score, fid, box) in enumerate(tqdm(dets, desc=f"AP 계산 중: {c}", leave=False)):
            gts = gt_boxes_by_frame_class[fid][c]
            if gts.shape[0] == 0:
                fp[i] = 1.0
                continue

            # IoU(1 x Ng) 계산
            ious = iou3d_nms_utils.boxes_iou3d_gpu(
                torch.from_numpy(box[None, :]).float().to(device),
                torch.from_numpy(gts).float().to(device)
            ).cpu().numpy()[0]

            max_iou_all = float(ious.max()) if ious.size > 0 else 0.0

            # 아직 매칭 안된 GT 중 최대 IoU 찾기
            best_j = -1
            best_iou = -1.0
            for j in range(gts.shape[0]):
                if not matched[fid][c][j] and ious[j] > best_iou:
                    best_iou = ious[j]
                    best_j = j

            if best_iou >= IOU_THRESHOLD:
                tp[i] = 1.0
                matched[fid][c][best_j] = True

                if c == ANALYZE_CLASS:
                    analysis['tp_scores'].append(float(score))
                    analysis['tp_iou'].append(float(best_iou))   # TP는 best_iou가 실제 매칭 IoU
            else:
                fp[i] = 1.0
                if c == ANALYZE_CLASS:
                    analysis['fp_scores'].append(float(score))
                    analysis['fp_iou_max'].append(max_iou_all)  # FP는 "가장 가까운 GT와의 IoU"로 분해용

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(npos, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        pr_by_class[c] = {
            "recall": recalls.copy(),
            "precision": precisions.copy()
        }

        # score 내림차순 리스트 만들었으니 scores도 같이 뽑아둠
        scores_sorted = np.array([d[0] for d in dets], dtype=np.float32)

        # ✅ 운영점(=score threshold) 위치 찾기: score >= CONFIDENCE_THRESHOLD인 마지막 인덱스
        op_idx = np.where(scores_sorted >= CONFIDENCE_THRESHOLD)[0]
        if len(op_idx) == 0:
            op_point = (0.0, 0.0)  # (recall, precision)
        else:
            k = op_idx[-1]  # threshold 이상인 마지막 det까지 포함한 지점
            op_point = (float(recalls[k]), float(precisions[k]))

        # 나중에 plot할 때 쓰려고 저장
        op_point_by_class[c] = op_point

        ap = compute_ap(recalls, precisions)
        ap_by_class[c] = ap

    mAP = float(np.mean([ap_by_class[c] for c in CLASS_NAMES])) if len(CLASS_NAMES) > 0 else 0.0

    # ------------------------------------------------------------
    # 4) 출력
    # ------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- 최종 평가 결과 ---")
    print(f"IoU 임계값: {IOU_THRESHOLD}, Confidence 점수 임계값(고정 지표용): {CONFIDENCE_THRESHOLD}")
    print(f"거리 제한: {DISTANCE_THRESHOLD}m 이내")
    print("-" * 90)
    print(f"{'클래스':<16} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-" * 90)

    for class_name, stats in stats_by_class.items():
        tpv = stats['TP']
        fpv = stats['FP']
        fnv = stats['FN']

        precision = tpv / (tpv + fpv) if (tpv + fpv) > 0 else 0.0
        recall = tpv / (tpv + fnv) if (tpv + fnv) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{class_name:<16} | {(precision * 100):11.4f}% | {(recall * 100):11.2f}% | {f1_score:10.4f} | {tpv:6} | {fpv:6} | {fnv:6}")

    print("-" * 90)
    print(f"AP@{IOU_THRESHOLD:.2f} (distance<= {DISTANCE_THRESHOLD}m, score>={MAP_MIN_SCORE})")
    for c in CLASS_NAMES:
        print(f"  {c:<12}: {ap_by_class[c]:.4f}")
    print(f"mAP@{IOU_THRESHOLD:.2f}: {mAP:.4f}")
    print("=" * 90)
    if len(analysis['tp_scores']) + len(analysis['fp_scores']) > 0:
        tp_scores = np.array(analysis['tp_scores'], dtype=np.float32)
        fp_scores = np.array(analysis['fp_scores'], dtype=np.float32)
        fp_iou_max = np.array(analysis['fp_iou_max'], dtype=np.float32)

        print("\n" + "=" * 90)
        print(f"[{ANALYZE_CLASS} Score/FP 분석] (distance<={DISTANCE_THRESHOLD}m, score>={MAP_MIN_SCORE})")
        print(f"TP count: {len(tp_scores)}, FP count: {len(fp_scores)}")

        if len(tp_scores) > 0:
            print(f"TP score: mean={tp_scores.mean():.4f}, median={np.median(tp_scores):.4f}, p10={np.percentile(tp_scores,10):.4f}, p90={np.percentile(tp_scores,90):.4f}")
        if len(fp_scores) > 0:
            print(f"FP score: mean={fp_scores.mean():.4f}, median={np.median(fp_scores):.4f}, p10={np.percentile(fp_scores,10):.4f}, p90={np.percentile(fp_scores,90):.4f}")

        # FP IoU 구간별 분해
        b0 = int(np.sum(fp_iou_max < 0.1))
        b1 = int(np.sum((fp_iou_max >= 0.1) & (fp_iou_max < 0.3)))
        b2 = int(np.sum((fp_iou_max >= 0.3) & (fp_iou_max < 0.5)))
        b3 = int(np.sum(fp_iou_max >= 0.5))
        print("[FP max IoU buckets]")
        print(f"  <0.1     : {b0}")
        print(f"  0.1-0.3  : {b1}")
        print(f"  0.3-0.5  : {b2}")
        print(f"  >=0.5    : {b3}  (보통은 NMS/중복/매칭경합 때문에 생길 수 있음)")

        # 파일 저장(npz)
        np.savez(
            os.path.join(ANALYSIS_DIR, f"{ANALYZE_CLASS}_tp_fp_scores.npz"),
            tp_scores=tp_scores,
            fp_scores=fp_scores,
            fp_iou_max=fp_iou_max,
            tp_iou=np.array(analysis['tp_iou'], dtype=np.float32)
        )
        print(f"[INFO] npz 저장: {os.path.join(ANALYSIS_DIR, f'{ANALYZE_CLASS}_tp_fp_scores.npz')}")

        # 히스토그램 저장(선택: matplotlib 있을 때)
        if HAS_PLT:
            # (1) TP/FP score 분포
            plt.figure()
            if len(tp_scores) > 0:
                plt.hist(tp_scores, bins=HIST_BINS, alpha=0.6, label='TP')
            if len(fp_scores) > 0:
                plt.hist(fp_scores, bins=HIST_BINS, alpha=0.6, label='FP')
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.title(f"{ANALYZE_CLASS} score distribution (TP vs FP)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(ANALYSIS_DIR, f"{ANALYZE_CLASS}_score_hist.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # (2) FP max IoU 분포
            plt.figure()
            if len(fp_iou_max) > 0:
                plt.hist(fp_iou_max, bins=HIST_BINS)
            plt.xlabel("max IoU to any GT")
            plt.ylabel("Count")
            plt.title(f"{ANALYZE_CLASS} FP max IoU distribution")
            plt.grid(True)
            plt.savefig(os.path.join(ANALYSIS_DIR, f"{ANALYZE_CLASS}_fp_iou_hist.png"), dpi=200, bbox_inches="tight")
            plt.close()

            print(f"[INFO] 이미지 저장: {ANALYSIS_DIR}/{ANALYZE_CLASS}_score_hist.png")
            print(f"[INFO] 이미지 저장: {ANALYSIS_DIR}/{ANALYZE_CLASS}_fp_iou_hist.png")
        else:
            print("[WARN] matplotlib이 없어 히스토그램 이미지는 저장하지 않았습니다. (pip install matplotlib)")


    if PLOT_PR_CURVES:
        os.makedirs(PR_SAVE_DIR, exist_ok=True)

        for c in CLASS_NAMES:
            if c not in pr_by_class:
                continue

            r = pr_by_class[c]["recall"]
            p = pr_by_class[c]["precision"]

            plt.figure()
            plt.plot(r, p, linewidth=2, label=c)
            op_r, op_p = op_point_by_class.get(c, (None, None))
            if op_r is not None:
                plt.scatter([op_r], [op_p], s=80)  # 색 지정 안 함(기본)
                plt.annotate(f"thr={CONFIDENCE_THRESHOLD}\n(P={op_p:.2f}, R={op_r:.2f})",
                            (op_r, op_p),
                            textcoords="offset points", xytext=(10, -10))  
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {c} (IoU={IOU_THRESHOLD}, dist<={DISTANCE_THRESHOLD}m)")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()

            save_path = os.path.join(PR_SAVE_DIR, f"pr_curve_{c}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            

        print(f"[INFO] PR curve 이미지 저장 완료: {PR_SAVE_DIR}/pr_curve_*.png")



if __name__ == '__main__':
    main()
