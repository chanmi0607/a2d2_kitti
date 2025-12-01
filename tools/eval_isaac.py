import argparse
import numpy as np
import torch
import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt   # <<< 추가

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# =========================================================================
# [핵심 설정] 클래스별 탐색할 Confidence 범위 설정
#   - 여기서는 [min_score, max_score] 안에서 min_score만 스윕하며
#     AP >= 0.95가 되는 최적 min_score를 찾음
# =========================================================================
CLASS_CONFIG = {
    'Truck': {
        'min_score': 0.3,  # 탐색 시작점 (0.3 ~ 1.0 사이에서 최적 min_score 탐색)
        'max_score': 1.0
    },
    'Pedestrian': {
        'min_score': 0.4,  # (0.4 ~ 1.0 사이에서 탐색)
        'max_score': 1.0
    }
}
FIXED_IOU_THRESH = 0.5
TARGET_AP = 0.95       # 목표 AP

# =========================================================================

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/isaacsim_models/second.yaml')
    parser.add_argument('--ckpt', type=str, default='output/isaacsim_models/second/default/ckpt/checkpoint_epoch_10.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dist_thresh', type=float, default=30.0)
    args = parser.parse_args()
    return args

def filter_by_distance(boxes, max_dist):
    if boxes.shape[0] == 0: 
        return boxes
    dist = torch.sqrt(boxes[:, 0]**2 + boxes[:, 1]**2)
    return boxes[dist <= max_dist]

def compute_ap(recall, precision):
    """
    연속형 AP (VOC2010+ 스타일)
    - recall, precision: 1D numpy array
    - precision-envelope를 만든 뒤, recall 축에 대해 적분
    """
    if len(recall) == 0:
        return 0.0

    # 1) recall 기준으로 정렬 (혹시라도 순서가 뒤섞여 있을 수 있으니)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    # 2) precision-envelope 만들기 (오른쪽에서 왼쪽으로 최대값 누적)
    #    → Precision-Recall 곡선을 단조 감소 형태로 보정
    mpre = precision.copy()
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 3) recall 앞뒤로 0,1 경계 추가 (필수는 아니지만 더 안정적)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([mpre[0]], mpre, [0.0]))

    # 4) 사다리꼴 적분(trapezoidal rule)로 AP 계산
    #    ∑ (recall[i] - recall[i-1]) * precision[i]
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap
def get_pr_curve_for_range(results, num_gt, min_s, max_s):
    """
    선택한 score 구간 [min_s, max_s] 에 대한
    PR 곡선(precision, recall)과 AP를 반환.
    """
    filtered = [(s, tp) for (s, tp) in results if (min_s <= s <= max_s)]

    if len(filtered) == 0 or num_gt == 0:
        return np.array([]), np.array([]), 0.0

    # 점수 내림차순 정렬
    filtered.sort(key=lambda x: x[0], reverse=True)

    tp_list = np.array([tp for (_, tp) in filtered], dtype=np.float32)
    fp_list = 1.0 - tp_list

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)

    # 연속형 AP (이미 정의된 compute_ap 사용)
    ap = compute_ap(recall, precision)

    return recall, precision, ap

# -------------------------------------------------------------------------
# [변경] 배치 평가 함수: threshold를 전혀 사용하지 않고
#       해당 class의 모든 prediction에 대해 (score, is_tp)만 계산
# -------------------------------------------------------------------------
def eval_one_batch(gt_boxes, pred_boxes, pred_scores, pred_labels, class_id):
    """
    gt_boxes: (N_gt, >=8) [x,y,z,dx,dy,dz,heading,class_id]
    pred_boxes: (N_pred, 7)
    pred_scores: (N_pred,)
    pred_labels: (N_pred,)
    """
    # 해당 클래스의 prediction만 선택 (threshold 적용 없음)
    pred_mask = (pred_labels == class_id)
    curr_pred = pred_boxes[pred_mask]
    curr_scores = pred_scores[pred_mask]

    # GT도 해당 클래스만 선택
    gt_mask = (gt_boxes[:, -1] == class_id)
    curr_gt = gt_boxes[gt_mask][:, :7]

    num_gt = curr_gt.shape[0]
    results = []  # [(score, is_tp), ...]

    # 예측이 없으면 바로 반환
    if curr_pred.shape[0] == 0:
        return results, num_gt

    # GT가 하나도 없으면 모든 예측은 FP
    if num_gt == 0:
        for score in curr_scores:
            results.append((float(score), 0))
        return results, num_gt

    # IoU 매트릭스 계산
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(curr_pred, curr_gt)

    # 점수 내림차순 정렬
    sorted_indices = torch.argsort(curr_scores, descending=True)
    gt_assigned = torch.zeros(num_gt, dtype=torch.bool, device=curr_gt.device)

    for pred_idx in sorted_indices:
        score = float(curr_scores[pred_idx])
        max_iou, max_gt_idx = torch.max(iou_matrix[pred_idx], dim=0)

        if max_iou >= FIXED_IOU_THRESH:
            if not gt_assigned[max_gt_idx]:
                results.append((score, 1))  # TP
                gt_assigned[max_gt_idx] = True
            else:
                results.append((score, 0))  # 같은 GT에 중복 매칭 → FP
        else:
            results.append((score, 0))      # IoU 미달 → FP

    return results, num_gt

def compute_metrics_for_range(results, num_gt, min_s, max_s):
    """
    results: [(score, is_tp), ...]  (모든 prediction, threshold 없음 상태)
    num_gt: 전체 GT 수
    min_s, max_s: 사용할 score 구간
    """
    # 선택한 score 구간에 해당하는 prediction만 사용
    filtered = [(s, tp) for (s, tp) in results if (min_s <= s <= max_s)]

    if len(filtered) == 0 or num_gt == 0:
        return 0.0, 0.0, 0.0, 0.0  # ap, prec, rec, f1

    # 점수 내림차순 정렬
    filtered.sort(key=lambda x: x[0], reverse=True)

    tp_list = np.array([tp for (_, tp) in filtered], dtype=np.float32)
    fp_list = 1.0 - tp_list

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)

    ap = compute_ap(recall, precision)  # 연속형 AP

    final_prec = precision[-1]
    final_rec = recall[-1]
    f1 = 2 * (final_prec * final_rec) / (final_prec + final_rec + 1e-6)

    return ap, final_prec, final_rec, f1

# -------------------------------------------------------------------------

def main():
    args = parse_config()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # [변경] 여기서는 "raw" 결과만 저장 (threshold 없음)
    all_eval_results = {cls: [] for cls in cfg.CLASS_NAMES}  # cls별 [(score, is_tp), ...]
    total_gt_counts = {cls: 0 for cls in cfg.CLASS_NAMES}    # cls별 GT 개수

    logger.info("Evaluating (collecting raw detections without confidence thresholds)...")
    for cls, conf in CLASS_CONFIG.items():
        logger.info(f" Search range for {cls}: {conf['min_score']} ~ {conf['max_score']}")

    with torch.no_grad():
        for i, batch_dict in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            for key, val in batch_dict.items():
                if not isinstance(val, np.ndarray):
                    continue
                if val.dtype.kind in {'U', 'S', 'O'}:
                    continue
                batch_dict[key] = torch.from_numpy(val).float().cuda()

            pred_dicts, _ = model.forward(batch_dict)

            for batch_idx, pred_dict in enumerate(pred_dicts):
                pred_boxes = pred_dict['pred_boxes']
                pred_scores = pred_dict['pred_scores']
                pred_labels = pred_dict['pred_labels']

                gt_boxes = batch_dict['gt_boxes'][batch_idx]
                gt_boxes = gt_boxes[gt_boxes[:, 3] > 0]

                # 거리 필터링
                if gt_boxes.shape[0] > 0:
                    gt_boxes = filter_by_distance(gt_boxes, args.dist_thresh)
                if pred_boxes.shape[0] > 0:
                    dist_mask = torch.sqrt(pred_boxes[:, 0]**2 + pred_boxes[:, 1]**2) <= args.dist_thresh
                    pred_boxes = pred_boxes[dist_mask]
                    pred_scores = pred_scores[dist_mask]
                    pred_labels = pred_labels[dist_mask]

                # 클래스별로 (score, is_tp) 계산 (threshold 없음)
                for class_idx, class_name in enumerate(cfg.CLASS_NAMES):
                    batch_res, num_gt = eval_one_batch(
                        gt_boxes, pred_boxes, pred_scores, pred_labels,
                        class_idx + 1
                    )
                    all_eval_results[class_name].extend(batch_res)
                    total_gt_counts[class_name] += num_gt

    # -----------------------------------------------------------
    # [최종 결과: AP 0.95 이상이 되는 최적 구간 탐색 후 출력]
    # -----------------------------------------------------------
    print(f"\n=========================================================================")
    print(f" Final Performance Report (Dist < {args.dist_thresh}m)")
    print(f" Auto-searched confidence ranges for TARGET AP >= {TARGET_AP:.2f}")
    print(f"=========================================================================")

    table = PrettyTable()
    table.field_names = ["Class", "Selected Range", "AP", "Precision", "Recall", "F1-Score", f"Pass ({TARGET_AP:.2f}+)"]

    for class_name in cfg.CLASS_NAMES:
        results = all_eval_results[class_name]
        num_gt = total_gt_counts[class_name]

        c_conf = CLASS_CONFIG.get(class_name, {'min_score': 0.0, 'max_score': 1.0})
        search_min = c_conf['min_score']
        search_max = c_conf['max_score']

        if len(results) == 0 or num_gt == 0:
            range_str = f"{search_min:.2f}~{search_max:.2f}"
            table.add_row([class_name, range_str, "0.0000", "-", "-", "-", "FAIL ❌"])
            continue

        # ------------------------------------------------------------------
        # [핵심] min, max 둘 다 스윕
        #   - step: 0.05 (원하면 0.01로 더 촘촘하게)
        #   - AP >= TARGET_AP 만족하는 구간 중
        #       1) 가장 폭이 좁은 구간 (max-min 최소)
        #       2) 동일 폭이면 min이 더 작은 구간
        #     을 선택
        #   + 동시에 (min,max) → AP 를 2D grid에 저장해서 heatmap으로 시각화
        # ------------------------------------------------------------------
        step = 0.05
        thr_values = np.arange(search_min, search_max + 1e-6, step)
        n_thr = len(thr_values)

        # AP 저장용 grid (min index = i, max index = j)
        ap_grid = np.full((n_thr, n_thr), np.nan, dtype=np.float32)

        best_pass = None    # AP >= TARGET_AP 인 구간 중 가장 좋은 것
        best_overall = None # 전체 AP 최대 구간 (PASS 없을 때용)

        for i, thr_min in enumerate(thr_values):
            for j, thr_max in enumerate(thr_values[i:], start=i):
                ap, prec, rec, f1 = compute_metrics_for_range(
                    results, num_gt, thr_min, thr_max
                )

                ap_grid[i, j] = ap  # heatmap용 저장

                # 전체 중 AP 최대 구간 기록
                if (best_overall is None) or (ap > best_overall['ap']):
                    best_overall = {
                        'min': thr_min, 'max': thr_max,
                        'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1
                    }

                # AP >= TARGET_AP 만족하는 후보
                if ap >= TARGET_AP:
                    width = thr_max - thr_min
                    if best_pass is None:
                        best_pass = {
                            'min': thr_min, 'max': thr_max,
                            'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1,
                            'width': width
                        }
                    else:
                        # 폭이 더 좁거나, 폭이 같으면 min이 더 작은 구간 선택
                        if (width < best_pass['width']) or (
                            np.isclose(width, best_pass['width']) and thr_min < best_pass['min']
                        ):
                            best_pass = {
                                'min': thr_min, 'max': thr_max,
                                'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1,
                                'width': width
                            }

        # 이제 best_pass가 있으면 PASS, 없으면 best_overall로 FAIL
        if best_pass is not None:
            sel = best_pass
            pass_status = "PASS ✅"
        else:
            sel = best_overall
            pass_status = "FAIL ❌"

        sel_min = sel['min']
        sel_max = sel['max']
        ap = sel['ap']
        final_prec = sel['prec']
        final_rec = sel['rec']
        f1 = sel['f1']

        range_str = f"{sel_min:.2f}~{sel_max:.2f}"

        table.add_row([
            class_name,
            range_str,
            f"{ap:.4f}",
            f"{final_prec:.4f}",
            f"{final_rec:.4f}",
            f"{f1:.4f}",
            pass_status
        ])

        # ------------------------------------------------------------------
        # [시각화] AP heatmap 저장
        #   - x축: max_score
        #   - y축: min_score
        #   - 선택된 구간(sel_min, sel_max)을 X 마커로 표시
        # ------------------------------------------------------------------
        plt.figure(figsize=(6, 5))

        # ap_grid의 i=thr_min index, j=thr_max index
        extent = [search_min, search_max, search_min, search_max]  # [x_min, x_max, y_min, y_max]
        img = plt.imshow(
            ap_grid,
            origin='lower',
            extent=extent,
            aspect='auto'
        )
        plt.colorbar(img, label='AP')

        plt.xlabel('max_score')
        plt.ylabel('min_score')
        plt.title(f'AP Heatmap - {class_name}')

        # 선택된 range 찍기 (max_score가 x축, min_score가 y축)
        plt.scatter([sel_max], [sel_min], marker='x')

        # 파일로 저장 (현재 실행 위치에 저장됨)
        out_name = f'ap_heatmap_{class_name}.png'
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved AP heatmap for {class_name} -> {out_name}")
        # ------------------------------------------------------------------
        # [시각화] 선택된 range [sel_min, sel_max]에 대한
        #          Precision-Recall 곡선과, 그 아래 면적(=AP)을 그림
        # ------------------------------------------------------------------
        recall_curve, precision_curve, ap_curve = get_pr_curve_for_range(
            results, num_gt, sel_min, sel_max
        )

        if recall_curve.size > 0:
            plt.figure(figsize=(5, 4))
            # PR 곡선
            plt.plot(recall_curve, precision_curve, label='PR curve')
            # 곡선 아래 면적 (AP) 채우기
            plt.fill_between(recall_curve, precision_curve, alpha=0.3)

            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(
                f'{class_name} (range={sel_min:.2f}~{sel_max:.2f}, AP={ap_curve:.3f})'
            )
            plt.grid(True)
            plt.legend()

            out_name = f'pr_area_{class_name}.png'
            plt.savefig(out_name, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved PR area plot for {class_name} -> {out_name}")
        else:
            print(f"[WARN] No valid PR curve for {class_name} in range {sel_min:.2f}~{sel_max:.2f}")
            
    print(table)
    print("\n")

if __name__ == '__main__':
    main()
