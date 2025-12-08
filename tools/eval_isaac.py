import argparse
import numpy as np
import torch
import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# =========================================================================
# [설정] 클래스별 탐색 범위 및 분석 기준
# =========================================================================
CLASS_CONFIG = {
    'Truck': {
        'min_score': 0.3, 
        'max_score': 1.0
    },
    'Pedestrian': {
        'min_score': 0.4,
        'max_score': 1.0
    }
}
FIXED_IOU_THRESH = 0.5
TARGET_AP = 0.95

# =========================================================================

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/isaacsim_models/second2.yaml')
    parser.add_argument('--ckpt', type=str, default='output/isaacsim_models/second/isaac_experiment/ckpt/checkpoint_epoch_162.pth')
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
    if len(recall) == 0:
        return 0.0
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    mpre = precision.copy()
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([mpre[0]], mpre, [0.0]))
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def get_pr_curve_for_range(results, num_gt, min_s, max_s):
    filtered = [(s, tp) for (s, tp) in results if (min_s <= s <= max_s)]
    if len(filtered) == 0 or num_gt == 0:
        return np.array([]), np.array([]), 0.0
    filtered.sort(key=lambda x: x[0], reverse=True)
    tp_list = np.array([tp for (_, tp) in filtered], dtype=np.float32)
    fp_list = 1.0 - tp_list
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)
    ap = compute_ap(recall, precision)
    return recall, precision, ap

def eval_one_batch(gt_boxes, pred_boxes, pred_scores, pred_labels, class_id):
    # (기존 함수와 동일: 전체 통계용 Raw 데이터 수집)
    pred_mask = (pred_labels == class_id)
    curr_pred = pred_boxes[pred_mask]
    curr_scores = pred_scores[pred_mask]

    gt_mask = (gt_boxes[:, -1] == class_id)
    curr_gt = gt_boxes[gt_mask][:, :7]

    num_gt = curr_gt.shape[0]
    results = [] 

    if curr_pred.shape[0] == 0:
        return results, num_gt
    if num_gt == 0:
        for score in curr_scores:
            results.append((float(score), 0))
        return results, num_gt

    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(curr_pred, curr_gt)
    sorted_indices = torch.argsort(curr_scores, descending=True)
    gt_assigned = torch.zeros(num_gt, dtype=torch.bool, device=curr_gt.device)

    for pred_idx in sorted_indices:
        score = float(curr_scores[pred_idx])
        max_iou, max_gt_idx = torch.max(iou_matrix[pred_idx], dim=0)

        if max_iou >= FIXED_IOU_THRESH:
            if not gt_assigned[max_gt_idx]:
                results.append((score, 1)) 
                gt_assigned[max_gt_idx] = True
            else:
                results.append((score, 0)) 
        else:
            results.append((score, 0)) 

    return results, num_gt

# -------------------------------------------------------------------------
# [추가] 프레임별 불량 데이터 분석 함수
# -------------------------------------------------------------------------
def analyze_per_frame(gt_boxes, pred_boxes, pred_scores, pred_labels, class_id, min_score_thresh):
    """
    특정 프레임에서 해당 클래스에 대해 분석 (FP, FN 개수 리턴)
    여기서는 'min_score_thresh'를 기준으로 판단함.
    """
    # 1. Prediction 필터링 (Min Score 이상만)
    pred_mask = (pred_labels == class_id) & (pred_scores >= min_score_thresh)
    curr_pred = pred_boxes[pred_mask]
    curr_scores = pred_scores[pred_mask]

    # 2. GT 필터링
    gt_mask = (gt_boxes[:, -1] == class_id)
    curr_gt = gt_boxes[gt_mask][:, :7]
    num_gt = curr_gt.shape[0]
    
    # Init stats
    tp_count = 0
    fp_count = 0
    
    if curr_pred.shape[0] == 0:
        # 예측이 하나도 없으면 GT 개수만큼 FN
        return 0, 0, num_gt  # TP, FP, FN

    if num_gt == 0:
        # GT가 없는데 예측이 있으면 전부 FP
        return 0, curr_pred.shape[0], 0

    # 3. 매칭 로직 (간소화)
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(curr_pred, curr_gt)
    
    # Score 높은 순 정렬
    sorted_indices = torch.argsort(curr_scores, descending=True)
    gt_assigned = torch.zeros(num_gt, dtype=torch.bool, device=curr_gt.device)

    for pred_idx in sorted_indices:
        max_iou, max_gt_idx = torch.max(iou_matrix[pred_idx], dim=0)
        
        if max_iou >= FIXED_IOU_THRESH:
            if not gt_assigned[max_gt_idx]:
                tp_count += 1
                gt_assigned[max_gt_idx] = True
            else:
                fp_count += 1 # 중복 검출
        else:
            fp_count += 1 # IoU 낮음 (오검출)
            
    fn_count = num_gt - tp_count
    return tp_count, fp_count, fn_count

def compute_metrics_for_range(results, num_gt, min_s, max_s):
    # (기존과 동일)
    filtered = [(s, tp) for (s, tp) in results if (min_s <= s <= max_s)]
    if len(filtered) == 0 or num_gt == 0:
        return 0.0, 0.0, 0.0, 0.0 
    filtered.sort(key=lambda x: x[0], reverse=True)
    tp_list = np.array([tp for (_, tp) in filtered], dtype=np.float32)
    fp_list = 1.0 - tp_list
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)
    ap = compute_ap(recall, precision) 
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

    all_eval_results = {cls: [] for cls in cfg.CLASS_NAMES}
    total_gt_counts = {cls: 0 for cls in cfg.CLASS_NAMES}

    # [추가] 프레임별 불량 데이터 저장용 리스트
    # 구조: {'frame_id': str, 'class': str, 'fp': int, 'fn': int, 'total_gt': int}
    bad_frame_logs = []

    logger.info("Evaluating (collecting raw detections)...")

    with torch.no_grad():
        for i, batch_dict in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            # 1. Load Data
            for key, val in batch_dict.items():
                if not isinstance(val, np.ndarray):
                    continue
                if val.dtype.kind in {'U', 'S', 'O'}:
                    continue
                batch_dict[key] = torch.from_numpy(val).float().cuda()

            # [중요] Frame ID 가져오기 (문자열이나 숫자)
            frame_ids = batch_dict.get('frame_id', [f'sample_{i}'])

            # 2. Forward
            pred_dicts, _ = model.forward(batch_dict)

            # 3. Process Batch
            for batch_idx, pred_dict in enumerate(pred_dicts):
                pred_boxes = pred_dict['pred_boxes']
                pred_scores = pred_dict['pred_scores']
                pred_labels = pred_dict['pred_labels']
                
                # frame_id 처리 (Batch size > 1일 경우 대응)
                if isinstance(frame_ids, (list, tuple, np.ndarray)):
                    current_frame_id = str(frame_ids[batch_idx])
                else:
                    current_frame_id = str(frame_ids)

                # GT 확인 (없으면 skip)
                if 'gt_boxes' not in batch_dict:
                    continue
                
                gt_boxes = batch_dict['gt_boxes'][batch_idx]
                gt_boxes = gt_boxes[gt_boxes[:, 3] > 0] # Valid GT only

                # 거리 필터링
                if gt_boxes.shape[0] > 0:
                    gt_boxes = filter_by_distance(gt_boxes, args.dist_thresh)
                if pred_boxes.shape[0] > 0:
                    dist_mask = torch.sqrt(pred_boxes[:, 0]**2 + pred_boxes[:, 1]**2) <= args.dist_thresh
                    pred_boxes = pred_boxes[dist_mask]
                    pred_scores = pred_scores[dist_mask]
                    pred_labels = pred_labels[dist_mask]

                # 클래스별 평가
                for class_idx, class_name in enumerate(cfg.CLASS_NAMES):
                    cls_id = class_idx + 1
                    
                    # (A) 전체 통계용 (기존 로직)
                    batch_res, num_gt = eval_one_batch(
                        gt_boxes, pred_boxes, pred_scores, pred_labels, cls_id
                    )
                    all_eval_results[class_name].extend(batch_res)
                    total_gt_counts[class_name] += num_gt

                    # (B) [추가] 프레임별 성능 분석 (Bad Case 찾기)
                    # 여기서는 Config에 설정된 'min_score'를 기준으로 판단합니다.
                    # 즉, "min_score"를 넘었는데 틀렸거나(FP), GT가 있는데 "min_score"를 못 넘은(FN) 경우
                    check_thresh = CLASS_CONFIG.get(class_name, {}).get('min_score', 0.3)
                    
                    tp, fp, fn = analyze_per_frame(
                        gt_boxes, pred_boxes, pred_scores, pred_labels, cls_id, check_thresh
                    )
                    
                    if fp > 0 or fn > 0:
                        bad_frame_logs.append({
                            'frame_id': current_frame_id,
                            'class': class_name,
                            'fp': fp,
                            'fn': fn,
                            'tp': tp,
                            'total_gt': num_gt,
                            'score_thresh': check_thresh
                        })

    # =========================================================================
    # [결과 출력 1] Worst Frames Report
    # =========================================================================
    print(f"\n=========================================================================")
    print(f" WORST PERFORMANCE FRAMES (Sorted by Misses(FN) -> Ghosts(FP))")
    print(f" * Analysis based on min_score threshold in config")
    print(f"=========================================================================")
    
    # 정렬 기준: FN(미검출) 내림차순 -> FP(오검출) 내림차순
    bad_frame_logs.sort(key=lambda x: (x['fn'], x['fp']), reverse=True)
    
    worst_table = PrettyTable()
    worst_table.field_names = ["Rank", "Frame ID", "Class", "GT Count", "Missed(FN)", "Ghost(FP)", "TP"]
    
    # 상위 20개만 출력
    top_k = 20
    for i, log in enumerate(bad_frame_logs[:top_k]):
        worst_table.add_row([
            i+1, 
            log['frame_id'], 
            log['class'], 
            log['total_gt'], 
            f"{log['fn']} ❌",  # Miss
            f"{log['fp']} 👻",  # Ghost
            log['tp']
        ])
    
    if len(bad_frame_logs) == 0:
        print("Amazing! No errors found with current thresholds.")
    else:
        print(worst_table)
        print(f"Total problematic frames found: {len(bad_frame_logs)}")


    # =========================================================================
    # [결과 출력 2] 기존 로직 (AP 계산 및 Plotting)
    # =========================================================================
    print(f"\n=========================================================================")
    print(f" Global Performance Report & Optimal Range Search")
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

        step = 0.05
        thr_values = np.arange(search_min, search_max + 1e-6, step)
        n_thr = len(thr_values)
        ap_grid = np.full((n_thr, n_thr), np.nan, dtype=np.float32)

        best_pass = None 
        best_overall = None

        for i, thr_min in enumerate(thr_values):
            for j, thr_max in enumerate(thr_values[i:], start=i):
                ap, prec, rec, f1 = compute_metrics_for_range(
                    results, num_gt, thr_min, thr_max
                )
                ap_grid[i, j] = ap 

                if (best_overall is None) or (ap > best_overall['ap']):
                    best_overall = {
                        'min': thr_min, 'max': thr_max,
                        'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1
                    }

                if ap >= TARGET_AP:
                    width = thr_max - thr_min
                    if best_pass is None:
                        best_pass = {
                            'min': thr_min, 'max': thr_max,
                            'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1,
                            'width': width
                        }
                    else:
                        if (width < best_pass['width']) or (
                            np.isclose(width, best_pass['width']) and thr_min < best_pass['min']
                        ):
                            best_pass = {
                                'min': thr_min, 'max': thr_max,
                                'ap': ap, 'prec': prec, 'rec': rec, 'f1': f1,
                                'width': width
                            }

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

        # Plotting (Heatmap & PR Curve)
        # (기존 코드와 동일하게 유지)
        plt.figure(figsize=(6, 5))
        extent = [search_min, search_max, search_min, search_max]
        img = plt.imshow(ap_grid, origin='lower', extent=extent, aspect='auto')
        plt.colorbar(img, label='AP')
        plt.xlabel('max_score')
        plt.ylabel('min_score')
        plt.title(f'AP Heatmap - {class_name}')
        plt.scatter([sel_max], [sel_min], marker='x')
        out_name = f'ap_heatmap_{class_name}.png'
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        plt.close()

        recall_curve, precision_curve, ap_curve = get_pr_curve_for_range(
            results, num_gt, sel_min, sel_max
        )

        if recall_curve.size > 0:
            plt.figure(figsize=(5, 4))
            plt.plot(recall_curve, precision_curve, label='PR curve')
            plt.fill_between(recall_curve, precision_curve, alpha=0.3)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{class_name} (range={sel_min:.2f}~{sel_max:.2f}, AP={ap_curve:.3f})')
            plt.grid(True)
            plt.legend()
            out_name = f'pr_area_{class_name}.png'
            plt.savefig(out_name, dpi=200, bbox_inches='tight')
            plt.close()
        
    print(table)
    print("\n")

if __name__ == '__main__':
    main()