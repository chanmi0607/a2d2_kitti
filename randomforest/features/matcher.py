import torch
import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils

def match_rpn_to_gt_for_training(rpn_boxes, gt_boxes, gt_labels, fg_iou_thresh=0.2, bg_iou_thresh=0.4):
    """
    RPN box를 GT box와 IoU 매칭.
    - IoU >= fg_thresh: Foreground (e.g., "Car")
    - IoU < bg_thresh: Background
    - 그 외: Ignore
    """
    num_rpn_boxes = rpn_boxes.shape[0]
    
    if gt_boxes.shape[0] == 0:
        # GT가 없으면 모두 "Background"
        return ["Background"] * num_rpn_boxes, np.zeros(num_rpn_boxes, dtype=np.float32)

    ious = iou3d_nms_utils.boxes_iou3d_gpu(
        torch.from_numpy(rpn_boxes).cuda(),
        torch.from_numpy(gt_boxes).cuda()
    ).cpu().numpy()

    best_gt_indices = np.argmax(ious, axis=1)
    best_ious_np = ious[np.arange(num_rpn_boxes), best_gt_indices]

    matched_labels = []
    for i in range(num_rpn_boxes):
        best_iou = best_ious_np[i]
        if best_iou >= fg_iou_thresh:
            matched_labels.append(gt_labels[best_gt_indices[i]])
        elif best_iou < bg_iou_thresh:
            matched_labels.append("Background")
        else:
            # (예: 0.3 <= IoU < 0.5) 애매한 영역은 "Ignore"
            matched_labels.append("Ignore")
            
    return matched_labels, best_ious_np

