import torch
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[0][0]
    box1_y1 = boxes_preds[0][1]
    box1_x2 = boxes_preds[0][2]
    box1_y2 = boxes_preds[0][3]
    
    box2_x1 = boxes_labels[0][0]
    box2_y1 = boxes_labels[0][1]
    box2_x2 = boxes_labels[0][2]
    box2_y2 = boxes_labels[0][3]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    
    return (intersection / (box1_area + box2_area - intersection + 1e-6)).item()

def f1_score(pred_boxes, true_boxes, iou_threshold=0.5, score_threshold=0.3, num_classes=7):
    precisions = []
    recalls = []
    f1 = []
    epsilon = 1e-6
    
    for c in range(1, num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                if detection[2] > score_threshold:
                    detections.append(detection)
            
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
            
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
    
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
    
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths 
                               if bbox[0] == detection[0]]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(detection[3:], gt[2:])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_sum = torch.sum(TP, dim=0)
        FP_sum = torch.sum(FP, dim=0)
            
        recall = (TP_sum / (total_true_bboxes + epsilon)).item()
        precision = (torch.divide(TP_sum, (TP_sum + FP_sum + epsilon))).item()
        recalls.append(recall)
        precisions.append(precision)
        f1.append(2 * (recall * precision) / (precision + recall + epsilon))
    
    return f1, recalls, precisions

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=7):
    average_precisions = []
    epsilon = 1e-6
    
    for c in range(1, num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths 
                               if bbox[0] == detection[0]]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(detection[3:], gt[2:])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
            
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
            
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
        
    return sum(average_precisions) / len(average_precisions)