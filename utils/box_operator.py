
def xywh2xyxy(box):
    r"""
        xmid, ymid, width, height, conf -> xmin, ymin, xmax, yamx, conf
    """

    xmin = box[0] - (box[2] / 2)
    ymin = box[1] - (box[3] / 2)
    xmax = box[0] + (box[2] / 2)
    ymax = box[1] + (box[3] / 2)
    conf = box[4]
    box = [xmin, ymin, xmax, ymax, conf]

    return box

def xywh_iou(pred, gt):
    pred_xmin = pred[0] - (pred[2] / 2)
    pred_xmax = pred[0] + (pred[2] / 2)
    pred_ymin = pred[1] - (pred[3] / 2) 
    pred_ymax = pred[1] + (pred[3] / 2)
    gt_xmin = gt[0] - (gt[2] / 2)
    gt_xmax = gt[0] + (gt[2] / 2)
    gt_ymin = gt[1] - (gt[3] / 2)
    gt_ymax = gt[1] + (gt[3] / 2)
    xmin = max(pred_xmin, gt_xmin)
    xmax = min(pred_xmax, gt_xmax)
    ymin = max(pred_ymin, gt_ymin)
    ymax = min(pred_ymax, gt_ymax)
    if pred_xmin > gt_xmax or pred_xmax < gt_xmin:
        return 0
    if pred_ymin > gt_ymax or pred_ymax < gt_ymin:
        return 0
    iou_area = (xmax - xmin) * (ymax - ymin)
    iou_pred = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    iou_gt = (gt_xmax - gt_xmin) * (pred_ymax - pred_ymin)
    return iou_area / (iou_gt + iou_pred - iou_area)

def nms(boxes):
    r"""
        在这里进行nms操作
    """
    #感觉测试阶段这部分就先不写了，直接就卡个阈值算了
    boxes_ = list()
    for box in boxes:
        if box[4] > 0.70:
            boxes_.append(box)
    return boxes_

    #先进行排序
    num = len(boxes)
    for i in range(num):
        for j in range(num - i):
            if boxes[i + j][4] > boxes[i][4]:
                box = boxes[i]
                boxes[i] = boxes[i + j]
                boxes[i + j] = box
    #进行iou 计算
    #其实这里进行事情上面的，这个模块暂且不实现，因为在前期的话这部分是没有什么影响的
 
    return boxes


 
