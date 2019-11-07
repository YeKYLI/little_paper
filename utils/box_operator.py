#to do

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

def nms(boxes):
    r"""
        在这里进行nms操作
    """
    #先进行排序
    num = len(boxes)
    for i in range(num):
        for j in range(num - i):
            if boxes[i + j][4] > boxes[i][4]:
                box = boxes[i]
                boxes[i] = boxes[i + j]
                boxes[i + j] = box
    #进行iou 计算
    
    return boxes


 
