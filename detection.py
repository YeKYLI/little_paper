#simple edetection model
#从目前来看，应该是哪里的编解码我哪里写错了

import sys
import torch
import torchvision
import cv2
import math
from utils.box_operator import xywh2xyxy
from utils.box_operator import nms

image = cv2.imread("data/1.png")
image = cv2.resize(image, (224, 224))
totensor = torchvision.transforms.ToTensor()
image = totensor(image)
image = torch.unsqueeze(image, 0)
image.requires_grad = True

#ground truth,[xmid, ymid, width, height]
gt = [[0.567, 0.469, 0.206, 0.53]]

#anchor, relative to the output dimension
anchor = [[0.2, 0.5]]

def iou(pred, gt):
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
    if pred_ymin > gt_ymax or pred_xmax < gt_ymin:
        return 0
    iou_area = (xmax - xmin) * (ymax - ymin)
    iou_pred = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    iou_gt = (gt_xmax - gt_xmin) * (pred_ymax - pred_ymin)
    return iou_area / (iou_gt + iou_pred - iou_area)

#define the detection model
from torch.nn.modules import Module
class net(Module):
    def __init__(self):
        super(net, self).__init__()
        self.init_hyperparameter()
        self.fun1 = torch.nn.Conv2d(3, 32, kernel_size = (3, 3))
        self.fun2 = torch.nn.MaxPool2d(2, 2, padding = 1)
        self.fun3 = torch.nn.Conv2d(32, 64, kernel_size = (3, 3))
        self.fun4 = torch.nn.MaxPool2d(2, 2, padding = 1)
        self.fun5 = torch.nn.Conv2d(64, 128, kernel_size = (3, 3), padding = 1)
        self.fun6 = torch.nn.Conv2d(128, 64, kernel_size = (1, 1))
        self.fun7 = torch.nn.Conv2d(64, 128, kernel_size = (3, 3), padding = 1)
        self.fun8 = torch.nn.MaxPool2d(2, 2)
        self.fun9 = torch.nn.Conv2d(128, 256, kernel_size = (3, 3), padding = 1)
        self.fun10 = torch.nn.Conv2d(256, 128, kernel_size = (1, 1))
        self.fun11 = torch.nn.Conv2d(128, 256, kernel_size = (3, 3))
        self.fun12 = torch.nn.MaxPool2d(2, 2, padding = 1)
        self.fun13 = torch.nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1)
        self.fun14 = torch.nn.Conv2d(512, 256, kernel_size = (1, 1))
        self.fun15 = torch.nn.Conv2d(256, 512, kernel_size = (3, 3), padding = 1)
        self.fun16 = torch.nn.Conv2d(512, 256, kernel_size = (1, 1))
        self.fun17 = torch.nn.Conv2d(256, 512, kernel_size= (3, 3))
        self.fun18 = torch.nn.MaxPool2d(2, 2, padding = 1)
        self.fun19 = torch.nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1)
        self.fun20 = torch.nn.Conv2d(1024, 512, kernel_size = (1, 1))
        self.fun21 = torch.nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1)
        self.fun22 = torch.nn.Conv2d(1024, 512, kernel_size = (1, 1))
        self.fun23 = torch.nn.Conv2d(512, 6, kernel_size = (3, 3), padding = 1)

    def init_hyperparameter(self):
        self.gt  = [[0.567, 0.469, 0.206, 0.53]] 
        self.anchor = [[0.2, 0.5]]  

    def forward(self, x):
        x = self.fun1(x)
        x = self.fun2(x)
        x = self.fun3(x)
        x = self.fun4(x)
        x = self.fun5(x)
        x = self.fun6(x)
        x = self.fun7(x)
        x = self.fun8(x)
        x = self.fun9(x)
        x = self.fun10(x)
        x = self.fun11(x)
        x = self.fun12(x)
        x = self.fun13(x)
        x = self.fun14(x)
        x = self.fun15(x)
        x = self.fun16(x)
        x = self.fun17(x)
        x = self.fun18(x)
        x = self.fun19(x)
        x = self.fun20(x)
        x = self.fun21(x)
        x = self.fun22(x)
        x = self.fun23(x)
        x = x.transpose(1, 3)
       
        #design the loss, x[batch][w][h][background, object, xmid, ymid, width, height]
        loss = 0
        for i in range(x.shape[0]): 
            for j in range(x.shape[1]): 
                for k in range(x.shape[2]): 
                    for v in range(int(x.shape[3] / 6)): 
                        best_index = -1
                        best_iou = -1
                        #decode the predict box to relative value
                        pred_x = (j + x[i][j][k][v * 6 + 2]) / x.shape[1]
                        pred_y = (k + x[i][j][k][v * 6 + 3]) / x.shape[2]
                        pred_w = torch.exp(x[i][j][k][v * 6 + 4]) * self.anchor[v][0] / x.shape[1]
                        pred_h = torch.exp(x[i][j][k][v * 6 + 5]) * self.anchor[v][1] / x.shape[2]
                        p = [pred_x, pred_y, pred_w, pred_h]
                        #encode the anchor box
                        anch_x = j + 0.5
                        anch_y = k + 0.5
                        anch_w = anchor[v][0] / x.shape[1]
                        anch_h = anchor[v][1] / x.shape[2]
                        #get the specific the gt box
                        best_box = list()
                        for g in gt:
                            pg_iou = iou(p, g)
                            if pg_iou > best_iou:
                                best_iou = pg_iou
                                best_box = g
                        #怎么计算softmax的功能，这样连续的几个怎么计算，怎么进行这个事情
                        y = torch.chunk(x, len(anchor), dim = 3)[v]
                        if best_iou < 0.3:
                            cls_tensor = y.split([2, 4], dim = 3)[0]
                            cls_tensor = torch.nn.functional.softmax(cls_tensor, dim = 3)
                            loss_cls = 1 - cls_tensor[i][j][k][0] 
                            loss += loss_cls
                        if best_iou > 0.7:
                            cls_tensor = y.split([2, 4], dim = 3)[0]
                            cls_tensor += torch.nn.functional.softmax(cls_tensor, dim = 3)
                            loss_cls = 1 - cls_tensor[i][j][k][1]
                            #encode the gt box
                            loss_x = (torch.Tensor([best_box[0]]) - x[i][j][k][v * 6 + 2]).abs()
                            loss_y = (torch.Tensor([best_box[1]]) - x[i][j][k][v * 6 + 3]).abs()
                            loss_w = (torch.Tensor([best_box[2]]) - x[i][j][k][v * 6 + 4]).abs()
                            loss_h = (torch.Tensor([best_box[3]]) - x[i][j][k][v * 6 + 5]).abs()
                            loss += loss_cls + loss_x + loss_y + loss_w + loss_h
        return x, loss

detection = net()

#define the optimizer parameter 
optimizer = torch.optim.SGD(detection.parameters(), lr = 0.01)

#define the test
#misaka的那张图调试出来，那张图调试出来，
def test():
    detection.load_state_dict(torch.load('data/params.pkl'))
    output, _ = detection.forward(image)
    print(output.shape)
    #decode the output to boxesww
    boxes = list()
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                for v in range(int(output.shape[3] / 6)):
                    v = v * 6
                    conf = (torch.exp(output[i][j][k][v + 1]) / 
                    (torch.exp(output[i][j][k][v]) + torch.exp(output[i][j][k][v + 1])))
                    pred_x = (j + output[i][j][k][v * 6 + 2]) / output.shape[1]
                    pred_y = (k + output[i][j][k][v * 6 + 3]) / output.shape[2]
                    pred_w = torch.exp(output[i][j][k][v * 6 + 4]) * anchor[v][0] / output.shape[1]
                    pred_h = torch.exp(output[i][j][k][v * 6 + 5]) * anchor[v][1] / output.shape[2]
                    box = [pred_x, pred_y, pred_w, pred_h, conf]
                    box = xywh2xyxy(box)
                    boxes.append(box)
    boxes = nms(boxes) 
    for box in boxes:
        print(box)
    #visiualization the specific output 
    misaka = cv2.imread("data/1.png")
    count = 0
    for box in boxes:
        count += 1
        if count > 20:
            break
        print(box)
        cv2.rectangle(misaka, (int(box[0] * 500), int(box[1] * 500)), (int(box[2] * 500), int(box[3] * 500)), (255, 0, 0), 4)
    cv2.imshow("misake", misaka)
    cv2.waitKey() 

#define the training
def train():
    for i in range(100):
        if i % 10 == 0:
            torch.save(detection.state_dict(), 'data/params.pkl')
        _, loss = detection.forward(image)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()
 
