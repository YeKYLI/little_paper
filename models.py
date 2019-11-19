import math
import torch
from utils.box_operator import xywh2xyxy
from utils.box_operator import xywh_iou
from utils.box_operator import nms

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
        self.fun19 = torch.nn.Conv2d(512, 256, kernel_size = (3, 3), padding = 1)
        self.fun20 = torch.nn.Conv2d(256, 128, kernel_size = (1, 1))
        self.fun21 = torch.nn.Conv2d(128, 64, kernel_size = (3, 3), padding = 1)
        self.fun22 = torch.nn.Conv2d(64, 32, kernel_size = (1, 1))
        self.fun23 = torch.nn.Conv2d(32, 6, kernel_size = (3, 3), padding = 1)

    def init_hyperparameter(self):
        #ground truth,[xmid, ymid, width, height]
        self.gt  = [[0.567, 0.469, 0.206, 0.53]]
        #anchor, relative to the output dimension
        #self.anchor = [[1.0, 1.0]]
        #self.anchor = [[0.2, 0.5]]  

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
        box_second = list()
        flag = [0, 0]
        loss = torch.zeros(1)
        for i in range(x.shape[0]): 
            for j in range(x.shape[1]):
                for k in range(x.shape[2]): 
                    for v in range(int(x.shape[3] / 6)):
                        #decode the predict box to relative value
                        pred_x = (j + x[i][j][k][v * 6 + 2]) / x.shape[1]
                        pred_y = (k + x[i][j][k][v * 6 + 3]) / x.shape[2]
                        pred_w = torch.exp(x[i][j][k][v * 6 + 4]) / x.shape[1] 
                        pred_h = torch.exp(x[i][j][k][v * 6 + 5]) / x.shape[2]
                        p = [float(pred_x), float(pred_y), float(pred_w), float(pred_h)]
                        #get the specific the gt box
                        best_box = [0] 
                        best_iou = -1.0
                        for g in self.gt:
                            pg_iou = xywh_iou(p, g)
                            if pg_iou > best_iou:
                                best_iou = pg_iou
                                best_box = g
                        loss_cls = torch.zeros(1) 
                        if best_iou < 0.3:
                            if flag[0] < 2:
                                p.append(-1)
                                flag[0] += 1
                                box_second.append(p)
                        #    cls = (torch.exp(x[i][j][k][v * 6]) /
                        #            (torch.exp(x[i][j][k][v * 6]) + torch.exp(x[i][j][k][v * 6 + 1])))
                        #    cls = torch.log((0.3 - cls).abs())
                        #    loss_cls = torch.mul(cls, -1)
                        #    print(str(0) + ' conf: ' + str(float(loss_cls)))
                        #    loss = torch.add(loss, loss_cls)
                        if best_iou > 0.5:
                            if best_iou > 0.7 and flag[1] < 2:
                                flag[1] += 1
                                p.append(1)
                                box_second.append(p)
                            print("iou = " + str(best_iou))
                        #    cls = (torch.exp(x[i][j][k][v * 6 + 1]) /
                        #            (torch.exp(x[i][j][k][v * 6]) + torch.exp(x[i][j][k][v * 6 + 1])))
                        #    cls = torch.log((best_iou - cls).abs())
                        #    loss_cls = torch.mul(cls, -1)
                            #encode the best_bobest_box
                            best_box_ = [0, 0, 0, 0]
                            best_box_[0] = best_box[0] * x.shape[1] - j
                            best_box_[1] = best_box[1] * x.shape[2] - k
                            best_box_[2] = math.log(best_box[2] * x.shape[1]) 
                            best_box_[3] = math.log(best_box[3] * x.shape[2])
                            loss_x = (torch.Tensor([best_box_[0]]) - x[i][j][k][v * 6 + 2]).abs()
                            loss_y = (torch.Tensor([best_box_[1]]) - x[i][j][k][v * 6 + 3]).abs()
                            loss_w = (torch.Tensor([best_box_[2]]) - x[i][j][k][v * 6 + 4]).abs()
                            loss_h = (torch.Tensor([best_box_[3]]) - x[i][j][k][v * 6 + 5]).abs()
                            #print(str(1) + ' conf: ' +  str(float(loss_cls)))
                            loss = torch.add(loss, loss_cls + loss_x + loss_y + loss_w + loss_h)
        return x, loss, box_second

from mnasnet import MnasNet
import random
import cv2
import torchvision
class two_stage(torch.nn.Module):
    def __init__(self):
        super(two_stage, self).__init__()
        self.fun1 = net()
        self.fun2 = MnasNet()

    def crop_image(self, image, xmin, ymin, xmax, ymax):
        xmin = max(0, image.shape[1] * xmin)
        xmax = max(0, image.shape[1] * xmax)
        ymin = max(0, image.shape[0] * ymin)
        ymax = max(0, image.shape[0] * ymax)
        image = image[int(ymin): int(ymax), int(xmin): int(xmax)]
        image = cv2.resize(image, (224, 224))
        totensor = torchvision.transforms.ToTensor()
        image = totensor(image)
        image = torch.unsqueeze(image, 0)
        image.requires_grad = True
        return image

    def forward(self, x):
        #直接在这里加一个判断条件，判断是否是train，如果是test的话，那么直接全部取出结果，将结果全部街道第二阶段的
        #的网络。来进行这个的操作进行事情第二次操作，在这里进行操作！！！！！！！！！！！！
        x, loss, boxes = self.fun1(x)
        image = cv2.imread('data/1.png') ##attection!!!!!!!!!!!!!!!!!!!!!!
        box = random.sample(boxes, 1)
        print(box)
        image = self.crop_image(image, (box[0][0] - box[0][2]), (box[0][1] - box[0][3]), (box[0][0] + box[0][2]), (box[0][1] + box[0][3])) 
        print(image)
        x = self.fun2(image) 
        x = torch.nn.functional.softmax(x)
        cls_loss = torch.zeros(1)
        cls_loss.requires_grad = True
        if box[0][4] > 0:
            cls_loss = torch.add(cls_loss, 1 - x[0][1])
        else:
            cls_loss = torch.add(cls_loss, 1 - x[0][0])
        return loss, cls_loss

            
