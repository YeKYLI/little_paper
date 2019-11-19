import sys
import torch
import torchvision
import cv2
import math
from utils.box_operator import xywh2xyxy
from utils.box_operator import xywh_iou
from utils.box_operator import nms
from models import net
import mnasnet
from utils.image_operator import crop_image


def load_image():
    image = cv2.imread("data/1.png")
    image = cv2.resize(image, (224, 224))
    totensor = torchvision.transforms.ToTensor()
    image = totensor(image)
    image = torch.unsqueeze(image, 0)
    image.requires_grad = True
    return image

def load_crop_image(xmin, ymin, xmax, ymax):
    image_path = "data/1.png"
    image = crop_image(image_path, xmin, ymin, xmax, ymax)
    image = cv2.resize(image, (224, 224))
    totensor = torchvision.transforms.ToTensor()
    image = totensor(image)
    image = torch.unsqueeze(image, 0)
    image.requires_grad = True
    return image

    
detection = net()
classi = mnasnet.MnasNet()

#define the optimizer parameter 
optimizer = torch.optim.SGD(detection.parameters(), lr = 0.01)
optimizer2 = torch.optim.SGD(classi.parameters(), lr = 0.01) 

#define the test
def train():
    for i in range(1000):
        if i % 10 == 0:
            torch.save(detection.state_dict(), 'data/params.pkl')
        image = load_image()
        _, loss, boxes = detection.forward(image)
        print(boxes)
        print("*******************************")
        #print(loss)
        #if loss.grad == None:
        #    continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #second stage
        cls_loss = torch.zeros(1)
        cls_loss.requires_grad = True
        for box in boxes:
            image = load_crop_image(box[0] - box[2], box[1] - box[3], box[0] + box[2], box[1] + box[3])
            output = classi.forward(image)
            print(output)
            output = torch.nn.functional.softmax(output)
            print(output)
            if box[4] > 0:
                cls_loss = torch.add(cls_loss, 1 - output[0][1])
            else:
                cls_loss = torch.add(cls_loss, 1 - output[0][0])
            print(cls_loss)
            optimizer2.zero_grad()
            cls_loss.backward()
            optimizer2.step()

from models import two_stage
detect = two_stage()
optimizer3 = torch.optim.SGD(detect.parameters(), lr = 0.01) 
def dete():
    image = load_image()
    for i in range(10):
        loss1, loss2 = detect.forward(image)
        print(loss1)
        print(loss2)
        optimizer3.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer3.step()
    
    #visiualization the output, the final test
   
 
def test():
    detection.load_state_dict(torch.load('data/params.pkl'))
    for i in range(30):
        image = load_image()
        output, loss = detection.forward(image)
        print("*******************************")
        print(loss)
        optimizer.zero_grad()
        if loss > 0: #在这里找到一个判断条件，这也是非常重要的！！！！
            loss.backward()
        optimizer.step()

        #decode the output
        boxes = list()
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    for v in range(int(output.shape[3] / 6)):
                        #conf =  output[i][j][k][v * 6 + 1]
                        conf = (torch.exp(output[i][j][k][v * 6 + 1]) / 
                            (torch.exp(output[i][j][k][v * 6]) + torch.exp(output[i][j][k][v * 6 + 1])))
                        pred_x = (j + output[i][j][k][v * 6 + 2]) / output.shape[1]
                        pred_y = (k + output[i][j][k][v * 6 + 3]) / output.shape[2]
                        pred_w = torch.exp(output[i][j][k][v * 6 + 4]) / output.shape[1]
                        pred_h = torch.exp(output[i][j][k][v * 6 + 5]) / output.shape[2]
                        box = [pred_x, pred_y, pred_w, pred_h, conf]
                        box = xywh2xyxy(box) 
                        boxes.append(box)
        boxes = nms(boxes) 
        #visiualization the specific output 
        misaka = cv2.imread("data/1.png")
        count = 0
        cv2.rectangle(misaka, (int(0.46 * 500), int(0.20 * 500)), (int(0.66 * 500), int(0.734 * 500)), (0, 255, 0), 4)
        for box in boxes:
            count += 1
            #if count > 20:
            #    break
            cv2.rectangle(misaka, (int(box[0] * 500), int(box[1] * 500)), (int(box[2] * 500), int(box[3] * 500)), (255, 0, 0), 4)
            print(box)
        cv2.imshow("misake", misaka)
        cv2.waitKey() 

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()
    if sys.argv[1] == 'dete':
        dete()

