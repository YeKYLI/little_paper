#simple detection model
#训练一张简单的图片进行物体的位置回归，通过将图片的回归效果做好后再做其它的回归效果
#配置一下简单的回归神经网络

import torch
import torchvision
import cv2

image = cv2.imread("image/1.png")
image = cv2.resize(image, (224, 224))
totensor = torchvision.transforms.ToTensor()
image = totensor(image)
image = torch.unsqueeze(image, 0)

#define the detection model
from torch.nn.modules import Module
class net(Module):
    def __init__(self):
        super(net, self).__init__()
        #在这里配置网络的参数进行操作的进行的事情
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
        self.fun23 = torch.nn.Conv2d(512, 1024, kernel_size = (3, 3), padding = 1)

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
        
        #here we define our loss, do our training 
        print(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                #下边针具体的grid进行操作x[i][index][j][k]
                
        return x        

detection = net()

#define the optimizer parameter 
optimizer = torch.optim.SGD(detection.parameters(), lr = 0.01)

#define the training
for i in range(1):
    loss = detection.forward(image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
