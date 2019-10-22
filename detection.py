#simple detection model
#训练一张简单的图片进行物体的位置回归，通过将图片的回归效果做好后再做其它的回归效果

import torch
import torchvision
import cv2
from PIL import Image

image = Image.open("1.png")
#image = cv2.imread("misaka.png", cv2.IMREAD_COLOR)
print(image)
#天啊，为什么这个图片读取进去全都是0啊。。


totensor = torchvision.transforms.ToTensor()
image = totensor(image)
print(image)
image = image * 255
print(image)

exit()

image = torch.ones(1, 3, 3, 3)

#define the detection model
from torch.nn.modules import Module
class net(Module):
    def __init__(self):
        super(net, self).__init__()
        self.fun1 = torch.nn.Linear(1, 10)
        self.fun2 = torch.nn.ReLU6()
        self.fun3 = torch.nn.Linear(10, 5)
        self.fun4 = torch.nn.ReLU6()
        self.fun5 = torch.nn.Linear(5, 1)
        #self.conv = torch.nn.Conv2d(1, 3, 3, 3, 2)
        #kernel = torch.ones(1, 3, 3, 3)
        #bias_ = torch.ones(1)
        #self.conv.weight = torch.nn.Parameter(kernel, requires_grad = True)
        #self.conv.bias = torch.nn.Parameter(bias_, requires_grad = True)

    def forward(self, x):
        x = self.fun1(x)
        x = self.fun2(x)
        x = self.fun3(x)
        x = self.fun4(x)
        x = self.fun5(x)
        return x
detection = net()

#define the basebone
class basebone(Module):
    def __init__(self):
        super(basebone, self).__init__()

#define the detection head
class head(Module):
    def __init__(self):
        super(head, self).__init__()


#********************************接下来，我们做一些参数更新的工作

#define the optimizer parameter 
optimizer = torch.optim.SGD(detection.parameters(), lr = 0.01)

#define the prediction
#output = detection.forward(image)
#print(output)

#do the training
for i in range(2000):
    inputs = torch.Tensor(1)
    inputs[0] = i
    outputs = detection.forward(inputs)
    print(outputs)
    loss = 2 * i - outputs
    loss_ = loss * loss
    optimizer.zero_grad()
    loss_.backward()
    optimizer.step()
    print(str(i) + "* 2 ***************")
    print(loss_)
    print(outputs)


