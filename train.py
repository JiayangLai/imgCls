import numpy as np
import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from tqdm import tqdm

def trans255to1(input):
    return input.astype(float)/255

class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有3类，所以最后一个全连接层输出数量是3
        self.fc3 = nn.Linear(84, 3)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    x_train = trans255to1(np.load("x_train.npy"))
    y_train = np.load("y_train.npy")
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    x_test = trans255to1(np.load("x_test.npy"))
    n_test = x_test.shape[0]
    y_test = np.load("y_test.npy")
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)



    imgcls_test = Data.TensorDataset(x_test, y_test)
    imgcls_train = Data.TensorDataset(x_train, y_train)

    testloader = torch.utils.data.DataLoader(imgcls_test, batch_size=32, shuffle=True)
    trainloader = torch.utils.data.DataLoader(imgcls_train, batch_size=32, shuffle=True)

    # 如果你没有GPU，那么可以忽略device相关的代码
    device = torch.device("cuda:0")
    net = LeNet().to(device)

    # optim中定义了各种各样的优化方法，包括SGD
    import torch.optim as optim

    # CrossEntropyLoss就是我们需要的损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Start Training...")
    for epoch in tqdm(range(2000)):

        # 我们用一个变量来记录每100个batch的平均loss
        loss100 = 0.0
        # 我们的dataloader派上了用场
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 注意需要复制到GPU
            optimizer.zero_grad()
            outputs = net(inputs.float())
            # print()
            # assert 1==0
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            loss100 += loss.item()
            if i % 100 == 99:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss100 / 100))
                loss100 = 0.0

    print("Done Training!")

    # 构造测试的dataloader
    dataiter = iter(testloader)
    # 预测正确的数量和总数量
    correct = 0
    total = 0
    # 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 预测
            outputs = net(images.float())
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    print('Accuracy of the network on the '+str(n_test)+' test images: %d %%' % (
            100 * correct / total))
