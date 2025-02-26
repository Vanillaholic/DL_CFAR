import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse         #传入  python参数的解析
from utils import progress_bar  #进度条

import swanlab

class Conv2D_BN_PReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2D_BN_PReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D_BN_PReLU(1, num_filters * 32, (3, 3))
        self.conv2 = Conv2D_BN_PReLU(num_filters * 32, num_filters * 16, (3, 3))
        self.conv3 = Conv2D_BN_PReLU(num_filters * 16, num_filters * 8, (3, 3))
        self.conv4 = Conv2D_BN_PReLU(num_filters * 8, num_filters, (3, 3))
        self.bn = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.bn(y)
        return x + y


class DLCFAR(nn.Module):
    def __init__(self):
        super(DLCFAR, self).__init__()
        self.res_block1 = ResidualBlock(1)
        self.res_block2 = ResidualBlock(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 1, 512)  # Assuming input size is (16, 16, 1)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.res_block1(x)
        x = F.prelu(x)  # PReLU activation
        x = self.res_block2(x)
        x = F.prelu(x)  # PReLU activation
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class MatDataset(Dataset):
    def __init__(self, mat_file_input, mat_file_label):
        # 加载训练数据和标签
        data = sio.loadmat(mat_file_input)
        self.x_data = data['RD_map_input_train']  # 输入数据
        labels = sio.loadmat(mat_file_label)
        self.y_data = labels['RD_map_label_train']  # 标签数据
        
        # 转换为 PyTorch 张量并处理数据形状
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.float32)

        # 如果需要，可以进行形状调整
        self.x_data = self.x_data.reshape(-1, 1, 16, 16)  # 转换为 (num_samples, 1, 16, 16)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


'''-------------------定义模型-----------------------------'''
# 模型实例化
model = DLCFAR()
#如果有多张 GPU，DataParallel 会自动分配负载
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model = model.to(device)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()

        # 将输入数据和目标数据移动到指定的设备上（如 GPU）
        inputs, targets = inputs.to(device), targets.to(device)
        #通过模型对输入数据进行预测播
        predictions = model(inputs)
        # 计算损失
        loss = criterion(predictions, targets)
        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
        # 累加训练损失
        train_loss += loss.item()
        # 对于回归任务，可以计算误差，平均绝对误差（MAE）
        total_error += torch.sum(torch.abs(predictions - targets)).item()  # MAE
        total_samples += targets.size(0)
        '''
        # 分类任务
        _, predicted = predictions.max(1)
        # 统计总样本数
        total += targets.size(0)
        # 统计正确预测的样本数
        correct += predicted.eq(targets).sum().item()  
        '''
        # 打印进度条，显示当前批次的损失和准确率
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Avg Error: %.3f' % (train_loss/(batch_idx+1), total_error/total_samples))
        # 记录训练指标
        swanlab.log({"train/loss": train_loss,"train/avg_error": total_error / total_samples})


def test(epoch):
    model.eval()

    # 初始化测试损失和正确预测的样本数
    test_loss = 0
    total_error, total_samples= 0,0
    # 关闭梯度计算，因为在测试阶段不需要计算梯度
    with torch.no_grad():
        # 遍历测试数据加载器中的所有批次
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # 将输入数据和目标数据移动到指定的设备上（如 GPU）
            inputs, targets = inputs.to(device), targets.to(device)
            # 通过模型对输入数据进行预测
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 累加测试损失
            test_loss += loss.item()
            # 对于回归任务，可以计算误差，平均绝对误差（MAE）
            total_error += torch.sum(torch.abs(outputs - targets)).item()  # MAE
            total_samples += targets.size(0)
            '''
            # 分类任务
            _, predicted = outputs.max(1)
            # 统计总样本数
            total += targets.size(0)
            # 统计正确预测的样本数
            correct += predicted.eq(targets).sum().item()
            '''

            # 打印进度条，显示当前批次的损失和准确率
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Avg Error: %.3f' % (test_loss/(batch_idx+1), total_error/total_samples))
            # 记录训练指标
            swanlab.log({"val/loss": test_loss,"val/avg_error": total_error / total_samples})


if __name__ == '__main__':
    #传入参数
    parser = argparse.ArgumentParser(description='PyTorch DL-CFAR Training')
    parser.add_argument('--lr', default=5 * 1e-5, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default = False,action='store_true',help='resume from checkpoint')
    args = parser.parse_args()

    # 登陆swanlab
    swanlab.init(config=args)

    # 输出模型摘要
    print(model)
    #TODO 选择损失函数，此处为均方误差
    criterion = nn.MSELoss()
    #TODO 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

    #TODO 此处加载数据集(由于缺失数据,此处待完成)
    train_set = MatDataset('training_input.mat', 'training_label.mat')
    test_set = MatDataset('validation_input.mat', 'validation_label.mat')

    trainloader = DataLoader( train_set, batch_size=128, shuffle=True) #读取数据trainset
    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)  #shuffle=False 不保证顺序

    #TODO 判断是否需要从 checkpoint 恢复模型
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0  # best test accuracy
    import os
    if args.resume:
        # 加载 checkpoint
        print('==> Resuming from checkpoint..')
        # 断言 checkpoint 目录存在
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # 加载 checkpoint 文件
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        # 加载模型参数
        model.load_state_dict(checkpoint['net'])
        # 加载最佳准确率
        best_acc = checkpoint['acc']
        # 加载起始 epoch
        start_epoch = checkpoint['epoch']

    #TODO 训练模型
    for epoch in range(start_epoch, 500):
        train(epoch)
        test(epoch)
        

       
