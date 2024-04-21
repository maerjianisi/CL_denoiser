import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    200*200*1
    100*100*2
    50*50*4
    25*25*8
    50*50*4
    100*100*2
    200*200*1
    """
    def __init__(self):
        super(CNN, self).__init__()
        # part_1
        self.part_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_1_m = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.ReLU())
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200*200 -> 100*100
        # part_2
        self.part_2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_2_m = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU())
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100*100 -> 50*50
        # part_3
        self.part_3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_3_m = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU())
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50*50 -> 25*25
        # part_feature
        self.part_fea = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU())
        # part_4
        self.up_sample_4 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)  # 25*25 -> 50*50
        self.part_4_m = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_4 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU())
        # part_5
        self.up_sample_5 = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1)  # 50*50 -> 100*100
        self.part_5_m = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_5 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU())
        # part_6
        self.up_sample_6 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)  # 100*100 -> 200*200
        self.part_6_m = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.ReLU())
        self.part_6 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.ReLU())
    
    def forward(self, x):
        out = self.part_1(x)
        out = self.part_1_m(out)
        out = self.part_1_m(out)
        out = self.pool_1(out)
        
        out = self.part_2(out)
        out = self.part_2_m(out)
        out = self.part_2_m(out)
        out = self.pool_2(out)
        
        out = self.part_3(out)
        out = self.part_3_m(out)
        out = self.part_3_m(out)
        out = self.pool_3(out)
        
        out = self.part_fea(out)
        # out = self.part_fea(out)
        out = self.part_fea(out)
        
        out = self.up_sample_4(out)
        out = self.part_4_m(out)
        out = self.part_4(out)
        
        out = self.up_sample_5(out)
        out = self.part_5_m(out)
        out = self.part_5(out)
        
        out = self.up_sample_6(out)
        out = self.part_6_m(out)
        out = self.part_6(out)
        
        return out



class UpsampleCNN(nn.Module):
    def __init__(self):
        super(UpsampleCNN, self).__init__()
        # 定义卷积层和反卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 输入尺寸: 200x200, 输出尺寸: 200x200
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 输入尺寸: 100x100, 输出尺寸: 100x100
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输入尺寸: 50x50, 输出尺寸: 50x50
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 输入尺寸: 25x25, 输出尺寸: 25x25
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1) # 上采样层，将图像尺寸扩大2倍

    def forward(self, x):
        # 卷积操作
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        # 上采样操作
        x = self.upsample(x)
        return x

