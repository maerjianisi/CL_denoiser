import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.models as models

import model
import utils

# Model related-------------
# cnn_model = models.resnet50(pretrained=False,)
cnn_model = model.CNN().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)

# procession for image-------------
batch_size = 2048
transforms = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])

dataset = utils.BMPDataset(root_dir='./data/Cr', transform=transforms)  # 原图
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset_blur = utils.BMPDataset(root_dir='./data/Cr_blur+gray', transform=transforms)  # 添加模糊后的图
data_loader_blur = DataLoader(dataset_blur, batch_size=batch_size, shuffle=True)


# Train-------------
num_epochs = 20000
loss_list = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for img_origin, img_blur in zip(data_loader, data_loader_blur):
        # 将输入数据传入模型
        img_origin = img_origin.cuda()
        img_blur = img_blur.cuda()
        outputs = cnn_model(img_blur)
        # 计算损失
        loss = criterion(outputs, img_origin)  # 输入和输出之间的重建损失
        loss = loss.to("cpu")
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * img_blur.size(0)

    epoch_loss = running_loss / len(dataset)
    loss_list.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
print('Finished Training')
    
