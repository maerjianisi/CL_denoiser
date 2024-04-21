import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os
import random


class BMPDataset(torch.utils.data.Dataset):
    """
    读取图像并做处理
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.bmp')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('L')  # 读取并转换为灰度图像
        if self.transform:
            image = self.transform(image)
        return image


def add_blur(input_folder, output_folder):
    """
    向图片添加高斯模糊，并保存在指定文件夹内
    :param input_folder: 原图片文件夹
    :param output_folder: 保存处理后图片的文件夹
    """
    min_blur = 15  # 最小模糊程度
    max_blur = 35  # 最大模糊程度
    
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]
    
    # 循环处理每张图片
    for image_file in image_files:
        # 读取图片
        img = cv2.imread(os.path.join(input_folder, image_file))
        
        # 生成随机模糊程度，确保是奇数
        blur_amount = random.randint(min_blur, max_blur)
        blur_amount = blur_amount + 1 if blur_amount % 2 == 0 else blur_amount
        
        # 对图片进行高斯模糊
        blurred_img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        
        
        # 保存处理后的图片到输出文件夹
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, blurred_img)


def add_gray(input_folder, output_folder):
    min_brightness = -30  # 最小灰度调整值
    max_brightness = 30  # 最大灰度调整值
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]
    
    # 循环处理每张图片
    for image_file in image_files:
        # 读取图片
        img = cv2.imread(os.path.join(input_folder, image_file))
        
        # 转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 生成随机灰度调整值
        brightness_adjustment = random.randint(min_brightness, max_brightness)
        
        # 调整灰度
        adjusted_gray_img = cv2.addWeighted(gray_img, 1, gray_img, 0, brightness_adjustment)
        
        # 保存处理后的图片到输出文件夹
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, adjusted_gray_img)
    
    print("图片处理完成。")
    
    
# add_blur('./data/Cr', './data/Cr_blur')
add_gray('./data/Cr_blur', './data/Cr_blur+gray')
