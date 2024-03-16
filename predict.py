import cv2
import os
from model import *
import numpy as np
from torchvision.transforms import ToTensor

# 待测试图片地址
img_path = "DATA/predict_in/cone120_in.bmp"
img = cv2.imread(img_path, 0)
# 归一化处理
norm = ToTensor()
img = norm(img)
# img转为张量
img = torch.unsqueeze(img, 0)
# 对img升维
# 加载模型
model = UNet(in_channels=1, out_channels=1)
# UNet模型输入输出均为单通道
state_dict = torch.load("save/train_best_Unet.pth")
model.load_state_dict(state_dict)
# 加载预训练好的模型
device = torch.device('cuda')  # cuda:0
model.eval()
# eval评估模型
model.to(device)
# 加载模型到gpu
with torch.no_grad():
    result = model(img.to(device))
    result = result[0].cpu().numpy().squeeze(0)
    # 去除1维的数据
    result = (result * 255.0).astype(np.uint8)
# result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 用于转成256*256*3的RGB图
name = os.path.basename(img_path).split('_')[0]
# 预测结果在：DATA/predict_out文件夹下
cv2.imwrite(f"DATA/predict_out/{name}_2predict.bmp", result)
print("(￣▽￣)~* (￣▽￣)预测完毕=￣ω￣= (￣３￣)a ￣▽￣")

