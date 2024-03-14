import cv2
import os
from model import *
import numpy as np
from torchvision.transforms import ToTensor

# 待测试图片地址
img_path = "DATA/predict_in/45_in.bmp"
img = cv2.imread(img_path, 0)
# 归一化处理
norm = ToTensor()
img = norm(img)
img = torch.unsqueeze(img, 0)
# 加载模型
model = UNet(in_channels=1, out_channels=1)
state_dict = torch.load("save/train_best_Unet.pth")
model.load_state_dict(state_dict)
device = torch.device('cuda')  # cuda:0
model.eval()
model.to(device)
with torch.no_grad():
    result = model(img.to(device))
    result = result[0].cpu().numpy().squeeze(0)
    result = (result * 255.0).astype(np.uint8)
# result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 用于转成256*256*3的RGB图
name = os.path.basename(img_path).split('_')[0]
# 预测结果在：DATA/predict_out文件夹下
cv2.imwrite(f"DATA/predict_out/{name}_predict6.bmp", result)
print("(￣▽￣)~* (￣▽￣)预测完毕=￣ω￣= (￣３￣)a ￣▽￣")

