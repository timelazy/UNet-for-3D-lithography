import os.path
from data import *
from torchvision.transforms import ToTensor
from model import *
import cv2
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE


def test(testdir):
    img_paths, gt_paths = load_img(testdir)
    # 载入test路径
    model = UNet(in_channels=1, out_channels=1)
    # 载入模型，输入输出均为单通道
    state_dict = torch.load("save/train_best_Unet.pth")
    # 载入训练好的模型
    model.load_state_dict(state_dict)
    # 加载模型
    device = torch.device('cuda')  # cuda:0
    # 启用第一块GPU
    model.eval()
    # 用eval函数来评估
    model.to(device)
    # 载入模型到GPU上
    for i in range(len(img_paths)):
    # 遍历路径下所有图片
        img = cv2.imread(img_paths[i], 0)
        gt = cv2.imread(gt_paths[i], 0)
        norm = ToTensor()
        img = norm(img)
        # 转为张量
        img = torch.unsqueeze(img, 0)
        # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度
        with torch.no_grad():
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，即不会求导。
            result = model(img.to(device))
            result = result[0].cpu().numpy().squeeze(0)
            # numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度。
            result = (result*255.0).astype(np.uint8)
            # 转为8位灰度
            ssim = SSIM(result, gt)
            # 求结构相似度ssim
            psnr = PSNR(result, gt)
            # 求峰值信噪比psnr
            mse = MSE(result, gt)
            # 求均方差mse
        name = os.path.basename(img_paths[i])
        # os.path.basename()返回path最后的文件名。若path以/或\结尾，则返回空值。 即os.path.split(path)的第二个元素。
        print(f"name:{name}", f"ssim:{ssim}", f"psnr:{psnr}", f"mse:{mse}")
        # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 用于转成256*256*3的RGB图
        name_ = os.path.basename(img_paths[i]).split('_')[0]
        cv2.imwrite(f"visual/{name_}_predict.bmp", result)
        # 将测试结果写入visual文件夹
