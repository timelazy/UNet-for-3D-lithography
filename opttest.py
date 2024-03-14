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
    model = UNet(in_channels=1, out_channels=1)
    state_dict = torch.load("save/train_best_Unet.pth")
    model.load_state_dict(state_dict)
    device = torch.device('cuda')  # cuda:0
    model.eval()
    model.to(device)
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i], 0)
        gt = cv2.imread(gt_paths[i], 0)
        norm = ToTensor()
        img = norm(img)
        img = torch.unsqueeze(img, 0)
        with torch.no_grad():
            result = model(img.to(device))
            result = result[0].cpu().numpy().squeeze(0)
            result = (result*255.0).astype(np.uint8)
            ssim = SSIM(result, gt)
            psnr = PSNR(result, gt)
            mse = MSE(result, gt)
        name = os.path.basename(img_paths[i])
        print(f"name:{name}", f"ssim:{ssim}", f"psnr:{psnr}", f"mse:{mse}")
        # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 用于转成256*256*3的RGB图
        name_ = os.path.basename(img_paths[i]).split('_')[0]
        cv2.imwrite(f"visual/{name_}_predict.bmp", result)
