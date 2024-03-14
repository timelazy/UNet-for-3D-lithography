import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def mse(tf_img1, tf_img2):
    return compare_mse(tf_img1, tf_img2)


def psnr(tf_img1, tf_img2):
    return compare_psnr(tf_img1, tf_img2)


def ssim(tf_img1, tf_img2):
    return compare_ssim(tf_img1, tf_img2)


def main():
    WSI_MASK_PATH1 = 'D:/UNet for 3D lithography/20221128/20221128/DATA/compare/compare_groundtruth17/'
    WSI_MASK_PATH2 = 'D:/UNet for 3D lithography/20221128/20221128/DATA/compare/compare_litho17/'
    path_real = glob(os.path.join(WSI_MASK_PATH1, '*.bmp'))
    path_fake = glob(os.path.join(WSI_MASK_PATH2, '*.bmp'))
    list_psnr = []
    list_ssim = []
    list_mse = []

    for i in range(len(path_real)):
        # t1 = read_img(path_real[i])
        # t2 = read_img(path_fake[i])
        t1 = cv2.imread(path_real[i], 0)
        t2 = cv2.imread(path_fake[i], 0)
        result1 = t1
        result2 = t2
        # result1 = np.zeros(t1.shape, dtype=np.float32)
        # result2 = np.zeros(t2.shape, dtype=np.float32)
        # cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
        ssim_num = ssim(result1, result2)
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        list_mse.append(mse_num)

        # 输出每张图像的指标：
        print("{}/".format(i + 1) + "{}:".format(len(path_real)))
        str = "\\"
        print("image:" + path_real[i][(path_real[i].index(str) + 1):])
        print("PSNR:", psnr_num)
        print("SSIM:", ssim_num)
        print("MSE:", mse_num)

    # 输出平均指标：
    print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr
    print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim
    print("平均MSE:", np.mean(list_mse))  # ,list_mse


if __name__ == '__main__':
    main()

