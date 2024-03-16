from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import cv2
import numpy as np
from scipy import ndimage

'''
Dataset
'''

def load_img(img_dir):
    img_paths, gt_paths = [], []
    img_gt = os.listdir(img_dir)
    # 遍历文件操作。os.listdir(path)中有一个参数，就传入相应的路径，将会返回那个目录下的所有文件名。
    for name in img_gt:
        if name == "groundtruth":
            gt_path = os.path.join(img_dir, name)
            # os.path.join()函数用于路径拼接文件路径，可以传入多个路径。
            gts = os.listdir(gt_path)
            gts.sort(key=lambda x: int(x.split('_')[0]))
            # sort排序，引入匿名函数lambda
            gt_paths = [os.path.join(gt_path, gt) for gt in gts]
            # 拼接路径为gt_paths
        if name == "images":
        # 重复上述操作，对images文件夹图片处理
            img_path = os.path.join(img_dir, name)
            imgs = os.listdir(img_path)
            imgs.sort(key=lambda x: int(x.split('_')[0]))
            img_paths = [os.path.join(img_path, img) for img in imgs]
    assert len(gt_paths) == len(img_paths), "数据集不匹配"
    # 断言并输出提示信息
    return img_paths, gt_paths
    # 返回images和groundtruth的图片


class CustomData(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths, self.gt_paths = load_img(img_dir)
        # load_img(img_dir)返回的值赋值
        self.transform = transform
        # 库自带的图片处理操作transform

    def __getitem__(self, item):
        # 得到相应的数值item
        img_path = self.img_paths[item]
        gt_path = self.gt_paths[item]
        image = cv2.imread(img_path, 0)
        # cv2.imread读取路径上图片，0为单通道即灰度
        gt = cv2.imread(gt_path, 0)
        """
        #  --求特征部分区域的二阶偏导数的离散梯度模量G----------------------------------------

        image_2 = image[117:137, 117:137]  # 裁剪中间区域20*20
        Kx = -1 * np.array([[-1, 0, 1]])  # -Derivative x
        Fx = ndimage.convolve(image_2, Kx)  # 仅沿x方向卷积
        Ky = -1 * np.array([[-1], [0], [1]])  # -Derivative y
        Fy = ndimage.convolve(image_2, Ky)  # 仅沿y方向卷积
        magnitude = np.sqrt(Fx ** 2 + Fy ** 2)  # # --Magnitude 模量G

        gt_2 = gt[117:137, 117:137]  # 裁剪中间区域20*20
        Kx_2 = -1 * np.array([[-1, 0, 1]])  # -Derivative x
        Fx_2 = ndimage.convolve(gt_2, Kx_2)  # 仅沿x方向卷积
        Ky_2 = -1 * np.array([[-1], [0], [1]])  # -Derivative y
        Fy_2 = ndimage.convolve(gt_2, Ky_2)  # 仅沿y方向卷积
        magnitude_2 = np.sqrt(Fx_2 ** 2 + Fy_2 ** 2)  # # --Magnitude 模量G

        # ---------------------------------------------------------------------------------
        """

        if self.transform is not None:
            transformed = self.transform(image=image, gt=gt)  # , image_2=image_2, gt_2=gt_2
            # 对图片进行变换，包括旋转90、180、270度、镜像等等。
            image = transformed['image']
            gt = transformed['gt']
            """
            # --专注于特征部分-------------------------------------------------------------------------------

            image_2 = image[117:137, 117:137]  # 裁剪中间区域20*20
            Kx = -1 * np.array([[-1, 0, 1]])  # -Derivative x
            Fx = ndimage.convolve(image_2, Kx)  # 仅沿x方向卷积
            Ky = -1 * np.array([[-1], [0], [1]])  # -Derivative y
            Fy = ndimage.convolve(image_2, Ky)  # 仅沿y方向卷积
            magnitude = np.sqrt(Fx ** 2 + Fy ** 2)  # # --Magnitude 模量G

            gt_2 = gt[117:137, 117:137]  # 裁剪中间区域20*20
            Kx_2 = -1 * np.array([[-1, 0, 1]])  # -Derivative x
            Fx_2 = ndimage.convolve(gt_2, Kx_2)  # 仅沿x方向卷积
            Ky_2 = -1 * np.array([[-1], [0], [1]])  # -Derivative y
            Fy_2 = ndimage.convolve(gt_2, Ky_2)  # 仅沿y方向卷积
            magnitude_2 = np.sqrt(Fx_2 ** 2 + Fy_2 ** 2)  # # --Magnitude 模量G

            #  -----------------------------------------------------------------------------------
            """
            # 归一化且转为张量tensor
            norm = ToTensor()
            # ToTensor作用：Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor，即图片转换张量。
            image = norm(image)
            gt = norm(gt)
            """
            magnitude = norm(magnitude)
            magnitude_2 = norm(magnitude_2)
            """
        if self.transform is None:
            # 若不进行图片变换，则直接进行张量转换。

            norm = ToTensor()
            image = norm(image)
            gt = norm(gt)
            """
            magnitude = norm(magnitude)
            magnitude_2 = norm(magnitude_2)
            """
        return image, gt         # , magnitude, magnitude_2
        # 返回张量image、gt、

    def __len__(self):
        return len(self.img_paths)
        # 定义len函数，即返回路径下所有图片个数。
