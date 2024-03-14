import albumentations as A

"""
Transform，对原始数据集进行数据增强操作，增加丰富性。
"""
# RandomRotate90（随机旋转90度）；Transpose（转置）
# HorizontalFlip（围绕Y轴水平翻转）；VerticalFlip（围绕X轴垂直翻转）
# 以给定概率p来执行操作


def Traintransformer():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.OneOf([A.HorizontalFlip(p=0.3),
                 A.VerticalFlip(p=0.3),
                 A.Transpose(p=0.2)], p=0.6),
    ])
    return transform

