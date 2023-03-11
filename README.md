# UNet-for-3D-lithography
UNet for 3D lithography
'''

2022.11.30
使用U-Net预测三维光刻微纳结构

'''

"""

Train: 在Terminal运行 python -m visdom.server 可视化训练过程；在新的Terminal运行 python main.py --opt train

Test: 在Terminal运行 python main.py --opt test

Predict: 单独运行predict.py

"""

import torch


'''
Device Check
'''

# 查看torchvision版本
print('torchvision版本:', torch.__version__)
# 查看gpu是否可用
print('GPU是否可用:', torch.cuda.is_available())
# 查看gpu个数
print('GPU个数:', torch.cuda.device_count())
# 查看当前gpu个数
print('当前GPU个数:', torch.cuda.current_device())
# 查看gpu设备名称
print('GPU设备名称:', torch.cuda.get_device_name(0))

