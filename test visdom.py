import visdom
import torch

"""
visdom容易安装失败，test visdom.py用于测试
"""

vis = visdom.Visdom()
x = torch.arange(1, 100, 0.01)
y = torch.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
