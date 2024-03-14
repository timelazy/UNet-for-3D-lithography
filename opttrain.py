import numpy as np
import visdom
from data import *
from transform import *
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from apex import amp


def train(train_path, epochs, batch, savedir, pretrain):
    # 加载数据集
    dataset = CustomData(train_path, Traintransformer())
    train_data = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    # 创建模型
    model = UNet(in_channels=1, out_channels=1)
    # 模型预训练
    if pretrain:
        pretrained_dict = torch.load(pretrain, map_location='cpu')
        pretrained_dict = pretrained_dict.get("unet", pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=True)
    # 训练参数
    lr = 0.0001
    lossmin = 1000
    ters_per_epoch = len(dataset) // batch  # example: 3288/8=411
    wind = visdom.Visdom(env='Loss Function')
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练
    model.train()
    model = model.to(device)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    for epoch in range(0, epochs):
        loss_lists = []
        for i, (img, gt) in enumerate(tqdm(train_data)):
            optimizer.zero_grad()
            img, gt = img.to(device), gt.to(device)
            pred = model(img)
            loss = loss_fn(pred, gt)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()
            loss_lists.append(loss.item())
            loss_content = [loss.item()]
            wind.line([loss_content],
                      [epoch * ters_per_epoch + i],  #  横轴不是epoch，而是epoch*411
                      # [epoch + i],
                      win='train unet',
                      update='append',
                      opts={'legend': ['train_loss']})
        mean_loss = np.array(loss_lists).mean()
        if mean_loss < lossmin:
            lossmin = mean_loss
            torch.save(model.state_dict(), f'{savedir}/train_best_Unet.pth')

