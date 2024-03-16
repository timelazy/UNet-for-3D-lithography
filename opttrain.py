import numpy as np
import visdom
from data import *
from transform import *
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from apex import amp
# amp混合精度训练

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
    # 可视化visdom部分env
    loss_fn = torch.nn.MSELoss()
    # 损失函数为mse
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 优化器为adam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gpu上训练
    # 训练
    model.train()
    model = model.to(device)
    # 模型加载到gpu上
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")  # "O0"单精度训练；"O1"混合精度训练；"O2""O3"
    for epoch in range(0, epochs):
        loss_lists = []
        for i, (img, gt) in enumerate(tqdm(train_data)):
            # enumerate为枚举类型。tqdm添加一个进度条提示信息。
            optimizer.zero_grad()
            # optimizer.zero_grad()梯度清零
            img, gt = img.to(device), gt.to(device)
            pred = model(img)
            # loss_1为微纳结构整体的MSE评估
            loss_1 = loss_fn(pred, gt)

            """
            #  -----------------------------------------------------------------------------
            # loss_2为微纳结构尖锐特征部分的梯度评估
            pred_2 = pred[0].cpu().detach().numpy().squeeze(0)
            # tensor是有梯度的, 要先detach去掉梯度, 然后再转到numpy类型
            # numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度。
            pred_2 = (pred_2 * 255.0).astype(np.float64)
            # 转为8位灰度
            pred_2 = pred_2[122:132, 122:132]  # 裁剪中间区域10*10（第一个有，最后一个没有）
            Kx = -1 * np.array([[-1, 0, 1]])  # -Derivative x
            Fx = ndimage.convolve(pred_2, Kx)  # 仅沿x方向卷积
            Ky = -1 * np.array([[-1], [0], [1]])  # -Derivative y
            Fy = ndimage.convolve(pred_2, Ky)  # 仅沿y方向卷积
            image_magnitude = np.sqrt(Fx ** 2 + Fy ** 2)  # # --Magnitude 模量G
            max_magnitude = image_magnitude.max()
            max_G = np.array(max_magnitude)
            max_G = torch.from_numpy(max_G)

            gt_2 = gt[0].cpu().detach().numpy().squeeze(0)
            gt_2 = (gt_2 * 255.0).astype(np.float64)
            gt_2 = gt_2[122:132, 122:132]  # 裁剪中间区域10*10
            Kx_2 = -1 * np.array([[-1, 0, 1]])  # -Derivative x
            Fx_2 = ndimage.convolve(gt_2, Kx_2)  # 仅沿x方向卷积
            Ky_2 = -1 * np.array([[-1], [0], [1]])  # -Derivative y
            Fy_2 = ndimage.convolve(gt_2, Ky_2)  # 仅沿y方向卷积
            image_magnitude_2 = np.sqrt(Fx_2 ** 2 + Fy_2 ** 2)  # # --Magnitude 模量G

            image_magnitude = torch.from_numpy(image_magnitude)
            image_magnitude_2 = torch.from_numpy(image_magnitude_2)
            loss_2 = loss_fn(image_magnitude.float(), image_magnitude_2.float())
            #  -----------------------------------------------------------------------------
            """

            """
            # --------------------------------------------------------------------------------
            # loss_3为设置的理想的非常尖锐的结构的梯度的模量最大值
            ideal_0 = cv2.imread('ideal2.bmp', 0).astype(np.float64)
            ideal_0 = ideal_0[122:132, 122:132]  # 裁剪中间区域10*10
            Kx_3 = -1 * np.array([[-1, 0, 1]])  # -Derivative x
            Fx_3 = ndimage.convolve(ideal_0, Kx_3)  # 仅沿x方向卷积
            Ky_3 = -1 * np.array([[-1], [0], [1]])  # -Derivative y
            Fy_3 = ndimage.convolve(ideal_0, Ky_3)  # 仅沿y方向卷积
            image_magnitude_3 = np.sqrt(Fx_3 ** 2 + Fy_3 ** 2)  # # --Magnitude 模量G
            max_magnitude3 = image_magnitude_3.max()
            max_G3 = np.array(max_magnitude3)
            max_G3 = torch.from_numpy(max_G3)

            loss_3 = loss_fn(max_G.float(), max_G3.float())
            # --------------------------------------------------------------------------------
            """

            # k1,k2,k3为权重系数
            # k1 = 1
            # k2 = 0.05
            # k3 = 0.05

            # loss = loss_1
            # loss = k1*loss_1 + k2*loss_2  # --改变loss function权重，让两个值尽量相近
            # loss = k1*loss_1 + k2*loss_2 + k3*loss_3  # --改变loss function权重，控制三种衡量标准

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                # loss.backward()反向传播
            # loss.backward()
            optimizer.step()
            # 更新网络参数
            loss_lists.append(loss.item())
            # 连续添加损失的数值点
            loss_content = [loss.item()]
            wind.line([loss_content],  # wind绘图
                      [epoch * ters_per_epoch + i],  # 横轴不是epoch，而是epoch*411
                      # [epoch + i],
                      win='train unet',
                      update='append',
                      opts={'legend': ['train_loss']})
        mean_loss = np.array(loss_lists).mean()
        # 计算平均损失
        if mean_loss < lossmin:
            lossmin = mean_loss
            torch.save(model.state_dict(), f'{savedir}/train_best_Unet.pth')
            # 保存训练好的模型到save文件夹

