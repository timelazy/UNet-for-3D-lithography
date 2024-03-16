import argparse
from opttrain import *
from opttest import *

"""
main
"""

def print_info():
    print('torchvision版本:', torch.__version__)
    print('GPU是否可用:', torch.cuda.is_available())
    print('GPU个数:', torch.cuda.device_count())
    print('当前GPU个数:', torch.cuda.current_device())
    print('GPU设备名称:', torch.cuda.get_device_name(0))


def run():
    # argparse模块，其实质就是将相关参数进行设置
    parser = argparse.ArgumentParser(description="U-Net预测三维光刻微纳结构")
    parser.add_argument("--opt", type=str, default="test", help="train or test")
    parser.add_argument("--traindir", type=str, default="DATA/train", help="训练集地址")
    parser.add_argument("--testdir", type=str, default="DATA/test", help="测试集地址")
    # parser.add_argument("--pretrain", type=str, default="save/train_best_Unet.pth", help="模型预训练")
    #  载入预训练模型，loss function一开始就非常低
    parser.add_argument("--pretrain", type=str, default="", help="模型预训练")
    #  不载入预训练模型
    parser.add_argument('--savedir', type=str, default="save", help="训练模型保存")
    parser.add_argument("--epoch", type=int, default=100, help="迭代次数")
    parser.add_argument("--batch", type=int, default=1, help="训练批次大小")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.savedir):
    # 若存在预训练模型，直接调用
        os.mkdir(args.savedir)
    if args.opt == "train":
        if args.traindir == "":
            print("训练集路径不存在")
    if args.opt == "test":
        if args.testdir == "":
            print("测试集路径不存在")
    if args.opt == "train":
    # 开始训练
        train(args.traindir, args.epoch, args.batch, args.savedir, args.pretrain)
    if args.opt == "test":
    # 开始测试
        test(args.testdir)


if __name__ == '__main__':
    run()
    # print_info()
