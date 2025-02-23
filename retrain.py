from train import train
import torch
import os
from torch import nn
from dataloader import train_dataloader
from timeit import default_timer as timer
from train import print_train_time
import matplotlib.pyplot as plt
import argparse
from model import fully_connect_model
from torch.optim import SGD


# 定义命令行参数
def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path', required=True, type=str, help='checkpoint file path')
    parser.add_argument("--epochs", default=10, type=int, help='total training rounds')
    return parser

# 加载检查点
def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, optimizer, epoch

def main():
    # checkpoint_path,epochs设置为命令行输入
    parser = config_parser()
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model=fully_connect_model(input_shape=784,
        hidden_units_1=128,
        hidden_units_2=64,
        output_shape=10)
    model.to(device)

    # 定义损失函数和优化器
    loss_function=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(),lr=0.01,momentum=0.9)

    # 初始化列表来保存训练损失
    train_losses=[] 

    # 如果检查点路径输入正确，加载检查点并恢复训练
    if os.path.exists(args.checkpoint_path):
        model, optimizer, start_epoch= load_checkpoint(model, optimizer, args.checkpoint_path, device)
        print(f"Resuming training from epoch {start_epoch+1}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # 定义随机数种子并开始计时
    torch.manual_seed(42)
    train_time_start=timer()

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch:{epoch}")
        print("------------------------------------")
        train_loss=train(model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=epoch)
        train_losses.append(train_loss)

    # 打印训练时间
    train_time_end=timer()
    print_train_time(start=train_time_start,
                    end=train_time_end,
                    device=str(next(model.parameters()).device))
    
    # 绘制loss曲线
    train_losses = torch.tensor(train_losses)
    train_losses_cpu = train_losses.cpu().detach().numpy()# 将张量从GPU移动到CPU,然后再转换为NumPy数组
    plt.figure(figsize=(6,4))
    plt.plot(train_losses_cpu,label='train_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.show()

# --------------------------------------main--------------------------------------------------
if __name__ == "__main__":
    main()