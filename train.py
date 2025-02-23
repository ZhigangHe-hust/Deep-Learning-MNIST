# 导入必要的模块
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import fully_connect_model
from torch.optim import SGD
from timeit import default_timer as timer
from tqdm import tqdm # tqdm支持整个迭代器和嵌套迭代器的进度条
from dataloader import train_dataloader

# 定义计时器，统计训练时间
def print_train_time(start:float,end:float,device:torch.device=None):
    total_time=end-start
    print(f"Train time on {device}:{total_time:.4f} seconds")

# 定义训练函数
def train(model, train_dataloader, optimizer, loss_function, device, epoch):

    train_loss=0

    # 添加一个循环来遍历训练批次
    for batch, (x, y) in enumerate(tqdm(train_dataloader)):
        x, y = x.to(device), y.to(device)
        model.train()
        
        # 1.前向传播
        y_pred=model(x)# X形状为 [batch_size, C, H, W]

        # 2.计算损失(每个批次)
        loss=loss_function(y_pred,y)
        train_loss+=loss# 累计每个batch的损失

        # 3.优化梯度清零
        optimizer.zero_grad()

        # 4.反向传播损失
        loss.backward()

        # 5.优化器更新参数
        optimizer.step()

    # 计算该轮次的平均训练损失
    train_loss/=len(train_dataloader)
    print(f"train_loss:{train_loss}")

    # 保存模型和优化器状态（断点续训）
    checkpoint_path = f'output\checkpoint\checkpoint_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    return train_loss

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================")
    print(f"Current device: {device}")

    # 创建模型并把模型放在设备上
    model=fully_connect_model(input_shape=784,
        hidden_units_1=128,
        hidden_units_2=64,
        output_shape=10)
    model.to(device)

    # 定义损失函数和优化器
    loss_function=nn.CrossEntropyLoss()# 交叉熵损失（Cross Entropy Loss）为常用的多分类损失函数
    optimizer=SGD(model.parameters(),lr=0.01,momentum=0.9)#使用SGD优化器，学习率设为0.01，动量设置0.9

    # 设置训练的轮数
    epochs=10

    # 初始化列表来保存训练损失
    train_losses=[] # 每个epoch的平均损失

    # 定义随机数种子并开始计时
    torch.manual_seed(42)
    train_time_start=timer()

    # 训练模型
    for epoch in range(epochs):
        print("------------------------------------")
        print(f"Epoch:{epoch}")
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