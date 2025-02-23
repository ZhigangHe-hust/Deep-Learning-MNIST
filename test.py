import torch
from torch import nn
import matplotlib.pyplot as plt
from model import fully_connect_model
from timeit import default_timer as timer
from tqdm import tqdm # tqdm支持整个迭代器和嵌套迭代器的进度条
from dataloader import test_dataloader
from retrain import config_parser


# 定义准确率
def accuracy_fn(y_true,y_pred):
    correct=torch.eq(y_true,y_pred).sum().item()
    acc=(correct/len(y_pred))*100
    return acc

# 定义计时器，统计测试时间
def print_test_time(start:float,end:float,device:torch.device=None):
    total_time=end-start
    print(f"Test time on {device}: {total_time:.4f} seconds")

# 定义测试函数
def test(model, test_dataloader, loss_function, device):

    test_loss=0
    test_acc=0

    # 添加一个循环来遍历训练批次
    for batch, (x, y) in enumerate(tqdm(test_dataloader)):
        x, y = x.to(device), y.to(device)
        model.eval()
        
        # 1.前向传播
        y_pred=model(x)# X形状为 [batch_size, C, H, W]

        # 2.计算损失(每个批次)
        loss=loss_function(y_pred,y)
        test_loss+=loss# 累计每个batch的损失

        # 3.计算准确率
        acc=accuracy_fn(y_true=y,y_pred=y_pred.argmax(dim=1))
        test_acc+=acc# 累计每个batch的准确率

    # 计算该轮次的平均测试损失，平均测试准确率
    test_loss/=len(test_dataloader)
    test_acc/=len(test_dataloader)

    return test_loss,test_acc

def main():

    parser = config_parser()
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # 初始化模型
    model=fully_connect_model(input_shape=784,
        hidden_units_1=128,
        hidden_units_2=64,
        output_shape=10)
    model.to(device)

    # 加载模型权重
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 定义损失函数
    loss_function=nn.CrossEntropyLoss()# 交叉熵损失（Cross Entropy Loss）为常用的多分类损失函数

    # 设置测试的轮数
    epochs=10

    # 初始化列表来保存测试损失
    test_losses=[] # 每个epoch的平均损失
    test_acces=[]

    # 定义随机数种子并开始计时
    torch.manual_seed(42)
    test_time_start=timer()

    # 评估模型
    test_loss,test_acc=test(model=model,
        test_dataloader=test_dataloader,
        loss_function=loss_function,
        device=device)
    test_losses.append(test_loss)
    test_acces.append(test_acc)

    # 打印训练时间
    test_time_end=timer()
    print_test_time(start=test_time_start,
                    end=test_time_end,
                    device=str(next(model.parameters()).device))
    
    print(f"test_loss:{test_loss}")
    print(f"test_acc:{test_acc}")


# --------------------------------------main--------------------------------------------------
if __name__ == "__main__":
    main()
