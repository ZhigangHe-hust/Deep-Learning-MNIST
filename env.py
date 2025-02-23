# 1.设置环境：
#########################################################################################################

# 导入必要的模块
import torch
from torch import nn
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import sys
# from data import train_data,test_data

# 查看环境相关信息
print(f"Python version:{sys.version}")
print(f"Pytorch version:{torch.__version__}")
print(f"torchvision version:{torchvision.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"Current GPU Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


