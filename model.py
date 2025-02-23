# 4.设置网络结构：
#########################################################################################################
import torch
from torch import nn

# 定义全连接神经网络（Fully Connected Neural Network, FCNN）的 PyTorch 模型，继承自 torch.nn.Module
# 1个输入层，2个隐藏层，1个输出层
class fully_connect_model(nn.Module):
    def __init__(self,input_shape:int,hidden_units_1:int,hidden_units_2:int,output_shape:int):
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units_1),
            nn.Linear(in_features=hidden_units_1,out_features=hidden_units_2),
            nn.Linear(in_features=hidden_units_2,out_features=output_shape),
            
        )
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)

