<p align="center">
    <h1 align="center">深度学习实践：MNIST手写数字识别</h1>
</p>


<br>
<div align="center">

[English](README.md) | 简体中文

</div>

<p align="center">
    🌟本项目使用<strong>全连接神经网络</strong>完成在<strong>MNIST数据集</strong>上的手写数字的分类任务，基于<strong>Pytorch</strong>实现。本项目可以作为基于Pytorch的深度学习项目开发的入门实践。
</p>


# 安装
## 依赖
```
Python 3.10
Pytorch 2.1.2
CUDA 12.1
torchvission 0.16.2
```
## 环境配置
### 1.创建环境并激活
```
conda create --name mnist python=3.10
conda activate mnist
```
你可以通过运行env_info.py查看Python，Pytorch，CUDA的版本信息
```
python env.py
```
运行结果如下（参考）：
![image](https://github.com/ZhigangHe-hust/Deep-Learning-MNIST/blob/main/figs/fig2.png)
### 2.安装Pytorch和cuda
Pytorch2.1.2和CUDA11.8是基于RTX3060安装的，你可以根据你的显卡型号安装合适版本的Pytorch和CUDA
```
conda install pytorch=2.1.2 torchvision cudatoolkit=11.8 -c pytorch -c nvidia
```
### 3.安装其他模块
```
conda install tqdm
conda install matplotlib
pip install numpy==1.23.5
```

# 准备数据
##  MNIST数据集简介
### 概述
MNIST（Modified National Institute of Standards and Technology）是一个经典的手写数字图像数据集，广泛用于机器学习和深度学习领域的入门教程和基准测试。
### 数据集内容
**图像数量**：70,000 张灰度图像<br>
**训练集**：60,000 张<br>
**测试集**：10,000 张<br>
**图像格式**：28x28 的灰度图像<br>
**标签**：每张图像对应一个 0 到 9 的数字标签<br>
### 数据示例
以下是 MNIST 数据集的示例图像：
![image](https://github.com/ZhigangHe-hust/Deep-Learning-MNIST/blob/main/figs/fig1.png)
## 数据集下载
MNIST已经集成在了Pytorch中，所以你可以直接通过脚本下载
```
python data.py
```

# 模型训练
在训练之前，你需要在当前目录下创建一个文件夹用来存储检查点文件
```
├─ data.py
├─ dataloader.py
├─ ···
├─ output
│    ├─ checkpoint
```
在终端输入
```
python train.py
```
如果模型训练意外终止，或者想添加训练的轮次，可以在终端输入如下命令：
```
python retrain.py --checkpoint_path output\checkpoint\checkpoint_epoch_N.pth --epochs M
# M是最终模型总的训练轮次数（默认为10）
# checkpoint_epoch_N是你最后一个检查点文件
```

# 模型测试
```
python test.py
```

# 联系我们

如果有其他问题❓，请及时与我们联系 👬
