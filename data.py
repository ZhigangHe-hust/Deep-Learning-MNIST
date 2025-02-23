# 2.准备数据：
#==============================================================================================

import torchvision

# 在 PyTorch 中，当使用 torchvision.datasets.MNIST 加载数据集时，返回的 train_data 是一个Dataset 对象，它的本质是继承自 torch.utils.data.Dataset 的子类
train_data=torchvision.datasets.MNIST(
    root='MNIST',# 数据存储路径
    train=True,# 表示加载训练集 (True) 或测试集 (False)
    transform=torchvision.transforms.ToTensor(),# 数据预处理，将图像转换为 Tensor 格式
    download=True # 如果数据集不存在，则自动下载
)

test_data=torchvision.datasets.MNIST(
    root='MNIST',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

