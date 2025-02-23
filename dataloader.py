# 3.设置数据加载器：
#########################################################################################################

# 设置数据加载器，将数据集转换为一个一个batch
from torch.utils.data import DataLoader
from data import train_data,test_data

# 设置batch_size
BATCH_SIZE=32

# 设置训练和测试的数据加载器
train_dataloader=DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_dataloader=DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)
