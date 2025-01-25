import copy
import time
import torch
import mat73
import scipy.io
from munch import DefaultMunch
import yaml
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from yolo_model import Neck_Head
from ViT_model import VisionTransformer
from models.t2t_vit import T2T_ViT
from models.t2t_vit_se import T2T_ViT_SE
from Residual_model import Residual, RESnet
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from revit.model.revit_model import ReViT
from DNN import DNN


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return torch.tensor(data=self.X[idx]).float(), torch.tensor(data=self.y[idx]).float()  # 相当于元组

    def __len__(self):
        return len(self.X)


data_dir = ''
model_dir = ''

read_temp = mat73.loadmat(data_dir)
X_Train = read_temp['X_Train']
X_Train = np.transpose(X_Train, (0, 3, 1, 2))
X_Validation = read_temp['X_Validation']
X_Validation = np.transpose(X_Validation, (0, 3, 1, 2))
Y_Train = read_temp['Y_Train']
Y_Validation = read_temp['Y_Validation']

# 创建训练集和验证集的数据集对象
train_dataset = MyDataset(X_Train, Y_Train)
val_dataset = MyDataset(X_Validation, Y_Validation)

model = VisionTransformer()
# 模型参数量
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:", total_params)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 30
batch_size = 128

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print('device:', device)

start_time = time.time()

val_losses = []
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 前向传播
        # outputs1, outputs2 = model(inputs)
        #
        # loss1 = criterion(outputs1, targets)
        # loss2 = criterion(outputs2, targets)
        # loss = loss1*0.8 + loss2*0.2
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

    print("Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}".format(epoch + 1, num_epochs, loss, val_loss))

    # 更新学习率
    scheduler.step()
    torch.cuda.empty_cache()

# 保存模型
torch.save(model.state_dict(), model_dir)
rmae = np.sqrt(np.mean(np.square(val_losses)))
print('RMAE:', rmae)
# 训练总用时
stop_time = time.time()
total_time = stop_time - start_time
print('total_time:', total_time)

#
# # 绘制训练曲线
# epochs = range(1, len(val_losses) + 1)
# plt.plot(epochs, val_losses, 'b-o')
# plt.xlabel('Epoch')
# plt.ylabel('Validation Loss')
# plt.title('Validation Loss vs. Epoch')
# plt.savefig('Result/R2_15_100_VIT.png')
# plt.show()
