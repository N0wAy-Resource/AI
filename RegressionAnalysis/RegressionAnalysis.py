import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('000300perf.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['t'] = range(len(df))  # 创建时间步

# 数据准备
x = df['t'].values.reshape(-1, 1)
# 使用指数移动平均预处理
df['Close_Smooth'] = df['Close'].ewm(span=10, adjust=False).mean()

# 重定义目标变量
y = df['Close_Smooth'].values.reshape(-1, 1)
#y = df['Close'].values.reshape(-1, 1)

# 归一化时间序列
scaler_x = MinMaxScaler(feature_range=(-1, 1))
x_scaled = scaler_x.fit_transform(x)

# 生成多项式特征
degree = 6
poly = PolynomialFeatures(degree=degree, include_bias=False)
poly_features = poly.fit_transform(x_scaled)

# 标准化多项式特征
scaler_poly = StandardScaler()
poly_features_scaled = scaler_poly.fit_transform(poly_features)

# 目标值归一化到正数区间
scaler_y = MinMaxScaler(feature_range=(0.1, 1))  # 保留10%安全边际
y_scaled = scaler_y.fit_transform(y)

# 转换为PyTorch张量
x_tensor = torch.tensor(poly_features_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# 构建带约束模型
class SafePolyRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus()  # 平滑的非线性激活

    def forward(self, x):
        return self.softplus(self.linear(x))  # 保证输出>0


# 训练配置调整
model = SafePolyRegression(input_dim=poly_features.shape[1])
optimizer = torch.optim.AdamW(model.parameters(),  # 使用AdamW优化器
                              lr=0.001,
                              weight_decay=1e-3)  # 更强的正则化

# 添加早停机制
best_loss = float('inf')
patience = 20
no_improve = 0

# 定义损失函数
criterion = nn.MSELoss()

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# 预测未来30天
future_days = 30
future_t = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
future_t_scaled = scaler_x.transform(future_t)
future_poly = poly.transform(future_t_scaled)
future_poly_scaled = scaler_poly.transform(future_poly)
future_tensor = torch.tensor(future_poly_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    future_pred_scaled = model(future_tensor).numpy()
future_pred = scaler_y.inverse_transform(future_pred_scaled)
df=pd.read_excel('000300perf.xlsx')

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Historical Price')
future_dates = [df['Date'].iloc[-1] + i for i in range(1,31)]
plt.plot(future_dates, future_pred, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 输出预测结果
print("\n接下来30天的收盘价:")
for date, price in zip(future_dates, future_pred):
    print(f"{date}: {price[0]:.2f}")
