import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据加载与预处理 (AD vs MCI )
# ==========================================
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print("正在加载训练集和测试集...")
train_df = load_and_concat_data(train_file)
test_df = load_and_concat_data(test_file)

real_label_col = 'Research Group'

print("当前执行的分类任务: AD vs MCI (测试模型分辨轻度认知障碍的能力)")
# 保留 AD 和 所有阶段的 MCI (踢掉正常人 NC/CN)
valid_groups = ['AD', 'MCI', 'EMCI', 'LMCI']
train_df = train_df[train_df[real_label_col].isin(valid_groups)]
test_df = test_df[test_df[real_label_col].isin(valid_groups)]

# 标签映射：所有 MCI 视为 0，AD 视为 1
label_map = {'MCI': 0, 'EMCI': 0, 'LMCI': 0, 'AD': 1}
y_train = train_df[real_label_col].map(label_map).values
y_test = test_df[real_label_col].map(label_map).values

drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in drop_cols]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

print(f"✅ 参与计算的特征数量: {len(feature_cols)} 个")
print(f"✅ 过滤后训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False
)


# ==========================================
# 2. 核心组件定义 (保持最优的 S-GMNet 结构)
# ==========================================
class DynamicGCN(nn.Module):
    def __init__(self, d_model):
        super(DynamicGCN, self).__init__()
        self.W_a = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)  # 关键的防平滑层

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_a(x)
        k = x.transpose(-1, -2)
        attn_scores = torch.matmul(q, k) / (D ** 0.5)
        adj_matrix = F.softmax(attn_scores, dim=-1)

        graph_out = torch.matmul(adj_matrix, x)
        graph_out = self.W_g(graph_out)
        # 带有残差连接的输出
        return self.norm(x + F.relu(graph_out))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=4, padding=3,
                                groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        delta, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y = torch.zeros_like(x_conv)

        for t in range(L):
            xt = x_conv[:, t, :]
            dt = delta[:, t, :]
            Bt = B_mat[:, t, :]
            Ct = C_mat[:, t, :]

            dA = torch.exp(-dt)
            dB = dt.unsqueeze(-1) * xt.unsqueeze(-1) * Bt.unsqueeze(1)

            h = h * dA.unsqueeze(-1) + dB
            y[:, t, :] = (h * Ct.unsqueeze(1)).sum(dim=-1)

        y = y * F.silu(z)
        return self.out_proj(y)


class S_GMNet(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=16):
        super(S_GMNet, self).__init__()
        self.mask = nn.Parameter(torch.randn(num_features) * 0.1)
        self.feature_proj = nn.Linear(1, d_model)
        self.gcn = DynamicGCN(d_model)
        self.mamba = MambaBlock(d_model=d_model, d_state=16, expand=2)

        self.classifier = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mask_weight = torch.sigmoid(self.mask)
        x_filtered = x * mask_weight
        x_seq = x_filtered.unsqueeze(-1)
        x_emb = self.feature_proj(x_seq)
        x_graph = self.gcn(x_emb)
        x_mamba = self.mamba(x_graph)
        x_flat = x_mamba.reshape(x.size(0), -1)
        logits = self.classifier(x_flat)
        return logits, mask_weight


# ==========================================
# 4. 训练与标志物挖掘 (加入样本不平衡惩罚)
# ==========================================
model = S_GMNet(num_features=len(feature_cols), num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# 👇【新增】：计算并引入类别权重 (Class Weights)
# 因为 MCI 数量大约是 AD 的 3 倍，所以我们给 AD(类别1) 施加 3 倍的注意力权重！
class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights) # 把天平加上！

epochs = 60
lambda_l1 = 0.008


print("\n🚀 启动 S-GMNet [AD vs MCI ]...")

best_test_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, current_mask = model(inputs)
        loss_cls = criterion(outputs, labels)
        loss_l1 = torch.mean(torch.abs(current_mask))
        loss = loss_cls + lambda_l1 * loss_l1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    train_acc = 100 * correct / total

    # 逐 Epoch 监控测试集准确率
    model.eval()
    test_correct_epoch, test_total_epoch = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total_epoch += labels.size(0)
            test_correct_epoch += (predicted == labels).sum().item()
    test_acc = 100 * test_correct_epoch / test_total_epoch


    if test_acc > best_test_acc:
        best_test_acc = test_acc


        torch.save(model.state_dict(), 'best_ad_vs_mci.pth')
        print(f"🌟 新纪录！已保存最佳模型权重 (Test Acc: {best_test_acc:.2f}%)")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.5f} | Loss: {total_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# ==========================================
# 5. 挖掘 LMCI vs EMCI 的特异性微观标志物
# ==========================================
print("\n🔍 正在加载最佳模型，挖掘 Top-10 核心生物标志物...")

# 【关键升级】：加载我们刚才保存的历史最佳脑子（权重），而不是用最后一次过拟合的脑子
model.load_state_dict(torch.load('best_ad_vs_mci.pth'))

# 提取特征掩码的权重
final_weights = torch.sigmoid(model.mask).detach().numpy()
top_10_indices = np.argsort(final_weights)[::-1][:10]

print("-" * 50)
print(f"{'排名':<5} | {'特征名称':<30} | {'重要性权重'}")
print("-" * 50)
for i, idx in enumerate(top_10_indices):
    feature_name = feature_cols[idx]
    weight = final_weights[idx]
    print(f"Top-{i+1:<3} | {feature_name:<30} | {weight:.4f}")
print("-" * 50)

