import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ==========================================
# 0. 核心组件定义 (必须与训练时一模一样，否则读不懂权重)
# ==========================================
class DynamicGCN(nn.Module):
    def __init__(self, d_model):
        super(DynamicGCN, self).__init__()
        self.W_a = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_a(x)
        k = x.transpose(-1, -2)
        attn_scores = torch.matmul(q, k) / (D ** 0.5)
        adj_matrix = F.softmax(attn_scores, dim=-1)
        graph_out = torch.matmul(adj_matrix, x)
        graph_out = self.W_g(graph_out)
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
# 1. 配置评估任务与数据加载
# ==========================================
# 🌟 【开关】：在这里选择你要评估哪个任务的模型！
TASK = 'AD_vs_NC'  # 可选: 'AD_vs_NC', 'AD_vs_MCI' ‘LMCI_vs_EMCI'
MODEL_WEIGHTS = f'best_{TASK.lower()}.pth'

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print(f"\n🔍 正在加载数据，准备评估任务: {TASK} ...")
train_df = load_and_concat_data(train_file)
test_df = load_and_concat_data(test_file)
real_label_col = 'Research Group'

if TASK == 'AD_vs_NC':
    valid_groups = ['AD', 'NC', 'CN']
    label_map = {'NC': 0, 'CN': 0, 'AD': 1}
elif TASK == 'AD_vs_MCI':
    valid_groups = ['AD', 'MCI', 'EMCI', 'LMCI']
    label_map = {'MCI': 0, 'EMCI': 0, 'LMCI': 0, 'AD': 1}
elif TASK == 'LMCI_vs_EMCI':
    valid_groups = ['EMCI', 'LMCI']
    label_map = {'EMCI': 0, 'LMCI': 1}

train_df = train_df[train_df[real_label_col].isin(valid_groups)]
test_df = test_df[test_df[real_label_col].isin(valid_groups)]
y_train = train_df[real_label_col].map(label_map).values
y_test = test_df[real_label_col].map(label_map).values

drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in drop_cols]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# 必须使用与训练时相同的 scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ==========================================
# 2. 加载满级模型并进行全维度测试
# ==========================================
model = S_GMNet(num_features=len(feature_cols), num_classes=2)

if not os.path.exists(MODEL_WEIGHTS):
    print(f"❌ 找不到权重文件 {MODEL_WEIGHTS}，请先跑一遍训练脚本生成它！")
    exit()

model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.eval()
print(f"✅ 成功加载 {TASK} 的历史最佳模型权重！")

with torch.no_grad():
    # 获取模型的原始输出 (logits)
    logits, _ = model(X_test_t)
    # 获取预测的概率值 (用于计算 AUC)
    probs = F.softmax(logits, dim=1)[:, 1].numpy()
    # 获取最终的类别预测 (0 或 1)
    _, predicted = torch.max(logits, 1)
    y_pred = predicted.numpy()

# ==========================================
# 3. 输出顶刊级多维性能报告
# ==========================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print(f" S-GMNet  ({TASK})")
print("=" * 50)
print(f"🔹 准确率 (Accuracy):  {acc * 100:.2f}%  <- 整体预测对的比例")
print(f"🔹 精确率 (Precision): {prec * 100:.2f}%  <- 预测是AD的人中，有多少真是AD")
print(f"🔹 召回率 (Recall):    {rec * 100:.2f}%  <- 真正是AD的人中，找出了多少")
print(f"🔹 F1-Score:         {f1 * 100:.2f}%  <- 综合考虑精确率和召回率")
print(f"🔹 AUC-ROC:          {auc:.4f}   <- 区分正负样本的综合能力(0.5瞎猜, 1完美)")
print("-" * 50)
print("📌 混淆矩阵 (Confusion Matrix):")
print(f"             预测 0(非AD)   预测 1(AD)")
print(f"实际 0(非AD) |   {cm[0][0]:<10} |  {cm[0][1]:<10} |")
print(f"实际 1(AD)   |   {cm[1][0]:<10} |  {cm[1][1]:<10} |")
print("=" * 50)
