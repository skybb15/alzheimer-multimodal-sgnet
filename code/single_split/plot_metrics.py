import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ==========================================
# 0. 核心组件定义 (必须与之前完全一致)
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
# 1. 全局配置与数据加载辅助函数
# ==========================================
# 设置 Matplotlib 的全局字体格式 (仿照 SCI 期刊风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print("\n🚀 正在加载原始数据集用于后续分割...")
full_train_df = load_and_concat_data(train_file)
full_test_df = load_and_concat_data(test_file)
real_label_col = 'Research Group'

# 定义三个任务的配置项 (模型权重路径, 数据过滤逻辑, 标签映射)
tasks_config = {
    'AD vs NC': {
        'weight_path': 'best_ad_vs_nc.pth',
        'valid_groups': ['AD', 'NC', 'CN'],
        'label_map': {'NC': 0, 'CN': 0, 'AD': 1},
        'color': '#d62728',  # 红色，代表基线任务高准确率
        'cm_labels': ['NC (0)', 'AD (1)']
    },
    'AD vs MCI': {
        'weight_path': 'best_ad_vs_mci.pth',
        'valid_groups': ['AD', 'MCI', 'EMCI', 'LMCI'],
        'label_map': {'MCI': 0, 'EMCI': 0, 'LMCI': 0, 'AD': 1},
        'color': '#1f77b4',  # 蓝色，代表核心攻坚任务
        'cm_labels': ['MCI (0)', 'AD (1)']
    },
    'LMCI vs EMCI': {
        'weight_path': 'best_lmci_vs_emci.pth',
        'valid_groups': ['EMCI', 'LMCI'],
        'label_map': {'EMCI': 0, 'LMCI': 1},
        'color': '#2ca02c',  # 绿色，代表极限界限任务
        'cm_labels': ['EMCI (0)', 'LMCI (1)']
    }
}

# 存储绘图需要的数据
roc_data = {}
cm_data = {}

# ==========================================
# 2. 遍历执行三个任务并收集预测数据
# ==========================================
for task_name, config in tasks_config.items():
    print(f"\n🔍 正在处理任务: {task_name}...")

    if not os.path.exists(config['weight_path']):
        print(f"⚠️ 警告: 找不到 {config['weight_path']}，跳过该任务绘图！请确保你已经跑完了训练保存了模型。")
        continue

    # 数据过滤与映射
    train_df = full_train_df[full_train_df[real_label_col].isin(config['valid_groups'])]
    test_df = full_test_df[full_test_df[real_label_col].isin(config['valid_groups'])]

    y_test = test_df[real_label_col].map(config['label_map']).values

    drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in drop_cols]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # 特征标准化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 加载模型
    model = S_GMNet(num_features=len(feature_cols), num_classes=2)
    model.load_state_dict(torch.load(config['weight_path']))
    model.eval()

    # 模型预测
    with torch.no_grad():
        logits, _ = model(X_test_t)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()  # 获取类别 1 的概率
        _, predicted = torch.max(logits, 1)
        y_pred = predicted.numpy()

    # 计算 ROC 和 混淆矩阵
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    roc_data[task_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'color': config['color']}
    cm_data[task_name] = {'cm': cm, 'labels': config['cm_labels']}

# ==========================================
# 3. 绘制顶刊级多任务 ROC 曲线图
# ==========================================
print("\n🎨 正在绘制综合 ROC 曲线...")
plt.figure(figsize=(8, 8))
for task_name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], color=data['color'], lw=2.5,
             label=f'{task_name} (AUC = {data["auc"]:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线 (瞎猜线)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) of S-GMNet', fontsize=16, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('ROC_Curves_Combined.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('ROC_Curves_Combined.png', format='png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ 已保存: ROC_Curves_Combined.pdf / .png")

# ==========================================
# 4. 绘制顶刊级混淆矩阵并排子图 (1x3)
# ==========================================
print("🎨 正在绘制混淆矩阵热力图...")
if len(cm_data) > 0:
    fig, axes = plt.subplots(1, len(cm_data), figsize=(6 * len(cm_data), 5))
    if len(cm_data) == 1:
        axes = [axes]  # 兼容只有一个任务的情况

    for ax, (task_name, data) in zip(axes, cm_data.items()):
        sns.heatmap(data['cm'], annot=True, fmt='d', cmap='Blues',
                    cbar=False, annot_kws={"size": 16, "weight": "bold"},
                    linewidths=1, linecolor='black', ax=ax)

        ax.set_title(task_name, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_xticklabels(data['labels'], fontsize=12)
        ax.set_yticklabels(data['labels'], fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig('Confusion_Matrices_Combined.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Confusion_Matrices_Combined.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ 已保存: Confusion_Matrices_Combined.pdf / .png")

print("\n🎉 所有学术图表已生成完毕！")
