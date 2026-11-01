import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, \
    roc_curve
from imblearn.over_sampling import SMOTE
import copy

# ==========================================
# 🎯 0. 核心任务控制器 (默认：大合并版 MCI vs AD)
# ==========================================
TASK_TYPE = 2

if TASK_TYPE == 1:
    task_name = "AD vs NC (剔除所有 MCI 干扰)"
    valid_groups = ['NC', 'AD']
    label_map = {'NC': 0, 'AD': 1}
elif TASK_TYPE == 2:
    task_name = "NC vs 全部 MCI (包含常规、早期 EMCI 与晚期 LMCI)"
    valid_groups = ['MCI', 'EMCI', 'LMCI', 'NC']
    label_map = {'MCI': 0, 'EMCI': 0, 'LMCI': 0, 'NC': 1}
else:
    task_name = "LMCI vs EMCI (地狱难度 Pro 级突破)"
    valid_groups = ['EMCI', 'LMCI']
    label_map = {'EMCI': 0, 'LMCI': 1}


# ==========================================
# 1. 基础组件定义 (Focal Loss, Dynamic GCN)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()


class DynamicGCN(nn.Module):
    def __init__(self, d_model):
        super(DynamicGCN, self).__init__()
        self.W_a = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        q = self.W_a(x)
        k = x.transpose(-1, -2)
        attn_scores = torch.matmul(q, k) / (x.shape[-1] ** 0.5)
        adj_matrix = F.softmax(attn_scores, dim=-1)
        graph_out = torch.matmul(adj_matrix, x)
        graph_out = self.W_g(graph_out)
        return self.norm(x + F.relu(graph_out))


# 🚀 替换 Mamba 的核心模块：Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=64, dropout=0.2):
        super().__init__()
        # batch_first=True 确保输入形状为 (Batch, Seq_Len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        return self.transformer(x)


# ==========================================
# 2. 🔥 核心：Transformer 版本的主网络
# ==========================================
class Transformer_S_GMNet(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=16):
        super(Transformer_S_GMNet, self).__init__()

        # 稀疏掩码
        self.mask = nn.Parameter(torch.randn(num_features) * 1.0)
        self.feature_proj = nn.Linear(1, d_model)

        # 动态图卷积
        self.gcn = DynamicGCN(d_model)

        # 🚀 序列处理模块换成 Transformer
        self.sequence_model = TransformerBlock(d_model=d_model, nhead=4)

        self.classifier = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 1. Mask 极化降噪
        mask_weight = torch.sigmoid(self.mask / 0.1)
        x_filtered = x * mask_weight

        # 特征升维 (B, N) -> (B, N, 1) -> (B, N, d_model)
        x_seq = x_filtered.unsqueeze(-1)
        x_emb = self.feature_proj(x_seq)

        # 2. GCN 建图
        x_graph = self.gcn(x_emb)

        # 3. 🚀 Transformer 全局注意力扫描
        x_seq_out = self.sequence_model(x_graph)

        # 4. 分类输出
        x_flat = x_seq_out.reshape(x.size(0), -1)
        logits = self.classifier(x_flat)

        return logits, mask_weight


# ==========================================
# 3. 数据加载
# ==========================================
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print(f"正在加载数据以进行 Transformer 基线对比实验...")
all_df = pd.concat([load_and_concat_data(train_file), load_and_concat_data(test_file)], ignore_index=True)
real_label_col = 'Research Group'

all_df = all_df[all_df[real_label_col].isin(valid_groups)]
y_all = all_df[real_label_col].map(label_map).values
drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
feature_cols = [col for col in all_df.select_dtypes(include=['number']).columns if col not in drop_cols]
X_all_unscaled = all_df[feature_cols].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前任务: {task_name} | 数据集大小: {X_all_unscaled.shape} | 设备: {device}")

# ==========================================
# 4. 十折交叉验证执行逻辑
# ==========================================
print("\n" + "=" * 50)
print(f"🚀 正在运行: Transformer_S_GMNet ")
print("=" * 50)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
epochs, lambda_l1, batch_size = 60, 0.01, 32
fold_metrics = {'acc': [], 'auc': [], 'f1': [], 'recall': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_unscaled, y_all)):
    X_train_fold, y_train_fold = X_all_unscaled[train_idx], y_all[train_idx]
    X_val_fold, y_val_fold = X_all_unscaled[val_idx], y_all[val_idx]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val_fold)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                       torch.tensor(y_train_resampled, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                       torch.tensor(y_val_fold, dtype=torch.long)),
        batch_size=batch_size, shuffle=False)

    # 🚀 实例化 Transformer 版本网络
    model = Transformer_S_GMNet(num_features=len(feature_cols), num_classes=2, d_model=16).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = FocalLoss(alpha=0.5, gamma=2.0).to(device)

    best_val_auc, best_model_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, current_mask = model(inputs)

            # 计算总 Loss (分类 Loss + L1 正则化惩罚)
            loss_cls = criterion(outputs, labels)
            loss_l1 = torch.mean(torch.abs(current_mask))
            loss = loss_cls + lambda_l1 * loss_l1

            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证寻找最佳 epoch
        model.eval()
        epoch_probs, epoch_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                epoch_probs.extend(probs)
                epoch_targets.extend(labels.cpu().numpy())
        val_auc = roc_auc_score(epoch_targets, epoch_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())

    # 加载最佳权重计算测试指标
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    # 动态阈值截断
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    all_preds = (np.array(all_probs) >= best_thresh).astype(int)

    fold_metrics['acc'].append(accuracy_score(all_targets, all_preds) * 100)
    fold_metrics['auc'].append(roc_auc_score(all_targets, all_probs))
    fold_metrics['recall'].append(recall_score(all_targets, all_preds, zero_division=0) * 100)
    fold_metrics['f1'].append(f1_score(all_targets, all_preds, zero_division=0) * 100)

    print(
        f"  ↳ 第 {fold + 1}/10 折完成 | AUC: {fold_metrics['auc'][-1]:.4f} | Recall: {fold_metrics['recall'][-1]:.2f}% | F1: {fold_metrics['f1'][-1]:.2f}%")

# ==========================================
# 5. 打印最终 Transformer 成绩单
# ==========================================
print("\n" + "🔥" * 25)
print(f"📊 Transformer_S_GMNet 最终成绩单 ({task_name})")
print("🔥" * 25)
print(f"Accuracy: {np.mean(fold_metrics['acc']):.2f} ± {np.std(fold_metrics['acc']):.2f} %")
print(f"AUC:      {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
print(f"Recall:   {np.mean(fold_metrics['recall']):.2f} ± {np.std(fold_metrics['recall']):.2f} %")
print(f"F1-Score: {np.mean(fold_metrics['f1']):.2f} ± {np.std(fold_metrics['f1']):.2f} %")
print("-" * 50)
print("📌 请拿这组数据与之前的 'Ours (Full S-GMNet Pro)' 也就是 Mamba 版本的数据进行对打！")
