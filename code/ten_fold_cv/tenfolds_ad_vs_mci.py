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
# 🎯 0. 核心任务控制器 (当前已锁定为大合并版 MCI vs AD)
# ==========================================
TASK_TYPE = 2

if TASK_TYPE == 1:
    task_name = "AD vs NC "
    valid_groups = ['NC', 'AD']
    label_map = {'NC': 0, 'AD': 1}
elif TASK_TYPE == 2:
    # 🚀 核心修改位置：将所有阶段的 MCI 全部合并，并统一定义为类别 0
    task_name = "NC vs 全部 MCI (包含常规、早期 EMCI 与晚期 LMCI)"
    valid_groups = ['MCI', 'EMCI', 'LMCI', 'NC']
    label_map = {'MCI': 0, 'EMCI': 0, 'LMCI': 0, 'NC': 1}
else:
    task_name = "LMCI vs EMCI "
    valid_groups = ['EMCI', 'LMCI']
    label_map = {'EMCI': 0, 'LMCI': 1}


# ==========================================
# 1. Focal Loss (焦点损失函数)
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

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ==========================================
# 2. 数据加载与预处理模块
# ==========================================
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print(f"正在加载并合并所有数据以进行十折交叉验证...")
train_df = load_and_concat_data(train_file)
test_df = load_and_concat_data(test_file)

all_df = pd.concat([train_df, test_df], ignore_index=True)
real_label_col = 'Research Group'

print(f"\n🔥 当前执行的分类任务: {task_name}")
# 严格过滤当前任务需要的群体
all_df = all_df[all_df[real_label_col].isin(valid_groups)]

y_all = all_df[real_label_col].map(label_map).values

drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
numeric_cols = all_df.select_dtypes(include=['number']).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in drop_cols]

X_all_unscaled = all_df[feature_cols].values

print(f"✅ 参与计算的特征数量: {len(feature_cols)} 个")
print(f"✅ 合并后总数据集大小: {X_all_unscaled.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前使用的计算设备: {device}")


# ==========================================
# 3. 核心组件定义 (S-GMNet )
# ==========================================
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
        self.mask = nn.Parameter(torch.randn(num_features) * 1.0)
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
        mask_weight = torch.sigmoid(self.mask / 0.1)
        x_filtered = x * mask_weight
        x_seq = x_filtered.unsqueeze(-1)
        x_emb = self.feature_proj(x_seq)
        x_graph = self.gcn(x_emb)
        x_mamba = self.mamba(x_graph)
        x_flat = x_mamba.reshape(x.size(0), -1)
        logits = self.classifier(x_flat)
        return logits, mask_weight


# ==========================================
# 4. 严格分层十折交叉验证流程
# ==========================================
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

epochs = 60
lambda_l1 = 0.01
batch_size = 32

fold_metrics = {'acc': [], 'auc': [], 'cm': [], 'precision': [], 'recall': [], 'f1': [], 'best_thresh': []}
fold_masks = []

print(f"\n🚀 启动 S-GMNet  混合MCI vs NC 十折交叉验证...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_unscaled, y_all)):
    print(f"\n[{'-' * 15} 正在训练第 {fold + 1}/{k_folds} 折 {'-' * 15}]")

    X_train_fold, y_train_fold = X_all_unscaled[train_idx], y_all[train_idx]
    X_val_fold, y_val_fold = X_all_unscaled[val_idx], y_all[val_idx]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val_fold)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                                   torch.tensor(y_train_resampled, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                                 torch.tensor(y_val_fold, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = S_GMNet(num_features=len(feature_cols), num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    criterion = FocalLoss(alpha=0.5, gamma=2.0).to(device)

    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, current_mask = model(inputs)
            loss_cls = criterion(outputs, labels)
            loss_l1 = torch.mean(torch.abs(current_mask))
            loss = loss_cls + lambda_l1 * loss_l1
            loss.backward()
            optimizer.step()

        scheduler.step()

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

    # 测试最佳权重
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

    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    youden_j = tpr - fpr
    best_thresh_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_thresh_idx]

    all_preds = (np.array(all_probs) >= best_threshold).astype(int)

    fold_acc = accuracy_score(all_targets, all_preds) * 100
    fold_auc = roc_auc_score(all_targets, all_probs)
    fold_cm = confusion_matrix(all_targets, all_preds)
    fold_precision = precision_score(all_targets, all_preds, zero_division=0) * 100
    fold_recall = recall_score(all_targets, all_preds, zero_division=0) * 100
    fold_f1 = f1_score(all_targets, all_preds, zero_division=0) * 100

    fold_metrics['acc'].append(fold_acc)
    fold_metrics['auc'].append(fold_auc)
    fold_metrics['cm'].append(fold_cm)
    fold_metrics['precision'].append(fold_precision)
    fold_metrics['recall'].append(fold_recall)
    fold_metrics['f1'].append(fold_f1)
    fold_metrics['best_thresh'].append(best_threshold)

    fold_mask_weight = torch.sigmoid(model.mask / 0.1).cpu().detach().numpy()
    fold_masks.append(fold_mask_weight)

    print(
        f"👉 第 {fold + 1} 折 | 最佳阈值: {best_threshold:.3f} | Acc: {fold_acc:.2f}% | AUC: {fold_auc:.4f} | Recall: {fold_recall:.2f}% | F1: {fold_f1:.2f}%")

# ==========================================
# 5. 十折交叉验证总结与极化标志物挖掘
# ==========================================
print("\n" + "=" * 50)
print(f"🏆 十折交叉验证最终成绩报告 (Pro版本 | 大合并版 MCI vs NC)")
print("=" * 50)

metrics_summary = {
    'Accuracy': (np.mean(fold_metrics['acc']), np.std(fold_metrics['acc'])),
    'AUC': (np.mean(fold_metrics['auc']), np.std(fold_metrics['auc'])),
    'Precision': (np.mean(fold_metrics['precision']), np.std(fold_metrics['precision'])),
    'Recall': (np.mean(fold_metrics['recall']), np.std(fold_metrics['recall'])),
    'F1-Score': (np.mean(fold_metrics['f1']), np.std(fold_metrics['f1']))
}

print(f"📊 平均 Accuracy:   {metrics_summary['Accuracy'][0]:.2f}% ± {metrics_summary['Accuracy'][1]:.2f}%")
print(f"📈 平均 AUC:        {metrics_summary['AUC'][0]:.4f}  ± {metrics_summary['AUC'][1]:.4f}")
print(f"🎯 平均 Precision:  {metrics_summary['Precision'][0]:.2f}% ± {metrics_summary['Precision'][1]:.2f}%")
print(f"🧲 平均 Recall:     {metrics_summary['Recall'][0]:.2f}% ± {metrics_summary['Recall'][1]:.2f}%")
print(f"⚖️ 平均 F1-Score:   {metrics_summary['F1-Score'][0]:.2f}% ± {metrics_summary['F1-Score'][1]:.2f}%")
print(f"⚙️ 平均最佳截断阈值: {np.mean(fold_metrics['best_thresh']):.3f}")

total_cm = np.sum(fold_metrics['cm'], axis=0)
print(f"\n🔍 全局累加混淆矩阵 (总计):\n{total_cm}")

avg_mask_weights = np.mean(fold_masks, axis=0)
top_10_indices = np.argsort(avg_mask_weights)[::-1][:10]

print("\n🧬 十折平均极度稳定 Top-10 核心生物标志物 (极化分布后)")
print("-" * 50)
print(f"{'排名':<5} | {'特征名称':<30} | {'平均重要性权重'}")
print("-" * 50)
for i, idx in enumerate(top_10_indices):
    feature_name = feature_cols[idx]
    weight = avg_mask_weights[idx]
    print(f"Top-{i + 1:<3} | {feature_name:<30} | {weight:.4f}")
print("-" * 50)


