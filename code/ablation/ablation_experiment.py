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
# 🎯 0. 核心任务控制器 (默认使用难度最大的大合并版)
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
# 1. 基础组件定义 (Focal Loss, GCN, Mamba)
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


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
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


# ==========================================
# 2. 🔥 核心：可配置消融模型 Ablation_S_GMNet
# ==========================================
class Ablation_S_GMNet(nn.Module):
    def __init__(self, num_features, num_classes=2, d_model=16,
                 use_mask=True, use_gcn=True, use_mamba=True):
        super(Ablation_S_GMNet, self).__init__()

        self.use_mask = use_mask
        self.use_gcn = use_gcn
        self.use_mamba = use_mamba

        # 1. 稀疏掩码层开关
        if self.use_mask:
            self.mask = nn.Parameter(torch.randn(num_features) * 1.0)

        self.feature_proj = nn.Linear(1, d_model)

        # 2. 动态图卷积开关
        if self.use_gcn:
            self.gcn = DynamicGCN(d_model)

        # 3. Mamba 序列扫描开关
        if self.use_mamba:
            self.mamba = MambaBlock(d_model=d_model, d_state=16, expand=2)

        self.classifier = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if self.use_mask:
            mask_weight = torch.sigmoid(self.mask / 0.1)
            x_filtered = x * mask_weight
        else:
            mask_weight = None  # 关闭 Mask 时不计算权重
            x_filtered = x

        x_seq = x_filtered.unsqueeze(-1)
        x_emb = self.feature_proj(x_seq)

        if self.use_gcn:
            x_graph = self.gcn(x_emb)
        else:
            x_graph = x_emb  # 旁路跳过 GCN

        if self.use_mamba:
            x_out = self.mamba(x_graph)
        else:
            x_out = x_graph  # 旁路跳过 Mamba

        x_flat = x_out.reshape(x.size(0), -1)
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


print(f"正在加载数据以进行全自动消融实验...")
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
# 4. 封装十折交叉验证函数
# ==========================================
def run_10fold_cv(variant_name, use_mask, use_gcn, use_mamba):
    print(f"\n" + "=" * 50)
    print(f"🚀 正在运行消融变体: {variant_name}")
    print(f"⚙️ 配置 -> Mask: {use_mask} | GCN: {use_gcn} | Mamba: {use_mamba}")
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

        model = Ablation_S_GMNet(len(feature_cols), 2, 16, use_mask, use_gcn, use_mamba).to(device)
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

                loss_cls = criterion(outputs, labels)
                # 🚀 消融逻辑：只有开启 Mask 时才计算 L1 惩罚
                if use_mask and current_mask is not None:
                    loss_l1 = torch.mean(torch.abs(current_mask))
                    loss = loss_cls + lambda_l1 * loss_l1
                else:
                    loss = loss_cls

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

        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        all_preds = (np.array(all_probs) >= best_thresh).astype(int)

        fold_metrics['acc'].append(accuracy_score(all_targets, all_preds) * 100)
        fold_metrics['auc'].append(roc_auc_score(all_targets, all_probs))
        fold_metrics['recall'].append(recall_score(all_targets, all_preds, zero_division=0) * 100)
        fold_metrics['f1'].append(f1_score(all_targets, all_preds, zero_division=0) * 100)

        print(f"  ↳ 第 {fold + 1}/10 折完成 | AUC: {fold_metrics['auc'][-1]:.4f}")

    return {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}


# ==========================================
# 5. 执行消融实验计划
# ==========================================
ablation_plan = [
    {"name": "w/o Mask (去除稀疏掩码)", "mask": False, "gcn": True, "mamba": True},
    {"name": "w/o GCN (去除动态图卷积)", "mask": True, "gcn": False, "mamba": True},
    {"name": "w/o Mamba (去除序列扫描)", "mask": True, "gcn": True, "mamba": False}
]

final_results = {}
for plan in ablation_plan:
    res = run_10fold_cv(plan['name'], plan['mask'], plan['gcn'], plan['mamba'])
    final_results[plan['name']] = res

# ==========================================
# 6. 打印全局对比总结表 (可直接截图用于周报)
# ==========================================
print("\n" + "🔥" * 25)
print(f"📊 最终消融实验性能对比总表 ({task_name})")
print("🔥" * 25)
print(f"{'模型变体 (Variant)':<30} | {'Accuracy (%)':<15} | {'AUC':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15}")
print("-" * 100)

for name, metrics in final_results.items():
    acc_str = f"{metrics['acc'][0]:.2f}±{metrics['acc'][1]:.2f}"
    auc_str = f"{metrics['auc'][0]:.4f}±{metrics['auc'][1]:.4f}"
    rec_str = f"{metrics['recall'][0]:.2f}±{metrics['recall'][1]:.2f}"
    f1_str = f"{metrics['f1'][0]:.2f}±{metrics['f1'][1]:.2f}"
    print(f"{name:<30} | {acc_str:<15} | {auc_str:<15} | {rec_str:<15} | {f1_str:<15}")
print("-" * 100)
print("📌 提示: 如果 w/o 系列指标出现显著下降，即证明了对应模块的不可替代性！")
