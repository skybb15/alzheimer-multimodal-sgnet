import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve
from imblearn.over_sampling import SMOTE

# 导入传统机器学习双雄
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
# 1. 数据加载与预处理
# ==========================================
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')


def load_and_concat_data(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return pd.concat(all_sheets.values(), ignore_index=True)


print(f"正在加载数据以进行 传统机器学习 (RF & SVM) 基线对比实验...")
all_df = pd.concat([load_and_concat_data(train_file), load_and_concat_data(test_file)], ignore_index=True)
real_label_col = 'Research Group'

all_df = all_df[all_df[real_label_col].isin(valid_groups)]
y_all = all_df[real_label_col].map(label_map).values
drop_cols = ['RID', 'PTID', real_label_col, 'Group', 'DX', 'Phase']
feature_cols = [col for col in all_df.select_dtypes(include=['number']).columns if col not in drop_cols]
X_all_unscaled = all_df[feature_cols].values

print(f"🔥 当前任务: {task_name} | 数据集大小: {X_all_unscaled.shape}")


# ==========================================
# 2. 核心评估流水线 (十折交叉验证)
# ==========================================
def evaluate_ml_model(model_name, clf):
    print(f"\n" + "=" * 50)
    print(f"🚀 正在运行基线模型: {model_name}")
    print("=" * 50)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = {'acc': [], 'auc': [], 'f1': [], 'recall': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_unscaled, y_all)):
        X_train_fold, y_train_fold = X_all_unscaled[train_idx], y_all[train_idx]
        X_val_fold, y_val_fold = X_all_unscaled[val_idx], y_all[val_idx]

        # 严格防穿越：先 SMOTE 训练集
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

        # 严格防穿越：基于扩增后的训练集进行标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val_fold)

        # 训练模型
        clf.fit(X_train_scaled, y_train_resampled)

        # 获取预测概率 (用于计算 AUC 和 寻找最佳阈值)
        all_probs = clf.predict_proba(X_val_scaled)[:, 1]
        all_targets = y_val_fold

        # 动态阈值截断 (保证与深度学习同样的防漏诊逻辑)
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        all_preds = (np.array(all_probs) >= best_thresh).astype(int)

        # 计算指标
        fold_metrics['acc'].append(accuracy_score(all_targets, all_preds) * 100)
        fold_metrics['auc'].append(roc_auc_score(all_targets, all_probs))
        fold_metrics['recall'].append(recall_score(all_targets, all_preds, zero_division=0) * 100)
        fold_metrics['f1'].append(f1_score(all_targets, all_preds, zero_division=0) * 100)

        print(f"  ↳ 第 {fold + 1}/10 折完成 | AUC: {fold_metrics['auc'][-1]:.4f}")

    # 返回均值和标准差
    return {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}


# ==========================================
# 3. 实例化模型并执行
# ==========================================
# 开启 probability=True 以支持 AUC 计算
ml_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
}

final_results = {}
for name, clf in ml_models.items():
    final_results[name] = evaluate_ml_model(name, clf)

# ==========================================
# 4. 打印全局对比总结表
# ==========================================
print("\n" + "🔥" * 25)
print(f"📊 传统机器学习基线性能对比总表 ({task_name})")
print("🔥" * 25)
print(f"{'模型变体 (Variant)':<20} | {'Accuracy (%)':<15} | {'AUC':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15}")
print("-" * 85)

for name, metrics in final_results.items():
    acc_str = f"{metrics['acc'][0]:.2f}±{metrics['acc'][1]:.2f}"
    auc_str = f"{metrics['auc'][0]:.4f}±{metrics['auc'][1]:.4f}"
    rec_str = f"{metrics['recall'][0]:.2f}±{metrics['recall'][1]:.2f}"
    f1_str = f"{metrics['f1'][0]:.2f}±{metrics['f1'][1]:.2f}"
    print(f"{name:<20} | {acc_str:<15} | {auc_str:<15} | {rec_str:<15} | {f1_str:<15}")
print("-" * 85)

