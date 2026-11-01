import matplotlib.pyplot as plt
import numpy as np

# 1. 基础设置：高分辨率与学术风格
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于中文显示，若为纯英文论文可注释
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据 (RF, SVM, S-GMNet)
tasks = ['AD vs NC', 'MCI vs NC', 'LMCI vs EMCI']
x = np.arange(len(tasks))
width = 0.20  # 柱子宽度

# 🔥 修改点：使用高对比度的学术配色组合 (蓝、橘、红)
colors = ['#4A90E2', '#F5A623','#50E3C2', '#D0021B']
models = ['Random Forest', 'SVM (RBF)', 'Transformer','S-GMNet (Ours)']

# 指标数据整理 [RF, SVM, Ours]
metrics_data = {
    'Accuracy (%)': {
        'means': [[90.70, 71.02, 71.61], [91.16, 67.64, 71.61], [91.05,69.15,73.34],[92.70, 72.13, 75.06]],
        'stds': [[2.97, 5.98, 8.25], [3.07, 4.24, 8.69],[3.97,11.66,6.34],[3.83, 5.43, 6.06]],
        'ylim': (60, 100)
    },
    'AUC': {
        'means': [[0.9556, 0.7444, 0.7139], [0.9491, 0.7213, 0.7009], [0.9543,0.7475,0.7266],[0.9636, 0.7654, 0.7449]],
        'stds': [[0.0248, 0.0626, 0.1053], [0.0260, 0.0646, 0.1200],[0.0264,0.0928, 0.0913], [0.0300, 0.0645, 0.0744]],
        'ylim': (0.6, 1.05)
    },
    'Recall (%)': {
        'means': [[90.92, 76.98, 65.00], [89.04, 77.95, 64.39],[91.15,78.14,64.90], [92.17, 81.45, 66.95]],
        'stds': [[5.53, 12.86, 15.12], [6.62, 16.02, 12.45],[3.71,13.63,16.41], [6.49, 10.86, 12.00]],
        'ylim': (50, 105)
    },
    'F1-Score (%)': {
        'means': [[90.72, 61.70, 67.96], [91.00, 59.34, 67.92], [91.60,61.80,69.65],[91.81, 64.39, 71.01]],
        'stds': [[3.26, 5.26, 7.63], [3.53, 4.84, 6.74], [4.03,6.99,8.21],[4.31, 5.18, 8.41]],
        'ylim': (50, 100)
    }
}

# 3. 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metric_names = list(metrics_data.keys())

for i, ax in enumerate(axes.flat):
    m_name = metric_names[i]
    means = metrics_data[m_name]['means']
    stds = metrics_data[m_name]['stds']

    # 绘制三组模型的柱子
    for j in range(4):
        pos = x + (j - 1) * width
        ax.bar(pos, means[j], width, yerr=stds[j], label=models[j],
               color=colors[j], edgecolor='black', capsize=4, zorder=3)

    # 子图细节优化
    ax.set_title(f'Comparison of {m_name}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11, fontweight='bold')
    ax.set_ylabel(m_name, fontsize=12, fontweight='bold')
    ax.set_ylim(metrics_data[m_name]['ylim'])
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 4. 统一设置图例 (位于顶部)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=3, fontsize=12, frameon=False)

# 5. 保存与展示
plt.tight_layout()
plt.savefig('Baseline_Performance_Comparison_HighContrast.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
