import matplotlib.pyplot as plt
import numpy as np

# 1. 全局字体与高分辨率设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签 (纯英文论文可注释此行)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 2. 准备数据字典
tasks = ['AD vs NC', 'MCI vs NC', 'LMCI vs EMCI']
x = np.arange(len(tasks))
width = 0.15  # 设置柱宽，5个模型总宽0.75

# 变体名称及对应配色 (突显完整模型)
variants = ['w/o Mask', 'w/o GCN', 'w/o Mamba', 'S-GMNet (Ours)']
colors = ['#8CA1CC', '#90C9B1', '#E6C687', '#D0021B']

# 各变体的指标数据 [Task1, Task2, Task3]
data = {
    'Accuracy (%)': {
        'means': [
            [91.59, 71.27, 71.21], [90.70, 70.64, 71.76],
            [91.05, 70.11, 70.10], [92.70, 72.13, 75.06]
        ],
        'stds': [
            [3.20, 3.62, 6.26], [3.83, 6.34, 7.83],
            [2.91, 6.73, 8.43], [3.83, 5.43, 6.06]
        ],
        'ylim': (55, 100)
    },
    'AUC': {
        'means': [
            [0.9651, 0.7443, 0.7332], [0.9535, 0.7115, 0.7133],
            [0.9437, 0.7382, 0.7228], [0.9636, 0.7654, 0.7449]
        ],
        'stds': [
            [0.0212, 0.0345, 0.0939], [0.0309, 0.0534, 0.1048],
            [0.0144, 0.0541, 0.0969], [0.0300, 0.0645, 0.0744]
        ],
        'ylim': (0.55, 1.05)
    },
    'Recall (%)': {
        'means': [
            [91.93, 79.73, 62.49], [90.90, 77.14, 63.96],
            [92.01, 74.82, 63.30], [92.17, 81.45, 66.95]
        ],
        'stds': [
            [6.03, 8.13, 18.67], [2.80, 10.77, 12.94],
            [4.66, 6.88, 14.30], [6.49, 10.86, 12.00]
        ],
        'ylim': (40, 105)
    },
    'F1-Score (%)': {
        'means': [
            [91.96, 62.21, 68.74], [90.10, 63.88, 66.76],
            [90.43, 61.48, 70.45], [91.81, 64.39, 71.01]
        ],
        'stds': [
            [3.59, 2.37, 10.68], [4.01, 5.26, 9.64],
            [3.13, 4.22, 7.70], [4.31, 5.18, 8.41]
        ],
        'ylim': (50, 100)
    }
}

# 3. 创建 2x2 子图矩阵
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = list(data.keys())

for idx, ax in enumerate(axes.flat):
    metric = metrics[idx]
    means = data[metric]['means']
    stds = data[metric]['stds']

    # 遍历5个模型进行绘制
    for i in range(4):
        # 计算每个柱子的偏移位置
        offset = (i - 2) * width
        ax.bar(x + offset, means[i], width, yerr=stds[i], label=variants[i],
               color=colors[i], edgecolor='black', capsize=3, zorder=3)

    # 子图修饰
    ax.set_title(f'Comparison of {metric}', fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=13, fontweight='bold')
    ax.set_ylim(data[metric]['ylim'])

    # 添加网格与边框调整
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 4. 统一设置全局图例
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=5, fontsize=13, frameon=False)

# 5. 调整布局并保存
plt.tight_layout()
plt.savefig('Ablation_and_Transformer_Comparison.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
