import matplotlib.pyplot as plt
import numpy as np

# 1. 基础设置：字体与分辨率
plt.rcParams['font.sans-serif'] = ['SimHei']  # 确保正常显示中文标签，如果是纯英文论文可以注释掉
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 设置高分辨率输出

# 2. 准备数据
tasks = ['AD vs NC\n', 'MCI vs NC\n', 'LMCI vs EMCI\n']
x = np.arange(len(tasks))  # 横坐标刻度位置
width = 0.2  # 柱子宽度

# 提取 S-GMNet 的指标均值（为了图表美观，将 AUC 转化为百分制）
acc_mean = [92.70, 72.13, 75.06]
auc_mean = [96.36, 76.54, 74.49]  # 0.9636 * 100
rec_mean = [92.17, 81.45, 66.95]
f1_mean  = [91.81, 64.39, 71.01]

# 提取对应的标准差，用于绘制误差棒 (Error Bar)
acc_std = [3.83, 5.43, 6.06]
auc_std = [3.00, 6.45, 7.44]      # 0.0300 * 100
rec_std = [6.49, 10.86, 12.00]
f1_std  = [4.31, 5.18, 8.41]

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制四组柱状图，并添加误差棒 (yerr), capsize控制误差棒顶端的横线宽度
rects1 = ax.bar(x - 1.5*width, acc_mean, width, yerr=acc_std, label='Accuracy', capsize=4, color='#4A90E2', edgecolor='black', zorder=3)
rects2 = ax.bar(x - 0.5*width, auc_mean, width, yerr=auc_std, label='AUC (×100)', capsize=4, color='#50E3C2', edgecolor='black', zorder=3)
rects3 = ax.bar(x + 0.5*width, rec_mean, width, yerr=rec_std, label='Recall', capsize=4, color='#F5A623', edgecolor='black', zorder=3)
rects4 = ax.bar(x + 1.5*width, f1_mean, width, yerr=f1_std, label='F1-Score', capsize=4, color='#D0021B', edgecolor='black', zorder=3)

# 4. 图表修饰
ax.set_ylabel('Performance Metric (%)', fontsize=12, fontweight='bold')
ax.set_title('S-GMNet Performance Across Progressive Clinical Tasks', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11, fontweight='bold')
ax.set_ylim(40, 105)  # 限制 Y 轴范围，使对比更加明显

# 添加图例
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')

# 添加背景网格虚线，便于阅读具体数值
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 隐藏顶部和右侧的边框线以增加学术感
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5. 显示并保存图表
plt.tight_layout()
plt.savefig('S_GMNet_Task_Comparison.png', format='png', dpi=300)
plt.show()