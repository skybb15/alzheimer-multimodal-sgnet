import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 基础设置：分辨率与字体
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文，纯英文论文可注释此行
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备全局混淆矩阵数据 (格式: [[TN, FP], [FN, TP]])
cm_task1 = np.array([[191, 14], [13, 152]])
cm_task2 = np.array([[312, 147], [38, 167]])
cm_task3 = np.array([[153, 33], [54, 109]])

# 定义横纵坐标标签
labels_task1 = ['NC', 'AD']
labels_task2 = ['MCI', 'NC']
labels_task3 = ['EMCI', 'LMCI']


# 3. 封装单张图绘制与保存的函数
def plot_and_save_single_cm(cm, labels, title, filename):
    # 为每张图创建独立的画布，尺寸调整为适合单张图的比例
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)

    # 设置标题和坐标轴标签
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold', labelpad=10)

    # 调整坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()  # 在 Notebook 中展示当前图
    plt.close(fig)  # 清理当前画布，防止下一张图画在同一张纸上


# 4. 分别执行生成并保存三张独立的图片
print("正在生成 Task 1 混淆矩阵...")
plot_and_save_single_cm(cm_task1, labels_task1, 'Task 1: AD vs NC\n', 'S_GMNet_CM_Task1.png')

print("正在生成 Task 2 混淆矩阵...")
plot_and_save_single_cm(cm_task2, labels_task2, 'Task 2: MCI vs NC\n', 'S_GMNet_CM_Task2.png')

print("正在生成 Task 3 混淆矩阵...")
plot_and_save_single_cm(cm_task3, labels_task3, 'Task 3: LMCI vs EMCI\n', 'S_GMNet_CM_Task3.png')

print("✅ 三张高清混淆矩阵图已成功分别保存！")

