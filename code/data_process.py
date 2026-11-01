import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 全局路径与参数设置 (改这里就能控制全局)
# ==========================================
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 输入文件
input_file = os.path.join(base_path, '三模态信息总表.xlsx')

# 输出文件
inter_file = os.path.join(base_path, '1_插值后数据_data_inter.xlsx')
report_file = os.path.join(base_path, '2_缺失值处理报告_NA_report.xlsx')
train_file = os.path.join(base_path, '3_训练集_train_0.8.xlsx')
test_file = os.path.join(base_path, '4_测试集_test_0.2.xlsx')

# 需要处理的组别
group_list = ['NC', 'AD', 'MCI', 'EMCI', 'LMCI']

# 用于临时存储处理结果的字典
dict_inter_data = {}
dict_report = {}
dict_train = {}
dict_test = {}

print("开始处理数据，请稍候...")

# ==========================================
# 2. 遍历各组，进行插值与切分
# ==========================================
for group_name in group_list:
    print(f"正在处理: {group_name} 组...")

    # --- A. 加载数据 ---
    # [cite_start]读取对应的 sheet
    data = pd.read_excel(input_file, sheet_name=group_name, engine='openpyxl')
    data_NA_num = data.isna().sum().sum()  # 统计原始NA个数

    # --- B. SNP数据线性插值 ---
    # [cite_start]假设第98列及以后是SNP数据
    data_SNP = data.iloc[:, 98:]
    data_SNP_NA_num = data_SNP.isna().sum().sum()

    # [cite_start]核心：实现患者个体内的SNP之间线性插值
    data_SNP_inter = data_SNP.interpolate(axis=1)
    data_SNP_inter_NA_num = data_SNP_inter.isna().sum().sum()

    # [cite_start]将插值后的数据替换回原表
    data_inter = data.copy()
    data_inter.iloc[:, 98:] = data_SNP_inter
    data_inter_NA_num = data_inter.isna().sum().sum()

    # 保存插值后的完整数据到字典
    dict_inter_data[group_name] = data_inter

    # --- C. 生成缺失值报告 ---
    NA_num = np.zeros(shape=(1, 4))
    NA_num[0][0] = data_NA_num
    NA_num[0][1] = data_SNP_NA_num
    NA_num[0][2] = data_SNP_inter_NA_num
    NA_num[0][3] = data_inter_NA_num
    df_report = pd.DataFrame(NA_num, columns=['原始数据NA', '原始SNP_NA', '插值后SNP_NA', '最终全表NA'])
    dict_report[group_name] = df_report

    # --- D. 划分训练集和测试集 (8:2) ---
    # [cite_start]统一使用 random_state=42 保证可复现 [cite: 4]
    x_train, x_test = train_test_split(data_inter, train_size=0.8, random_state=42)
    dict_train[group_name] = x_train
    dict_test[group_name] = x_test

# ==========================================
# 3. 统一保存所有结果到 Excel 的不同 Sheet 中
# ==========================================
print("正在将结果保存到 Excel 文件...")


# 批量写入函数，避免重复写 with open 导致代码臃肿
def save_dict_to_excel(data_dict, file_path):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# 执行保存
save_dict_to_excel(dict_inter_data, inter_file)
save_dict_to_excel(dict_report, report_file)
save_dict_to_excel(dict_train, train_file)
save_dict_to_excel(dict_test, test_file)

print(f"处理完成！所有文件已成功保存在: {base_path}")

