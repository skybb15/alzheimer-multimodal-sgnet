# 数据目录（不包含在仓库中）

本项目的全部实验都基于 **ADNI（Alzheimer's Disease Neuroimaging Initiative）** 的多组学数据。ADNI 是受**数据使用协议（DUA）**约束的受控访问数据库，受试者级数据不得公开再分发，因此本仓库不提供任何数据文件。

## 获取数据

1. 在 ADNI 官网注册并提交数据使用申请：<https://adni.loni.usc.edu>
2. 申请通过后，从 LONI IDA 下载 ADNI-1 / ADNI-2 / ADNI-GO 三个队列的：
   - 临床诊断与人口学信息（`DXSUM`、`PTDEMOG` 等）
   - CSF 生物标志物（`UPENNBIOMK` 系列，取 ABETA / TAU / PTAU）
   - SNP 基因型（PLINK 格式，需自行完成质控，见论文 2.2.2 节）
   - 结构 MRI（按论文 2.2.1 节流程用 SPM8 + AAL 模板提取 90 个脑区 ROI）
3. 按下列结构整理成 Excel：

```
code/data/
└── 三模态信息总表.xlsx
      ├── sheet "NC"    # 认知正常
      ├── sheet "AD"    # 阿尔茨海默病
      ├── sheet "MCI"   # 轻度认知障碍
      ├── sheet "EMCI"  # 早期 MCI
      └── sheet "LMCI"  # 晚期 MCI
```

每个 sheet 的列约定：

- 第 1–98 列为受试者标识、临床与影像/蛋白特征（含 `RID`、`PTID`、`Research Group` 等，脚本会自动排除非数值列）；
- **第 99 列及之后为 SNP 位点**，`data_process.py` 会对这一段做个体内线性插值。

## 脚本产出

运行 `python code/data_process.py` 后会在本目录生成：

| 文件 | 内容 |
| --- | --- |
| `1_插值后数据_data_inter.xlsx` | SNP 插值后的完整数据 |
| `2_缺失值处理报告_NA_report.xlsx` | 各组插值前后的缺失值统计 |
| `3_训练集_train_0.8.xlsx` | 8:2 划分的训练集 |
| `4_测试集_test_0.2.xlsx` | 8:2 划分的测试集 |

这些文件同样被 `.gitignore` 排除。

## 引用要求

使用 ADNI 数据发表成果时，需按 ADNI 的致谢规范声明数据来源，例如：

> Data used in preparation of this work were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.
