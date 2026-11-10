# 数据预处理契约

本文件规定：原始下载文件不能直接训练或评价。每个数据源必须有可批处理入口、固定的规范化输出、质量检查和可重复运行证据；小样本只用于验收该流程，不能替代全期处理。

## 共同规则

- 原始文件只读保留在`data/raw/`；可再生中间结果写入`data/processed/`，不提交Git。
- 所有空间字段统一为热带太平洋20°S–20°N、120°E–280°E及递增的0–360°经度；训练目标深度为5–500m。
- 不以缺失值填充为观测。每一步保留来源、时间覆盖、有效值/有效观测数和缺测掩膜。
- 训练期之外的数据不得参与背景、标准化、OI协方差、质量阈值或模型选择统计量的拟合。
- 绝对盐度`SA`、保守温度`CT`、`sigma0`和`N²`由GSW 3.6.23生成；同时保留原始TEMP/PSAL及其定义。

## 数据源处理矩阵

| 数据源 | 可复用入口 | 规范输出与固定规则 | 必做QA | 当前状态 |
|---|---|---|---|---|
| CORA OA | `scripts/preprocess_cora.py` | 每月T/S、误差、SA/CT、sigma0、N²；裁剪5–500m并保留原生层 | 配对、坐标、变量语义、有限值、N²分母与阈值敏感性 | **2018-01/07已验收** |
| TAO/TRITON日T/S | `preprocess_tao.py`＋`pair_tao.py` | QC=1/2/3逐观测；精确同站/同日/同深度配对并派生SA/CT/sigma0 | QC、区域、深度、配对数、有限值 | **2018-01/07已验收**；深层覆盖仍不足 |
| Argo剖面 | `preprocess_argo.py`＋`grid_argo_monthly.py` | A/D调整值QC=1；GSW；60层有界插值；平台固定输入/留出角色；月稀疏网格 | 往返、缺口/外推、重复深度、去重和角色确定性 | **100文件、100唯一剖面已验收** |
| OISST日SST | `preprocess_surface.py oisst`＋`regrid_surface.py` | 完整日历月均、有效数、掩膜归一双线性至CORA网格 | 日历、单位、网格、均值、共同坐标 | **2018-01/07已验收** |
| SMAP 8日SSS | `preprocess_smap.py`＋`regrid_surface.py` | 31个标称窗口月代表场，不使用`sss_ref`，保留不确定性与支持边界 | 日历、窗口、nobs、跨月支持、共同坐标 | 合成测试通过；真实缺56窗口，**未验收** |
| DUACS日ADT | `preprocess_surface.py adt`＋`regrid_surface.py` | 日期线两块合并去重、完整日历月均、共同网格 | 日期、拼接、单位、共同坐标 | **2018-01/07已验收** |
| ERA5月U/V | `preprocess_surface.py era5`＋`regrid_surface.py` | 月均U/V裁剪并映射共同网格；不称风应力 | 时间、单位、纬向、共同坐标 | **2018-01/07已验收** |

## 已运行命令

```bash
"CORA OA/.venv/bin/python" scripts/run_preprocessing.py --mode common
"CORA OA/.venv/bin/python" scripts/run_preprocessing.py --mode full  # 需完整SMAP
"CORA OA/.venv/bin/python" scripts/training_data.py --index 0
"CORA OA/.venv/bin/python" scripts/validate_pipeline.py
"CORA OA/.venv/bin/python" scripts/smoke_train.py  # 需PyTorch
```

`validate_profiles.py`包含D20合成剖面断言，并对目录中所有匹配的CORA月TEMP场及TAO日温度文件批量生成审计结果；它是指标验证入口，不替代TEOS-10转换或N²实现。

## 当前退出门

在以下条件满足前，不批量生成训练张量、不开始模型训练：

1. SMAP两个月完整窗口通过真实月聚合与共同网格验收；
2. 服务器PyTorch/CUDA及一个batch前后向通过；
3. 全期产品版本、月份覆盖、训练/验证/测试角色和跨月支持窗口冻结。
