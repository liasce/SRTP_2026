# 全量数据下载交接清单

> 面向数据下载协作者 · 更新日期：2026-11-10（Asia/Shanghai）
> 项目：热带太平洋稀疏观测条件下的月尺度三维温盐重建

## 1. 接手后先确认的边界

- 研究区：20°S–20°N、120°E–80°W；统一使用0–360°经度时为120–280°E。
- 跨日期变更线的数据源必须拆成两块下载：西块120–180°E，东块180–80°W（即−180至−80°）；合并时180°重复经线必须一致去重。
- 目标深度：5–500 m；下载CORA和剖面资料时保留500 m下方紧邻层，便于禁止外推的插值。
- 目标时间：2005-01至2025-06，共246个月。
- 海表输入为`t−2:t`三个月历史窗口，因此OISST、ADT和ERA5要额外下载2004-11、2004-12两个缓冲月，共248个月。
- SMAP仅用于2015-04以后的完整模态；从2015-04下载至2025-06，不向前伪造SSS。
- 数据角色：
  - 2005-01—2015-03：共同模态预训练；
  - 2015-04—2020-11：含SMAP的近期微调；
  - 2020-12：隔离月，不拟合；
  - 2021-01—2022-12：验证；
  - 2023-01—2024-12：主测试；
  - 2025-01—2025-06：补充测试。
- 下载验证/测试数据是允许的，但下载人员不得查看模型效果、改变划分或据此选择模型。建议将2021年以后目录设为只读或单独OSS前缀。
- 不混用MY与NRT产品，不静默更换数据集版本。若当前固定版本停止提供，先记录并通知负责人。

## 2. 账号准备

| 服务 | 用途 | 是否必须 | 凭据字段 | 官方入口 |
|---|---|---:|---|---|
| Copernicus Marine | DUACS ADT | 是 | `COPERNICUSMARINE_SERVICE_USERNAME`、`COPERNICUSMARINE_SERVICE_PASSWORD` | 注册说明：https://help.marine.copernicus.eu/en/articles/4220332-how-to-register-for-copernicus-marine-service |
| NASA Earthdata Login | SMAP SSS | 是 | `EARTHDATA_USERNAME`、`EARTHDATA_PASSWORD` | 注册：https://urs.earthdata.nasa.gov/documentation/for_users/how_to_register |
| ECMWF Climate Data Store | ERA5 U/V | 是 | `CDSAPI_URL`、`CDSAPI_KEY` | API设置：https://cds.climate.copernicus.eu/how-to-api |
| Argo GDAC/AWS公开镜像 | Argo Core剖面 | 否 | 无 | 索引：https://data-argo.ifremer.fr/ar_index_global_prof.txt.gz |
| NOAA NCEI | OISST | 否 | 无 | https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/ |
| Copernicus/CloudFerro公开对象存储 | CORA5.2 | 否 | 无 | 下载URL已经固定在脚本中 |
| NOAA PMEL ERDDAP | TAO/TRITON外部评价 | 否 | 无 | https://data.pmel.noaa.gov/generic/erddap |

安全要求：

1. 协作者使用自己的官方账号；不要通过聊天、邮件或Git共享个人密码。
2. 从`.env.example`复制本机`.env`，只在下载机器上填写；`.env`不得提交Git、上传OSS或写入日志。
3. CDS账号须登录ERA5数据集页面并手动接受许可，否则API可能返回403。
4. CDS使用个人访问令牌，不使用旧式UID:key组合；固定URL为`https://cds.climate.copernicus.eu/api`。
5. 失败日志只记录异常类型和HTTP状态，不复制带签名的下载URL。

`.env`格式：

```dotenv
COPERNICUSMARINE_SERVICE_USERNAME=replace_me
COPERNICUSMARINE_SERVICE_PASSWORD=replace_me
EARTHDATA_USERNAME=replace_me
EARTHDATA_PASSWORD=replace_me
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=replace_me
```

## 3. 必须收集的数据

| 优先级 | 数据源/固定版本 | 时间范围 | 内容 | 预计对象数 | 原始落盘目录 | 账号 |
|---:|---|---|---|---:|---|---|
| 1 | CORA5.2 OA，`cmems_obs-ins_glo_phy-temp-sal_my_cora-oa_P1M_202603` | 2005-01—2025-06 | 每月`fld`与`dat`，TEMP和PSAL；`fld`含温盐、误差及分析场，`dat`保留分析输入候选 | 984个NetCDF | `data/raw/cora/original_202603/` | 无 |
| 1 | Argo GDAC Core profiles | 2005-01—2025-06 | PRES/TEMP/PSAL及ADJUSTED、QC、ERROR、JULD、经纬度、WMO、循环、方向、模式 | 索引候选531,903个文件；实际以冻结索引为准 | `data/raw/argo/<dac>/...` | 无 |
| 1 | NOAA OISST v2.1 AVHRR | 2004-11—2025-06 | 逐日SST、误差`err`、海冰辅助字段`ice` | 7,547个日文件 | `data/raw/oisst/YYYYMM/` | 无 |
| 1 | DUACS vNov2024，`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`，version `202411` | 2004-11—2025-06 | 逐日`adt`；每月west/east两块 | 496个NetCDF | `data/raw/adt/` | Copernicus Marine |
| 1 | ERA5 single-level monthly means | 2004-11—2025-06 | 月平均10 m风`u10`、`v10`；每月west/east两块 | 496个NetCDF | `data/raw/era5/` | CDS |
| 2 | SMAP RSS L3 SSS V6，`SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6` | 2015-04—2025-06 | 每个标称日对应8日滑动窗口；保留`sss_smap`、不确定性与观测支持，不用`sss_ref`作观测输入 | 3,744个窗口文件 | `data/raw/smap/` | Earthdata |
| 暂缓 | TAO/TRITON日T/S | 评价用途冻结后再定；不作为训练输入 | 日温度、盐度及原始QC，用于独立D20/温盐确认性评价 | 待定 | `data/raw/tao_daily/` | 无 |

说明：当前531,903是2026-09-08 Argo索引在既定时间、海区和Pacific海盆条件下的候选文件数，不等于最终通过温盐、深度和QC的剖面数。

## 4. 环境与目录

在下载用ECS上执行：

```bash
set -euo pipefail
cd /path/to/project
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-preprocess.txt
cp .env.example .env
chmod 600 .env
```

推荐存储：

- OSS保存不可变原始主副本和年度清单；建议同GPU/ECS地域。
- ECS下载/预处理盘至少1 TiB；若只保留原始数据主副本，OSS 500 GiB为最低规划值。
- 预计全量原始数据约270–310 GiB；预处理、中间缓存、双份校验和临时文件会继续占空间。
- 下载到`data/raw/`，不要直接写入`data/processed/`。
- 每个任务日志写入`logs/download/<source>_<year>.log`，日志不得包含凭据。

## 5. 标准下载方法

统一入口：

```bash
python scripts/download_samples.py SOURCE --months YYYYMM [YYYYMM ...]
```

脚本每次最多接收12个月，支持断点重跑：已存在且可读的文件会复用；不完整文件使用`.part`后缀，验证成功后才改为正式文件名。

生成一年月份参数的Shell函数：

```bash
months_for_year() {
  local year="$1"
  for month in {01..12}; do printf '%s%s ' "$year" "$month"; done
}
mkdir -p logs/download
```

### 5.1 Argo：先冻结索引，再下载全量剖面

```bash
python scripts/download_samples.py argo-index
sha256sum data/raw/profiles/ar_index_global_prof.txt.gz \
  > research/audits/argo_index_download.sha256
python scripts/build_argo_inventory.py \
  | tee logs/download/argo_inventory.log

for year in $(seq 2005 2024); do
  python scripts/download_samples.py argo-full --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/argo_${year}.log"
done
python scripts/download_samples.py argo-full \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/argo_2025.log
```

不要使用`argo`子命令代替`argo-full`；`argo`仅下载历史固定的每月50文件审计样本。

### 5.2 CORA

```bash
for year in $(seq 2005 2024); do
  python scripts/download_samples.py cora --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/cora_${year}.log"
done
python scripts/download_samples.py cora \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/cora_2025.log
```

CORA服务会限流；脚本已固定单并发。不要同时启动多个CORA年份任务。

### 5.3 OISST

```bash
python scripts/download_samples.py oisst --months 200411 200412 \
  2>&1 | tee logs/download/oisst_2004_buffer.log
for year in $(seq 2005 2024); do
  python scripts/download_samples.py oisst --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/oisst_${year}.log"
done
python scripts/download_samples.py oisst \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/oisst_2025.log
```

### 5.4 ADT

```bash
python scripts/download_samples.py adt --months 200411 200412 \
  2>&1 | tee logs/download/adt_2004_buffer.log
for year in $(seq 2005 2024); do
  python scripts/download_samples.py adt --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/adt_${year}.log"
done
python scripts/download_samples.py adt \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/adt_2025.log
```

### 5.5 ERA5

主方法是CDS预计算月平均产品：

```bash
python scripts/download_samples.py era5 --months 200411 200412 \
  2>&1 | tee logs/download/era5_2004_buffer.log
for year in $(seq 2005 2024); do
  python scripts/download_samples.py era5 --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/era5_${year}.log"
done
python scripts/download_samples.py era5 \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/era5_2025.log
```

ERA5纪律：

- 只允许一个CDS下载进程，west/east和月份由脚本顺序提交。
- 不要因排队而重复提交同一目标；先检查CDS任务页和本地审计。
- 截至2026-11-10，CDS可能有明显排队；排队不是失败。
- `download_era5_planetary_fallback.py`只作为经交叉验证的应急方案。它从小时ERA5镜像重新计算月均，来源语义不同，使用前必须记录审计，不可与CDS文件混称同一产品。

### 5.6 SMAP

```bash
python scripts/download_samples.py smap \
  --months 201504 201505 201506 201507 201508 201509 201510 201511 201512 \
  2>&1 | tee logs/download/smap_2015.log
for year in $(seq 2016 2024); do
  python scripts/download_samples.py smap --months $(months_for_year "$year") \
    2>&1 | tee "logs/download/smap_${year}.log"
done
python scripts/download_samples.py smap \
  --months 202501 202502 202503 202504 202505 202506 \
  2>&1 | tee logs/download/smap_2025.log
```

每月应找到与自然月日数相同的“标称日”窗口；这些是相互重叠的8日滑动平均，不能当作独立日观测。

## 6. 调度顺序

1. 先完成账号登录小测试：ADT一个月、SMAP一个月、ERA5一个月。
2. 冻结Argo索引及SHA-256，生成候选清单。
3. 优先下载训练侧：2005-01—2020-11；这部分先支持预训练和微调。
4. 下载2020-12隔离月，但不加入拟合。
5. 最后下载2021-01—2025-06，并放到只读/锁定前缀。
6. OISST、Argo和ADT可与其他来源有限并行；CORA始终单进程；ERA5始终单进程；SMAP建议最多4线程（脚本默认）。
7. 不要在同一数据源上同时运行多个年份任务，否则重试和临时文件可能互相干扰。

## 7. 每年完成后的验收

每个年度/批次至少检查：

```bash
find data/raw -type f -name '*.part*' -print
find data/raw -type f -size 0 -print
df -h
```

完成条件：

- 没有`.part`残留和零字节文件；
- NetCDF全部可打开；
- OISST逐日覆盖完整自然月；
- ADT每月west/east各一份，日数匹配自然月；
- ERA5每月west/east各一份，只有一个正确月时间，u10/v10均非全缺失；
- CORA每月四份：fld/dat × TEMP/PSAL；
- SMAP每月标称日数量等于自然月日数；
- Argo文件路径与冻结索引唯一对应，下载数不能冒充通过QC的剖面数。

可在批次完成后运行本地审计入口：

```bash
python scripts/download_samples.py audit \
  2>&1 | tee logs/download/final_raw_audit.log
```

注意：全量SHA-256和NetCDF变量扫描会耗时数小时，应在年度批次或最终收尾时运行，不要每下载一个文件就全库重扫。

## 8. 向负责人回报的固定格式

每完成一个来源/年度，回报以下字段：

```text
数据源：
时间范围：
产品ID/版本：
完成文件数：
总字节数：
缺失月份/日期：
.part或零字节文件数：
失败目标与异常类型：
原始目录：
日志路径：
索引/清单SHA-256：
是否接触验证/测试模型指标：否
```

## 9. 明确禁止事项

- 不修改时间划分、空间范围、深度层或C/Q/E角色。
- 不以插值、复制月份或全球粗样本填补缺失文件。
- 不把CORA称为无误差真值，也不把Argo索引数称为有效剖面数。
- 不把SMAP重叠8日窗口称为独立日观测。
- 不在下载日志中输出账号、密码、CDS令牌或签名URL。
- 不上传`.env`、凭据、验证/测试指标或原始数据到公共Git仓库。
- 不因新版产品出现就混用版本；需要升级时先形成版本差异审计。

## 10. 现有参考证据

- 当前唯一状态入口：`PROJECT_STATUS.md`
- 统一下载脚本：`scripts/download_samples.py`
- 下载依赖：`requirements-preprocess.txt`
- 预处理契约：`research/audits/preprocessing_contract.md`
- Argo全期索引审计：`research/audits/argo_inventory_summary.json`
- 2018原始闭合审计：`research/audits/raw_2018_common_inventory_summary.json`
- 2018全管线验收：`research/audits/pipeline_validation_common.json`
- ERA5应急镜像交叉验证：`research/audits/era5_planetary_fallback_audit.json`
