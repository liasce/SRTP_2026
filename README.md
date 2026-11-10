# 稀疏观测条件下的海洋三维温盐重建

先读 [项目状态与执行台账](PROJECT_STATUS.md)。它是当前方案、实际进展和下一步任务的唯一状态入口。

## 目录

| 位置 | 内容 |
|---|---|
| `PROJECT_STATUS.md` | 长期维护的状态与执行台账 |
| `AGENTS.md` | 工作区协作与状态更新规则 |
| `research/` | 当前研究方案、文献审计与实验审计 |
| `research/figures/argo_tao_distribution/` | 已有观测分布图、统计、绘图程序和检查记录 |
| `literature/references.csv` | 文献登记；论文全文因版权和体积不入库 |
| `scripts/` | 下载、预处理、审计和最小训练检查 |
| `DATA_DOWNLOAD_HANDOFF.md` | 全量数据范围、账号、下载命令和验收交接 |

原始数据、处理数据、论文PDF、原始PPT、模型权重和账号配置均由`.gitignore`排除。它们仍保留在本地，但克隆仓库后需要按数据来源重新获取。

## 使用约定

- 原始数据保持只读；当前已有2018全年共同模态开发数据及少量跨年代审计样本，但不等于2005—2025全期训练资料。
- 新资料按产品放入`data/raw/`；处理结果和缓存按需创建，不提前搭空目录。
- 不把原始数据、缓存、账号秘密或大权重提交Git；忽略规则不等于备份。
- 过期方案不在工作区重复保留；设计演变只在`PROJECT_STATUS.md`变更日志中追踪。

## 预处理入口

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-preprocess.txt
Copy-Item .env.example .env

python scripts/run_preprocessing.py --mode common
python scripts/run_preprocessing.py --mode full
python scripts/training_data.py --index 0
```

`common`用于不含SMAP的长期共同模态；`full`要求每月31个SMAP标称窗口。两个模式都会检查月份配对、共同网格，生成训练文件清单，并仅用预训练/微调月份拟合归一化统计。

## 自检

```powershell
Get-ChildItem scripts\test_*.py | ForEach-Object { python $_.FullName; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
```

本机`.venv`用于数据下载与预处理；正式GPU训练环境应在服务器上按`requirements-train.txt`重建并验证CUDA，不复制本机虚拟环境。

## GitHub协作

协作仓库：[liasce/SRTP_2026](https://github.com/liasce/SRTP_2026)。

```bash
git add .
git status --short
git commit -m "Initial research workflow"
git remote add origin https://github.com/liasce/SRTP_2026.git
git push -u origin main
```

只推送`main`，不要使用`--all`或`--mirror`。每次提交前再次确认`.env`和`data/`出现在忽略列表中。

当前详细方案：[研究方案](research/稀疏观测条件下海洋三维温盐重建_研究方案.md)。
