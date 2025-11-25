# Aetheria

本仓库包含论文《Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration》的源代码实现，覆盖文中描述的简化多智能体评估流水线、BFF 服务与可视化前端。整体复用原始 Aetheria 数据与输出接口，重点展示如何借助 Azure OpenAI、Top-K RAG 以及多轮辩论式协作来提供可解释的内容安全审核体验，可直接支撑论文实验复现或内部落地。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `aetheria_simple/` | Python 版简化评估器：多智能体图 (`graph.py`)、角色定义 (`agents.py`)、配置与提示 (`config.py`, `prompts.py`)、命令行入口 (`main.py`)。 |
| `aetheria_simple/bff/` | FastAPI 后端（`app.py`），通过 `ReviewService` 暴露 `/api/review` 供前端调用，可并发处理文本 + Base64 图片请求。 |
| `aetheria_simple/scripts/` | 维护工具：构建/再平衡案例库、生成 Chroma 向量库、提取误判样本等。 |
| `aetheria_simple/case_libraries/` | 预制案例库 JSON，支持 Supporter/RAG 模块直接消费。 |
| `aetheria_simple/chroma_db/` | 默认 Chroma 向量数据库持久化目录（可用脚本重建）。 |
| `frontend/` | Vue 3 + Vite 前端，含审核表单、RAG 结果、样例展示等视图。 |
| `result/` (运行生成) | CLI 评估输出的 JSON 报告与 CSV 详单。 |
| `logs/` (运行生成) | `ReviewService` 记录的单次审核日志与模型对话。 |

## 运行前提

- **Python** 3.10+，建议使用虚拟环境（`venv`/`uv`/`conda` 均可）。
- **Node.js** 20.19+（与 `frontend/package.json` 的 `engines` 对齐）、`npm` 10+。
- **Azure OpenAI** 资源，至少包含 GPT-4o/GPT-4o mini 与 text-embedding-3-large（或自定义部署映射）。
- 可访问的 **Chroma 持久化目录**（默认 `aetheria_simple/chroma_db`）。首次使用可通过脚本构建。

## 安装步骤

```bash
# 1. Python 依赖
cd aetheria_simple
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r ../requirements.txt

# 2. 前端依赖
cd ../frontend
npm install
```

> 可根据需要将依赖固化到自定义 `pyproject.toml` 或锁定文件中以便共享。

## 环境变量

所有 Python 子模块会自动加载根目录 `.env`（依赖 `python-dotenv`），并支持以下键值。

### 必填

| 变量 | 备用名 | 说明 |
| --- | --- | --- |
| `AETHERIA_SIMPLE_AZURE_ENDPOINT` | `AZURE_ENDPOINT` | Azure OpenAI 终端地址（https://...openai.azure.com/）。 |
| `AETHERIA_SIMPLE_API_KEY` | `API_KEY`, `AZURE_API_KEY` | Azure OpenAI API Key。 |

### 可选

| 变量 | 用途 | 默认 |
| --- | --- | --- |
| `AETHERIA_SIMPLE_API_VERSION` | Azure OpenAI API 版本 | `2024-12-01-preview` |
| `AETHERIA_SIMPLE_SUPPORTER_MODEL` 等（`STRICT/LOOSE/ARBITER`） | 指定各角色模型 | `gpt-4o-mini` / `gpt-4o` |
| `AETHERIA_SIMPLE_RAG_TOP_K` | Supporter 检索案例数量 | `3` |
| `AETHERIA_SIMPLE_ROUNDS` | 辩论轮次 | `2` |
| `AETHERIA_SIMPLE_EMBEDDING_MODEL` | 案例库嵌入模型 | `text-embedding-3-large` |
| `AETHERIA_SIMPLE_CHROMA_PERSIST_DIR` | Chroma 持久化目录 | `./aetheria_simple/chroma_db` |
| `AETHERIA_SIMPLE_CHROMA_COLLECTION` | 默认集合名 | `usb_only_img_case_library` |
| `AETHERIA_SIMPLE_CASE_LIBRARY_PATH` | Supporter 案例库 JSON | `./aetheria_simple/case_libraries/default_case_library.json` |
| `AETHERIA_SIMPLE_DEPLOYMENT_MAP` | 模型→部署 JSON 映射 | identity |
| `AETHERIA_SIMPLE_DEPLOYMENT_<MODEL>` | 单模型部署 ID 覆盖 | — |
| `AETHERIA_SIMPLE_ENABLE_SUPPORTER/STRICT/LOOSE` | 打开/关闭单个角色 | `True` |

前端可在 `frontend/.env` 设置 `VITE_API_BASE_URL=http://localhost:8000` 指向对应的 BFF。

## 使用方式

### 1. 命令行批量评估

```bash
cd aetheria_simple
python -m aetheria_simple.main \
  --dataset /path/to/usb_text_img_relabeled.json \
  --limit 200 --workers 8 --skip 0
```

- 运行结束后，`result/` 会生成 `*.json` 汇总和 `*_details.csv` 详单。
- 可通过 `aetheria_simple/data_configs.py` 选择不同模态映射。

### 2. FastAPI BFF

```bash
uvicorn aetheria_simple.bff.app:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /api/review`：处理 `input_1`（文本）和 `input_2`（文本或 Base64 图像），日志存入 `logs/`。
- `GET /health`：健康检查。

### 3. Vue 评估面板

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

- 默认端口 `5173`，支持审核界面与样例展示视图。

## 案例库与向量数据库

1. **案例库维护**
   - `python aetheria_simple/scripts/build_balanced_case_library.py`
   - `python aetheria_simple/scripts/rebalance_dataset.py`
   - `python aetheria_simple/scripts/case_maintainer.py`

2. **构建 Chroma 向量库**

```bash
python aetheria_simple/scripts/build_database.py \
  --library-path aetheria_simple/case_libraries/default_case_library.json \
  --persist-dir aetheria_simple/chroma_db \
  --collection-name safety_cases_default
```

## 日志与排障

- CLI 指标输出至 stdout，详细结果在 `result/`。
- `ReviewService` 会将完整推理写入 `logs/<request_id>.jsonl`。
- 常见问题：
  - `Chroma collection not found`：确认集合名与持久化目录一致。
  - `Vision model is not configured`：处理图像输入时需配置多模态部署映射。
  - Azure 429 / 内容策略拒绝会在 CLI 中标记 `api_error`，需检查限流或输入内容。

## 开发提示

- 在 `aetheria_simple/prompts.py` 或 `prompts_en.py` 调整角色提示即可实验不同策略。
- `SimpleRunConfig` 支持通过环境变量热切换模型、RAG 参数、辩手开关。
- 扩展前端时可修改 `frontend/src/services/reviewService.js` 及相关组件保持一致。

欢迎根据团队需求扩展脚本、补充测试或将依赖固化为内部制品库。
