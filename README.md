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
pip install fastapi uvicorn[standard] langchain langchain-openai \
            langchain-community chromadb openai python-dotenv tqdm

# 2. 前端依赖
cd ../frontend
npm install
```

> 根据需要可将 Python 依赖写入自定义 `requirements.txt`/`pyproject.toml` 以便团队共享。

## 环境变量

所有 Python 端模块会自动加载根目录下的 `.env`（通过 `python-dotenv`），同时支持以下键：

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

前端另可在 `frontend/.env` 设置 `VITE_API_BASE_URL=http://localhost:8000` 指向 BFF。

## 使用方式

### 1. 命令行批量评估

```bash
cd aetheria_simple
python -m aetheria_simple.main \
  --dataset /path/to/usb_text_img_relabeled.json \
  --limit 200 --workers 8 --skip 0
```

- 任务完成后会在 `result/` 写入 `*.json` 汇总与 `*_details.csv` 详单（参见 `evaluate.py`）。
- 数据集字段映射可通过 `aetheria_simple/data_configs.py` 的预设（文字/图像/纯文本）定制。

### 2. FastAPI BFF

```bash
uvicorn aetheria_simple.bff.app:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /api/review`：参见 `aetheria_simple/bff/app.py` 与 `services/review_service.py`。`input_2` 支持 Base64 图像，服务会在 `logs/` 保存每次推理的对话及 RAG 结果。
- `GET /health`：健康检查。

### 3. Vue 评估面板

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

- 默认监听 `5173` 端口，见 `frontend/src/App.vue`。
- UI 包含「审核工作台」与「样例展示」两种视图，实时展示模型得分、RAG 命中案例、辩手轮次曲线等。

## 案例库与向量数据库

1. **构建/维护案例库**  
   - `python aetheria_simple/scripts/build_balanced_case_library.py`：融合多数据源并平衡标签分布。  
   - `python aetheria_simple/scripts/rebalance_dataset.py`：根据策略重抽样。  
   - `python aetheria_simple/scripts/case_maintainer.py`：增删改单条案例的元数据。

2. **构建 Chroma 向量库**  
   ```bash
   python aetheria_simple/scripts/build_database.py \
     --library-path aetheria_simple/case_libraries/default_case_library.json \
     --persist-dir aetheria_simple/chroma_db \
     --collection-name safety_cases_default
   ```
   运行结束后，Supporter 会直接从新的集合中检索相似案例并生成 `background_info`。

## 日志与排障

- **CLI 输出**：进度由 `tqdm` 展示，最终指标（准确率、召回、F1）与配置摘要会写入 `stdout`，详情写入 `result/`。
- **BFF/服务端日志**：`ReviewService` 默认写入 `logs/<request_id>.jsonl`，含所有消息（`messages` 字段）与 RAG 细节，便于复现问题。
- **常见问题**  
  - `Chroma collection not found`：确认 `AETHERIA_SIMPLE_CHROMA_PERSIST_DIR`、`AETHERIA_SIMPLE_CHROMA_COLLECTION` 与构建脚本保持一致。  
  - `Vision model is not configured`：当 `input_2` 为图像时需设置 `AETHERIA_SIMPLE_DEPLOYMENT_MAP` 或 `AETHERIA_SIMPLE_DEPLOYMENT_GPT_4O` 指向可用的多模态部署。
  - Azure 429/内容策略拦截会在 CLI 中标记为 `api_error`，BFF 会返回 500，需检查速率限制或输入内容。

## 开发提示

- 可在 `aetheria_simple/prompts.py` 中调整角色语气或输出格式，以快速实验不同的评估策略。
- `SimpleRunConfig`（`config.py`）支持通过环境变量热切换模型、RAG Top-K 与辩手开关，便于做消融实验。
- 若需扩展前端字段，可在 `frontend/src/services/reviewService.js` 中更新响应 Schema，并同步修改组件。

欢迎根据团队需求扩展脚本、补充测试或将依赖固化为内部制品库。

---

## English Version

### Overview

This repository hosts the reference implementation for the paper *“Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration.”* It reproduces the simplified multi-agent evaluator, the FastAPI BFF, and the Vue 3 workstation UI used in the experiments, showing how Azure OpenAI, top-k RAG retrieval, and role-based debates combine to deliver transparent content moderation.

### Repository Layout

| Path | Description |
| --- | --- |
| `aetheria_simple/` | Python evaluator with the agent graph (`graph.py`), node definitions (`agents.py`), config/prompts, and CLI entry (`main.py`). |
| `aetheria_simple/bff/` | FastAPI backend (`app.py`) exposing `/api/review` via `ReviewService`, handling text plus Base64 images. |
| `aetheria_simple/scripts/` | Maintenance utilities for building/rebalancing case libraries, creating the Chroma DB, and extracting misclassified samples. |
| `aetheria_simple/case_libraries/` | Seed case libraries consumed by the Supporter/RAG component. |
| `aetheria_simple/chroma_db/` | Default Chroma persistence directory (rebuild with the provided scripts as needed). |
| `frontend/` | Vue 3 + Vite UI with the review console and showcase view. |
| `result/` (runtime) | CLI evaluation reports (`*.json`) and detailed logs (`*_details.csv`). |
| `logs/` (runtime) | Per-request traces produced by `ReviewService`. |

### Prerequisites

- Python 3.10+ (use `venv`, `uv`, or Conda for isolation).
- Node.js ≥ 20.19 (matches `package.json` engines) and npm ≥ 10.
- Access to Azure OpenAI deployments for GPT-4o/GPT-4o mini and `text-embedding-3-large` (or provide your own deployment map).
- Writable storage for the Chroma DB (`aetheria_simple/chroma_db` by default).

### Installation

```bash
# Python dependencies
cd aetheria_simple
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r ../requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

Feel free to pin dependencies in a custom `pyproject.toml` or lock file if your workflow requires it.

### Environment Variables

All Python services load `.env` at the repo root (thanks to `python-dotenv`) and honor the following keys.

**Required**

| Variable | Aliases | Description |
| --- | --- | --- |
| `AETHERIA_SIMPLE_AZURE_ENDPOINT` | `AZURE_ENDPOINT` | Azure OpenAI endpoint (`https://...openai.azure.com/`). |
| `AETHERIA_SIMPLE_API_KEY` | `API_KEY`, `AZURE_API_KEY` | Azure OpenAI API key. |

**Optional Highlights**

| Variable | Purpose | Default |
| --- | --- | --- |
| `AETHERIA_SIMPLE_API_VERSION` | Azure API version | `2024-12-01-preview` |
| `AETHERIA_SIMPLE_SUPPORTER_MODEL`, etc. | Model per role (supporter/strict/loose/arbiter) | `gpt-4o-mini` / `gpt-4o` |
| `AETHERIA_SIMPLE_RAG_TOP_K` | Supporter retrieval depth | `3` |
| `AETHERIA_SIMPLE_ROUNDS` | Debate rounds | `2` |
| `AETHERIA_SIMPLE_EMBEDDING_MODEL` | Case-library embedding model | `text-embedding-3-large` |
| `AETHERIA_SIMPLE_CHROMA_PERSIST_DIR` | Chroma storage directory | `./aetheria_simple/chroma_db` |
| `AETHERIA_SIMPLE_CHROMA_COLLECTION` | Default Chroma collection | `usb_only_img_case_library` |
| `AETHERIA_SIMPLE_CASE_LIBRARY_PATH` | Case-library JSON path | `./aetheria_simple/case_libraries/default_case_library.json` |
| `AETHERIA_SIMPLE_DEPLOYMENT_MAP` | JSON deployment map | identity |
| `AETHERIA_SIMPLE_DEPLOYMENT_<MODEL>` | Per-model override | — |
| `AETHERIA_SIMPLE_ENABLE_SUPPORTER/STRICT/LOOSE` | Toggle individual roles | `True` |

Set `VITE_API_BASE_URL` inside `frontend/.env` to point the UI at your BFF (defaults to `http://localhost:8000`).

### Usage

**1. CLI Evaluation**

```bash
cd aetheria_simple
python -m aetheria_simple.main \
  --dataset /path/to/usb_text_img_relabeled.json \
  --limit 200 --workers 8 --skip 0
```

- Writes summary JSON and detailed CSV files into `result/`.
- Switch dataset schemas with the presets in `aetheria_simple/data_configs.py`.

**2. FastAPI BFF**

```bash
uvicorn aetheria_simple.bff.app:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /api/review`: accepts `input_1` (text) and optional `input_2` (text or Base64 image).
- `GET /health`: readiness probe.

**3. Vue Dashboard**

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

- Serves the audit console on port `5173` with real-time scoring, RAG insights, and debate traces.

### Case Library & Chroma DB

1. **Case library maintenance**
   - `python aetheria_simple/scripts/build_balanced_case_library.py`
   - `python aetheria_simple/scripts/rebalance_dataset.py`
   - `python aetheria_simple/scripts/case_maintainer.py`

2. **Build the Chroma vector store**

```bash
python aetheria_simple/scripts/build_database.py \
  --library-path aetheria_simple/case_libraries/default_case_library.json \
  --persist-dir aetheria_simple/chroma_db \
  --collection-name safety_cases_default
```

### Logs & Troubleshooting

- CLI metrics print to stdout and store artifacts under `result/`.
- `ReviewService` writes per-request traces to `logs/<request_id>.jsonl` for reproducibility.
- Common issues:
  - `Chroma collection not found`: ensure the collection name and persistence dir align with your build.
  - `Vision model is not configured`: configure `AETHERIA_SIMPLE_DEPLOYMENT_MAP` or `AETHERIA_SIMPLE_DEPLOYMENT_GPT_4O` when accepting images.
  - Azure throttling/policy errors surface as `api_error` in CLI runs and HTTP 500 on the BFF; check quotas and inputs.

### Development Notes

- Adjust role prompts in `aetheria_simple/prompts.py` to explore new reasoning styles.
- Override `SimpleRunConfig` via environment variables for quick ablations (models, RAG depth, debate toggles).
- Extend the UI by updating `frontend/src/services/reviewService.js` and related Vue components when the API schema changes.

Feel free to adapt the stack for your own experiments, add tests, or publish an internal package feed for the dependencies.
