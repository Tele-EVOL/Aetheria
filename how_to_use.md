# Aetheria 快速使用清单

## 目录速览
- `aetheria_simple/`：Python 简化评估器与 FastAPI BFF。主要文件：`main.py` CLI 入口；`graph.py` 多智能体流程与裁决；`agents.py` 角色定义；`config.py`/`prompts.py`/`prompts_en.py` 配置与提示；`evaluate.py` 批量评估逻辑；`data_configs.py` 数据字段映射；`services/` 业务服务；`bff/app.py` FastAPI 接口；`scripts/` 数据库与案例库维护工具。
- `frontend/`：Vue 3 + Vite 前端界面，调用 BFF 提供可视化审核体验。
- `data/`：示例数据集与辅助脚本，便于本地快速跑通与实验。
- `logs/`：运行时生成的审核日志与模型对话记录。
- `result/`：批量评估的输出报告与明细（运行生成）。

## 1) 选择提供商并配置环境变量
- 任选其一：
  - **Azure OpenAI**：`AETHERIA_SIMPLE_AZURE_ENDPOINT=https://xxx.openai.azure.com/`，`AETHERIA_SIMPLE_API_KEY=...`，可选 `AETHERIA_SIMPLE_API_VERSION`、`AETHERIA_SIMPLE_DEPLOYMENT_*`。
  - **OpenAI**：`OPENAI_API_KEY=...`，可选 `OPENAI_BASE`（自托管/兼容站点）。
- 如需强制指定：`AETHERIA_SIMPLE_PROVIDER=azure|openai`。
- 其他常用可选项：`AETHERIA_SIMPLE_RAG_TOP_K`、`AETHERIA_SIMPLE_ROUNDS`、`AETHERIA_SIMPLE_SUPPORTER/STRICT/LOOSE/ARBITER_MODEL`、`AETHERIA_SIMPLE_CHROMA_PERSIST_DIR`。

## 2) 安装依赖
```bash
# Python
cd aetheria_simple
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r ../requirements.txt

# 前端
cd ../frontend
npm install
```

## 3) 启动后端（BFF）
```bash
cd aetheria_simple
source .venv/bin/activate
uvicorn aetheria_simple.bff.app:app --host 0.0.0.0 --port 8000 --reload
```
- 健康检查：`GET /health`
- 核心接口：`POST /api/review`，JSON 示例：
```json
{
  "input_1": "待审核文本",
  "input_2": "data:image/png;base64,....",  // 可选，文本或图片 Base64
  "metadata": {"source": "demo"}            // 可选
}
```

## 4) 启动前端
```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
# 浏览器访问 http://localhost:5173
```

## 5) 离线批量评估（可选）
```bash
cd aetheria_simple
source .venv/bin/activate
python -m aetheria_simple.main --dataset /path/to/usb_text_img_relabeled.json --limit 200 --workers 8
```
- 结果输出到 `result/`，日志在 `logs/`。

## 6) 构建/更新向量库（可选，需案例库 JSON）
```bash
cd aetheria_simple
source .venv/bin/activate
python aetheria_simple/scripts/build_database.py \
  --library-path aetheria_simple/case_libraries/default_case_library.json \
  --persist-dir aetheria_simple/chroma_db \
  --collection-name safety_cases_default
```
