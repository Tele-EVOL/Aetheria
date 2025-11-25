# Aetheria

This repository hosts the reference implementation for the paper *“Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration.”* It reproduces the simplified multi-agent evaluator, the FastAPI BFF, and the Vue 3 workstation UI used in the experiments, demonstrating how Azure OpenAI, top-k RAG retrieval, and role-based debates deliver transparent content moderation.

## Repository Layout

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

## Prerequisites

- Python 3.10+ (use `venv`, `uv`, or Conda for isolation).
- Node.js ≥ 20.19 (matches `package.json` engines) and npm ≥ 10.
- Access to Azure OpenAI deployments for GPT-4o/GPT-4o mini and `text-embedding-3-large` (or configure your own deployment map).
- Writable storage for the Chroma DB (`aetheria_simple/chroma_db` by default).

## Installation

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

You can pin dependencies via a custom `pyproject.toml` or lockfile if desired.

## Environment Variables

All Python services load `.env` at the repo root (via `python-dotenv`) and honor the following keys.

### Required

| Variable | Aliases | Description |
| --- | --- | --- |
| `AETHERIA_SIMPLE_AZURE_ENDPOINT` | `AZURE_ENDPOINT` | Azure OpenAI endpoint (`https://...openai.azure.com/`). |
| `AETHERIA_SIMPLE_API_KEY` | `API_KEY`, `AZURE_API_KEY` | Azure OpenAI API key. |

### Optional Highlights

| Variable | Purpose | Default |
| --- | --- | --- |
| `AETHERIA_SIMPLE_API_VERSION` | Azure API version | `2024-12-01-preview` |
| `AETHERIA_SIMPLE_SUPPORTER_MODEL`, etc. | Model per role (supporter/strict/loose/arbiter) | `gpt-4o-mini` / `gpt-4o` |
| `AETHERIA_SIMPLE_RAG_TOP_K` | Supporter retrieval depth | `3` |
| `AETHERIA_SIMPLE_ROUNDS` | Debate rounds | `2` |
| `AETHERIA_SIMPLE_EMBEDDING_MODEL` | Case-library embedding model | `text-embedding-3-large` |
| `AETHERIA_SIMPLE_CHROMA_PERSIST_DIR` | Chroma persistence directory | `./aetheria_simple/chroma_db` |
| `AETHERIA_SIMPLE_CHROMA_COLLECTION` | Default collection name | `usb_only_img_case_library` |
| `AETHERIA_SIMPLE_CASE_LIBRARY_PATH` | Supporter case-library JSON | `./aetheria_simple/case_libraries/default_case_library.json` |
| `AETHERIA_SIMPLE_DEPLOYMENT_MAP` | Model→deployment JSON map | identity |
| `AETHERIA_SIMPLE_DEPLOYMENT_<MODEL>` | Per-model deployment override | — |
| `AETHERIA_SIMPLE_ENABLE_SUPPORTER/STRICT/LOOSE` | Toggle per-role participation | `True` |

Set `VITE_API_BASE_URL` in `frontend/.env` if your BFF runs on a non-default host/port.

## Usage

### 1. CLI Evaluation

```bash
cd aetheria_simple
python -m aetheria_simple.main \
  --dataset /path/to/usb_text_img_relabeled.json \
  --limit 200 --workers 8 --skip 0
```

- Outputs summary JSON and detailed CSV artifacts under `result/`.
- Switch dataset schemas using `aetheria_simple/data_configs.py`.

### 2. FastAPI BFF

```bash
uvicorn aetheria_simple.bff.app:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /api/review`: handles `input_1` (text) plus optional `input_2` (text/Base64 image) and logs to `logs/`.
- `GET /health`: readiness probe.

### 3. Vue Dashboard

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

- Serves the audit console on port `5173`, including review and showcase views.

## Case Library & Chroma DB

1. **Maintain case libraries**
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

## Logging & Troubleshooting

- CLI runs print metrics to stdout and write artifacts to `result/`.
- `ReviewService` stores full traces at `logs/<request_id>.jsonl` for debugging.
- Common issues:
  - `Chroma collection not found`: ensure the collection name and persistence dir match your build script.
  - `Vision model is not configured`: configure `AETHERIA_SIMPLE_DEPLOYMENT_MAP` (or per-model overrides) when handling Base64 images.
  - Azure 429 / content-policy failures appear as `api_error` in CLI runs and HTTP 500 via the BFF; review quotas and inputs.

## Development Notes

- Edit `aetheria_simple/prompts.py` or `prompts_en.py` to experiment with new role prompts.
- Override `SimpleRunConfig` via environment variables for ablations (models, RAG depth, debate toggles).
- When extending the UI, update `frontend/src/services/reviewService.js` and components to keep schemas in sync.

Feel free to adapt the stack for your own experiments, add tests, or package the dependencies for internal distribution.
