# Aetheria Simple Pipeline

The `aetheria_simple` package offers a slimmed-down variant of the original
multi-agent debate framework. It keeps the same dataset and output interfaces,
but removes the experimental guardrail logic and instrumentation. The Holistic
Arbiter's verdict becomes the final system prediction after a fixed number of
debate rounds.

## Key Differences

- **Evidence-weighted arbiter:** the arbiter vote is adopted when present and
  supported with medium/high confidence plus at least one policy reference.
  Otherwise the system falls back to the panel majority (or, if unavailable,
  defaults to safe).
- **Top-K RAG:** the supporter performs a direct similarity search over the
  existing Chroma index and surfaces the top *k* historical cases (default 3),
  without semantic filtering or re-balancing.
- **Fixed configuration:** a single `SimpleRunConfig` drives the pipeline; no
  ablation toggles or prompt variants.
- **Minimal prompts:** each agent uses short role-specific system prompts that
  emphasise score formatting.
- **Same CLI contract:** `python -m aetheria_simple.main` accepts `--limit` and
  `--workers` flags like the original entry point, writes reports to
  `./result`, and operates on the same dataset schema.

## Environment

The simplified stack now reads its own environment variables (`.env` is loaded
automatically via `python-dotenv`) and falls back to generic Azure/OpenAI names
when possible. Configure at least the mandatory credentials before running:

| Variable | Fallbacks | Description |
| --- | --- | --- |
| `AETHERIA_SIMPLE_AZURE_ENDPOINT` | `AZURE_ENDPOINT` | Azure OpenAI endpoint |
| `AETHERIA_SIMPLE_API_KEY` | `API_KEY`, `AZURE_API_KEY` | Azure OpenAI API key |

Optional overrides are exposed via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `AETHERIA_SIMPLE_RAG_TOP_K` | Number of retrieved cases shown to Supporter | `3` |
| `AETHERIA_SIMPLE_ROUNDS` | Debate rounds before arbitration | `2` |
| `AETHERIA_SIMPLE_SUPPORTER_MODEL` | Model for Supporter | `gpt-4o-mini` |
| `AETHERIA_SIMPLE_STRICT_MODEL` | Model for Strict Debater | `gpt-4o-mini` |
| `AETHERIA_SIMPLE_LOOSE_MODEL` | Model for Loose Debater | `gpt-4o-mini` |
| `AETHERIA_SIMPLE_ARBITER_MODEL` | Model for Holistic Arbiter | `gpt-4o` |
| `AETHERIA_SIMPLE_EMBEDDING_MODEL` | Embedding model used for Chroma index | `text-embedding-3-large` |
| `AETHERIA_SIMPLE_CHROMA_PERSIST_DIR` | Directory for the Chroma database | `./aetheria_simple/chroma_db` |
| `AETHERIA_SIMPLE_CHROMA_COLLECTION` | Default Chroma collection name | `usb_only_img_case_library` |
| `AETHERIA_SIMPLE_CASE_LIBRARY_PATH` | Supporter case library JSON path | `./aetheria_simple/case_libraries/default_case_library.json` |
| `AETHERIA_SIMPLE_DEPLOYMENT_MAP` | JSON mapping model names to Azure deployments | identity mapping |

Per-model overrides such as `AETHERIA_SIMPLE_DEPLOYMENT_GPT_4O` can also be set to
pin specific Azure deployment IDs without supplying a full map.

## Usage

```bash
python -m aetheria_simple.main --skip 100 --limit 50 --workers 8
```

The command prints a summary to stdout and saves JSON/CSV artefacts in the
`result/` directory, mirroring the behaviour of the full pipeline.

`--skip` can be used to ignore the first *N* samples before evaluation starts, which is
helpful when you want to resume from a previous run or focus on a later segment of the dataset.


### Batch image annotation

Use the helper script in `aetheria_simple/scripts/annotate_images.py` when you only
have raw image files and want the simplified evaluator to label them directly:

```bash
python aetheria_simple/scripts/annotate_images.py \
  --input-dir /path/to/images \
  --recursive \
  --text-template "Please review this image: {name}"
```

The script base64-encodes each image, feeds it through `ReviewService`, and writes
both JSON and CSV artefacts under `result/image_annotations/` (override with
`--output-dir`). Per-request logs still land in `logs/` unless you pass `--log-dir`.

Handy flags:

- `--limit`: stop after *N* files (quick smoke test).
- `--prompt-profile` / `--rag-collection`: align prompts and retrieval with your
  chosen case library (defaults match the image-only preset).
- `--text-template`: customise the textual input shown to the agents; `{name}` and
  `{path}` placeholders expand to the file name or absolute path.
