import argparse
import json
from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from .. import config

# --- 配置 ---
if config.USING_AZURE:
    EMBEDDING_MODEL = AzureOpenAIEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.API_KEY,
        api_version=config.API_VERSION,
    )
else:
    EMBEDDING_MODEL = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.OPENAI_BASE or None,
    )


def _to_str_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            return [text]
        return [text]
    return []


def _trim(text: object, limit: int = 600) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if limit > 0 and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _compose_document_text(case: Dict[str, object]) -> str:
    parts: List[str] = []
    decision = str(case.get("final_label") or "unknown").strip().lower()
    dataset_label = str(case.get("dataset_label") or "unknown").strip().lower()
    parts.append(f"Final label: {decision} | Dataset label: {dataset_label}")

    summary = case.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"Summary: {summary.strip()}")

    key_cues = _to_str_list(case.get("key_cues"))
    if key_cues:
        parts.append(
            "Key cues:\n- " + "\n- ".join(_trim(item, 200) for item in key_cues)
        )

    prompt_tips = _to_str_list(case.get("prompt_tips"))
    if prompt_tips:
        parts.append(
            "Prompt tips:\n- "
            + "\n- ".join(_trim(item, 200) for item in prompt_tips)
        )

    source_excerpt = case.get("source_excerpt")
    if isinstance(source_excerpt, str) and source_excerpt.strip():
        parts.append(f"Source excerpt: {_trim(source_excerpt, 400)}")

    return "\n\n".join(parts).strip()


def _sanitize_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            try:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                cleaned[key] = str(value)
    return cleaned


def build_and_persist_db(
    *,
    library_path: Path,
    persist_dir: Path,
    collection_name: str,
) -> None:
    """
    加载案例库，构建 Chroma 向量数据库，并将其持久化。
    """
    print("--- 开始构建向量数据库 ---")
    print(f"1. 从 {library_path} 加载案例库...")
    if not library_path.exists():
        print(
            f"错误: 未找到案例库文件 '{library_path}'。"
            "请先运行 'build_balanced_case_library.py'。"
        )
        return

    with library_path.open("r", encoding="utf-8") as fh:
        cases: List[Dict[str, object]] = json.load(fh)

    if not cases:
        print("案例库为空，跳过构建。")
        return

    documents: List[Document] = []
    for case in cases:
        case_id = case.get("case_id")
        if not case_id:
            continue
        text = _compose_document_text(case)
        if not text:
            continue
        metadata = {
            "case_id": case_id,
            "final_label": case.get("final_label"),
            "dataset_label": case.get("dataset_label"),
            "summary": case.get("summary"),
            "key_cues": case.get("key_cues"),
            "prompt_tips": case.get("prompt_tips"),
            "source_excerpt": case.get("source_excerpt"),
            "next_actions": case.get("next_actions"),
            "updated_at": case.get("updated_at"),
        }
        metadata = _sanitize_metadata(metadata)
        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    if not documents:
        print("没有可用于嵌入的案例文本。")
        return

    print(f"2. 成功加载并准备了 {len(documents)} 份文档。")

    vectorstore = Chroma(
        embedding_function=EMBEDDING_MODEL,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )

    # 清理旧集合，确保不会重复添加
    vectorstore.delete_collection()
    vectorstore = Chroma(
        embedding_function=EMBEDDING_MODEL,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )

    print("3. 正在写入向量数据库...")
    total_docs = len(documents)
    batch_size = max(1, config.BUILD_DATABASE_BATCH_SIZE)
    for start in range(0, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        batch = documents[start:end]
        print(
            f"   - 处理文档 {start + 1}-{end} / {total_docs}"
        )
        vectorstore.add_documents(documents=batch)

    print(f"4. 数据库创建成功，并已持久化至 '{persist_dir}'。")
    print("--- 构建完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Chroma vector database from a case library."
    )
    parser.add_argument(
        "--library-path",
        default=config.CASE_LIBRARY_PATH,
        help="Path to the case library JSON file.",
    )
    parser.add_argument(
        "--persist-dir",
        default=config.CHROMA_PERSIST_DIR,
        help="Directory where the Chroma collection should be stored.",
    )
    parser.add_argument(
        "--collection-name",
        default=config.CHROMA_COLLECTION_NAME,
        help="Name of the Chroma collection to create/overwrite.",
    )
    args = parser.parse_args()

    build_and_persist_db(
        library_path=Path(args.library_path).expanduser(),
        persist_dir=Path(args.persist_dir).expanduser(),
        collection_name=args.collection_name,
    )
