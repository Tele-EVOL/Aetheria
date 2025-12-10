from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from aetheria_simple import config, prompts
from aetheria_simple.utils.usage_tracker import LLMUsageTracker
from aetheria_simple.utils.voting import (
    compute_majority_vote,
    compute_weighted_score,
    format_vote_snapshot,
)


SupporterNode = Callable[[Dict[str, object]], Dict[str, object]]
DebaterNode = Callable[[Dict[str, object]], Dict[str, object]]
ArbiterNode = Callable[[Dict[str, object]], Dict[str, object]]


_EMBEDDINGS: Optional[AzureOpenAIEmbeddings] = None


def _get_embeddings() -> AzureOpenAIEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = AzureOpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.API_KEY,
            api_version=config.API_VERSION,
        )
    return _EMBEDDINGS


def _build_retriever(
    k: int, collection_name: str
) -> Optional[Callable[[str], List[Document]]]:
    try:
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=_get_embeddings(),
            collection_name=collection_name,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(
            f"[aetheria_simple] 无法加载向量数据库 '{collection_name}': {exc}"
        )
        return None
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke


def _format_case(doc: Document, index: int) -> str:
    metadata = doc.metadata or {}

    outcome = metadata.get("final_label") or "unknown"
    dataset_label = metadata.get("dataset_label") or "unknown"

    similarity = metadata.get("similarity")
    if similarity is None:
        score_value = metadata.get("score")
        if isinstance(score_value, (int, float)):
            similarity = float(score_value)
        else:
            distance = metadata.get("distance") or metadata.get("_distance")
            if isinstance(distance, (int, float)):
                similarity = max(0.0, 1.0 - float(distance))

    summary = metadata.get("summary")
    source_excerpt = metadata.get("source_excerpt") or ""

    def _to_list(value: object) -> List[str]:
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

    parts: List[str] = [f"Case {index}: dataset_label={dataset_label}, final_label={outcome}"]
    if isinstance(similarity, (int, float)):
        parts[-1] += f", similarity={similarity:.3f}"
    if summary:
        parts.append(f"Summary: {summary.strip()}")
    if source_excerpt:
        parts.append(f"Source excerpt: {source_excerpt.strip()}")

    library_notes = _to_list(metadata.get("key_cues"))
    if library_notes:
        parts.append("Key cues:\n- " + "\n- ".join(library_notes))

    prompt_notes = _to_list(metadata.get("prompt_tips"))
    if prompt_notes:
        parts.append("Prompt tips:\n- " + "\n- ".join(prompt_notes))

    return "\n".join(parts)


def _summarise_cases(docs: List[Document]) -> str:
    if not docs:
        return "暂未检索到相似案例，请直接依据当前输入和辩论内容评估风险。"
    lines = ["以下是最相似的历史案例："]
    for idx, doc in enumerate(docs, start=1):
        lines.append("")
        lines.append(_format_case(doc, idx))
    return "\n".join(lines)


def create_supporter_node(
    settings: config.SimpleRunConfig,
    system_prompt: str,
    llm: AzureChatOpenAI,
    usage_tracker: Optional[LLMUsageTracker] = None,
) -> SupporterNode:
    collection_name = settings.rag.collection_name or config.CHROMA_COLLECTION_NAME
    retriever = _build_retriever(settings.rag.top_k, collection_name)

    def agent_node(state: Dict[str, object]) -> Dict[str, object]:
        query_primary = str(state.get("input_1", ""))
        query_secondary = str(state.get("input_2", ""))
        query = f"{query_primary}\n{query_secondary}".strip()

        docs: List[Document] = []
        runtime_mode = "top_k"
        if retriever is None:
            runtime_mode = "stub"
        else:
            try:
                docs = retriever(query or query_primary)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[aetheria_simple] 检索失败: {exc}")
                runtime_mode = "error"
                docs = []

        cases_overview = _summarise_cases(docs)

        prompt_body = (
            "--- CURRENT INPUT ---\n"
            f"Text Input:\n{query_primary or '(空)'}\n\n"
            f"Secondary Input:\n{query_secondary or '(空)'}\n"
            "--- END OF INPUT ---\n\n"
            "--- RETRIEVED CASES ---\n"
            f"{cases_overview}\n"
            "--- END OF CASES ---"
        )

        supporter_brief = cases_overview
        llm_status = "skipped"

        if system_prompt.strip():
            llm_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_body),
            ]
            try:
                ai_response = _invoke_with_tracking(llm, llm_messages, usage_tracker)
                supporter_brief = ai_response.content
                llm_status = "success"
            except Exception as exc:  # pragma: no cover - defensive logging
                # print(f"[aetheria_simple] Supporter LLM 调用失败: {exc}")
                llm_status = "error"
                supporter_brief = (
                    "Supporter 简报不可用：LLM 请求被拦截或失败。"
                    "请基于当前输入和其他角色的讨论继续评估。"
                )

        rag_details = {
            "requested_mode": "top_k",
            "runtime_mode": runtime_mode,
            "top_k": settings.rag.top_k,
            "num_cases": len(docs),
            "collection_name": collection_name,
            "case_ids": [
                (doc.metadata or {}).get("case_id") for doc in docs if doc.metadata
            ],
            "cases_overview": cases_overview,
            "supporter_llm_status": llm_status,
        }

        return {"background_info": supporter_brief, "rag_details": rag_details}

    return agent_node


def _parse_risk_score(text: str) -> float:
    if not text:
        return -1.0
    import re

    match = re.search(r"Risk Score:\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"Risk Score:\s*([-+]?\d*\.?\d+)\s*%", text, re.IGNORECASE)
    value = -1.0
    if match:
        try:
            value = float(match.group(1))
            if value > 1.0:
                value /= 100.0
        except ValueError:
            value = -1.0
    else:
        if re.search(r"Risk Score:\s*(safe|low)", text, re.IGNORECASE):
            value = 0.0
        elif re.search(r"Risk Score:\s*(unsafe|high)", text, re.IGNORECASE):
            value = 1.0

    if value < 0.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def _format_round_series(label: str, history: List[float]) -> str:
    if not history:
        return f"{label}: no scores recorded."
    points = ", ".join(f"Round {idx + 1}: {value:.2f}" for idx, value in enumerate(history))
    return f"{label}: {points}"


def _format_messages(messages: List[BaseMessage]) -> str:
    if not messages:
        return "No history yet."
    return "\n".join([msg.content for msg in messages])


def create_regular_node(
    role: str,
    system_prompt: str,
    llm: AzureChatOpenAI,
    usage_tracker: Optional[LLMUsageTracker] = None,
) -> DebaterNode:
    def agent_node(state: Dict[str, object]) -> Dict[str, object]:
        messages = state["messages"]
        background_info = state["background_info"]
        current_round_index = int(state.get("round", 0))
        round_number = current_round_index + 1

        previous_score = None
        if current_round_index > 0:
            if role == "Strict Debater":
                previous_score = state.get("strict_score")
            elif role == "Loose Debater":
                previous_score = state.get("loose_score")

        formatted_history = _format_messages(messages)
        if "{round_number}" in system_prompt or "{previous_score}" in system_prompt:
            rendered_system_prompt = system_prompt.format(
                round_number=round_number,
                previous_score="N/A" if previous_score is None else f"{previous_score:.2f}",
            )
        else:
            rendered_system_prompt = system_prompt

        prompt_template = f"""{rendered_system_prompt}

--- CONTEXTUAL BRIEFING ---
{background_info}
--- END OF BRIEFING ---

Here is the conversation history so far:
{formatted_history}

All content awaiting review is as follows:
{state['input_1']}
{state['input_2']}

Provide your analysis and score now."""

        full_prompt_messages = messages + [HumanMessage(content=prompt_template)]
        ai_response = _invoke_with_tracking(llm, full_prompt_messages, usage_tracker)
        new_score = _parse_risk_score(ai_response.content)

        if new_score < 0.0:
            if isinstance(previous_score, (int, float)):
                new_score = float(previous_score)
            else:
                new_score = 0.5

        new_score = max(0.0, min(1.0, new_score))

        formatted_message = AIMessage(content=f"[{role}]: {ai_response.content}")
        return_state: Dict[str, object] = {"messages": [formatted_message]}

        if role == "Strict Debater":
            return_state["strict_score"] = new_score
            history_key = "strict_round_scores"
        elif role == "Loose Debater":
            return_state["loose_score"] = new_score
            history_key = "loose_round_scores"
        else:
            history_key = None

        if history_key:
            history = list(state.get(history_key, []))
            history.append(new_score)
            return_state[history_key] = history

        return return_state

    return agent_node


def create_arbiter_node(
    role: str,
    system_prompt: str,
    llm: AzureChatOpenAI,
    *,
    strict_bias: float = 0.6,
    usage_tracker: Optional[LLMUsageTracker] = None,
) -> ArbiterNode:
    def agent_node(state: Dict[str, object]) -> Dict[str, object]:
        messages = state["messages"]
        background_info = state["background_info"]
        strict_final = state.get("strict_score")
        loose_final = state.get("loose_score")
        strict_history = state.get("strict_round_scores", []) or []
        loose_history = state.get("loose_round_scores", []) or []
        panel_disabled = bool(state.get("panel_disabled", False))
        case_input_section = (
            "--- CASE INPUT ---\n"
            f"Text Input:\n{state.get('input_1') or '(空)'}\n\n"
            f"Secondary Input:\n{state.get('input_2') or '(空)'}\n"
            "--- END OF CASE INPUT ---"
        )

        majority = compute_majority_vote(strict_final, loose_final)
        weighted_score = compute_weighted_score(
            strict_final, loose_final, strict_bias=strict_bias
        )
        panel_snapshot = format_vote_snapshot(majority, weighted_score)

        trajectory_section = "\n".join(
            [
                _format_round_series("Strict Debater trajectory", strict_history),
                _format_round_series("Loose Debater trajectory", loose_history),
            ]
        )

        formatted_history = _format_messages(messages)
        if panel_disabled:
            panel_note = (
                "\n(Panel note: Debater roles were disabled in this evaluation. "
                "You must directly assess the CASE INPUT and available briefing to reach a verdict. "
                "不要仅因为缺少辩论记录就默认判定 SAFE。)"
            )
        else:
            panel_note = ""

        prompt_template = f"""{system_prompt}

--- SUPPORTER BRIEFING ---
{background_info}
--- END OF SUPPORTER BRIEFING ---

{case_input_section}

--- PANEL SNAPSHOT ---
{panel_snapshot}
{trajectory_section}
--- END OF SNAPSHOT ---
{panel_note}

--- DEBATE TRANSCRIPT ---
{formatted_history}
--- END OF TRANSCRIPT ---

Evaluate the case and return your final decision."""

        full_prompt_messages = messages + [HumanMessage(content=prompt_template)]
        ai_response = _invoke_with_tracking(llm, full_prompt_messages, usage_tracker)

        return {
            "final_decision": ai_response.content,
            "messages": [AIMessage(content=f"[{role}]: {ai_response.content}")],
        }

    return agent_node


@dataclass
class AgentBundle:
    supporter: Optional[SupporterNode]
    strict: Optional[DebaterNode]
    loose: Optional[DebaterNode]
    arbiter: ArbiterNode


def _resolve_model_name(llm: AzureChatOpenAI) -> str:
    for attr in ("model_name", "model", "deployment_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _extract_usage_payload(message: AIMessage) -> Dict[str, object]:
    usage_payload = getattr(message, "usage_metadata", None)
    if isinstance(usage_payload, dict) and usage_payload:
        return usage_payload

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            return token_usage

    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        token_usage = additional.get("usage")
        if isinstance(token_usage, dict):
            return token_usage

    return {}


def _invoke_with_tracking(
    llm: AzureChatOpenAI,
    messages: List[BaseMessage],
    tracker: Optional[LLMUsageTracker],
) -> AIMessage:
    start = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - start
    if tracker is not None:
        tracker.record(
            _resolve_model_name(llm),
            _extract_usage_payload(response),
            elapsed,
        )
    return response


class _LLMRegistry:
    """Lazy loader that reuses chat models across roles."""

    def __init__(self) -> None:
        self._cache: Dict[str, AzureChatOpenAI] = {}

    def get(self, model_name: str) -> AzureChatOpenAI:
        if model_name not in self._cache:
            deployment = config.AZURE_DEPLOYMENT_MAP.get(model_name, model_name)
            self._cache[model_name] = AzureChatOpenAI(
                model=deployment,
                azure_endpoint=config.AZURE_ENDPOINT,
                api_key=config.API_KEY,
                api_version=config.API_VERSION,
                temperature=0,
            )
        return self._cache[model_name]


def build_agents(
    settings: config.SimpleRunConfig,
    usage_tracker: Optional[LLMUsageTracker] = None,
) -> AgentBundle:
    models = settings.models.as_dict()
    llms = _LLMRegistry()

    profile_key = (settings.prompt_profile or "default").strip().lower()
    if profile_key not in prompts.PROMPT_SETS:
        print(f"[aetheria_simple] 未知提示配置 '{settings.prompt_profile}', 回退到 default。")
    prompt_pack = prompts.get_prompt_pack(profile_key)

    supporter_node: Optional[SupporterNode] = None
    supporter_model = models.get("Supporter")
    if settings.use_supporter and supporter_model:
        supporter_node = create_supporter_node(
            settings,
            prompt_pack["Supporter"],
            llms.get(supporter_model),
            usage_tracker,
        )

    loose_node: Optional[DebaterNode] = None
    loose_model = models.get("Loose Debater")
    if settings.use_loose_debater and loose_model:
        loose_node = create_regular_node(
            "Loose Debater",
            prompt_pack["Loose Debater"],
            llms.get(loose_model),
            usage_tracker,
        )

    strict_node: Optional[DebaterNode] = None
    strict_model = models.get("Strict Debater")
    if settings.use_strict_debater and strict_model:
        strict_node = create_regular_node(
            "Strict Debater",
            prompt_pack["Strict Debater"],
            llms.get(strict_model),
            usage_tracker,
        )

    arbiter_model = models.get("Holistic Arbiter")
    if not arbiter_model:
        raise ValueError("Holistic Arbiter 模型未配置，无法继续运行评估。")

    arbiter_node = create_arbiter_node(
        "Holistic Arbiter",
        prompt_pack["Holistic Arbiter"],
        llms.get(arbiter_model),
        strict_bias=settings.strict_bias,
        usage_tracker=usage_tracker,
    )

    return AgentBundle(
        supporter=supporter_node,
        strict=strict_node,
        loose=loose_node,
        arbiter=arbiter_node,
    )
