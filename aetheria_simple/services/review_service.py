"""Application-facing service wrapper for the simplified evaluator."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import imghdr

from openai import AzureOpenAI, OpenAI

from aetheria_simple import config
from aetheria_simple.config import SimpleRunConfig
from aetheria_simple.graph import SimpleMultiAgentEvaluator

logger = logging.getLogger(__name__)


def _normalise_content(content: Any) -> str:
    """Convert LangChain message payloads into compact strings."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                value = chunk.get("text") or chunk.get("content")
                if value:
                    parts.append(str(value))
            elif chunk:
                parts.append(str(chunk))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _serialise_messages(messages: List[Any]) -> List[Dict[str, str]]:
    serialised: List[Dict[str, str]] = []
    for message in messages:
        role = getattr(message, "type", None) or getattr(message, "role", "assistant")
        serialised.append({
            "role": str(role),
            "content": _normalise_content(getattr(message, "content", "")),
        })
    return serialised


def _extract_base64_image(raw_value: str | None) -> Tuple[str, str] | None:
    """Return canonical base64 payload + mime type when the value looks like an image."""

    if not raw_value:
        return None

    value = raw_value.strip()
    mime_type: Optional[str] = None
    base64_payload = value

    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
        except ValueError:
            return None
        base64_payload = encoded.strip()
        try:
            mime_type = header.split(";")[0].split(":", maxsplit=1)[1]
        except IndexError:
            mime_type = None

    try:
        decoded = base64.b64decode(base64_payload, validate=True)
    except (binascii.Error, ValueError):
        return None

    detected_format = imghdr.what(None, decoded)
    if mime_type is None:
        if detected_format:
            mime_type = f"image/{detected_format}"
        else:
            return None
    elif not mime_type.startswith("image/"):
        return None

    canonical_payload = base64.b64encode(decoded).decode("ascii")
    return canonical_payload, mime_type


@dataclass
class ReviewResult:
    request_id: str
    created_at: datetime
    predicted_score: Optional[int]
    reasoning: str
    background_info: str
    messages: List[Dict[str, str]]
    log_path: Path
    metadata: Dict[str, Any]
    rag_details: Dict[str, Any]
    strict_score: Optional[float]
    loose_score: Optional[float]
    strict_round_scores: List[float]
    loose_round_scores: List[float]
    panel_vote_source: str
    threshold_note: str
    majority_vote: Dict[str, Any]
    weighted_score: Optional[float]
    arbiter_vote: Optional[int]
    arbiter_payload: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["log_path"] = str(self.log_path)
        return payload


class ReviewService:
    """High-level helper that adapts the simplified evaluator to product use cases."""

    def __init__(
        self,
        *,
        log_dir: str | Path = "logs",
        evaluator: Optional[SimpleMultiAgentEvaluator] = None,
        settings: Optional[SimpleRunConfig] = None,
        vision_model: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._evaluator = evaluator or SimpleMultiAgentEvaluator(settings)
        self._vision_model = vision_model or config.DEPLOYMENT_MAP.get("gpt-4o")
        self._vision_client: object | None = None

    def _get_vision_client(self) -> object:
        if not self._vision_model:
            raise RuntimeError("Vision model is not configured for image captioning.")
        if self._vision_client is None:
            if config.USING_AZURE:
                self._vision_client = AzureOpenAI(
                    api_key=config.API_KEY,
                    azure_endpoint=config.AZURE_ENDPOINT,
                    api_version=config.API_VERSION,
                )
            else:
                self._vision_client = OpenAI(
                    api_key=config.API_KEY,
                    base_url=config.OPENAI_BASE or None,
                )
        return self._vision_client

    def _describe_image(self, base64_payload: str, mime_type: str) -> str:
        client = self._get_vision_client()
        data_url = f"data:{mime_type};base64,{base64_payload}"
        response = client.chat.completions.create(
            model=self._vision_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You provide objective, detailed descriptions of images without speculation.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate a concise yet detailed description of the image content.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        )

        if not response.choices:
            return ""

        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _prepare_input_2(self, raw_input_2: str | None) -> Tuple[str, Dict[str, Any]]:
        if not raw_input_2:
            return "", {}

        image_payload = _extract_base64_image(raw_input_2)
        if not image_payload:
            return raw_input_2, {}

        base64_payload, mime_type = image_payload
        metadata: Dict[str, Any] = {
            "source": "image_base64",
            "mime_type": mime_type,
            "original_length": len(raw_input_2),
        }

        if not self._vision_model:
            metadata["status"] = "skipped_no_model"
            metadata["description"] = ""
            logger.warning("Image input detected but no vision model configured; skipping caption generation.")
            return "[Image content unavailable]", metadata

        try:
            description = self._describe_image(base64_payload, mime_type)
        except Exception as exc:  # pragma: no cover - defensive logging
            metadata["status"] = "description_failed"
            metadata["error"] = str(exc)
            metadata["description"] = ""
            logger.exception("Failed to generate image description for input_2")
            return "[Image content could not be described]", metadata

        if not description:
            metadata["status"] = "description_empty"
            metadata["description"] = ""
            return "[Image description unavailable]", metadata

        metadata["status"] = "described"
        metadata["description"] = description
        return description, metadata

    def review(
        self,
        input_1: str,
        input_2: str | None = "",
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        if not input_1:
            raise ValueError("input_1 must not be empty.")

        request_id = uuid.uuid4().hex
        created_at = datetime.now(tz=timezone.utc)

        logger.info("Running simplified review %s", request_id, extra={"request_id": request_id})

        transformed_input_2, processing_metadata = self._prepare_input_2(input_2)
        enriched_metadata = dict(metadata or {})
        if processing_metadata:
            enriched_metadata["input_2_processing"] = processing_metadata

        result = self._evaluator.evaluate(
            input_1=input_1,
            input_2=transformed_input_2,
        )

        serialised_messages = _serialise_messages(result.get("messages", []))

        log_payload = {
            "request_id": request_id,
            "created_at": created_at.isoformat(),
            "input_1": input_1,
            "input_2": transformed_input_2,
            "predicted_score": result.get("predicted_score"),
            "reasoning": result.get("reasoning"),
            "background_info": result.get("background_info"),
            "messages": serialised_messages,
            "metadata": enriched_metadata,
            "rag_details": result.get("rag_details", {}),
            "strict_score": result.get("strict_score"),
            "loose_score": result.get("loose_score"),
            "strict_round_scores": result.get("strict_round_scores", []),
            "loose_round_scores": result.get("loose_round_scores", []),
            "panel_vote_source": result.get("panel_vote_source"),
            "threshold_note": result.get("threshold_note"),
            "majority_vote": result.get("majority_vote", {}),
            "weighted_score": result.get("weighted_score"),
            "arbiter_vote": result.get("arbiter_vote"),
            "arbiter_payload": result.get("arbiter_payload", {}),
        }

        log_path = self._log_dir / f"{created_at.strftime('%Y%m%d_%H%M%S')}_{request_id}.json"
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(log_payload, handle, ensure_ascii=False, indent=2)

        logger.info(
            "Review %s completed with score %s",
            request_id,
            result.get("predicted_score"),
            extra={"request_id": request_id, "log_path": str(log_path)},
        )

        return ReviewResult(
            request_id=request_id,
            created_at=created_at,
            predicted_score=result.get("predicted_score"),
            reasoning=result.get("reasoning", ""),
            background_info=result.get("background_info", ""),
            messages=serialised_messages,
            log_path=log_path,
            metadata=enriched_metadata,
            rag_details=result.get("rag_details", {}),
            strict_score=result.get("strict_score"),
            loose_score=result.get("loose_score"),
            strict_round_scores=result.get("strict_round_scores", []),
            loose_round_scores=result.get("loose_round_scores", []),
            panel_vote_source=result.get("panel_vote_source", "unknown"),
            threshold_note=result.get("threshold_note", ""),
            majority_vote=result.get("majority_vote", {}),
            weighted_score=result.get("weighted_score"),
            arbiter_vote=result.get("arbiter_vote"),
            arbiter_payload=result.get("arbiter_payload", {}),
        )
