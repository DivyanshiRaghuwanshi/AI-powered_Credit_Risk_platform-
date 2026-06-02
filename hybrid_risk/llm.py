from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception as exc:  # pragma: no cover - import-time guard for clearer runtime errors
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("LLM did not return valid JSON.")


def _build_chat_model(temperature: float = 0.0) -> Any:
    if ChatOpenAI is None:
        raise RuntimeError(f"LangChain OpenAI support is not available: {_IMPORT_ERROR}")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    return ChatOpenAI(model=model_name, temperature=temperature)


def invoke_llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    model = _build_chat_model(temperature=temperature)
    response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return str(getattr(response, "content", "") or "").strip()


def invoke_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
    return parse_json_object(invoke_llm_text(system_prompt, user_prompt, temperature=temperature))