import json
import os
from src.utils.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from src.talk_to_data.prompt_templates import (
    SQL_SYSTEM_PROMPT, SQL_USER_PROMPT_TEMPLATE,
    SQL_REPAIR_SYSTEM_PROMPT, SQL_REPAIR_USER_PROMPT_TEMPLATE
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def _get_llm_model():
    if not OPENAI_API_KEY:
        raise ValueError("Missing LLM API key. Set OPENAI_API_KEY in .env.")
        
    # Standard ChatOpenAI natively supports any OpenAI-compatible base URL (Groq, Gemini, etc.)
    return ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL or None,
        temperature=0.0
    )

def generate_sql(user_question: str, semantic_context: str, conversation_context: str = "") -> str:
    llm = _get_llm_model()
    prompt = SQL_USER_PROMPT_TEMPLATE.format(
        semantic_context=semantic_context,
        prior_block=conversation_context,
        user_question=user_question
    )
    
    resp = llm.invoke([
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    txt = resp.content.strip()
    # Strip fences
    if txt.startswith("```"):
        txt = txt.strip("`").replace("json", "").strip()
        
    payload = json.loads(txt)
    return payload["sql"]

def repair_sql(user_question: str, semantic_context: str, failed_sql: str, db_error: str, conversation_context: str = "") -> str:
    llm = _get_llm_model()
    prompt = SQL_REPAIR_USER_PROMPT_TEMPLATE.format(
        semantic_context=semantic_context,
        prior_block=conversation_context,
        user_question=user_question,
        failed_sql=failed_sql,
        db_error=db_error
    )
    
    resp = llm.invoke([
        SystemMessage(content=SQL_REPAIR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    txt = resp.content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").replace("json", "").strip()
        
    payload = json.loads(txt)
    return payload["sql"]
