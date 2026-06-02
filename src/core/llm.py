from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    openrouter_key = os.getenv("OPENROUTER") or os.getenv("OPENROUTER_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        import time

        class RateLimitedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                time.sleep(5)
                retries = 3
                while retries > 0:
                    try:
                        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e) or "UNAVAILABLE" in str(e):
                            time.sleep(65)
                            retries -= 1
                            if retries == 0:
                                raise e
                        else:
                            raise e

        return RateLimitedChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=google_key,
            max_retries=2,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model=model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
            temperature=temperature,
            max_tokens=2048,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY")
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=groq_key,
            max_retries=2,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("CUSTOM_MODEL", "deepseek-v4-flash"),
            temperature=temperature,
            api_key=os.getenv("CUSTOM_API_KEY"),
            base_url=os.getenv("CUSTOM_LLM_ENDPOINT"),
            max_tokens=2048,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError("This lab supports only the `google` and `ollama` providers.")


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    prompt = f"""
You are grading a student order-agent answer.
Return JSON only with:
- score: integer from 0 to 10
- verdict: short string
- feedback: short list of strings

Rubric:
{rubric}

User query:
{query}

Student answer:
{answer}
""".strip()
    
    import time
    retries = 5
    while retries > 0:
        try:
            raw_content = model.invoke(prompt).content
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                time.sleep(45)
                retries -= 1
                if retries == 0:
                    raise e
            else:
                raise e

    payload = extract_json_object(raw_content)
    score = max(0, min(10, int(payload.get("score", 0))))
    return {
        "score": score,
        "verdict": str(payload.get("verdict", "")).strip(),
        "feedback": [str(item).strip() for item in payload.get("feedback", []) if str(item).strip()],
    }
