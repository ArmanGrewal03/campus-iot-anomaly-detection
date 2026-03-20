from datetime import datetime, timezone
import asyncio
import os

from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


app = FastAPI(title="Campus IoT LLM Service", version="1.0.0")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
DEFAULT_TIMEOUT = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
CONTEXT_TIMEOUT = float(os.getenv("APP_CONTEXT_TIMEOUT_SECONDS", "5"))

DATA_INGESTION_SERVICE = os.getenv("DATA_INGESTION_SERVICE", "http://127.0.0.1:8000")
MODEL_SERVICE = os.getenv("MODEL_SERVICE", "http://127.0.0.1:8001")
USER_SERVICE = os.getenv("USER_SERVICE", "http://127.0.0.1:8002")
SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are an assistant for a campus IoT anomaly detection platform. "
    "Prioritize factual answers from provided application context. "
    "If context is unavailable or incomplete, say so explicitly and avoid guessing.",
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=8000)
    model: str | None = Field(default=None, max_length=100)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    include_application_context: bool = True
    history_limit: int = Field(default=20, ge=1, le=100)


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str = "groq"
    timestamp: str
    used_application_context: bool = False
    context_sources: list[str] = []


def get_llm(model_name: str, temperature: float) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY environment variable",
        )

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        timeout=DEFAULT_TIMEOUT,
    )


async def _safe_get_json(client: httpx.AsyncClient, url: str, source: str) -> tuple[str, dict]:
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return source, {"ok": True, "data": data}
    except Exception as exc:
        return source, {"ok": False, "error": str(exc)}


async def build_application_context(history_limit: int) -> tuple[str, list[str]]:
    context_payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "services": {},
        "kpis": {},
        "model": {},
        "dataset": {},
        "recent_history": {},
    }
    successful_sources: list[str] = []

    async with httpx.AsyncClient(timeout=CONTEXT_TIMEOUT) as client:
        calls = [
            ("data_ingestion_health", f"{DATA_INGESTION_SERVICE}/health"),
            ("model_service_health", f"{MODEL_SERVICE}/health"),
            ("user_service_health", f"{USER_SERVICE}/health"),
            ("user_dashboard_kpis", f"{USER_SERVICE}/dashboard-kpis"),
            ("model_status", f"{MODEL_SERVICE}/model/status"),
            ("model_list", f"{MODEL_SERVICE}/models"),
            ("data_stats", f"{DATA_INGESTION_SERVICE}/stats"),
            ("type_stats", f"{DATA_INGESTION_SERVICE}/type-stats"),
            ("recent_history", f"{USER_SERVICE}/history?limit={history_limit}&offset=0"),
        ]

        results = await asyncio.gather(
            *[_safe_get_json(client, url, source) for source, url in calls]
        )

    result_map = {source: payload for source, payload in results}

    context_payload["services"] = {
        "data_ingestion": result_map["data_ingestion_health"],
        "model_service": result_map["model_service_health"],
        "user_service": result_map["user_service_health"],
    }
    context_payload["kpis"] = result_map["user_dashboard_kpis"]
    context_payload["model"] = {
        "status": result_map["model_status"],
        "models": result_map["model_list"],
    }
    context_payload["dataset"] = {
        "stats": result_map["data_stats"],
        "type_stats": result_map["type_stats"],
    }
    context_payload["recent_history"] = result_map["recent_history"]

    for source, payload in result_map.items():
        if payload.get("ok"):
            successful_sources.append(source)

    context_text = (
        "APPLICATION_CONTEXT_JSON\n"
        "Use this real-time system context when answering. "
        "If any field is missing or failed, state that clearly.\n"
        f"{context_payload}"
    )
    return context_text, successful_sources


@app.get("/health")
@app.get("/llm/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LLM Service",
        "provider": "groq",
        "default_model": DEFAULT_MODEL,
        "api_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "application_context_sources": {
            "data_ingestion_service": DATA_INGESTION_SERVICE,
            "model_service": MODEL_SERVICE,
            "user_service": USER_SERVICE,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model_name = req.model or DEFAULT_MODEL
    temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature

    llm = get_llm(model_name=model_name, temperature=temperature)

    context_sources: list[str] = []
    used_application_context = False
    user_prompt_content = req.prompt
    if req.include_application_context:
        try:
            context_text, context_sources = await build_application_context(req.history_limit)
            used_application_context = True
            user_prompt_content = f"{context_text}\n\nUSER_QUESTION\n{req.prompt}"
        except Exception:
            # Continue without app context if context aggregation fails.
            used_application_context = False

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return ChatResponse(
            content=content,
            model=model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            used_application_context=used_application_context,
            context_sources=context_sources,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq request failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8004)
