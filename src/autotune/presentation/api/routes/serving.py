"""Serve a trained model with vLLM, and ask any OpenAI-compatible model a question."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autotune.core.domain.models import ModelEndpoint, ServingStatus
from autotune.defaults import CHECKPOINT_DIR, SERVE_PORT
from autotune.infrastructure.evaluators.openai_client import OpenAIChatClient
from autotune.presentation.api.schemas import PlaygroundAnswer, ServingStarted, ServingStopped
from autotune.presentation.api.state import State

router = APIRouter()


class ServingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: str = CHECKPOINT_DIR
    port: int = SERVE_PORT


class PromptGenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = Field(min_length=1)
    # The OpenAI-compatible server to ask, such as the base or the trained model.
    model: ModelEndpoint
    max_tokens: int = Field(default=1024, ge=1, le=131072)

    @model_validator(mode="after")
    def _prompt_has_text(self) -> "PromptGenerationRequest":
        if not self.prompt.strip():
            raise ValueError("prompt must contain text")
        return self


@router.post("/api/serve/start", status_code=202, response_model=ServingStarted)
async def start_vllm_server(req: ServingRequest, state: State) -> dict[str, Any]:
    try:
        endpoint = state.serving.start(req.model_path, req.port)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "starting",
        "model_path": req.model_path,
        "port": req.port,
        "endpoint_url": f"{endpoint.base_url}/v1",
    }


@router.get("/api/serve/status", response_model=ServingStatus)
async def serving_status(state: State) -> dict[str, Any]:
    return state.serving.status().model_dump()


@router.post("/api/serve/stop", response_model=ServingStopped)
async def stop_vllm_server(state: State) -> dict[str, Any]:
    state.serving.stop()
    return {"status": "stopped"}


@router.post("/api/playground/generate", response_model=PlaygroundAnswer)
def playground_generate(req: PromptGenerationRequest) -> dict[str, str]:
    answer = OpenAIChatClient().complete(
        req.model, [{"role": "user", "content": req.prompt.strip()}], max_tokens=req.max_tokens,
    )
    return {"response": answer}
