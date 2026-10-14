"""Resolve custom Hugging Face metadata without loading model weights."""
from __future__ import annotations

import json
import re
import urllib.request
from autotune.core.domain.model_catalog import ModelSpec, get_model, memory_baseline
from autotune.core.domain.models import QuantizationMethod
from autotune.infrastructure.evaluators.openai_client import Transport, _urlopen
from autotune.settings import HF_TOKEN, credential_value, is_configured


def resolve_model(hf_id: str, transport: Transport | None = None) -> ModelSpec:
    """Return reviewed metadata or require explicit parameter and context data from the Hub."""
    known = get_model(hf_id)
    if known:
        return known
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", hf_id) or '..' in hf_id:
        raise ValueError("Use a Hugging Face ID in owner/model form")
    transport = transport or _urlopen
    token = credential_value(HF_TOKEN)
    headers = {"Authorization": f"Bearer {token}"} if is_configured(token) else {}

    def fetch(url: str) -> dict:
        try:
            body = json.loads(transport(urllib.request.Request(url, headers=headers), 20))
            if not isinstance(body, dict):
                raise ValueError("Expected metadata object")
            return body
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read Hugging Face Hub metadata for {hf_id}. Check the ID, network, and HF_TOKEN: {exc}") from exc

    info = fetch(f"https://huggingface.co/api/models/{hf_id}")
    count = (info.get("safetensors") or {}).get("total")
    if type(count) is not int or count <= 0:
        raise ValueError(f"Hub has no reliable parameter count for {hf_id}. Choose a catalog model.")
    config = fetch(f"https://huggingface.co/{hf_id}/resolve/main/config.json")
    context = config.get("max_position_embeddings")
    if type(context) is not int or context <= 0:
        raise ValueError(f"Hub has no context length for {hf_id}. Choose a catalog model.")
    billions = count / 1e9
    return ModelSpec(hf_id, hf_id, count, str((info.get("cardData") or {}).get("license", "See model card")),
                     bool(info.get("gated")), False, context,
                     memory_baseline(billions, QuantizationMethod.QLORA_4BIT_NF4),
                     memory_baseline(billions, QuantizationMethod.LORA_16BIT))
