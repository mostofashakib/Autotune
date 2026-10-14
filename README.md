# Autotune

**LLM Post-Training & Benchmarking Engine — Turn raw domain documents and datasets into fine-tuned models without deep learning infrastructure expertise.**

Autotune gives developers, researchers, and domain specialists a complete post-training pipeline without requiring machine learning engineering experience. Upload your own documents (`.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.zip`) or structured data (`.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`). Autotune does the heavy lifting: extracting and chunking text, generating grounded training pairs, fitting model weights to your GPU's VRAM to prevent CUDA Out-Of-Memory errors, choosing optimal tuning recipes (SFT, DPO, or GRPO), running training, and benchmarking quality against the base model with pass@k evaluations.

Works directly with [Forge](https://github.com/mostofashakib/Forge) for execution-verified synthetic datasets.

---

## Why Autotune?

- **Unstructured Document Ingestion:** Upload raw PDF, Word, HTML, text, Markdown files, or ZIP archives without manual preprocessing. Autotune extracts clean text, chunks by token counts, generates grounded training pairs through an LLM, filters duplicates and unsupported answers, and splits train/validation data cleanly by document.
- **Automatic Format Detection:** Drop in CSV, JSONL, or Parquet files without converting schemas. Autotune detects conversational turns, prompt/response pairs, tool use, reward labels, and preference pairs (`prompt` + `chosen` + `rejected`) automatically.
- **Hardware-Aware Model Catalog:**
  - Choose from 11 verified open-weight instruct models (0.5B to 14.7B parameters across Qwen, Llama, Gemma, Mistral, and Phi) or provide a custom Hugging Face ID.
  - **QLoRA (4-bit NF4):** Uses bitsandbytes 4-bit NormalFloat with estimated minima of 5GB to 19GB VRAM.
  - **LoRA (16-bit):** Selected when the model fits at 16-bit precision within available GPU memory.
- **Beginner Presets & Expert Controls:** Basic view offers Quick, Balanced, and Thorough training intensity presets, while Advanced exposes full hyperparameter control (LoRA rank/alpha, batch size, gradient accumulation, sequence length) backed by server-side preflight VRAM estimation.
- **Reinforcement & Preference Learning:** Train alignment and reasoning models with Direct Preference Optimization (DPO) or Group Relative Policy Optimization (GRPO).
- **vLLM Serving:** Serve checkpoints with an OpenAI-compatible endpoint in one command.
- **Cloud GPUs on RunPod:** Rent on-demand RTX 4090, A100, and H100 GPUs directly from your terminal or UI.
- **Side-by-Side Benchmark:** Measure your trained model against its base on held-out data or on Harbor agent tasks, reporting per-task pass@1 and pass@k from real samples.

---

## Architecture

```
autotune/
├── core/
│   ├── domain/          # Entities: Recipe, ModelSpec, TrainingSettings, DatasetRecord, ServingEndpoint
│   ├── ports/           # Interface protocols (DataSanitizer, TrainingBackend, ServingProvider, CloudComputeProvider)
│   └── services/        # Hardware-aware AutoRecipeSelector (LoRA vs QLoRA, memory fitting, GRPO/DPO)
├── infrastructure/
│   ├── backends/        # TrlPeftBackend and reward mapping (GRPO, DPO, SFT)
│   ├── sanitizers/      # AutoFormatSanitizer, DocumentIngestor, and dataset loaders
│   ├── models/          # Hub metadata resolution for custom Hugging Face models
│   ├── storage/         # PolicyCheckpoint serialization and SQLite JobStore
│   ├── serving/         # VLLMServingProvider (OpenAI-compatible inference server)
│   ├── cloud/           # RunPodCloudProvider (GraphQL remote GPU lifecycle)
│   └── evaluators/      # Dataset and Harbor benchmarks, model client, scoring, judge, metrics (pass@k)
└── presentation/
    ├── api/             # FastAPI cockpit routes and background workers
    └── cli/             # Typer CLI commands (inspect, train, serve, benchmark, cloud)
```

---

## Quickstart

### Credentials
Every credential Autotune reads is declared in `src/autotune/settings.py` and listed in `.env.example`:

```bash
cp .env.example .env   # then replace the mock_key placeholders you need
```

| Variable | Needed for |
|---|---|
| `RUNPOD_API_KEY` | Renting cloud GPUs on RunPod |
| `HF_TOKEN` | Downloading gated Hugging Face base models such as `meta-llama/*` |
| `AUTOTUNE_DOCUMENT_MODEL` | Generator model name for document ingestion (e.g. `gemma3:1b`, `qwen2.5:7b`) |
| `AUTOTUNE_DOCUMENT_ENDPOINT` | Generator endpoint (default: `http://localhost:11434/v1` for local Ollama) |
| `AUTOTUNE_DOCUMENT_API_KEY_ENV` | Optional env var holding the API key for authenticated generator endpoints |

A value set in the environment wins over `.env`. A credential left at `mock_key` counts as not configured, so the feature that needs it refuses to run and names the variable. Model servers for benchmarks and the playground take their keys from env vars you name per run (`--base-key-env`, `--trained-key-env`, `--judge-key-env`).

### 1. Web UI
The web UI in `frontend/` is a Next.js app built on Forge's design system. It talks to the API that `autotune api` runs.

```bash
./run.sh
# or run the two halves yourself
uv run autotune api --port 7860
npm --prefix frontend install && npm --prefix frontend run dev   # http://localhost:3001
```

| Page | What you do there |
|---|---|
| Home | See the device, VRAM, recommended quantization, missing credentials, serving state, and recent runs |
| Training | Upload structured files or documents (ZIP for folders), review the detected format and recipe, choose presets or advanced overrides, and start a run with live loss and step logs. Training needs the `training` extras and a CUDA GPU. |
| Benchmark | Compare a base and a trained model on a held-out dataset or Harbor tasks, with an optional LLM judge. Open a run for its measured report. |
| Serving | Start or stop vLLM on a checkpoint, watch its state and log, and send one prompt to two models side by side |
| Cloud | Browse RunPod's live GPU catalog, launch a pod, and terminate the pods you launched. Needs `RUNPOD_API_KEY`. |
| Settings | See which credentials are configured (never their values) and where to set them |

Runs and uploads persist under `AUTOTUNE_DATA_DIR` (default `./data`, gitignored), so they survive a restart. A run that was in progress when the API restarted shows as interrupted. The API accepts cross-origin requests only from the UI, at `AUTOTUNE_UI_ORIGIN` (default `http://localhost:3001`). Point the UI at a different API with `NEXT_PUBLIC_API_URL` (see `frontend/.env.example`).

### 2. Inspect a Dataset
Inspect a structured dataset file or document archive to see detected task types and recommended tuning settings:
```bash
uv run autotune inspect my_data.jsonl
```

### 3. Train a Model (SFT, DPO, or GRPO)
Run post-training with automated model selection and LoRA/QLoRA tuning:
```bash
# Standard Supervised Fine-Tuning (SFT)
uv run autotune train my_data.jsonl --output-dir ./checkpoints/my-model

# Direct Preference Optimization (DPO)
uv run autotune train my_preference_data.jsonl --objective dpo --output-dir ./checkpoints/my-dpo-model

# Reinforcement Learning with GRPO
uv run autotune train my_rl_data.parquet --objective grpo --output-dir ./checkpoints/my-rl-model
```

### 4. Serve with vLLM
Launch an OpenAI-compatible vLLM server:
```bash
uv run autotune serve ./checkpoints/my-model --port 8000
```

### 5. Cloud GPUs on RunPod
List available GPU instances and deploy on-demand training pods:
```bash
# List GPUs and pricing
uv run autotune cloud list-gpus

# Provision remote RTX 4090 pod
uv run autotune cloud launch --gpu "NVIDIA GeForce RTX 4090" --name "rl-trainer"

# Check pod connection status
uv run autotune cloud status <pod_id>

# Stop pod after training
uv run autotune cloud stop <pod_id>
```

### 6. Benchmark Against the Base Model
Serve both models from any OpenAI-compatible server (vLLM, RunPod, Ollama), then compare them on the same tasks. Every number in the report comes from a recorded sample.

Held-out dataset mode answers each prompt `k` times and scores it against the reference (exact match, or token F1 of at least 0.8). Add `--judge-url` and `--judge-model` for an LLM judge as a second score.

```bash
uv run autotune benchmark dataset ./heldout/sft_pairs.jsonl \
  --base-url http://localhost:8000/v1 --base-model Qwen/Qwen2.5-7B-Instruct \
  --trained-url http://localhost:8001/v1 --trained-model my-model -k 3
```

Harbor mode runs each model as an agent on [Harbor](https://harborframework.com) tasks in Docker and grades it with each task's own verifier. Install Harbor first with `uv tool install harbor`.

```bash
uv run autotune benchmark harbor ./tasks \
  --base-url http://localhost:8000/v1 --base-model Qwen/Qwen2.5-7B-Instruct \
  --trained-url http://localhost:8001/v1 --trained-model my-model -k 3
```

Both write `benchmark_results/benchmark_report.json`. A model that stops answering, a judge that fails, or a Harbor run with no trial results stops the benchmark with an error instead of reporting partial numbers. Crashed Harbor trials count as failures and are reported as errors. Reasoning models need room to think, so raise `--max-tokens` (default 4096) if one runs out of tokens. API keys stay in the env vars you name with `--base-key-env`, `--trained-key-env`, and `--judge-key-env`.

### 7. Development Scripts
```bash
./run.sh    # Starts development environment (API and frontend)
./kill.sh   # Stops all Autotune background processes
```

---

## Detailed Capabilities

### Document Ingestion

Install `uv sync --extra documents` for PDF (`pypdf`), Word (`python-docx`), and HTML (`beautifulsoup4`) extraction. Upload `.pdf`, `.docx`, `.txt`, `.md`, or `.html`, or ZIP a folder of documents. Scanned PDFs need OCR before upload. Archives are limited to 1,000 files, 20 MiB per document, and 100 MiB total uncompressed size.

Set `AUTOTUNE_DOCUMENT_MODEL` to the exact model name on your generator server. `AUTOTUNE_DOCUMENT_ENDPOINT` defaults to `http://localhost:11434/v1` for local Ollama. For authenticated servers, set `AUTOTUNE_DOCUMENT_API_KEY_ENV` to the name of your API key variable. Unset keys and `mock_key` stop ingestion. Values load from the environment or `.env`.

Chunk settings are `AUTOTUNE_DOCUMENT_CHUNK_TOKENS=800`, `AUTOTUNE_DOCUMENT_OVERLAP_TOKENS=100`, and `AUTOTUNE_DOCUMENT_MAX_CHUNKS=200`. Counts use local lexical tokens (words and punctuation), independent of the generator's tokenizer. Pair bounds are `AUTOTUNE_DOCUMENT_MIN_PAIR_TOKENS=8` and `AUTOTUNE_DOCUMENT_MAX_PAIR_TOKENS=1024`. Exceeding the chunk cap stops ingestion instead of truncating input.

The generator produces Q&A and instruction/response pairs. Length bounds, normalized hashes, token-set similarity, lexical support, and an LLM grounding check filter them. These checks reduce unsupported answers but are not a factual guarantee. Each record retains its source filename and chunk ID. Splits keep documents separate. A single document uses chunk groups and reports a warning about overlapping text. Fewer than two surviving groups produces no held-out split and a warning.

The UI displays documents read, chunks made, pairs generated, and pairs kept. `POST /api/dataset/upload` retains its synchronous response. Add `?background=true` to receive a job ID, then poll `GET /api/dataset/status/{job_id}`. Generated splits persist beside the upload as `.cleaned.json`, with a source hash to prevent stale reuse. Training reuses those exact records. The `.heldout.jsonl` file works with the existing dataset benchmark. Generator or grounding failures fail the whole upload.

### Raw Preference Data

CSV, TSV, JSON, JSONL, and Parquet accept `prompt`, `chosen`, and `rejected`. Responses are strings or message lists containing assistant text. Both alternatives survive normalization and feed the DPO trainer. If reward scores are supplied, both must be finite and `chosen_reward` must exceed `rejected_reward`. Empty or identical alternatives are rejected. Task precedence is reward labels, preference pairs, tool messages, then chat, independent of row order. Mixed files select the highest-priority objective, but DPO training requires preference fields on every row and fails clearly otherwise.

### Model Catalog

`GET /api/models?vram=24` returns all 11 entries with parameter counts, licenses, gating, tool support, context limits, estimated VRAM, `fits`, and recommended quantization. Omitting `vram` uses detected hardware. The UI groups models by size and disables entries that exceed the estimate. Default selection chooses the largest fitting ungated model, restricting tool datasets to tool-capable entries. With no detected GPU, inspection still works and training reports its hardware requirement.

IDs were checked on the Hugging Face Hub on 2026-10-14: [Qwen 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen 14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [Llama 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [Llama 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Llama 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Gemma 9B](https://huggingface.co/google/gemma-2-9b-it), [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), and [Phi 4 Mini](https://huggingface.co/microsoft/Phi-4-mini-instruct). Context limits use the standard model configuration, without optional context extension. Licenses differ, and several models are open-weight rather than OSI open source.

Estimates in `core/domain/model_catalog.py` follow the memory components in [Hugging Face's memory guide](https://huggingface.co/docs/transformers/model_memory_anatomy). They are engineering budgets, not measured guarantees: frozen weights and scales, adapter state, checkpointed activations, and 4 GiB workspace. Baselines assume batch 1, 2048 tokens, and rank 16. A 14.7B model needs an estimated 19GB for QLoRA or 38GB for 16-bit LoRA. Gated entries need `HF_TOKEN` and license acceptance. The custom-model option resolves parameter count and context from Hub metadata without downloading weights and fails if reliable size metadata is unavailable.

### Basic and Advanced Training Settings

Basic shows model, objective, and intensity. Presets live in `core/domain/training_settings.py` and scale the data-driven recommendation:

| Intensity | Epoch multiplier (rounded up) | Learning-rate multiplier |
|---|---:|---:|
| Quick | 0.5 | 0.5 |
| Balanced | 1 | 1 |
| Thorough | 1.5 | 0.75 |

GRPO stays at or below `0.00002`. Advanced starts collapsed and exposes quantization, LoRA rank and alpha, learning rate, epochs, batch size, gradient accumulation, and sequence length. Changed values are marked against the upload recommendation. Reset restores that recommendation. Selecting a new model or objective refreshes its recommended settings from the server.

`POST /api/recipe/recommend` accepts `dataset_name` and optional `base_model` and `objective`, returning a recipe and all presets. `POST /api/training/start` accepts `intensity` (default `balanced`), `gradient_accumulation_steps`, and `max_seq_length` alongside its existing fields. Omitted settings retain auto values. Explicit numeric overrides take precedence over the intensity preset.

Preflight checks reject invalid numbers, sequences beyond the model context, and memory estimates above detected VRAM. Memory scales with parameter count, precision, batch size, sequence length, rank, and objective. Gradient accumulation adds steps rather than concurrent activations. Auto selection reduces batch size and increases accumulation to fit. Errors explain which settings to reduce. CPU-only inspection remains available, but the real trainer requires CUDA before downloading a model. These settings add no environment variables.

Training extras require TRL 0.18 or newer and Transformers 4.49 or newer. SFT and DPO pass the configured sequence length, batch size, accumulation, and epochs to their trainers. DPO uses conversational alternatives so TRL applies the model's chat template. A GRPO initialization error now stops the job instead of switching the objective to SFT.

---

## Integration with Forge

Need execution-verified training data? Use [Forge](https://github.com/mostofashakib/Forge) to synthesize sandboxed tasks, simulated personas, and multi-turn agent trajectories. Autotune natively ingests Forge export bundles (`sft_pairs.jsonl`, `preference_pairs.jsonl`, `grpo_rollouts.parquet`). The two never import or call each other. Forge data reaches Autotune as exported files or Harbor task directories.

---

## License

MIT. See [LICENSE](https://github.com/mostofashakib/VidQ/blob/main/LICENSE).

Built by [Mostofa Shakib](https://www.mostofashakib.com/).

