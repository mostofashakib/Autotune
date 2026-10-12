# Autotune

**LLM Post Training & Benchmarking Engine — Plug in your data, get a frontier model out**

Autotune handles the entire post-training workflow for large language models. Feed it raw CSV, JSONL, or Parquet datasets. Autotune cleans the inputs, selects the right base model and tuning recipe, runs training, and verifies quality against baseline models with automated benchmarks.

Works directly with [Forge](https://github.com/mostofashakib/Forge) for execution-verified synthetic datasets.

---

## Why Autotune?

- **Automatic Ingestion:** Drop in raw CSV, JSONL, or Parquet files without converting schemas. Autotune identifies conversation turns, prompt and response pairs, or tool calls automatically.
- **Hardware-Aware LoRA and QLoRA:**
  - **QLoRA (4-bit NF4):** Uses bitsandbytes 4-bit NormalFloat to tune 7B to 14B models on 16GB to 24GB GPUs without memory errors.
  - **LoRA (16-bit):** Selected on GPUs with 40GB or more VRAM for faster training.
- **Reinforcement Learning with GRPO:** Train reasoning models with Group Relative Policy Optimization, customizable rollouts, and group rewards.
- **vLLM Serving:** Serve models with an OpenAI-compatible endpoint in one command.
- **Cloud GPUs on RunPod:** Rent on-demand RTX 4090, A100, and H100 GPUs directly from your terminal.
- **Side-by-Side Benchmark:** Measure your trained model against its base on held-out data or on Harbor agent tasks, and get per-task pass@1 and pass@k from real samples.

---

## Architecture

```
autotune/
├── core/
│   ├── domain/          # Entities: Recipe, RolloutRecord, PreferenceRecord, ServingEndpoint, RunPodConfig
│   ├── ports/           # Interface protocols (TrainingBackend, ServingProvider, CloudComputeProvider, Storage)
│   └── services/        # Hardware-aware AutoRecipeSelector (LoRA vs QLoRA, GRPO)
├── infrastructure/
│   ├── backends/        # TrlPeftBackend and reward mapping (GRPO, DPO, SFT)
│   ├── sanitizers/      # AutoFormatSanitizer and dataset loaders (Parquet, JSONL, CSV)
│   ├── storage/         # PolicyCheckpoint serialization and manifest management
│   ├── serving/         # VLLMServingProvider (OpenAI-compatible inference server)
│   ├── cloud/           # RunPodCloudProvider (GraphQL remote GPU lifecycle)
│   └── evaluators/      # Dataset and Harbor benchmarks, model client, scoring, judge, metrics (pass@k)
└── presentation/
    └── cli/             # Typer CLI commands (inspect, train, serve, benchmark, cloud)
```

---

## Quickstart

### 1. Interactive Web Cockpit
Launch the full web dashboard in your browser:
```bash
./run.sh
# or
uv run autotune ui --port 7860
```
Open `http://localhost:7860` to access:
- **Training Cockpit**: Drag-and-drop datasets, automatic recipe configuration, and real-time loss telemetry.
- **RunPod Cloud**: Browse GPU availability, compare hourly pricing, and deploy cloud instances.
- **Side-by-Side Playground**: Test prompts simultaneously on base and fine-tuned models with benchmark metrics.
- **vLLM Serving**: One-click inference serving with an OpenAI-compatible API endpoint.

### 2. Inspect a Dataset
Inspect a dataset file to see detected task types and recommended tuning settings:
```bash
uv run autotune inspect my_data.jsonl
```

### 2. Train a Model (SFT or GRPO)
Run post-training with automated model selection and LoRA/QLoRA tuning:
```bash
uv run autotune train my_data.jsonl --output-dir ./checkpoints/my-model
```

To run GRPO reinforcement learning:
```bash
uv run autotune train my_rl_data.parquet --objective grpo --output-dir ./checkpoints/my-rl-model
```

### 3. Serve with vLLM
Launch an OpenAI-compatible vLLM server:
```bash
uv run autotune serve ./checkpoints/my-model --port 8000
```

### 4. Cloud GPUs on RunPod
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

### 5. Benchmark Against the Base Model
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

### 6. Development Scripts
```bash
./run.sh    # Starts development environment
./kill.sh   # Stops all Autotune background processes
```

---

## Integration with Forge

Need execution-verified training data? Use [Forge](https://github.com/mostofashakib/Forge) to synthesize sandboxed tasks, simulated personas, and multi-turn agent trajectories. Autotune natively ingests Forge export bundles (`sft_pairs.jsonl`, `preference_pairs.jsonl`, `grpo_rollouts.parquet`). The two never import or call each other. Forge data reaches Autotune as exported files or Harbor task directories.
