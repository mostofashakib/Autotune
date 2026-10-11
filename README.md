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
- **Side-by-Side Evaluation:** Compare your checkpoint directly against the base model on held-out data to verify pass@1, pass@3, and win rates before deployment.

---

## Architecture

```
autotune/
├── core/
│   ├── domain/          # Pure entities: TrainingJob, CleanedDataset, Recipe, EvalReport, ServingEndpoint, RunPodConfig
│   ├── ports/           # Interface protocols (TrainingBackend, ServingProvider, CloudComputeProvider)
│   └── services/        # Hardware-aware AutoRecipeSelector (LoRA vs QLoRA, GRPO)
├── infrastructure/
│   ├── backends/        # TrlPeftBackend (SFT, DPO, GRPO via TRL + PEFT) and MockTrainingBackend
│   ├── sanitizers/      # AutoFormatSanitizer (CSV, JSONL, Parquet)
│   ├── serving/         # VLLMServingProvider (OpenAI-compatible inference server)
│   ├── cloud/           # RunPodCloudProvider (GraphQL remote GPU lifecycle)
│   └── evaluators/      # SideBySideEvaluator
└── presentation/
    └── cli/             # Typer CLI commands (inspect, train, serve, evaluate, cloud)
```

---

## Quickstart

### 1. Inspect a Dataset
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

To run a fast simulation locally without GPUs:
```bash
uv run autotune train my_data.jsonl --output-dir ./checkpoints/my-model --mock
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

### 5. Evaluate Results
Check how your model performs against the baseline:
```bash
uv run autotune evaluate ./checkpoints/my-model --eval-dataset ./eval.jsonl
```

### 6. Development Scripts
```bash
./run.sh    # Starts development environment
./kill.sh   # Stops all Autotune background processes
```

---

## Integration with Forge

Need execution-verified training data? Use [Forge](https://github.com/mostofashakib/Forge) to synthesize sandboxed tasks, simulated personas, and multi-turn agent trajectories. Autotune natively ingests Forge export bundles (`sft_pairs.jsonl`, `preference_pairs.jsonl`, `grpo_rollouts.parquet`).
