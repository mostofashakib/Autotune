# Autotune

**Autonomous AI Training & Benchmarking Engine — Plug in your data, get a frontier model out.**

Autotune automates the entire LLM post-training lifecycle. It takes raw datasets (CSV, JSONL, Parquet), auto-sanitizes them, automatically selects the optimal base model and LoRA/QLoRA configuration, trains the model, and validates improvements with side-by-side benchmark evals.

Pairs seamlessly with [Forge](https://github.com/mostofashakib/Forge) for execution-verified synthetic data generation.

---

## Why Autotune?

- **Zero-Friction Ingestion:** Drop in any raw dataset without converting schemas. Autotune detects conversational turns, prompts/responses, or tool-use trajectories automatically.
- **Hardware-Aware Dual Engine (LoRA & QLoRA):**
  - **QLoRA (4-bit NF4):** Uses `bitsandbytes` double quantization to fine-tune 7B–14B parameter models on 16GB–24GB consumer GPUs without CUDA Out-Of-Memory errors.
  - **LoRA (16-bit):** Automatically selected on datacenter GPUs (40GB+) for maximum training throughput.
- **Reinforcement Learning with GRPO:** Group Relative Policy Optimization support with configurable rollouts and group sizes for reasoning & math tasks.
- **High-Throughput Serving (vLLM):** One-command OpenAI-compatible model serving powered by vLLM KV-cache optimizations.
- **Cloud Scale with RunPod:** On-demand cloud GPU provisioning (RTX 4090, A100, H100) directly from the CLI.
- **Side-by-Side Proof of Improvement:** Never deploy a fine-tuned model blindly. Autotune compares the trained checkpoint against the base model on held-out tasks and reports pass@1, pass@3, and improvement percentages.
- **Senior Engineering Architecture:** Clean Architecture with Python `typing.Protocol` ports for dependency injection.

---

## Architecture

```
autotune/
├── core/
│   ├── domain/          # Pure entities: TrainingJob, CleanedDataset, Recipe, EvalReport, ServingEndpoint, RunPodConfig
│   ├── ports/           # Dependency Injection Protocols (TrainingBackend, ServingProvider, CloudComputeProvider)
│   └── services/        # Hardware-aware AutoRecipeSelector (LoRA vs QLoRA, GRPO)
├── infrastructure/
│   ├── backends/        # TrlPeftBackend (SFT, DPO, GRPO via TRL + PEFT) & MockTrainingBackend
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
Inspect an arbitrary dataset file to see detected tasks and recommended training recipes:
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
Launch a production-grade OpenAI-compatible vLLM serving server:
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
Prove the model improved over the baseline:
```bash
uv run autotune evaluate ./checkpoints/my-model --eval-dataset ./eval.jsonl
```

### 6. Running the Development Environment
```bash
./run.sh    # Starts development environment
./kill.sh   # Stops all Autotune background processes
```

---

## Integration with Forge

Need high-volume, execution-verified training data? Use [Forge](https://github.com/mostofashakib/Forge) to synthesize sandboxed tasks, simulated personas, and multi-turn agent trajectories. Autotune natively ingests Forge export bundles (`sft_pairs.jsonl`, `preference_pairs.jsonl`, `grpo_rollouts.parquet`).

