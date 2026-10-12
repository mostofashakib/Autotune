"""Command-line interface for Autotune."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from autotune.core.domain.models import (
    ComparisonReport, JobStatus, ModelEndpoint, QuantizationMethod, RunPodConfig, TrainingJob, TrainingObjective,
)
from autotune.core.services.recipe_selector import AutoRecipeSelector
from autotune.infrastructure.backends.mock_backend import MockTrainingBackend
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider
from autotune.infrastructure.evaluators.comparison import run_comparison, write_report
from autotune.infrastructure.evaluators.dataset_benchmark import DatasetBenchmark
from autotune.infrastructure.evaluators.harbor_benchmark import HarborBenchmark, HarborError
from autotune.infrastructure.evaluators.judge import JudgeError, LLMJudge
from autotune.infrastructure.evaluators.openai_client import ModelCallError, OpenAIChatClient
from autotune.infrastructure.sanitizers.dataset_loader import MalformedExportError, load_sft
from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
from autotune.infrastructure.serving.vllm_provider import VLLMServingProvider

app = typer.Typer(
    name="autotune",
    help="LLM Post Training & Benchmarking Engine — Plug in your data, get a frontier model out",
    add_completion=False,
)
cloud_app = typer.Typer(
    name="cloud",
    help="Manage remote cloud GPU instances on RunPod for large-scale training.",
    add_completion=False,
)
app.add_typer(cloud_app, name="cloud")
benchmark_app = typer.Typer(
    name="benchmark",
    help="Measure a trained model against its base on held-out data or Harbor tasks.",
    add_completion=False,
)
app.add_typer(benchmark_app, name="benchmark")
console = Console()


@app.command()
def inspect(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file (.csv, .jsonl, .parquet)"),
) -> None:
    """Inspect dataset and display auto-detected training recipe."""
    sanitizer = AutoFormatSanitizer()
    selector = AutoRecipeSelector()

    try:
        dataset = sanitizer.sanitize(dataset_path)
    except Exception as exc:
        console.print(f"[red]Error reading dataset:[/] {exc}")
        raise typer.Exit(code=1)

    recipe = selector.select_recipe(dataset)

    # Dataset Summary Table
    ds_table = Table(title="Dataset Analysis", show_header=True, header_style="bold magenta")
    ds_table.add_column("Property", style="dim")
    ds_table.add_column("Value")
    ds_table.add_row("Detected Task Type", dataset.task_type.value)
    ds_table.add_row("Total Valid Records", str(dataset.total_records))
    ds_table.add_row("Training Split", str(len(dataset.train_records)))
    ds_table.add_row("Validation Split", str(len(dataset.val_records)))
    ds_table.add_row("Dropped / Corrupt", str(dataset.dropped_count))
    console.print(ds_table)

    # Auto-Recipe Table
    r_table = Table(title="Auto-Selected Training Recipe", show_header=True, header_style="bold green")
    r_table.add_column("Parameter", style="dim")
    r_table.add_column("Recommendation")
    r_table.add_row("Base Model", recipe.base_model)
    r_table.add_row("Objective", recipe.objective.value)
    r_table.add_row("Quantization Method", recipe.quantization.value)
    r_table.add_row("LoRA Rank (r)", str(recipe.lora_r))
    r_table.add_row("LoRA Alpha", str(recipe.lora_alpha))
    r_table.add_row("Learning Rate", f"{recipe.learning_rate:.2e}")
    r_table.add_row("Batch Size", str(recipe.batch_size))
    r_table.add_row("Epochs", str(recipe.epochs))
    if recipe.objective == TrainingObjective.GRPO:
        r_table.add_row("GRPO Group Size", str(recipe.grpo_group_size))
        r_table.add_row("GRPO Epsilon", str(recipe.grpo_epsilon))
    console.print(r_table)


@app.command()
def train(
    dataset_path: Path = typer.Argument(..., help="Path to training dataset"),
    output_dir: Path = typer.Option("./checkpoints/autotune-model", "--output-dir", "-o", help="Where to save model weights"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Override default base model"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Override: 'lora_16bit' or 'qlora_4bit_nf4'"),
    objective: Optional[str] = typer.Option(None, "--objective", help="Override: 'sft', 'dpo', or 'grpo'"),
    mock: bool = typer.Option(False, "--mock", help="Run simulated training without GPUs"),
) -> None:
    """Ingest data, auto-tune parameters, and train model."""
    sanitizer = AutoFormatSanitizer()
    selector = AutoRecipeSelector()

    console.print(f"[bold cyan]Autotune:[/] Analyzing dataset {dataset_path}...")
    try:
        dataset = sanitizer.sanitize(dataset_path)
    except Exception as exc:
        console.print(f"[red]Error parsing dataset:[/] {exc}")
        raise typer.Exit(code=1)

    quant_enum = QuantizationMethod(quantization) if quantization else None
    obj_enum = TrainingObjective(objective.lower()) if objective else None
    recipe = selector.select_recipe(
        dataset,
        forced_model=base_model,
        forced_quantization=quant_enum,
        forced_objective=obj_enum,
    )

    console.print(f"[green]✓[/] Selected model: [bold]{recipe.base_model}[/]")
    console.print(f"[green]✓[/] Objective: [bold]{recipe.objective.value.upper()}[/]")
    console.print(f"[green]✓[/] Quantization: [bold]{recipe.quantization.value}[/]")

    job = TrainingJob(
        id=f"job_{dataset_path.stem}",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        recipe=recipe,
    )

    backend = MockTrainingBackend() if mock else TrlPeftBackend()
    console.print(f"[bold yellow]Starting training loop into {output_dir}...[/]")
    try:
        result = backend.train(job, dataset)
    except Exception as exc:
        console.print(f"[red]Training error:[/] {exc}")
        raise typer.Exit(code=1)

    if result.status == JobStatus.COMPLETED:
        console.print(f"[bold green]✓ Training completed successfully![/]")
        console.print(f"Checkpoint saved to: [underline]{result.checkpoint_dir}[/]")
    else:
        console.print(f"[yellow]Training ended with status:[/] {result.status.value}")


@app.command()
def serve(
    model_path: Path = typer.Argument(..., help="Path to checkpoint directory or Hugging Face model id"),
    name: str = typer.Option("", "--name", "-n", help="Alias for served model"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    gpu_memory_utilization: float = typer.Option(0.85, "--gpu-memory-utilization", help="GPU memory fraction for vLLM"),
) -> None:
    """Serve trained model with high-throughput vLLM OpenAI-compatible engine."""
    provider = VLLMServingProvider()
    console.print(f"[bold cyan]Launching vLLM serving engine for {model_path} on port {port}...[/]")
    try:
        endpoint = provider.start(
            model_path=model_path,
            model_name=name,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        console.print(f"[bold green]✓ Server spawned at:[/] [underline]{endpoint.base_url}[/]")
        console.print("API Base URL: [bold]http://127.0.0.1:{port}/v1[/]")
    except Exception as exc:
        console.print(f"[red]Failed to start vLLM serving engine:[/] {exc}")
        raise typer.Exit(code=1)


BENCHMARK_FAILURES = (ModelCallError, JudgeError, HarborError, MalformedExportError, ValueError)

BaseUrl = typer.Option(..., "--base-url", help="OpenAI-compatible URL serving the base model")
BaseModel = typer.Option(..., "--base-model", help="Base model name on that server")
BaseKeyEnv = typer.Option(None, "--base-key-env", help="Env var holding the base server's API key")
TrainedUrl = typer.Option(..., "--trained-url", help="OpenAI-compatible URL serving the trained model")
TrainedModel = typer.Option(..., "--trained-model", help="Trained model name on that server")
TrainedKeyEnv = typer.Option(None, "--trained-key-env", help="Env var holding the trained server's API key")
Samples = typer.Option(3, "--k", "-k", min=1, help="Samples per task (pass@k)")
Output = typer.Option(Path("benchmark_results"), "--output", "-o", help="Directory for benchmark_report.json")


@benchmark_app.command("dataset")
def benchmark_dataset(
    data: Path = typer.Argument(..., help="Held-out export: sft_pairs.jsonl, messages JSONL, or a directory"),
    base_url: str = BaseUrl, base_model: str = BaseModel, base_key_env: Optional[str] = BaseKeyEnv,
    trained_url: str = TrainedUrl, trained_model: str = TrainedModel, trained_key_env: Optional[str] = TrainedKeyEnv,
    k: int = Samples, output: Path = Output,
    judge_url: Optional[str] = typer.Option(None, "--judge-url", help="OpenAI-compatible URL for an optional LLM judge"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Judge model name"),
    judge_key_env: Optional[str] = typer.Option(None, "--judge-key-env", help="Env var holding the judge's API key"),
    max_tokens: int = typer.Option(4096, "--max-tokens", min=1, help="Answer token budget. Reasoning models need room to think"),
) -> None:
    """Answer each held-out prompt k times and score it against the reference."""
    if bool(judge_url) != bool(judge_model):
        console.print("[red]--judge-url and --judge-model go together.[/]")
        raise typer.Exit(code=1)
    client = OpenAIChatClient()
    judge = LLMJudge(client, ModelEndpoint(base_url=judge_url, model=judge_model, api_key_env=judge_key_env)) if judge_url else None
    _run_benchmark(
        "dataset", lambda: DatasetBenchmark(load_sft(data), client, k, judge=judge, max_tokens=max_tokens), k, output,
        ModelEndpoint(base_url=base_url, model=base_model, api_key_env=base_key_env),
        ModelEndpoint(base_url=trained_url, model=trained_model, api_key_env=trained_key_env),
    )


@benchmark_app.command("harbor")
def benchmark_harbor(
    tasks: Path = typer.Argument(..., help="A Harbor task directory, or a directory of tasks"),
    base_url: str = BaseUrl, base_model: str = BaseModel, base_key_env: Optional[str] = BaseKeyEnv,
    trained_url: str = TrainedUrl, trained_model: str = TrainedModel, trained_key_env: Optional[str] = TrainedKeyEnv,
    k: int = Samples, output: Path = Output,
    jobs_dir: Path = typer.Option(Path("benchmark_results/harbor_jobs"), "--jobs-dir", help="Where Harbor writes its trials"),
    agent: str = typer.Option("terminus-2", "--agent", help="Harbor agent that drives the model"),
) -> None:
    """Run each model as an agent on Harbor tasks and grade it with each task's verifier."""
    _run_benchmark(
        "harbor", lambda: HarborBenchmark(tasks, k, jobs_dir, agent=agent), k, output,
        ModelEndpoint(base_url=base_url, model=base_model, api_key_env=base_key_env),
        ModelEndpoint(base_url=trained_url, model=trained_model, api_key_env=trained_key_env),
    )


def _run_benchmark(mode, make_runner, k: int, output: Path, base: ModelEndpoint, trained: ModelEndpoint) -> None:
    console.print(f"[bold cyan]Benchmarking {base.model} against {trained.model} ({mode}, k={k})...[/]")
    try:
        report = run_comparison(mode, make_runner(), k, base, trained)
    except BENCHMARK_FAILURES as exc:
        console.print(f"[red]Benchmark failed:[/] {exc}")
        raise typer.Exit(code=1)
    path = write_report(report, output)
    _print_report(report)
    console.print(f"Report written to {path}")


def _print_report(report: ComparisonReport) -> None:
    table = Table(title=f"Benchmark ({report.mode}, k={report.k})", show_header=True, header_style="bold blue")
    for column in ("Task", "Base pass@1", "Trained pass@1", "Change"):
        table.add_column(column)
    for task in report.tasks:
        color = "green" if task.delta_pass_at_1 > 0 else "red" if task.delta_pass_at_1 < 0 else "white"
        table.add_row(task.task_id, f"{task.base.pass_at_1:.2f}", f"{task.trained.pass_at_1:.2f}",
                      f"[{color}]{task.delta_pass_at_1:+.2f}[/]")
    console.print(table)
    for label, model in (("Base", report.base), ("Trained", report.trained)):
        judge = f", judge {model.mean_judge_score:.2f}" if model.mean_judge_score is not None else ""
        console.print(
            f"{label}: pass@1 {model.pass_at_1:.2f}, pass@{report.k} {model.pass_at_k:.2f}, "
            f"mean score {model.mean_score:.2f}{judge}, {model.errors} errored of {model.samples} samples"
        )
    console.print(f"[bold]Change: pass@1 {report.delta_pass_at_1:+.2f}, pass@{report.k} {report.delta_pass_at_k:+.2f}[/]")


@cloud_app.command("list-gpus")
def cloud_list_gpus(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="RunPod API Key"),
) -> None:
    """List available GPU models and hourly pricing on RunPod."""
    provider = RunPodCloudProvider(api_key=api_key)
    try:
        gpus = provider.list_available_gpus()
        table = Table(title="RunPod GPU Availability & Pricing", show_header=True, header_style="bold cyan")
        table.add_column("GPU Model ID")
        table.add_column("Display Name")
        table.add_column("VRAM (GB)")
        table.add_column("Community Price ($/hr)")
        for g in gpus:
            table.add_row(
                str(g.get("id")),
                str(g.get("name")),
                str(g.get("memory_gb")),
                f"${g.get('community_price', 0):.2f}",
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Error fetching GPU types:[/] {exc}")
        raise typer.Exit(code=1)


@cloud_app.command("launch")
def cloud_launch(
    gpu_type: str = typer.Option("NVIDIA GeForce RTX 4090", "--gpu", "-g", help="RunPod GPU Type ID"),
    pod_name: str = typer.Option("autotune-pod", "--name", "-n", help="Name of the pod"),
    disk_gb: int = typer.Option(50, "--disk", help="Disk size in GB"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="RunPod API Key"),
) -> None:
    """Launch a remote GPU pod on RunPod for training."""
    provider = RunPodCloudProvider(api_key=api_key)
    config = RunPodConfig(
        gpu_type_id=gpu_type,
        pod_name=pod_name,
        container_disk_in_gb=disk_gb,
        volume_in_gb=disk_gb,
    )
    console.print(f"[bold cyan]Provisioning remote {gpu_type} pod on RunPod...[/]")
    try:
        instance = provider.create_instance(config)
        console.print(f"[bold green]✓ Pod created successfully![/]")
        console.print(f"Pod ID: [bold]{instance.pod_id}[/]")
        console.print(f"Status: {instance.status}")
    except Exception as exc:
        console.print(f"[red]Error provisioning pod:[/] {exc}")
        raise typer.Exit(code=1)


@cloud_app.command("status")
def cloud_status(
    pod_id: str = typer.Argument(..., help="RunPod Pod ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="RunPod API Key"),
) -> None:
    """Check status and connection info for a remote RunPod instance."""
    provider = RunPodCloudProvider(api_key=api_key)
    try:
        inst = provider.get_instance_status(pod_id)
        console.print(f"Pod ID: [bold]{inst.pod_id}[/]")
        console.print(f"Name: {inst.name}")
        console.print(f"Status: {inst.status}")
        console.print(f"GPU: {inst.gpu_name}")
        console.print(f"Cost: ${inst.cost_per_hr:.2f}/hr")
        if inst.ssh_ip:
            console.print(f"SSH Target: ssh root@{inst.ssh_ip} -p {inst.ssh_port}")
    except Exception as exc:
        console.print(f"[red]Error fetching status:[/] {exc}")
        raise typer.Exit(code=1)


@cloud_app.command("stop")
def cloud_stop(
    pod_id: str = typer.Argument(..., help="RunPod Pod ID to terminate"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="RunPod API Key"),
) -> None:
    """Terminate and deallocate a remote RunPod instance."""
    provider = RunPodCloudProvider(api_key=api_key)
    try:
        stopped = provider.stop_instance(pod_id)
        if stopped:
            console.print(f"[bold green]✓ Pod {pod_id} successfully terminated.[/]")
        else:
            console.print(f"[yellow]Could not terminate pod {pod_id}.[/]")
    except Exception as exc:
        console.print(f"[red]Error terminating pod:[/] {exc}")
        raise typer.Exit(code=1)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to bind dashboard server"),
) -> None:
    """Launch the Autotune interactive Web Cockpit UI."""
    import uvicorn
    from autotune.presentation.api.app import create_app

    console.print(f"[bold cyan]Launching Autotune Web Cockpit on http://{host}:{port}...[/]")
    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

