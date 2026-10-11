"""Command-line interface for Autotune."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from autotune.core.domain.models import JobStatus, QuantizationMethod, RunPodConfig, TrainingJob, TrainingObjective
from autotune.core.services.recipe_selector import AutoRecipeSelector
from autotune.infrastructure.backends.mock_backend import MockTrainingBackend
from autotune.infrastructure.backends.trl_peft_backend import TrlPeftBackend
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider
from autotune.infrastructure.evaluators.side_by_side import SideBySideEvaluator
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


@app.command()
def evaluate(
    checkpoint_dir: Path = typer.Argument(..., help="Path to trained checkpoint directory"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Base model for comparison"),
    eval_dataset: Optional[Path] = typer.Option(None, "--eval-dataset", "-d", help="Path to held-out eval dataset"),
) -> None:
    """Evaluate trained checkpoint side-by-side against base model."""
    evaluator = SideBySideEvaluator()

    # Read base model from metadata if not explicitly provided
    resolved_base = base_model
    if not resolved_base:
        meta_path = checkpoint_dir / "autotune_meta.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                resolved_base = data.get("base_model")
            except Exception:
                pass
    if not resolved_base:
        resolved_base = "meta-llama/Llama-3.1-8B-Instruct"

    console.print(f"[bold cyan]Running benchmark comparison for {checkpoint_dir}...[/]")
    try:
        report = evaluator.evaluate(
            base_model=resolved_base,
            checkpoint_dir=checkpoint_dir,
            eval_dataset_path=eval_dataset,
        )
    except Exception as exc:
        console.print(f"[red]Evaluation error:[/] {exc}")
        raise typer.Exit(code=1)

    table = Table(title="Benchmark Evaluation Results", show_header=True, header_style="bold blue")
    table.add_column("Task Probe")
    table.add_column("Base Score")
    table.add_column("Trained Score")
    table.add_column("Improvement")

    for m in report.metrics:
        delta_color = "green" if m.delta > 0 else "red"
        table.add_row(
            m.task_name,
            f"{m.base_score:.2f}",
            f"{m.trained_score:.2f}",
            f"[{delta_color}]{m.delta:+.2f}[/]",
        )
    console.print(table)
    console.print(f"[bold green]{report.summary}[/]")


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


if __name__ == "__main__":
    app()

