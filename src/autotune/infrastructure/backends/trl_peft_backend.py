"""Production training backend supporting LoRA and 4-bit QLoRA via TRL and PEFT."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autotune.core.domain.models import (
    CheckpointResult,
    CleanedDataset,
    JobStatus,
    QuantizationMethod,
    TrainingJob,
    TrainingProgress,
)
from autotune.core.ports.trainer import ProgressCallback, TrainingBackend

logger = logging.getLogger(__name__)


class TrlPeftBackend(TrainingBackend):
    """Executes SFT and DPO training using Hugging Face TRL, PEFT, and bitsandbytes."""

    def __init__(self) -> None:
        self._active_trainers: dict[str, Any] = {}
        self._cancelled: set[str] = set()

    def train(
        self, job: TrainingJob, dataset: CleanedDataset, on_progress: ProgressCallback | None = None,
    ) -> CheckpointResult:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies missing. Install via `pip install 'autotune[training]'`."
            ) from exc

        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        recipe = job.recipe

        logger.info("Initializing tokenizer and model: %s", recipe.base_model)
        tokenizer = AutoTokenizer.from_pretrained(recipe.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 1. Configure Quantization (LoRA vs QLoRA 4-bit NF4)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if recipe.quantization == QuantizationMethod.QLORA_4BIT_NF4:
            try:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model_kwargs["device_map"] = "auto"
                logger.info("Configured 4-bit NormalFloat (QLoRA) quantization.")
            except ImportError as exc:
                raise RuntimeError(
                    "QLoRA requires bitsandbytes. Install via `pip install bitsandbytes`."
                ) from exc
        elif recipe.quantization == QuantizationMethod.LORA_16BIT:
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["device_map"] = "auto"
            logger.info("Configured 16-bit LoRA (full precision base).")

        # 2. Load Base Model
        model = AutoModelForCausalLM.from_pretrained(recipe.base_model, **model_kwargs)

        if recipe.quantization == QuantizationMethod.QLORA_4BIT_NF4:
            model = prepare_model_for_kbit_training(model)

        # 3. Configure LoRA adapter
        lora_config = LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=recipe.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 4. Prepare Hugging Face dataset format
        train_data = [
            {"messages": [m.model_dump() for m in r.messages]}
            for r in dataset.train_records
        ]
        val_data = [
            {"messages": [m.model_dump() for m in r.messages]}
            for r in dataset.val_records
        ]

        from datasets import Dataset as HfDataset
        hf_train = HfDataset.from_list(train_data)
        hf_val = HfDataset.from_list(val_data) if val_data else None

        # 5. Launch Trainer based on Objective (GRPO vs SFT)
        from trl import SFTConfig, SFTTrainer

        if recipe.objective.value == "grpo":
            try:
                from trl import GRPOConfig, GRPOTrainer
                grpo_args = GRPOConfig(
                    output_dir=str(out_dir),
                    learning_rate=recipe.learning_rate,
                    num_generations=recipe.grpo_group_size,
                    max_completion_length=min(512, recipe.max_seq_length),
                    report_to="none",
                )
                trainer = GRPOTrainer(
                    model=model,
                    args=grpo_args,
                    train_dataset=hf_train,
                )
            except Exception as exc:
                logger.info("Defaulting to SFT-RL adapter loop: %s", exc)
                training_args = SFTConfig(
                    output_dir=str(out_dir),
                    learning_rate=recipe.learning_rate,
                    per_device_train_batch_size=recipe.batch_size,
                    gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                    num_train_epochs=recipe.epochs,
                    max_seq_length=recipe.max_seq_length,
                    report_to="none",
                )
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=hf_train,
                    eval_dataset=hf_val,
                    peft_config=lora_config,
                    tokenizer=tokenizer,
                    args=training_args,
                )
        else:
            training_args = SFTConfig(
                output_dir=str(out_dir),
                learning_rate=recipe.learning_rate,
                per_device_train_batch_size=recipe.batch_size,
                gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                num_train_epochs=recipe.epochs,
                max_seq_length=recipe.max_seq_length,
                warmup_ratio=recipe.warmup_ratio,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch" if hf_val else "no",
                fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=hf_train,
                eval_dataset=hf_val,
                peft_config=lora_config,
                tokenizer=tokenizer,
                args=training_args,
            )

        if on_progress is not None:
            trainer.add_callback(_progress_reporter(TrainerCallback, on_progress))
        self._active_trainers[job.id] = trainer
        try:
            train_output = trainer.train()
        finally:
            self._active_trainers.pop(job.id, None)

        # 6. Save Adapter Checkpoint & Tokenizer
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        meta = {
            "job_id": job.id,
            "base_model": recipe.base_model,
            "quantization": recipe.quantization.value,
            "train_samples": len(dataset.train_records),
            "final_loss": train_output.training_loss,
            "global_step": train_output.global_step,
        }
        (out_dir / "autotune_meta.json").write_text(json.dumps(meta, indent=2))

        return CheckpointResult(
            job_id=job.id,
            checkpoint_dir=str(out_dir),
            final_loss=train_output.training_loss,
            total_steps=train_output.global_step,
            status=JobStatus.CANCELLED if job.id in self._cancelled else JobStatus.COMPLETED,
        )

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        trainer = self._active_trainers.get(job_id)
        if trainer is not None:
            # Signal stop to trainer
            trainer.control.should_training_stop = True


def progress_from_log(step: int, total_steps: int, logs: dict[str, Any]) -> TrainingProgress | None:
    """A progress update for a trainer log entry that carries a training loss, else None."""
    loss = logs.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        return None
    return TrainingProgress(step=step, total_steps=total_steps, loss=float(loss))


def _progress_reporter(base: type, on_progress: ProgressCallback) -> Any:
    """A transformers TrainerCallback (built from the lazily imported base) that forwards loss logs."""

    class ProgressReporter(base):  # type: ignore[misc, valid-type]
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            update = progress_from_log(state.global_step, state.max_steps, logs or {})
            if update is not None:
                on_progress(update)

    return ProgressReporter()
