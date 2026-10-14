/** Shapes the Autotune API returns. Kept in step with src/autotune/presentation/api/app.py. */

export type SystemStatus = {
  vram_gb: number;
  cuda_available: boolean;
  device: string;
  recommended_quantization: string;
};

export type Recipe = {
  base_model: string;
  objective: "sft" | "dpo" | "grpo";
  quantization: "none" | "lora_16bit" | "qlora_4bit_nf4";
  lora_r: number;
  lora_alpha: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
};

export type UploadedDataset = {
  file_name: string;
  file_path: string;
  task_type: string;
  total_records: number;
  train_records: number;
  val_records: number;
  dropped_count: number;
  recipe: Recipe;
  presets: Record<"quick" | "balanced" | "thorough", Recipe>;
  warnings?: string[];
  preview: { prompt: string; completion: string }[];
};

export type TrainingSummary = {
  job_id: string;
  status: string;
  created_at?: string;
  dataset_name?: string;
  base_model?: string;
  objective?: string;
  output_dir?: string;
  progress_percent: number;
  current_step: number;
  total_steps: number | null;
  current_loss: number | null;
  checkpoint_dir: string | null;
  error: string | null;
};

export type TrainingRun = TrainingSummary & { logs: string[]; loss_history: number[] };

export type ModelEndpoint = { base_url: string; model: string; api_key_env?: string | null };

export type ModelScore = {
  samples: number;
  passed: number;
  errors: number;
  pass_at_1: number;
  pass_at_k: number;
  mean_score: number;
  mean_judge_score: number | null;
};

export type TaskComparison = {
  task_id: string;
  base: ModelScore;
  trained: ModelScore;
  delta_pass_at_1: number;
  delta_mean_score: number;
};

export type ComparisonReport = {
  mode: "dataset" | "harbor";
  k: number;
  base_endpoint: ModelEndpoint;
  trained_endpoint: ModelEndpoint;
  base: ModelScore;
  trained: ModelScore;
  tasks: TaskComparison[];
  delta_pass_at_1: number;
  delta_pass_at_k: number;
  delta_mean_score: number;
};

export type BenchmarkSummary = {
  job_id: string;
  status: string;
  created_at?: string;
  mode: "dataset" | "harbor";
  k?: number;
  path?: string;
  base_model?: string;
  trained_model?: string;
  error: string | null;
  headline: { base_pass_at_1: number; trained_pass_at_1: number; delta_pass_at_1: number } | null;
};

export type BenchmarkRun = Omit<BenchmarkSummary, "headline"> & { logs: string[]; report: ComparisonReport | null };

export type ServingState = { status: "stopped" | "starting" | "online" | "failed"; base_url: string | null; log_tail: string };

export type Gpu = { id: string; name: string; memory_gb: number; community_price: number };

export type Pod = {
  pod_id: string;
  name: string;
  status: string;
  gpu_name: string;
  cost_per_hr: number;
  ssh_ip: string | null;
  ssh_port: number | null;
  launched_at?: string;
  live_error: string | null;
};

export type CredentialStatus = { name: string; purpose: string; configured: boolean; env_file: string };

export type CatalogModel = {
  hf_id: string;
  display_name: string;
  parameters_b: number;
  license: string;
  gated: boolean;
  supports_tool_calling: boolean;
  context_length: number;
  qlora_min_vram_gb: number;
  lora_min_vram_gb: number;
  fits: boolean;
  quantization: Recipe["quantization"] | null;
  reason: string | null;
};
