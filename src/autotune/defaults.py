"""Defaults the CLI and the API share, so both start from the same values."""
CHECKPOINT_DIR = "./checkpoints/autotune-model"
BENCHMARK_DIR = "benchmark_results"
HARBOR_JOBS_DIR = "benchmark_results/harbor_jobs"
SERVE_PORT = 8000
ANSWER_MAX_TOKENS = 4096
PASS_K = 3
RUNPOD_GPU = "NVIDIA GeForce RTX 4090"
RUNPOD_POD_NAME = "autotune-pod"
RUNPOD_DISK_GB = 50
