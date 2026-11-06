"""CudaProbe reports free CUDA memory and degrades to 0.0 without a usable device."""
import sys
from types import SimpleNamespace

from autotune.core.ports.hardware import HardwareProbe
from autotune.infrastructure.hardware.cuda_probe import CudaProbe


def fake_torch(*, available=True, free_gb=8):
    cuda = SimpleNamespace(is_available=lambda: available, current_device=lambda: 0,
                           mem_get_info=lambda *_: (free_gb * 1024**3, 24 * 1024**3))
    return SimpleNamespace(cuda=cuda)


def test_probe_satisfies_the_port():
    assert isinstance(CudaProbe(), HardwareProbe)


def test_missing_torch_reports_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert CudaProbe().available_vram_gb() == 0.0


def test_reports_free_cuda_memory_in_gib(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(free_gb=8))
    assert CudaProbe().available_vram_gb() == 8.0


def test_no_cuda_device_reports_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False))
    assert CudaProbe().available_vram_gb() == 0.0


def test_mps_is_not_reported_as_a_cuda_training_gpu(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert CudaProbe().available_vram_gb() == 0.0


def test_cuda_runtime_error_reports_zero(monkeypatch):
    def fail(*_):
        raise RuntimeError("CUDA driver error")
    torch = fake_torch()
    torch.cuda.mem_get_info = fail
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert CudaProbe().available_vram_gb() == 0.0
