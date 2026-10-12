import json
import pytest
from autotune.core.domain.models import RunPodConfig
from autotune.infrastructure.cloud.runpod_provider import RunPodCloudProvider


@pytest.fixture
def no_runpod_key(tmp_path, monkeypatch):
    from autotune import settings
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)


def _never_called(*args, **kwargs):
    pytest.fail("RunPod must not be called without a configured key")


@pytest.mark.parametrize("call", [
    lambda p: p.create_instance(RunPodConfig(gpu_type_id="RTX 4090")),
    lambda p: p.get_instance_status("pod_1"),
    lambda p: p.stop_instance("pod_1"),
    lambda p: p.list_available_gpus(),
])
def test_every_call_needs_a_configured_key_and_names_it(no_runpod_key, call):
    from autotune.settings import CredentialNotConfigured

    with pytest.raises(CredentialNotConfigured, match="RUNPOD_API_KEY"):
        call(RunPodCloudProvider(request_fn=_never_called))


def test_the_placeholder_key_is_not_a_configured_key(no_runpod_key):
    from autotune.settings import PLACEHOLDER, CredentialNotConfigured

    with pytest.raises(CredentialNotConfigured):
        RunPodCloudProvider(api_key=PLACEHOLDER, request_fn=_never_called).list_available_gpus()


def test_the_key_comes_from_the_central_settings(no_runpod_key, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-live-123")
    seen = []

    def graphql(method, url, headers, body):
        seen.append(headers["Authorization"])
        return 200, json.dumps({"data": {"gpuTypes": []}}).encode()

    RunPodCloudProvider(request_fn=graphql).list_available_gpus()

    assert seen == ["Bearer rp-live-123"]


def test_create_instance_success():
    def mock_graphql(method: str, url: str, headers: dict, body: bytes | None):
        assert headers.get("Authorization") == "Bearer test_secret_key"
        payload = {
            "data": {
                "podFindAndDeployOnDemand": {
                    "id": "pod_abc123",
                    "name": "autotune-trainer",
                    "desiredStatus": "RUNNING",
                    "imageName": "runpod/pytorch:2.2.0",
                }
            }
        }
        return 200, json.dumps(payload).encode("utf-8")

    provider = RunPodCloudProvider(api_key="test_secret_key", request_fn=mock_graphql)
    config = RunPodConfig(gpu_type_id="NVIDIA GeForce RTX 4090", pod_name="autotune-trainer")
    instance = provider.create_instance(config)

    assert instance.pod_id == "pod_abc123"
    assert instance.status == "RUNNING"
    assert instance.name == "autotune-trainer"
    assert instance.gpu_name == "NVIDIA GeForce RTX 4090"


def test_get_instance_status():
    def mock_graphql(method: str, url: str, headers: dict, body: bytes | None):
        payload = {
            "data": {
                "pod": {
                    "id": "pod_abc123",
                    "name": "autotune-trainer",
                    "desiredStatus": "RUNNING",
                    "costPerHr": 0.44,
                    "runtime": {
                        "uptimeInSeconds": 300,
                        "ports": [
                            {"ip": "192.168.1.100", "isIpPublic": True, "privatePort": 22, "publicPort": 22022}
                        ],
                    },
                    "machine": {
                        "gpuDisplayName": "NVIDIA RTX 4090",
                    },
                }
            }
        }
        return 200, json.dumps(payload).encode("utf-8")

    provider = RunPodCloudProvider(api_key="test_secret_key", request_fn=mock_graphql)
    instance = provider.get_instance_status("pod_abc123")

    assert instance.pod_id == "pod_abc123"
    assert instance.status == "RUNNING"
    assert instance.cost_per_hr == 0.44
    assert instance.ssh_ip == "192.168.1.100"
    assert instance.ssh_port == 22022


def test_stop_instance():
    def mock_graphql(method: str, url: str, headers: dict, body: bytes | None):
        payload = {"data": {"podTerminate": True}}
        return 200, json.dumps(payload).encode("utf-8")

    provider = RunPodCloudProvider(api_key="test_secret_key", request_fn=mock_graphql)
    assert provider.stop_instance("pod_abc123") is True


def test_list_available_gpus():
    def mock_graphql(method: str, url: str, headers: dict, body: bytes | None):
        payload = {
            "data": {
                "gpuTypes": [
                    {"id": "RTX 4090", "displayName": "GeForce RTX 4090", "memoryInGb": 24, "communityPrice": 0.44},
                    {"id": "A100-80GB", "displayName": "A100 SXM4 80GB", "memoryInGb": 80, "communityPrice": 1.89},
                ]
            }
        }
        return 200, json.dumps(payload).encode("utf-8")

    provider = RunPodCloudProvider(api_key="test_secret_key", request_fn=mock_graphql)
    gpus = provider.list_available_gpus()
    assert len(gpus) == 2
    assert gpus[0]["id"] == "RTX 4090"
    assert gpus[1]["community_price"] == 1.89
