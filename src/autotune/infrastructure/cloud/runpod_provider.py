"""RunPod Cloud Compute Provider adapter implementation."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

from autotune.core.domain.models import RunPodConfig, RunPodInstance
from autotune.core.ports.cloud import CloudComputeProvider
from autotune.settings import RUNPOD_API_KEY, require_credential

logger = logging.getLogger(__name__)

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


class RunPodCloudProvider(CloudComputeProvider):
    """Orchestration adapter for RunPod on-demand GPU instances via GraphQL."""

    def __init__(
        self,
        api_key: str | None = None,
        request_fn: Callable[..., Any] | None = None,
    ) -> None:
        """
        Initialize RunPod provider.

        Args:
            api_key: RunPod API key. If omitted, RUNPOD_API_KEY comes from autotune.settings.
            request_fn: Optional injected HTTP client function for testability.
        """
        self._key_override = api_key
        self._request_fn = request_fn or self._default_graphql_request

    def create_instance(self, config: RunPodConfig) -> RunPodInstance:
        """
        Provision an on-demand GPU pod on RunPod.

        Args:
            config: Configuration specifying GPU type, container disk, and volume.

        Returns:
            Provisioned RunPodInstance.
        """
        key = self._key(config.api_key)

        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                machine {
                    podHostId
                }
            }
        }
        """
        variables = {
            "input": {
                "cloudType": config.cloud_type,
                "gpuCount": 1,
                "volumeInGb": config.volume_in_gb,
                "containerDiskInGb": config.container_disk_in_gb,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuTypeId": config.gpu_type_id,
                "name": config.pod_name,
                "imageName": config.image_name,
            }
        }

        data = self._execute_graphql(mutation, variables, api_key=key)
        pod_data = data.get("data", {}).get("podFindAndDeployOnDemand")
        if not pod_data:
            errors = data.get("errors", [])
            raise RuntimeError(f"RunPod instance creation failed: {errors}")

        return RunPodInstance(
            pod_id=pod_data["id"],
            name=pod_data.get("name", config.pod_name),
            status=pod_data.get("desiredStatus", "RUNNING"),
            gpu_name=config.gpu_type_id,
        )

    def get_instance_status(self, pod_id: str) -> RunPodInstance:
        """
        Fetch the current status of an existing RunPod instance.

        Args:
            pod_id: RunPod instance identifier.

        Returns:
            RunPodInstance with current status and connection info.
        """
        key = self._key()

        query = """
        query GetPod($podId: String!) {
            pod(input: { podId: $podId }) {
                id
                name
                desiredStatus
                lastStatusChange
                costPerHr
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
                machine {
                    gpuDisplayName
                }
            }
        }
        """
        variables = {"podId": pod_id}
        data = self._execute_graphql(query, variables, api_key=key)
        pod_data = data.get("data", {}).get("pod")
        if not pod_data:
            errors = data.get("errors", [])
            raise RuntimeError(f"Could not retrieve pod '{pod_id}': {errors}")

        ssh_ip: str | None = None
        ssh_port: int | None = None
        runtime = pod_data.get("runtime") or {}
        for port_info in runtime.get("ports", []):
            if port_info.get("privatePort") == 22:
                ssh_ip = port_info.get("ip")
                ssh_port = port_info.get("publicPort")
                break

        machine = pod_data.get("machine") or {}
        gpu_name = machine.get("gpuDisplayName", "Unknown GPU")

        return RunPodInstance(
            pod_id=pod_data["id"],
            name=pod_data.get("name", "autotune-pod"),
            status=pod_data.get("desiredStatus", "UNKNOWN"),
            gpu_name=gpu_name,
            cost_per_hr=float(pod_data.get("costPerHr") or 0.0),
            ssh_ip=ssh_ip,
            ssh_port=ssh_port,
        )

    def stop_instance(self, pod_id: str) -> bool:
        """
        Terminate and deallocate a running RunPod instance.

        Args:
            pod_id: RunPod instance identifier.

        Returns:
            True if successfully stopped, False otherwise.
        """
        key = self._key()

        mutation = """
        mutation TerminatePod($podId: String!) {
            podTerminate(input: { podId: $podId })
        }
        """
        variables = {"podId": pod_id}
        try:
            data = self._execute_graphql(mutation, variables, api_key=key)
            return "errors" not in data
        except Exception as e:
            logger.error("Failed to terminate RunPod instance %s: %s", pod_id, e)
            return False

    def list_available_gpus(self) -> list[dict[str, str | float]]:
        """
        Retrieve available GPU models and their approximate pricing.

        Returns:
            List of GPU specs with id, display name, and memory.
        """
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
            }
        }
        """
        data = self._execute_graphql(query, {}, api_key=self._key())
        gpus = data.get("data", {}).get("gpuTypes", [])
        return [
            {
                "id": g.get("id", ""),
                "name": g.get("displayName", ""),
                "memory_gb": g.get("memoryInGb", 0),
                "community_price": g.get("communityPrice", 0.0),
            }
            for g in gpus
        ]

    def _key(self, override: str | None = None) -> str:
        """The configured RunPod key. Raises CredentialNotConfigured when there is none."""
        return require_credential(RUNPOD_API_KEY, override or self._key_override)

    def _execute_graphql(self, query: str, variables: dict[str, Any], api_key: str) -> dict[str, Any]:
        """Execute a GraphQL request against RunPod API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"query": query, "variables": variables}
        body_bytes = json.dumps(payload).encode("utf-8")
        status_code, resp_bytes = self._request_fn(
            "POST",
            RUNPOD_GRAPHQL_URL,
            headers=headers,
            body=body_bytes,
        )
        if status_code != 200:
            raise RuntimeError(f"RunPod API returned status {status_code}: {resp_bytes.decode('utf-8', errors='ignore')}")
        payload = json.loads(resp_bytes.decode("utf-8"))
        if payload.get("errors"):
            raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
        return payload

    @staticmethod
    def _default_graphql_request(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float = 15.0,
    ) -> tuple[int, bytes]:
        """Default HTTP requester using urllib."""
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            raise ConnectionError(f"HTTP request to {url} failed: {e}") from e
