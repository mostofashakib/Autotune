"""The API describes its responses, so the frontend types can be generated from it."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.routing import APIRoute

from autotune.presentation.api.app import create_app

def _api_routes(app: FastAPI) -> list[APIRoute]:
    """Every /api route, looking inside the routers the app includes."""
    found = []
    for route in app.routes:
        nested = route.original_router.routes if hasattr(route, "original_router") else [route]
        found += [r for r in nested if isinstance(r, APIRoute) and r.path.startswith("/api")]
    return found


def test_every_route_declares_a_response_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOTUNE_DATA_DIR", str(tmp_path))
    routes = _api_routes(create_app())
    assert routes
    undeclared = [f"{sorted(r.methods)} {r.path}" for r in routes if r.response_model is None]
    assert undeclared == []


def test_openapi_describes_the_dashboard_types(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOTUNE_DATA_DIR", str(tmp_path))
    schemas = create_app().openapi()["components"]["schemas"]
    for name in ("SystemStatus", "UploadedDataset", "TrainingSummary", "TrainingRun", "BenchmarkSummary",
                 "BenchmarkRun", "Pod", "CredentialStatus", "CatalogModel"):
        assert name in schemas
