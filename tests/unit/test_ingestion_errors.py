"""Every unreadable upload raises IngestionError, and the API answers 422."""
import io

import pytest
from fastapi.testclient import TestClient


def _encrypted_pdf() -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("name, data, message", [
    ("broken.pdf", b"%PDF-1.4 garbage", "broken.pdf"),
    ("broken.docx", b"PK garbage", "broken.docx"),
    ("locked.pdf", _encrypted_pdf(), "encrypted"),
])
def test_unreadable_documents_raise_ingestion_error(name, data, message):
    from autotune.core.ports.sanitizer import IngestionError
    from autotune.infrastructure.sanitizers.document_text import extract_text
    with pytest.raises(IngestionError, match=message):
        extract_text(name, data)


@pytest.mark.parametrize("name, data", [
    ("empty.csv", b""),
    ("broken.parquet", b"PAR1 garbage"),
    ("ragged.csv", b'prompt,completion\n"unterminated,quote\n'),
])
def test_unreadable_structured_files_raise_ingestion_error(tmp_path, name, data):
    from autotune.core.ports.sanitizer import IngestionError
    from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(IngestionError, match=name):
        AutoFormatSanitizer().sanitize(path)


def test_ingestion_error_is_a_value_error():
    from autotune.core.ports.sanitizer import IngestionError
    assert issubclass(IngestionError, ValueError)


@pytest.mark.parametrize("name, data", [("broken.pdf", b"%PDF-1.4 garbage"), ("empty.csv", b"")])
def test_upload_of_unreadable_file_is_422(monkeypatch, name, data):
    from autotune.presentation.api import app as api
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "local")
    response = TestClient(api.create_app()).post("/api/dataset/upload", files={"file": (name, data)})
    assert response.status_code == 422, response.text
    assert name in response.json()["detail"]


def test_recommend_with_corrupt_document_is_422(monkeypatch, isolated_data_dir):
    from autotune.presentation.api import app as api
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "local")
    uploads = isolated_data_dir / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    response = TestClient(api.create_app()).post("/api/recipe/recommend", json={"dataset_name": "broken.pdf"})
    assert response.status_code == 422, response.text


def test_upload_without_document_extras_is_503_with_the_install_command(monkeypatch):
    import sys
    from autotune.presentation.api import app as api
    monkeypatch.setitem(sys.modules, "pypdf", None)
    response = TestClient(api.create_app()).post("/api/dataset/upload", files={"file": ("notes.md", b"policy")})
    assert response.status_code == 503, response.text
    assert "uv sync --extra documents" in response.json()["detail"]


def test_recommend_without_document_extras_is_503_with_the_install_command(monkeypatch, isolated_data_dir):
    import sys
    from autotune.presentation.api import app as api
    monkeypatch.setitem(sys.modules, "pypdf", None)
    uploads = isolated_data_dir / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "notes.md").write_text("policy")
    response = TestClient(api.create_app()).post("/api/recipe/recommend", json={"dataset_name": "notes.md"})
    assert response.status_code == 503, response.text
    assert "uv sync --extra documents" in response.json()["detail"]
