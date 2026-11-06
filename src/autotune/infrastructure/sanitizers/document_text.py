"""Extract plain text from one document's bytes."""
from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.sanitizers.text_decoding import decode_text, html_charset

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
INSTALL = "Document ingestion requires extras. Run: uv sync --extra documents"


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        from pypdf import PdfReader
        from docx import Document
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(INSTALL) from exc
    return PdfReader, Document, BeautifulSoup


def extract_text(name: str, data: bytes) -> str:
    """Extract text only. Scanned PDFs need OCR before upload."""
    PdfReader, Document, BeautifulSoup = _dependencies()
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        text = _parsed(name, lambda: _pdf_text(name, PdfReader(io.BytesIO(data))))
    elif suffix == ".docx":
        text = _parsed(name, lambda: _docx_text(Document(io.BytesIO(data))))
    elif suffix in (".html", ".htm"):
        soup = BeautifulSoup(decode_text(data, html_charset(data)), "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(" ", strip=True)
    elif suffix == ".md":
        text = strip_markdown(decode_text(data))
    else:
        text = decode_text(data)
    text = unicodedata.normalize("NFKC", text).replace("\x00", "").strip()
    if not text:
        raise IngestionError(f"{name} has no extractable text. Apply OCR to scanned documents first.")
    return text


def _parsed(name: str, parse: Callable[[], str]) -> str:
    """Run a third-party parser, reporting any failure as an IngestionError that names the file."""
    try:
        return parse()
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not read {name}. The file is corrupt or not a valid document ({exc}).") from exc


def _pdf_text(name: str, reader: Any) -> str:
    if reader.is_encrypted and not reader.decrypt(""):
        raise IngestionError(f"{name} is encrypted. Remove the password and upload it again.")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _docx_text(doc: Any) -> str:
    """Paragraphs and table rows in the order they appear in the document."""
    from docx.table import Table
    lines = []
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            lines += [" | ".join(cell.text for cell in row.cells) for row in block.rows]
        else:
            lines.append(block.text)
    return "\n".join(lines)


_MARKDOWN_RULES = [
    (re.compile(r"^[ \t]*(```|~~~).*\n?", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$\n?", re.MULTILINE), ""),
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"`([^`\n]+)`"), r"\1"),
    (re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"), r"\2"),
    (re.compile(r"(?<!\w)([*_])(?=\S)(.+?)(?<=\S)\1(?!\w)"), r"\2"),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def strip_markdown(text: str) -> str:
    """Drop Markdown syntax and keep its words, including the contents of code blocks."""
    for pattern, replacement in _MARKDOWN_RULES:
        text = pattern.sub(replacement, text)
    return text
