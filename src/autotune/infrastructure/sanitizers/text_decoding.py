"""Decode uploaded text bytes without assuming UTF-8."""
from __future__ import annotations

import codecs
import re

_BOMS = ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"))
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w.:-]+)""", re.IGNORECASE)


def decode_text(data: bytes, declared: str | None = None) -> str:
    """Honor a byte order mark, then a declared charset, then UTF-8, then Latin-1."""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding)
    for encoding in filter(None, (declared, "utf-8")):
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("latin-1")


def html_charset(data: bytes) -> str | None:
    """The charset an HTML document declares in its first 4 KiB, if any."""
    match = _META_CHARSET.search(data[:4096])
    return match.group(1).decode("ascii") if match else None
