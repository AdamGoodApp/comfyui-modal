"""URL classification and filename helpers shared by comfyapp.py and tests.

Pure functions only - no modal/httpx imports, so tests can import this module
directly under conftest's stubbed environment.
"""

import os
import re
from urllib.parse import urlparse, parse_qs, unquote


def classify(url: str) -> str:
    """Return one of: "hf", "civitai_download", "civitai_page", "generic"."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        return "hf"
    if host == "civitai.com" or host.endswith(".civitai.com"):
        if parsed.path.startswith("/api/download/"):
            return "civitai_download"
        return "civitai_page"
    return "generic"


def hf_normalize(url: str) -> str:
    """Rewrite HF /blob/ page URLs to /resolve/ so they download raw bytes."""
    if "/blob/" in urlparse(url).path:
        return url.replace("/blob/", "/resolve/", 1)
    return url


def civitai_version_id(url: str) -> str:
    """Model-version id from /api/download/models/<id> or ?modelVersionId=. "" if absent."""
    parsed = urlparse(url)
    m = re.match(r"^/api/download/models/(\d+)", parsed.path)
    if m:
        return m.group(1)
    vals = parse_qs(parsed.query).get("modelVersionId", [])
    if vals and vals[0].isdigit():
        return vals[0]
    return ""


def civitai_model_id(url: str) -> str:
    """Model id from a civitai /models/<id>/... page URL, "" if absent."""
    m = re.match(r"^/models/(\d+)", urlparse(url).path)
    return m.group(1) if m else ""


def filename_from_content_disposition(header: str) -> str:
    """Parse filename from a Content-Disposition header. Prefers RFC 5987 filename*."""
    if not header:
        return ""
    m = re.search(r"filename\*=(?:UTF-8''|utf-8'')([^;]+)", header)
    if m:
        return sanitize_filename(unquote(m.group(1).strip().strip('"')))
    m = re.search(r'filename="([^"]+)"', header)
    if m:
        return sanitize_filename(m.group(1))
    m = re.search(r"filename=([^;]+)", header)
    if m:
        return sanitize_filename(m.group(1).strip())
    return ""


def filename_from_url(url: str) -> str:
    """Best-effort filename from URL path basename; "" when it has no extension."""
    path = urlparse(url).path
    name = unquote(path.rstrip("/").rsplit("/", 1)[-1]) if path else ""
    return sanitize_filename(name) if "." in name else ""


def sanitize_filename(name: str) -> str:
    """Strip directories/query junk; return "" for anything not a plain file name."""
    name = name.strip().strip('"').split("?")[0].split("#")[0]
    name = name.replace("\\", "/")
    name = os.path.basename(name)
    if name in ("", ".", ".."):
        return ""
    return name
