"""Persistent model wishlist: entries auto-download to the Modal volume and
get 0-byte local stubs so this ComfyUI's dropdowns keep listing them.

State lives in auto_models.json next to this file (gitignored). Functions are
called from the aiohttp event loop; ensure_pending serializes itself.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_NODE_DIR, "auto_models.json")

# Folders offered by the sidebar plus the legacy aliases ComfyUI still scans.
ALLOWED_FOLDERS = {
    "checkpoints", "diffusion_models", "unet", "loras", "vae", "controlnet",
    "upscale_models", "embeddings", "clip", "text_encoders", "model_patches",
    "clip_vision", "style_models", "vae_approx", "hypernetworks", "gligen",
    "photomaker", "latent_upscale_models", "audio_encoders", "frame_interpolation",
}

# Folder pairs ComfyUI treats as one pool: presence checks use the canonical
# name so a "unet" entry matches a file stored under diffusion_models/.
_CANON = {"unet": "diffusion_models", "clip": "text_encoders"}

MAX_AUTO_ATTEMPTS = 2  # failing entries stop consuming queue-time Modal calls

_lock = asyncio.Lock()


def canon(folder: str) -> str:
    return _CANON.get(folder, folder)


def load_entries() -> list:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("entries", [])
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        print(f"[comfyui-modal] auto_models.json unreadable ({e}); starting empty")
        try:
            os.replace(STATE_FILE, STATE_FILE + ".corrupt")
        except OSError:
            pass
        return []


def save_entries(entries: list) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)
    os.replace(tmp, STATE_FILE)


def add_entries(items: list) -> tuple:
    """items: [{"url", "folder", "filename"?}] -> (all entries, rejected reasons).

    Silently skips exact duplicates (same url, or same canonical folder+filename).
    """
    entries = load_entries()
    have_names = {(canon(e["folder"]), e["filename"]) for e in entries if e.get("filename")}
    have_urls = {e["url"] for e in entries}
    rejected = []
    for it in items:
        url = (it.get("url") or "").strip()
        folder = (it.get("folder") or "").strip()
        filename = (it.get("filename") or "").strip()
        if not url.startswith(("http://", "https://")):
            rejected.append(f"not a URL: {url[:80]}")
            continue
        if folder not in ALLOWED_FOLDERS:
            rejected.append(f"unknown folder: {folder or '(empty)'}")
            continue
        if url in have_urls or (filename and (canon(folder), filename) in have_names):
            continue
        entries.append({
            "id": uuid.uuid4().hex,
            "url": url,
            "folder": folder,
            "filename": filename,
            "status": "pending",
            "error": "",
            "attempts": 0,
            "added_at": int(time.time()),
        })
        have_urls.add(url)
        if filename:
            have_names.add((canon(folder), filename))
    save_entries(entries)
    return entries, rejected


def remove_entry(entry_id: str) -> list:
    entries = [e for e in load_entries() if e["id"] != entry_id]
    save_entries(entries)
    return entries


def create_stub(comfyui_root: str, folder: str, filename: str) -> None:
    """0-byte marker so local dropdowns list the model. Never touches real files."""
    path = Path(comfyui_root) / "models" / folder / filename
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _volume_index(status: dict) -> set:
    return {(canon(m["folder"]), m["name"]) for m in status.get("models", [])}


async def ensure_pending(get_status_fn, batch_download_fn, comfyui_root: str,
                         notify=None, force: bool = False) -> dict:
    """Bring wishlist entries onto the Modal volume; stub confirmed ones locally.

    get_status_fn: modal_client.get_sync_status   () -> {"models": [...]}
    batch_download_fn: modal_client.batch_download_models   (items) -> [dict,...]
    Zero Modal calls when nothing is actionable. Per-item failures are recorded
    on the entry; only Modal-level errors propagate to the caller.
    """
    def _notify(msg):
        if notify:
            notify(msg)

    async with _lock:
        entries = load_entries()
        candidates = [e for e in entries
                      if e["status"] != "present"
                      and (force or e["attempts"] < MAX_AUTO_ATTEMPTS)]
        if not candidates:
            return {"checked": 0, "downloaded": 0, "cached": 0, "failed": 0}

        _notify(f"Auto-models: checking {len(candidates)} model(s) on Modal volume...")
        have = _volume_index(await get_status_fn())

        cached = downloaded = failed = 0
        to_download = []
        for e in candidates:
            if e["filename"] and (canon(e["folder"]), e["filename"]) in have:
                e["status"], e["error"] = "present", ""
                create_stub(comfyui_root, e["folder"], e["filename"])
                cached += 1
            else:
                to_download.append(e)

        if to_download:
            names = ", ".join(e["filename"] or e["url"].rsplit("/", 1)[-1] for e in to_download)
            _notify(f"Auto-models: downloading {len(to_download)} model(s): {names}")
            items = [{"url": e["url"], "filename": e["filename"], "save_path": e["folder"]}
                     for e in to_download]
            results = await batch_download_fn(items)
            for e, res in zip(to_download, results):
                if isinstance(res, dict) and res.get("status") == "ok":
                    e["filename"] = res.get("filename") or e["filename"]
                    e["status"], e["error"] = "present", ""
                    create_stub(comfyui_root, e["folder"], e["filename"])
                    if res.get("skipped"):
                        cached += 1
                    else:
                        downloaded += 1
                else:
                    msg = res.get("message", "download failed") if isinstance(res, dict) else str(res)
                    e["status"], e["error"] = "error", msg
                    e["attempts"] += 1
                    failed += 1

        save_entries(entries)
        _notify(f"Auto-models: {downloaded} downloaded, {cached} cached, {failed} failed")
        return {"checked": len(candidates), "downloaded": downloaded,
                "cached": cached, "failed": failed}
