from __future__ import annotations

import asyncio
import json
import os

import pytest

import auto_models


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point the wishlist state file at a temp dir for every test."""
    monkeypatch.setattr(auto_models, "STATE_FILE", str(tmp_path / "auto_models.json"))
    return tmp_path


def _boom(*_args, **_kwargs):
    raise AssertionError("remote call should not have happened")


async def _fake_status(models):
    return {"models": models}


def test_add_entries_rejects_bad_input():
    entries, rejected = auto_models.add_entries([
        {"url": "not-a-url", "folder": "loras"},
        {"url": "https://x.com/m.safetensors", "folder": "nonsense_folder"},
        {"url": "https://x.com/ok.safetensors", "folder": "loras"},
    ])
    assert len(entries) == 1
    assert entries[0]["url"] == "https://x.com/ok.safetensors"
    assert entries[0]["status"] == "pending"
    assert entries[0]["attempts"] == 0
    assert len(rejected) == 2


def test_add_entries_dedupes_by_url():
    auto_models.add_entries([{"url": "https://x.com/a.safetensors", "folder": "loras"}])
    entries, _ = auto_models.add_entries([{"url": "https://x.com/a.safetensors", "folder": "loras"}])
    assert len(entries) == 1


def test_add_entries_dedupes_across_folder_aliases():
    auto_models.add_entries([
        {"url": "https://x.com/one", "folder": "unet", "filename": "m.safetensors"},
    ])
    entries, _ = auto_models.add_entries([
        {"url": "https://x.com/two", "folder": "diffusion_models", "filename": "m.safetensors"},
    ])
    assert len(entries) == 1, "unet/ and diffusion_models/ are one pool"


def test_ensure_pending_makes_no_remote_calls_when_idle(isolated_state):
    summary = asyncio.run(auto_models.ensure_pending(_boom, _boom, str(isolated_state)))
    assert summary == {"checked": 0, "downloaded": 0, "cached": 0, "failed": 0}


def test_ensure_pending_skips_download_when_already_on_volume(isolated_state):
    auto_models.add_entries([
        {"url": "https://x.com/m", "folder": "unet", "filename": "m.safetensors"},
    ])

    async def status():
        # Stored under the canonical alias folder on the volume.
        return {"models": [{"folder": "diffusion_models", "name": "m.safetensors"}]}

    summary = asyncio.run(auto_models.ensure_pending(status, _boom, str(isolated_state)))
    assert summary["cached"] == 1
    assert summary["downloaded"] == 0
    stub = isolated_state / "models" / "unet" / "m.safetensors"
    assert stub.is_file() and stub.stat().st_size == 0
    assert auto_models.load_entries()[0]["status"] == "present"


def test_ensure_pending_writes_back_resolved_filename(isolated_state):
    auto_models.add_entries([
        {"url": "https://civitai.com/api/download/models/1", "folder": "loras"},
    ])
    calls = []

    async def status():
        return {"models": []}

    async def batch(items):
        calls.append(items)
        return [{"status": "ok", "skipped": False, "folder": "loras",
                 "filename": "resolved.safetensors", "path": "/x", "message": ""}]

    summary = asyncio.run(auto_models.ensure_pending(status, batch, str(isolated_state)))
    assert summary["downloaded"] == 1
    assert calls == [[{"url": "https://civitai.com/api/download/models/1",
                       "filename": "", "save_path": "loras"}]]
    entry = auto_models.load_entries()[0]
    assert entry["filename"] == "resolved.safetensors"
    assert entry["status"] == "present"
    assert (isolated_state / "models" / "loras" / "resolved.safetensors").is_file()


def test_ensure_pending_counts_skipped_as_cached(isolated_state):
    auto_models.add_entries([{"url": "https://x.com/m.pth", "folder": "vae"}])

    async def status():
        return {"models": []}

    async def batch(items):
        return [{"status": "ok", "skipped": True, "folder": "vae",
                 "filename": "m.pth", "path": "/x", "message": ""}]

    summary = asyncio.run(auto_models.ensure_pending(status, batch, str(isolated_state)))
    assert summary == {"checked": 1, "downloaded": 0, "cached": 1, "failed": 0}


def test_failed_entry_is_dropped_from_auto_path_then_retried_by_force(isolated_state):
    auto_models.add_entries([{"url": "https://x.com/bad.safetensors", "folder": "loras"}])

    async def status():
        return {"models": []}

    async def failing_batch(items):
        return [{"status": "error", "skipped": False, "folder": "loras",
                 "filename": "", "path": "", "message": "HTTP 404"}]

    for expected_attempts in (1, 2):
        summary = asyncio.run(auto_models.ensure_pending(status, failing_batch, str(isolated_state)))
        assert summary["failed"] == 1
        entry = auto_models.load_entries()[0]
        assert entry["status"] == "error"
        assert entry["error"] == "HTTP 404"
        assert entry["attempts"] == expected_attempts

    # Budget exhausted: the automatic path must not call Modal again.
    summary = asyncio.run(auto_models.ensure_pending(_boom, _boom, str(isolated_state)))
    assert summary["checked"] == 0

    # Manual sync overrides the budget.
    summary = asyncio.run(
        auto_models.ensure_pending(status, failing_batch, str(isolated_state), force=True)
    )
    assert summary["checked"] == 1
    assert summary["failed"] == 1


def test_ensure_pending_notifies(isolated_state):
    auto_models.add_entries([{"url": "https://x.com/m.pth", "folder": "vae"}])
    messages = []

    async def status():
        return {"models": []}

    async def batch(items):
        return [{"status": "ok", "skipped": False, "folder": "vae",
                 "filename": "m.pth", "path": "/x", "message": ""}]

    asyncio.run(auto_models.ensure_pending(
        status, batch, str(isolated_state), notify=messages.append
    ))
    assert any("checking" in m for m in messages)
    assert any("downloading" in m for m in messages)
    assert any("1 downloaded" in m for m in messages)


def test_create_stub_never_truncates_real_file(isolated_state):
    real = isolated_state / "models" / "loras" / "real.safetensors"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"weights" * 100)
    size_before = real.stat().st_size

    auto_models.create_stub(str(isolated_state), "loras", "real.safetensors")

    assert real.stat().st_size == size_before


def test_remove_entry():
    entries, _ = auto_models.add_entries([
        {"url": "https://x.com/a.pth", "folder": "vae"},
        {"url": "https://x.com/b.pth", "folder": "vae"},
    ])
    remaining = auto_models.remove_entry(entries[0]["id"])
    assert len(remaining) == 1
    assert remaining[0]["url"] == "https://x.com/b.pth"


def test_corrupt_state_file_is_quarantined(isolated_state):
    path = isolated_state / "auto_models.json"
    path.write_text("{not json")
    assert auto_models.load_entries() == []
    assert (isolated_state / "auto_models.json.corrupt").is_file()
    assert not path.exists()


def test_save_entries_is_atomic_json(isolated_state):
    auto_models.save_entries([{"id": "x", "url": "u", "folder": "vae", "filename": "f"}])
    data = json.loads((isolated_state / "auto_models.json").read_text())
    assert data["entries"][0]["id"] == "x"
    assert not os.path.exists(str(isolated_state / "auto_models.json.tmp"))
