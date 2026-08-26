from __future__ import annotations

import model_urls


def test_classify_hf():
    assert model_urls.classify("https://huggingface.co/org/repo/resolve/main/m.safetensors") == "hf"
    assert model_urls.classify("https://cdn-lfs.huggingface.co/repo/file.bin") == "hf"


def test_classify_civitai_download_vs_page():
    assert model_urls.classify("https://civitai.com/api/download/models/12345") == "civitai_download"
    assert model_urls.classify("https://civitai.com/models/999?modelVersionId=1") == "civitai_page"


def test_classify_generic():
    assert model_urls.classify("https://github.com/x/y/releases/download/v1/m.pth") == "generic"


def test_hf_normalize_rewrites_blob_once():
    url = "https://huggingface.co/org/repo/blob/main/blob/m.safetensors"
    assert model_urls.hf_normalize(url) == (
        "https://huggingface.co/org/repo/resolve/main/blob/m.safetensors"
    )


def test_hf_normalize_leaves_resolve_untouched():
    url = "https://huggingface.co/org/repo/resolve/main/m.safetensors"
    assert model_urls.hf_normalize(url) == url


def test_civitai_version_id_from_api_path():
    assert model_urls.civitai_version_id("https://civitai.com/api/download/models/123") == "123"


def test_civitai_version_id_from_query():
    assert model_urls.civitai_version_id("https://civitai.com/models/9?modelVersionId=456") == "456"


def test_civitai_version_id_absent():
    assert model_urls.civitai_version_id("https://civitai.com/models/9") == ""


def test_civitai_model_id():
    assert model_urls.civitai_model_id("https://civitai.com/models/778/some-lora") == "778"
    assert model_urls.civitai_model_id("https://civitai.com/api/download/models/1") == ""


def test_filename_from_content_disposition_forms():
    assert model_urls.filename_from_content_disposition('attachment; filename="a b.safetensors"') == "a b.safetensors"
    assert model_urls.filename_from_content_disposition("attachment; filename=plain.pth") == "plain.pth"
    assert model_urls.filename_from_content_disposition(
        "attachment; filename*=UTF-8''my%20model.safetensors"
    ) == "my model.safetensors"
    assert model_urls.filename_from_content_disposition("") == ""


def test_filename_from_content_disposition_prefers_rfc5987():
    header = "attachment; filename=\"fallback.bin\"; filename*=UTF-8''real.safetensors"
    assert model_urls.filename_from_content_disposition(header) == "real.safetensors"


def test_filename_from_url_requires_extension():
    assert model_urls.filename_from_url("https://x.com/a/b/model.safetensors") == "model.safetensors"
    assert model_urls.filename_from_url("https://civitai.com/api/download/models/123") == ""
    assert model_urls.filename_from_url("https://x.com/a/b/") == ""


def test_filename_from_url_decodes_percent_escapes():
    assert model_urls.filename_from_url("https://x.com/a/my%20model.pth") == "my model.pth"


def test_sanitize_filename_strips_paths_and_query():
    assert model_urls.sanitize_filename("../../etc/passwd") == "passwd"
    assert model_urls.sanitize_filename("model.safetensors?download=true") == "model.safetensors"
    assert model_urls.sanitize_filename("dir\\sub\\model.pth") == "model.pth"
    assert model_urls.sanitize_filename('"quoted.pth"') == "quoted.pth"


def test_sanitize_filename_rejects_dot_names():
    assert model_urls.sanitize_filename("..") == ""
    assert model_urls.sanitize_filename(".") == ""
    assert model_urls.sanitize_filename("   ") == ""
