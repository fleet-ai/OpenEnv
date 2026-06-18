"""Tests for `_openai_block_to_gemini_part` in fleet trace upload.

The Fleet /v1/traces/logs server pydantic-binds each `parts[*]` as a Gemini
`Part` (`text`, `inline_data: {data, mime_type}`, ...). The earlier shape
`{"type":"image","mime_type":...,"data":...}` validated successfully because
all `Part` fields are Optional, but `part.inline_data` was None — so images
were silently dropped before the server-side S3 upload that renders them in
the UI. This module pins the corrected shape.

Source contract: theseus/orchestrator/private_api/traces.py
  - TracePart (line ~152): only `text`, `thought`, `inline_data`,
    `function_call`, `function_response` are read.
  - _extract_content_from_parts (line ~431): pulls `part.inline_data.data`
    and `part.inline_data.mime_type`.
"""

from __future__ import annotations

from envs.fleet_env.trace import _openai_block_to_gemini_part


class TestImageBlockToGeminiPart:
    def test_data_url_png_becomes_inline_data(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAA"},
        }
        part = _openai_block_to_gemini_part(block)
        # Server reads part.inline_data — that key MUST be present.
        assert "inline_data" in part
        assert part["inline_data"]["mime_type"] == "image/png"
        assert part["inline_data"]["data"] == "iVBORw0KGgoAAA"
        # The legacy top-level keys MUST NOT remain; they're red herrings the
        # server ignores.
        assert "type" not in part
        assert "data" not in part
        assert "mime_type" not in part

    def test_data_url_jpeg_preserves_mime_type(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
        }
        part = _openai_block_to_gemini_part(block)
        assert part["inline_data"]["mime_type"] == "image/jpeg"
        assert part["inline_data"]["data"] == "/9j/4AAQ"

    def test_remote_https_url_falls_back_to_text(self):
        # Server has no fetcher; remote URLs become text so the trace still
        # carries the reference rather than disappearing.
        block = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
        part = _openai_block_to_gemini_part(block)
        assert "inline_data" not in part
        assert "text" in part
        assert "https://x/y.png" in part["text"]


class TestTextBlock:
    def test_text_block_becomes_bare_text_part(self):
        block = {"type": "text", "text": "[Turn 5/64]"}
        part = _openai_block_to_gemini_part(block)
        assert part == {"text": "[Turn 5/64]"}
        # The earlier shape `{"type":"text","text":"..."}` worked because the
        # server reads `part.text` regardless. We drop the redundant `type`
        # key to keep payloads consistent with the image part shape.
        assert "type" not in part


class TestUnknownBlock:
    def test_unknown_block_flattens_to_text_for_durability(self):
        block = {"type": "video_url", "video_url": {"url": "..."}}
        part = _openai_block_to_gemini_part(block)
        # Server would ignore this entirely; better to surface it as text so
        # the trace still tells us what was attempted.
        assert "text" in part
