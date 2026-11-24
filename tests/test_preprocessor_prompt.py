"""
Unit tests for the configurable user prompt and the panel-count hint.
"""

from PIL import Image

from src.data.preprocessor import DEFAULT_USER_PROMPT, build_user_prompt, format_qwen_vl_conversation


def _user_text(messages):
    return messages[1]["content"][1]["text"]


def test_without_hint_returns_base():
    assert build_user_prompt("Extract.", None, "Has {n} panel(s).") == "Extract."
    assert build_user_prompt("Extract.", 0, "Has {n} panel(s).") == "Extract."
    assert build_user_prompt("Extract.", 2, None) == "Extract."


def test_hint_is_appended_with_the_panel_count():
    assert build_user_prompt("Extract.", 1, "Has {n} panel(s).") == "Extract. Has 1 panel(s)."


def test_user_prompt_reaches_the_message():
    messages = format_qwen_vl_conversation(Image.new("RGB", (8, 8)), user_prompt="X")
    assert _user_text(messages) == "X"


def test_default_user_prompt_is_unchanged():
    messages = format_qwen_vl_conversation(Image.new("RGB", (8, 8)))
    assert _user_text(messages) == DEFAULT_USER_PROMPT
    assert DEFAULT_USER_PROMPT.endswith("extract a separate table for each panel.")
