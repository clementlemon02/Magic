"""The offline backend switch. It must be opt-in, and it must announce itself."""

import pytest

from src.config import get_settings
from src.llm.fake import FakeChatModel, UnknownPrompt
from src.llm.factory import get_chat_model


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    get_chat_model.cache_clear()
    yield
    get_settings.cache_clear()
    get_chat_model.cache_clear()


def test_fake_backend_is_opt_in_and_warns(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "fake")
    with pytest.warns(RuntimeWarning, match="canned"):
        model = get_chat_model()
    assert isinstance(model, FakeChatModel)


def test_default_backend_is_hunyuan(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert get_settings().llm_backend == "hunyuan"


def test_unrecognised_prompt_raises_rather_than_guessing():
    """A reworded PROMPT_TEMPLATE should break the fake loudly, not silently mis-answer."""
    with pytest.raises(UnknownPrompt):
        FakeChatModel().invoke("some prompt this fake has never seen")
