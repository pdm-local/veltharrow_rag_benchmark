from veltharrow.config import (
    DOCUMENTS,
    DOCUMENTS_DIR,
    MEDIUM_MODEL_SPECS,
    MCQ_DIR,
    SMALL_MODEL_SPECS,
    TIER_ALIASES,
    model_specs_for,
    required_models_for,
    resolve_tier,
    results_dir_for,
    retrieval_conditions,
)


EXPECTED_SMALL_MODELS = {
    "qwen3:0.6b",
    "qwen3.5:0.8b",
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    "gemma3:1b",
    "gemma4:e2b",
    "qwen3.5:2b",
    "llama3.2:3b",
    "phi4-mini-reasoning",
    "gemma3:4b",
}


def test_documents_resolve_under_data_dir():
    assert DOCUMENTS_DIR.name == "documents"
    assert MCQ_DIR.name == "mcq"
    for spec in DOCUMENTS:
        assert spec.source_path.exists(), spec.source_path
        assert spec.questions_path.exists(), spec.questions_path


EXPECTED_MEDIUM_MODELS = {
    "qwen3:8b",
    "qwen3.5:9b",
    "gemma4:e4b",
    "gemma4:12b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "deepseek-r1:7b",
}


def test_tiers_are_the_canonical_two():
    assert set(TIER_ALIASES.values()) == {"small", "medium"}


SMALL_THINKING_OFF_ONLY = {"phi4-mini-reasoning"}
SMALL_THINKING_ON_ONLY  = {"llama3.2:3b", "gemma3:4b"}


def test_small_tier_matches_roster_with_both_thinking_states():
    got = {s.model for s in SMALL_MODEL_SPECS}
    assert got == EXPECTED_SMALL_MODELS, got ^ EXPECTED_SMALL_MODELS
    for tag in EXPECTED_SMALL_MODELS:
        states = {s.thinking for s in SMALL_MODEL_SPECS if s.model == tag}
        if tag in SMALL_THINKING_OFF_ONLY:
            expected_states = {False}
        elif tag in SMALL_THINKING_ON_ONLY:
            expected_states = {True}
        else:
            expected_states = {True, False}
        assert states == expected_states, (tag, states)
    assert len(SMALL_MODEL_SPECS) == 21


# Medium-tier models that ship with thinking ON and OFF; the rest are OFF-only
# unless listed in MEDIUM_THINKING_ON_ONLY.
MEDIUM_BOTH_THINKING = {"qwen3:8b", "qwen3.5:9b", "gemma4:e4b", "gemma4:12b"}
MEDIUM_THINKING_ON_ONLY = {"deepseek-r1:7b"}


def test_medium_tier_has_expected_models():
    got = {s.model for s in MEDIUM_MODEL_SPECS}
    assert got == EXPECTED_MEDIUM_MODELS, got ^ EXPECTED_MEDIUM_MODELS
    # mistral:7b is intentionally excluded from the standard set; deepseek-r1:7b
    # was re-added (thinking-on only).
    assert "mistral:7b" not in got
    for tag in EXPECTED_MEDIUM_MODELS:
        states = {s.thinking for s in MEDIUM_MODEL_SPECS if s.model == tag}
        if tag in MEDIUM_BOTH_THINKING:
            expected = {True, False}
        elif tag in MEDIUM_THINKING_ON_ONLY:
            expected = {True}
        else:
            expected = {False}
        assert states == expected, (tag, states)


def test_every_tier_alias_resolves_and_is_non_empty():
    for alias in TIER_ALIASES:
        canonical = resolve_tier(alias)
        assert canonical in {"small", "medium"}
        assert model_specs_for(alias), f"no model specs for {alias}"
        assert required_models_for(alias), f"no required models for {alias}"
        assert retrieval_conditions(alias), f"no conditions for {alias}"
        assert results_dir_for(alias).name == canonical
