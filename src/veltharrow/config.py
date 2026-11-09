"""Single source of truth for the benchmark's static configuration.

Everything the pipeline keys off lives here: filesystem paths, the Ollama endpoint,
chunking and decoding parameters, the document set, and the per-tier model rosters and
retrieval conditions. Other modules import these constants rather than redefining them,
so changing a roster or a path happens in exactly one place.

The identifiers defined here (model ``label``s, tier names, the ``retrieval_condition``
strings built by :func:`retrieval_conditions`) are **load-bearing**: they are written
into the result CSVs and figure paths, so renaming them silently orphans existing
results. Add to them; don't rename them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = ROOT / "data" / "documents"
MCQ_DIR = ROOT / "data" / "mcq"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

# Ollama endpoint. Override by exporting OLLAMA_HOST (e.g. for a remote host or
# a non-standard port); otherwise the standard local install on 11434 is used.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_FALLBACK_HOSTS: tuple[str, ...] = ()

CHUNK_WORDS = 450
CHUNK_OVERLAP_WORDS = 80
EMBED_BATCH_SIZE = 24

GENERATION_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    # Matches the thinking-mode budget. 256 was enough for a model that answers with a
    # bare letter, but some reasoning families emit an inline <think> block even when
    # ``think=False`` and were cut off before reaching an answer — phi4-mini-reasoning
    # stopped at exactly 256 tokens on 97% of its MCQs. The cap is a harness artifact,
    # not a property of the model, so it must not decide whether a model answers at all.
    "num_predict": 1024,
    # Pin the serving context explicitly. Recent Ollama versions choose a VRAM-based
    # default (4096 on this host), which is smaller than the largest k=5 MCQ prompt.
    # 8192 matches the published protocol and prevents silent prompt truncation.
    "num_ctx": 8192,
}

THINKING_GENERATION_OPTIONS = {
    **GENERATION_OPTIONS,
    "num_predict": 1024,
}

FINALIZE_OPTIONS = {
    **GENERATION_OPTIONS,
    "num_predict": 16,
}

# Some reasoning-family models continue producing an internal chain of thought even
# when ``think=False``.  A 256-token retry can therefore end before the requested
# answer letter.  Recovery calls keep the same deterministic decoding and context,
# but allow enough output for the model to reach its final answer.
RETRY_GENERATION_OPTIONS = {
    **GENERATION_OPTIONS,
    # 2048 was still the binding constraint for the slowest reasoning families:
    # deepseek-r1:7b (thinking) ended at exactly 2048 on every one of its 24 unrecovered
    # k-condition rows. 4096 leaves room for the largest k=5 prompt inside num_ctx=8192.
    "num_predict": 4096,
}

# Retries reuse the same terse prompt, so with the pinned ``seed`` they reproduce the
# previous attempt byte-for-byte: across 55k scored rows attempt 3 recovered 13 rows to
# attempt 2's 583. Offsetting the seed per attempt makes each retry an independent draw,
# which is what makes a third and fourth attempt worth spending at all.
RETRY_SEED_STRIDE = 1000

# Final rung of the escalation ladder, used from ``FINAL_RETRY_FROM_ATTEMPT`` onward.
# 92 of the 114 rows still unanswered after five attempts ended at *exactly* 4096, so for
# those the budget was still the binding constraint rather than the model. Doubling the
# output budget needs the serving context doubled with it: at num_ctx=8192 a k=5 prompt
# plus 8192 generated tokens would not fit, and the sliding window would push the question
# out of context before the model answered.
#
# The rungs below 6 are deliberately left untouched so that every response already scored
# reproduces byte-for-byte; only attempts that never happened use the larger budget.
FINAL_RETRY_GENERATION_OPTIONS = {
    **GENERATION_OPTIONS,
    "num_predict": 8192,
    "num_ctx": 16384,
}
FINAL_RETRY_FROM_ATTEMPT = 6



@dataclass(frozen=True)
class DocumentSpec:
    """One corpus document and its question set.

    ``document_id`` is the short stable name used throughout results (e.g.
    ``charter``); ``source_path`` is the markdown source and ``questions_path`` the
    matching MCQ JSON.
    """

    document_id: str
    source_path: Path
    questions_path: Path


@dataclass(frozen=True)
class ModelSpec:
    """One evaluated configuration: a display ``label`` (e.g.
    ``qwen3_8b_thinking_on``), the Ollama ``model`` tag to call, and whether
    reasoning ``thinking`` mode is on. The ``label`` is what appears in result CSVs.
    """

    label: str
    model: str
    thinking: bool


DOCUMENTS = (
    DocumentSpec(
        document_id="charter",
        source_path=DOCUMENTS_DIR / "doc1.md",
        questions_path=MCQ_DIR / "mcp1.json",
    ),
    DocumentSpec(
        document_id="budget",
        source_path=DOCUMENTS_DIR / "doc2.md",
        questions_path=MCQ_DIR / "mcp2.json",
    ),
    DocumentSpec(
        document_id="zoning",
        source_path=DOCUMENTS_DIR / "doc3.md",
        questions_path=MCQ_DIR / "mcp3.json",
    ),
    DocumentSpec(
        document_id="ordinances",
        source_path=DOCUMENTS_DIR / "doc4.md",
        questions_path=MCQ_DIR / "mcp4.json",
    ),
)


# ---------------------------------------------------------------------------
# MEDIUM tier: 7-9B class models (plus the 12B gemma4:12b, ~9.9 GB loaded,
# which fits a 16 GB card) on an 8-16 GB VRAM GPU.
# k=1/3/5 matches the small tier so the two tiers are cross-comparable.
# Some earlier runs left legacy k=6/9 rows in raw_results.csv; they are kept
# for archival purposes but are not targeted by new runs and are excluded
# from the primary qwen3_0_6b/ analysis.
# mistral_7b_thinking_off is intentionally absent from the standard model set
# (see README "Known limitations"). deepseek_r1_7b_thinking_on was re-added to
# the roster and run to the full 400-question standard grid.
# ---------------------------------------------------------------------------
MEDIUM_TOP_K_VALUES = (1, 3, 5)
MEDIUM_EMBEDDING_MODELS = (
    "qwen3-embedding:0.6b",
)

MEDIUM_MODEL_SPECS = (
    ModelSpec("qwen3_8b_thinking_off",       "qwen3:8b",        False),
    ModelSpec("qwen3_8b_thinking_on",        "qwen3:8b",        True),
    ModelSpec("qwen35_9b_thinking_off",      "qwen3.5:9b",      False),
    ModelSpec("qwen35_9b_thinking_on",       "qwen3.5:9b",      True),
    ModelSpec("gemma4_e4b_thinking_off",     "gemma4:e4b",      False),
    ModelSpec("gemma4_e4b_thinking_on",      "gemma4:e4b",      True),
    ModelSpec("gemma4_12b_thinking_off",     "gemma4:12b",      False),
    ModelSpec("gemma4_12b_thinking_on",      "gemma4:12b",      True),
    ModelSpec("qwen25_7b_thinking_off",      "qwen2.5:7b",      False),
    ModelSpec("llama31_8b_thinking_off",     "llama3.1:8b",     False),
    ModelSpec("deepseek_r1_7b_thinking_on",  "deepseek-r1:7b",  True),
)

MEDIUM_REQUIRED_MODELS = (
    "qwen3:8b",
    "qwen3.5:9b",
    "gemma4:e4b",
    "gemma4:12b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "deepseek-r1:7b",
    "qwen3-embedding:0.6b",
)


# ---------------------------------------------------------------------------
# SMALL tier: 13 hand-picked models ≤ ~3.3 GB RAM, each run with thinking
# OFF and ON (26 specs). Uses qwen3-embedding:0.6b only. k=1/3/5 because
# earlier sweeps showed large k hurts sub-3B models (lost-in-the-middle).
# Models that don't natively support thinking ignore the flag (Ollama
# behaviour); thinking_on rows for non-reasoning families act as a
# sanity-check duplicate.
# ---------------------------------------------------------------------------
SMALL_TOP_K_VALUES = (1, 3, 5)
SMALL_EMBEDDING_MODELS = (
    "qwen3-embedding:0.6b",
)


def _both(label_root: str, tag: str) -> tuple[ModelSpec, ModelSpec]:
    """Build the thinking-off and thinking-on :class:`ModelSpec` pair for one model
    tag, with labels ``<label_root>_thinking_off`` / ``_on``."""
    return (
        ModelSpec(f"{label_root}_thinking_off", tag, False),
        ModelSpec(f"{label_root}_thinking_on",  tag, True),
    )


SMALL_MODEL_SPECS = (
    *_both("qwen3_0_6b",            "qwen3:0.6b"),
    *_both("qwen35_0_8b",           "qwen3.5:0.8b"),
    *_both("llama32_1b",            "llama3.2:1b"),
    *_both("qwen25_1_5b",           "qwen2.5:1.5b"),
    *_both("deepseek_r1_1_5b",      "deepseek-r1:1.5b"),
    *_both("qwen3_1_7b",            "qwen3:1.7b"),
    *_both("gemma3_1b",             "gemma3:1b"),
    *_both("gemma4_e2b",            "gemma4:e2b"),
    *_both("qwen35_2b",             "qwen3.5:2b"),
    ModelSpec("llama32_3b_thinking_on",           "llama3.2:3b",         True),
    ModelSpec("phi4_mini_reasoning_thinking_off", "phi4-mini-reasoning", False),
    ModelSpec("gemma3_4b_thinking_on",            "gemma3:4b",           True),
)

SMALL_REQUIRED_MODELS = (
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
    "qwen3-embedding:0.6b",
)


# ---------------------------------------------------------------------------
# Tier selection: two public tiers, no aliases.
# ---------------------------------------------------------------------------
TIER_ALIASES = {
    "small": "small",
    "medium": "medium",
}


def resolve_tier(tier: str) -> str:
    """Normalize a tier name to its canonical form (``small`` or ``medium``),
    raising ``ValueError`` on anything else."""
    key = tier.strip().lower()
    if key not in TIER_ALIASES:
        raise ValueError(
            f"Unknown tier {tier!r}. Use one of: {sorted(set(TIER_ALIASES))}"
        )
    return TIER_ALIASES[key]


def model_specs_for(tier: str) -> tuple[ModelSpec, ...]:
    """The roster of model configurations to evaluate for ``tier``."""
    name = resolve_tier(tier)
    if name == "small":
        return SMALL_MODEL_SPECS
    return MEDIUM_MODEL_SPECS


def required_models_for(tier: str) -> tuple[str, ...]:
    """Ollama tags that must already be installed for ``tier`` (preflight hard-fails
    on any missing entry; embedders are auto-pulled instead)."""
    name = resolve_tier(tier)
    if name == "small":
        return SMALL_REQUIRED_MODELS
    return MEDIUM_REQUIRED_MODELS


def results_dir_for(tier: str) -> Path:
    """Per-tier results directory (``results/<tier>``)."""
    name = resolve_tier(tier)
    return RESULTS_DIR / name


def plots_dir_for(tier: str) -> Path:
    """Per-tier plots directory (``results/<tier>/plots``)."""
    return results_dir_for(tier) / "plots"


def top_k_values_for(tier: str) -> tuple[int, ...]:
    """Retrieval depths ``k`` swept for ``tier`` (both tiers use 1/3/5)."""
    name = resolve_tier(tier)
    if name == "small":
        return SMALL_TOP_K_VALUES
    return MEDIUM_TOP_K_VALUES


def embedding_models_for(tier: str) -> tuple[str, ...]:
    """Embedder tag(s) used to build retrieval conditions for ``tier``."""
    name = resolve_tier(tier)
    if name == "small":
        return SMALL_EMBEDDING_MODELS
    return MEDIUM_EMBEDDING_MODELS


def auto_pull_models_for(tier: str) -> tuple[str, ...]:
    """Models the pipeline may pull automatically — just the (small, cheap)
    embedders the tier uses; the LLMs themselves must be pre-installed."""
    return embedding_models_for(tier)


DEFAULT_TIER = "medium"

# Backwards-compat aliases (older scripts may still reference these names).
MODEL_SPECS = MEDIUM_MODEL_SPECS
REQUIRED_MODELS = MEDIUM_REQUIRED_MODELS
TOP_K_VALUES = MEDIUM_TOP_K_VALUES
EMBEDDING_MODELS = MEDIUM_EMBEDDING_MODELS
AUTO_PULL_MODELS = MEDIUM_EMBEDDING_MODELS


def retrieval_conditions(tier: str = "medium") -> list[dict[str, object]]:
    """All retrieval conditions for ``tier``: the ``no_rag`` baseline plus one
    ``rag_<embedder>_k<k>`` entry per (embedder, depth) combination. Each is a dict
    of ``retrieval_condition`` / ``embedding_model`` / ``top_k``; the condition
    string is the key written into result CSVs."""
    conditions: list[dict[str, object]] = [
        {"retrieval_condition": "no_rag", "embedding_model": "", "top_k": 0}
    ]
    for embedding_model in embedding_models_for(tier):
        for top_k in top_k_values_for(tier):
            conditions.append({
                "retrieval_condition": f"rag_{embedding_model}_k{top_k}",
                "embedding_model": embedding_model,
                "top_k": top_k,
            })
    return conditions
