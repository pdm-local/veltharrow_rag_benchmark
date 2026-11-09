# Veltharrow RAG Benchmark

## How this benchmark came to be

This started as a nagging feeling about my bachelor thesis. I had built a RAG benchmark on
real-world documents, and when I went back to read the results, they didn't say what I
expected. The numbers were *too good* in places where retrieval shouldn't have mattered. It
took me a while to see the problem. If you ask an open-weight model about real, public
documents, you can never be sure it isn't simply *recalling* them. They may well have been in
its training data. So a "no-RAG" baseline isn't a blind baseline at all, and if the baseline
is contaminated, you aren't measuring retrieval, you're measuring memory.

To honestly measure what RAG adds, the model has to face documents it has *never seen*. Around
the same time I happened to be talking with a few municipalities, potential clients for exactly
this kind of document Q&A, which is where the idea clicked. Instead of hunting for "clean" real
data that probably doesn't exist, why not invent a municipality from scratch? Somewhere
genuinely unseen, that no model could have memorized.

So I built **Veltharrow**, a fictional town in the fictional Republic of Drenmoor, with its own
history, geography, and government. I wrote four documents describing it: who governs it and by
what rules (the charter), where its money comes from and goes (the budget), what you're allowed
to build and where (the zoning code), and the day-to-day rules residents live under (the
ordinances). Then I wrote **400 multiple-choice questions** about those documents, graded along
a Bloom's-taxonomy rubric so the set spans simple recall up to cross-provision reasoning.
Because every answer is known and fixed, scoring needs no LLM judge. Accuracy is just the
fraction correct, and the no-RAG baseline finally measures pure parametric guessing on data the
model truly cannot have seen.

That gave me the core result. But multiple-choice flattered the models, since in real use
nobody picks the right answer out of six. So I ran a second, harder round: **70 open-ended
free-form questions** across seven categories (including unanswerable, typo-laden, and
multi-chunk questions) to see how the same models hold up when there's no menu to choose from.

This repo is the reproducible result of that work: the documents, the questions, the code, and
everything you need to get numbers close to mine on your own machine.

> A post-thesis extension of a bachelor thesis submitted May 2026. The thesis used three
> real-world domains, free-text answers, and a cloud judge. This project replaces them with
> synthetic, provably-unseen documents and judge-free MCQ accuracy, scaled across many more
> models and retrieval configurations.

Made by Daniel-Mihai Pocatilu at [pdm.ai](https://www.pdmzerotrustbusiness.com), 27 October 2026.

- Demo: https://www.pdmzerotrustbusiness.com/projects/government/
- Paper: https://www.pdmzerotrustbusiness.com/papers/2026-10-veltharrow.pdf, DOI [10.5281/zenodo.24084989](https://doi.org/10.5281/zenodo.24084989)

---

## Quickstart

Run the whole benchmark on a single machine:

```bash
git clone https://github.com/pdm-local/veltharrow_rag_benchmark.git
cd veltharrow_rag_benchmark
bash setup.sh
source .venv/bin/activate

# Pull the models you want (see "Models" below); at minimum the small tier:
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:0.6b qwen3.5:0.8b llama3.2:1b qwen2.5:1.5b deepseek-r1:1.5b \
            qwen3:1.7b gemma3:1b gemma4:e2b qwen3.5:2b llama3.2:3b \
            phi4-mini-reasoning gemma3:4b

# Cache retrievals once, then run:
python3 scripts/precompute_retrievals.py --tier small
python3 scripts/run_benchmark.py --tier small
```

`--tier all` runs both tiers (`small` then `medium`) sequentially on one host. Ollama is assumed at `http://localhost:11434`; override with `export OLLAMA_HOST=...`. No path editing is needed: everything is relative to the repo root.

---

## Results & dataset

- **Plots** (publication figures) are committed under [`figures/`](figures/): `figures/small/` and `figures/medium/` mirror each tier's primary `qwen3_0_6b/` analysis folder, `figures/freeform/` holds the free-form (LLM-judged) figures, and `figures/main/` holds the paper's headline recognition-vs-production figure.
- **The dataset and raw result CSVs** will be published on Hugging Face (link added at release). That will be the canonical copy of the documents, the 400 MCQs, the 70 free-form questions, and the result CSVs (`raw_results.csv` / `retrieval_quality.csv` per tier, plus the free-form answers and 10x-averaged judge grades).
- This GitHub repo holds the **code and inputs** needed to reproduce or extend the benchmark. `results/` and `cache/` are gitignored: clone the repo to run it, and (once released) pull the CSVs from Hugging Face to analyze the published output.

---

## What is being measured

The question driving everything here is simple: **when a local model is asked about documents
it has never seen, how much does retrieval actually help, and for which models?**

Four Veltharrow documents carry the whole benchmark. Between them they describe the town's
people, its money, its government, and its rules, the kinds of things a real municipal
assistant would be asked about:

| ID | Document | Description |
|----|----------|-------------|
| `charter` | Municipal Charter | Governance structure, officers, bylaws |
| `budget` | FY2024 Budget | Revenue and expenditure line items |
| `zoning` | Zoning Code | Land-use districts, setbacks, permitted uses |
| `ordinances` | Municipal Ordinances | Regulations across 13 topic areas |

Each document has **100 multiple-choice questions** (6 choices: A to F) across four Bloom's Taxonomy levels:

| Level | Count | What it tests |
|-------|------:|---------------|
| remember | 76 | Direct recall of a stated fact |
| understand | 145 | Paraphrase or interpret a provision |
| apply | 96 | Use a rule in a specific scenario |
| analyze | 83 | Compare or reason across multiple provisions |

---

## Tiers

Two tiers, distinguished only by model size. Both use **qwen3-embedding:0.6b** as the embedder and **k = 1, 3, 5** retrieval depths, so results are directly cross-comparable. The canonical analysis for each is `results/<tier>/plots/qwen3_0_6b/` (mirrored to `figures/<tier>/`).

| Tier | Target hardware | Model size |
|------|-----------------|------------|
| `small` | Any machine with Ollama (>= 6 GB VRAM) | Small-footprint, roughly 0.6B to 4B-class |
| `medium` | 8 to 16 GB VRAM GPU | 7 to 12B parameters |

---

## Models

### Small-footprint tier (roughly 0.6B to 4B-class)
Each run with thinking ON and OFF (where supported by Ollama):

`qwen3:0.6b`, `qwen3.5:0.8b`, `llama3.2:1b`, `qwen2.5:1.5b`, `deepseek-r1:1.5b`, `qwen3:1.7b`, `gemma3:1b`, `gemma4:e2b`\*, `qwen3.5:2b`, `llama3.2:3b`, `phi4-mini-reasoning`, `gemma3:4b`

\* `gemma4:e2b` is a MatFormer model: 5.1B total params, ~2B active at inference.

### Medium tier (7 to 12B)
`qwen3:8b` (think off/on), `qwen3.5:9b` (think off/on), `gemma4:e4b` (think off/on), `gemma4:12b` (think off/on)\*, `qwen2.5:7b`, `llama3.1:8b`, `deepseek-r1:7b` (think on)

\* `gemma4:12b` (12B, q4_K_M) is the largest model in the suite; it needs ~9.9 GB loaded (fits a 16 GB GPU). It tops the MCQ tier (95.6% at k=5, thinking on).

---

## Retrieval conditions

Every model x question is run under 4 conditions:

| Condition | Description |
|-----------|-------------|
| `no_rag` | No context; model answers from parametric knowledge only |
| `rag_qwen3-embedding:0.6b_k1` | Top-1 chunk retrieved |
| `rag_qwen3-embedding:0.6b_k3` | Top-3 chunks retrieved |
| `rag_qwen3-embedding:0.6b_k5` | Top-5 chunks retrieved |

The medium tier's raw CSV also contains legacy rows at k=6 and k=9 from earlier exploratory runs. These are kept for archival purposes but excluded from the primary `qwen3_0_6b/` analysis plots.

---

## Project structure

```
Benchmark/
├── src/veltharrow/
│   ├── config.py               # ALL tier/model/document config; single source of truth
│   ├── rag.py                  # Chunker, embedder, retriever, cache
│   ├── ollama_client.py        # HTTP client (host from OLLAMA_HOST, default localhost:11434)
│   └── stats.py                # Shared stdlib-only stats helpers (Wilson score CI)
│
├── scripts/
│   ├── run_benchmark.py        # Main MCQ runner (resume-safe, file-locked); --tier all supported
│   ├── precompute_retrievals.py# Cache top-k chunk lists per question
│   ├── check_speed.py          # Live progress monitor
│   ├── plot_results.py         # All MCQ result plots (11 overview + top5 + per-model)
│   ├── compute_retrieval_quality.py   # Post-hoc recall scoring per row
│   ├── plot_retrieval_quality.py      # Retrieval quality plots (rq_01 to 04)
│   ├── validate_plot_numbers.py       # Cross-check every plot number from raw CSV
│   ├── run_freeform_eval.py           # Free-form answer generation (RAG)
│   ├── grade_freeform_api.py          # Free-form grading via OpenAI judge (gpt-5-mini)
│   ├── plot_freeform.py               # Free-form figures (ff_01 to 03, ff_05)
│   ├── prepare_grading_batch.py       # Split/merge free-form answers for manual blind grading
│   ├── generate_answers.py            # Free-text generation over the MCQ questions (exploratory)
│   └── evaluate_generation.py         # Judge for generate_answers.py output (local or OpenAI)
│
├── tools/
│   ├── merge_results.py        # Merge incoming CSV into primary (dedup by key); advanced/optional
│   ├── status_board.py         # Terminal dashboard
│   ├── paper_tables.py         # LaTeX tables for the paper from result CSVs
│   ├── stats_tests.py          # Paired exact McNemar tests (Holm-corrected)
│   ├── plot_quality_latency.py # Paper figure: free-form quality vs latency Pareto
│   └── plot_recognition_gap.py # Paper figure: recognition vs production gap
│
├── data/
│   ├── documents/              # doc1 to 4.md (Charter / Budget / Zoning / Ordinances)
│   ├── mcq/
│   │   ├── mcp1 to 4.json      # 100 six-option MCQs per document
│   │   ├── gold_chunks.json    # question_id -> [chunk_id] mapping (auto-generated)
│   │   └── README.md           # MCQ dataset documentation
│   └── freeform/
│       └── questions.json      # 70 open-ended questions across 7 buckets
│
├── figures/                    # Publication plots (committed)
│   ├── small/                  # mirror of results/small/plots/qwen3_0_6b/
│   ├── medium/                 # mirror of results/medium/plots/qwen3_0_6b/
│   ├── freeform/               # free-form (LLM-judged) figures
│   └── main/                   # headline figure: recognition_vs_production.png
│
├── results/                    # Benchmark output; GITIGNORED; CSVs to be published on Hugging Face
│   ├── small/{data,plots}/     # raw_results.csv, retrieval_quality.csv, plots/qwen3_0_6b/
│   ├── medium/{data,plots}/    # + plots/archive/ for full multi-condition results
│   └── freeform/               # generated answers + graded.csv (judge output, 10x averaged)
│
└── cache/                      # Regeneratable embeddings + retrievals; GITIGNORED
```

---

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates a `.venv`, installs pinned dependencies from `requirements.txt`, and installs the `veltharrow` package in editable mode so `from veltharrow...` works anywhere. The core runner is stdlib-only; numpy/pandas/matplotlib are needed only for plotting.

---

## Running the benchmark

### 1. Precompute retrievals (once per tier)

Embeds all 400 questions and caches top-k chunk lists. No GPU pressure beyond the embedding model.

```bash
python3 scripts/precompute_retrievals.py --tier small
python3 scripts/precompute_retrievals.py --tier medium
# or both at once:
python3 scripts/precompute_retrievals.py --tier all
```

### 2. Pull required models

```bash
# Shared:
ollama pull qwen3-embedding:0.6b

# Small tier:
ollama pull qwen3:0.6b qwen3.5:0.8b llama3.2:1b qwen2.5:1.5b deepseek-r1:1.5b \
            qwen3:1.7b gemma3:1b gemma4:e2b qwen3.5:2b llama3.2:3b \
            phi4-mini-reasoning gemma3:4b

# Medium tier:
ollama pull qwen3:8b qwen3.5:9b gemma4:e4b gemma4:12b qwen2.5:7b llama3.1:8b deepseek-r1:7b
```

### 3. Run

```bash
# Both tiers, sequentially, on one machine:
python3 scripts/run_benchmark.py --tier all

# A single tier:
python3 scripts/run_benchmark.py --tier small
python3 scripts/run_benchmark.py --tier medium

# Specific models only:
python3 scripts/run_benchmark.py --tier medium \
  --model qwen35_9b_thinking_on \
  --model gemma4_e4b_thinking_off

# Smoke test (2 questions):
python3 scripts/run_benchmark.py --tier medium --limit-questions 2 \
  --model qwen3_8b_thinking_off --condition no_rag
```

The runner is **resume-safe**: already-completed `(model, document, question, condition)` rows are skipped. It uses `fcntl` file locking so parallel workers can append concurrently.

**Evict a model from VRAM** before switching to a new model:
```bash
curl -s -X POST http://localhost:11434/api/generate \
  -d '{"model": "gemma4:e4b", "keep_alive": 0}' > /dev/null
```

### 4. Monitor progress

```bash
python3 scripts/check_speed.py                                            # default: medium tier
python3 scripts/check_speed.py --file results/small/data/raw_results.csv --total 1600
```

---

## Generating plots

```bash
# MCQ analysis plots:
python3 scripts/plot_results.py --tier small
python3 scripts/plot_results.py --tier medium

# Retrieval quality (compute first if retrieval_quality.csv is stale/missing):
python3 scripts/compute_retrieval_quality.py --tier medium
python3 scripts/plot_retrieval_quality.py --tier medium

# Cross-check every plot number against the raw CSV:
python3 scripts/validate_plot_numbers.py --tier medium --fair
```

### Plot catalogue

Primary plots live in `results/<tier>/plots/qwen3_0_6b/` and are mirrored to `figures/<tier>/`.

**Overview** (`overview/`):

| File | Description |
|------|-------------|
| `01_norag_vs_bestrag.png` | No-RAG vs best-RAG accuracy per model (95% CI) |
| `02_top3_per_model.png` | Top-3 conditions per model, grouped |
| `03_heatmap.png` | Full model x condition accuracy heatmap |
| `04_k_effect.png` | Accuracy vs retrieval depth k |
| `05_embedder_by_k.png` | Embedder comparison across k values |
| `06_rag_gain.png` | RAG gain over no-RAG baseline (pp) |
| `07_by_document.png` | Per-document accuracy |
| `08_bloom_level.png` | Accuracy by Bloom's Taxonomy level |
| `09_latency_vs_accuracy.png` | Pareto frontier: latency vs accuracy |
| `09_latency_vs_accuracy_tokens_per_sec.png` | Throughput (tok/s) vs accuracy |
| `10_condition_ranking.png` | All conditions ranked by average accuracy |
| `11_thinking_comparison.png` | Thinking ON vs OFF across conditions |

**Top-5** (`top5/`): `top5_01_ranked.png` (leaderboard with 95% CI overlap banding), `top5_02_by_document.png`, `top5_03_by_bloom.png`, `top5_04_latency.png`, `top5_04_tokens_per_sec.png`.

**Per-model** (`per_model/<label>.png`): 6-panel dashboard with accuracy by condition, by document, by Bloom level, latency distribution, k-effect, and a stats summary.

**Retrieval quality** (`retrieval/`): `rq_01_recall_by_condition.png`, `rq_02_per_document.png`, `rq_03_failure_decomposition.png`, `rq_04_precision_recall_fair.png`.

**Free-form** (`figures/freeform/`, via `scripts/plot_freeform.py`): `ff_01_overall.png` (per-model scores on the three judge metrics), `ff_02_heatmap_<metric>.png` (model x bucket heatmaps for fact-correct / hallucination-free / source-grounded), `ff_03_buckets_<metric>.png` (per-bucket bars), `ff_05_latency.png`. `ff_04_long_context.png` is a committed one-off figure (RAG vs full-document context) not regenerated by the script.

**Headline** (`figures/main/`): `recognition_vs_production.png`, best-RAG MCQ accuracy against free-form source-grounding for the six free-form models.

---

## Raw CSV columns

`results/<tier>/data/raw_results.csv` (to be published on Hugging Face):

| Column | Description |
|--------|-------------|
| `run_id` | Timestamp-based session identifier |
| `document_id` | `charter` / `budget` / `zoning` / `ordinances` |
| `question_id` | Unique question identifier |
| `model_label` | e.g. `qwen3_8b_thinking_off` |
| `model` | Ollama model tag e.g. `qwen3:8b` |
| `retrieval_condition` | `no_rag` or `rag_qwen3-embedding:0.6b_k<n>` |
| `is_correct` | `1` / `0` |
| `retrieved_chunk_ids` | Pipe-separated chunk IDs used as context |
| `latency_seconds` | Wall-clock time for the LLM call |
| `eval_count` | Tokens generated (throughput = eval_count / latency_seconds) |
| `prompt_eval_count` | Prompt tokens processed |
| `thinking` | Raw chain-of-thought (thinking mode only) |
| `attempts` | Retries before a parseable answer was extracted |
| `fallback_used` | `1` if a second LLM call was needed to parse the letter |

---

## Known limitations

- **`deepseek_r1_7b_thinking_on` is now part of the standard medium-tier set** (thinking-on only), run to the full 400-question grid. It posts the lowest no-RAG baseline of the 7 to 9B models (~0.44) but one of the largest RAG lifts (~+0.36), reaching ~0.80 at k=3.
- **Legacy k=6/9 rows** exist in the medium-tier raw CSV from earlier exploratory runs. They are kept for archival purposes and are excluded from the primary `qwen3_0_6b/` analysis.
- **170 of 400 MCQs are unmapped** in `gold_chunks.json`: their evidence text spans a chunk boundary or is not verbatim in any single chunk, so they are excluded from recall scoring (but still scored for accuracy).

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Ollama only | No API keys, no cloud spend, fully reproducible on local hardware |
| Temperature 0, seed 42 | Deterministic generation; the same question always gets the same answer |
| fcntl file locking | Multiple workers can append to the same CSV safely |
| Resume-safe by composite key | Any crash is recoverable; re-run picks up where it left off |
| Fair subset (`qwen3_0_6b/`) | Both tiers share qwen3-embedding:0.6b @ k=1/3/5; cross-tier comparison is valid |
| Wilson 95% CI on all accuracy figures | n=60 at 93% and n=400 at 93% are not equally certain |
| CI overlap banding on leaderboard | Visually groups statistically tied models so within-band ranking is not over-interpreted |
| Gold chunk mapping (2-pass) | Direct substring, then markdown-stripped. No fuzzy fallback (false-positive risk outweighs coverage gain) |

---

## What I found

The first surprise was a small model. **gemma4:e2b lands around 93% accuracy**, on par with the
7 to 9B models despite being a fraction of their size. Its MatFormer design (5.1B total
parameters, ~2B active at inference) clearly does more than its size suggests, and it changed
how I think about needing a big model for this kind of task.

The second surprise was that **retrieval is not a free win**. On average it helps, but for a few
of the smallest models RAG actively *hurt*: feeding them context made them worse than when they
answered from memory alone. RAG is a tool, not a guarantee.

Then there's **how much context to retrieve**. I expected "more is better"; the data says
otherwise. Recall is essentially saturated by **k=3**, and pushing to k=5 buys only a little
extra recall while dragging in more noise for the model to trip over. k=3 turned out to be the
sweet spot almost everywhere.

The most clarifying finding came from asking *why* the wrong answers were wrong. Most of the
time **the right chunk had been retrieved and the model still got it wrong** (see
`rq_03_failure_decomposition`). The bottleneck for these local models isn't finding the
evidence, it's reasoning over it. That reframed retrieval quality as necessary but far from
sufficient.

One technical footnote so the precision numbers aren't misread: **precision@k looks low by
construction** (about recall/k), because each question has exactly one gold chunk among the k
retrieved. That is expected; for this benchmark recall is the metric that actually matters.

The free-form round echoed all of this and added its own twist, the finding that ended up
naming the paper: **recognizing is not grounding**. The six free-form models all *recognize*
the right MCQ answer within a narrow band (best-RAG accuracy between 89.0% and 95.6%), but
their ability to *produce* a grounded answer, one that cites a section that really appears in
the retrieved context, fans out from 46% to 94%. The ordering changes too: gemma4:12b tops the
MCQ leaderboard, yet the 4B-active gemma4:e4b leads the free-form composite (93.6%). And the
hard buckets aren't the ones you'd guess: typos barely matter (100% fact-correct), while vague
and long-context prompts are the real stress points (~70%).

---

## Advanced: distributed runs

The default workflow is single-machine (`--tier all`). If you split work across machines, for example the small tier on a laptop and the medium tier on a desktop, each writes its own `results/<tier>/data/raw_results.csv`, and you can merge per-model CSVs afterward.

Transfer the retrieval cache to skip re-embedding on a second machine:
```bash
scp -r cache/ user@<other-machine>:/path/to/Benchmark/
```

Merge incoming results into a primary CSV (dedup key `(model_label, document_id, question_id, retrieval_condition)`):
```bash
python3 tools/merge_results.py \
  results/medium/data/raw_results.csv \
  results/medium/by_model/<label>.csv
```

---

## Citation

```bibtex
@misc{pocatilu2026veltharrow,
  title     = {Recognition and Grounded Answering: Retrieval-Augmented Small Language
               Models on a Fictional Municipal Corpus},
  author    = {Pocatilu, Daniel-Mihai},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.24084989},
  url       = {https://doi.org/10.5281/zenodo.24084989}
}
```

---

## Maintainer

Daniel-Mihai Pocatilu, pdm.ai (https://www.pdmzerotrustbusiness.com)

---

## License

- **Code**: GNU Affero General Public License 3.0 (see [`LICENSE`](LICENSE)). Versions released before November 2026 stay under MIT.
- **Dataset & results** (`data/`, `results/`, `figures/`, and the Hugging Face dataset): Creative Commons Attribution 4.0 International (CC BY 4.0; see [`LICENSE-DATA`](LICENSE-DATA)).
