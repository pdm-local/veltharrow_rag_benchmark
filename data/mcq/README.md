# MCQ Dataset: Veltharrow Benchmark

## How questions were generated

These questions exist because the documents do. Once Veltharrow was written, four municipal
documents about a town no model could have seen, the goal was to probe how well a model can
answer questions about them at every level of difficulty, from "look it up" to "reason across
several provisions."

So for each of the four documents (charter, budget, zoning, ordinances) I authored 100
questions, **400 in total**, deliberately spread across the four Bloom's-taxonomy levels below
rather than clustered at easy recall. Every question is pinned to a verbatim `evidence` span
from its source document, which is what makes the answer key objective and lets retrieval
recall be scored automatically. The six options per question (one correct, five distractors)
are described under "Distractor strategy" below.

## Format

Each question file (`mcp1.json` to `mcp4.json`) contains a JSON array. Each element has:

| Field       | Description |
|-------------|-------------|
| `id`        | Unique question ID (e.g. `charter_q001`) |
| `type`      | Question sub-type within the Bloom level (e.g. `definition`, `calculation`) |
| `bloom`     | Bloom's Taxonomy level (see distribution below) |
| `question`  | The question stem |
| `options`   | Object with keys A to F, six answer choices |
| `answer`    | Correct letter (A to F) |
| `evidence`  | Verbatim text from the source document that supports the correct answer |
| `reasoning` | Explanation of why the correct answer is correct |

## Distractor strategy

Every question has exactly **6 choices (A to F)**; 5 are distractors. Distractors are
drawn from other facts within the same document (e.g. neighbouring article numbers,
adjacent budget line items) so they are plausible to a naive model but unambiguously
wrong given the evidence.

## Distribution

**100 questions per document, 400 total.**

| Bloom level | charter | budget | zoning | ordinances | total |
|-------------|--------:|-------:|-------:|-----------:|------:|
| remember    |      19 |     19 |     19 |         19 |    76 |
| understand  |      32 |     38 |     37 |         38 |   145 |
| apply       |      24 |     24 |     24 |         24 |    96 |
| analyze     |      25 |     19 |     20 |         19 |    83 |
| **total**   |  **100**| **100**| **100**|    **100** | **400** |

Bloom levels test progressively deeper reasoning:
- **remember**: direct recall of a stated fact
- **understand**: paraphrase or interpret a provision
- **apply**: use a rule in a specific scenario
- **analyze**: compare, infer, or reason across multiple provisions

## Gold chunk mapping

`gold_chunks.json` maps each `question_id` to the list of chunk IDs (from the RAG
chunker) whose text contains the `evidence` string as a substring. This mapping is
used by `scripts/compute_retrieval_quality.py` to score retrieval recall.

Out of 400 questions, **230 are mappable** (evidence found in at least one chunk);
170 have no match because the evidence text either spans a chunk boundary, belongs to
a very short chunk that was merged, or is otherwise not verbatim in any single chunk.
