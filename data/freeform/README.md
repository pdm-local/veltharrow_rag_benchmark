# Free-form Dataset: Veltharrow Benchmark

## Why this round exists

The multiple-choice benchmark gave the core result, but it flattered the models. In real use
nobody answers a question by picking one of six pre-written options. They have to *produce* an
answer, and they have to know when there isn't one. So after seeing the MCQ numbers I added a
second, harder round: **70 open-ended free-form questions** about the same four Veltharrow
documents, where the model writes the answer itself and is graded on whether it is factually
correct, grounded in the source, and free of hallucination.

## Buckets

The 70 questions are split evenly across **7 categories (10 each)**, chosen to stress different
failure modes, not just whether a model can answer, but whether it knows the limits of what it
can answer:

| Bucket | What it probes |
|--------|----------------|
| `answerable` | Straightforward questions with a clear answer in the documents |
| `unanswerable` | Questions whose answer is *not* in the documents; the model should decline, not invent |
| `typo` | Questions with misspellings/noise, testing robustness to imperfect input |
| `math` | Questions requiring arithmetic over budget/figures, not just recall |
| `multichunk` | Answers that require combining evidence from more than one passage |
| `vague` | Under-specified questions that test whether the model asks/handles ambiguity |
| `longcontext` | Questions whose answer is spread across a long stretch of the document, retrieved via RAG (k = 3) |

## Format

`questions.json` is a JSON array. Each element has:

| Field | Description |
|-------|-------------|
| `id` | Unique question ID (e.g. `FF-A01`) |
| `bucket` | One of the 7 categories above |
| `document_id` | `charter` / `budget` / `zoning` / `ordinances` |
| `question` | The question stem |
| `gold_answer` | Reference answer (for `unanswerable`, this records that no answer exists) |
| `gold_keywords` | Key terms a correct answer should contain |
| `gold_chunks` | Chunk IDs holding the supporting evidence |

Grading is handled by `scripts/grade_freeform_api.py`. For each row the judge (an OpenAI model,
default `gpt-5-mini`) receives the question, the `gold_answer`, the actual retrieved context
paragraphs, and the answer text, never the model's thinking, and applies the rubric in
`scripts/grade_freeform_prompt.txt` (three binary metrics: fact-correct, hallucination-free,
source-grounded).
