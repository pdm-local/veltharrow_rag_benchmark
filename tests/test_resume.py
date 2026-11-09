import csv
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_completed_keys_skips_done_rows(tmp_path: Path) -> None:
    from run_benchmark import completed_keys

    rows = [
        {
            "document_id": "charter",
            "question_id": "C001",
            "model_label": "gemma3_e4b_thinking_on",
            "retrieval_condition": "no_rag",
            "predicted_letter": "A",
            "error": "",
        },
        {
            "document_id": "charter",
            "question_id": "C002",
            "model_label": "gemma3_e4b_thinking_on",
            "retrieval_condition": "no_rag",
            "predicted_letter": "B",
            "error": "",
        },
        {
            "document_id": "charter",
            "question_id": "C003",
            "model_label": "gemma3_e4b_thinking_on",
            "retrieval_condition": "no_rag",
            "predicted_letter": "",
            "error": "timeout",
        },
    ]
    csv_path = tmp_path / "raw_results.csv"
    _write_csv(csv_path, rows)

    keys = completed_keys(csv_path)
    # 2 good rows are skipped; the errored row should NOT be skipped (so it
    # gets retried on resume).
    assert ("charter", "C001", "gemma3_e4b_thinking_on", "no_rag") in keys
    assert ("charter", "C002", "gemma3_e4b_thinking_on", "no_rag") in keys
    assert ("charter", "C003", "gemma3_e4b_thinking_on", "no_rag") not in keys
    assert len(keys) == 2


def test_completed_keys_empty_when_file_missing(tmp_path: Path) -> None:
    from run_benchmark import completed_keys

    assert completed_keys(tmp_path / "nope.csv") == set()


def test_adjudicated_unanswered_rows_count_as_done(tmp_path: Path) -> None:
    """A row scored "never answered" is a final verdict, not a gap to refill.

    Regression for 2026-09-07: score_unanswered.py writes these rows to complete a cell,
    but completed_keys rejected them (invalid letter + non-empty error), so a resumed run
    re-answered the question and *appended* the result. One cell reached 402 rows with
    duplicate composite keys, and a published accuracy shifted.
    """
    from run_benchmark import NO_ANSWER_LETTER, SCORED_UNANSWERED, completed_keys

    base = {
        "document_id": "budget",
        "model_label": "deepseek_r1_7b_thinking_on",
        "retrieval_condition": "no_rag",
    }
    rows = [
        {**base, "question_id": "Q1", "predicted_letter": "B", "error": ""},
        {**base, "question_id": "Q2", "predicted_letter": NO_ANSWER_LETTER,
         "error": f"{SCORED_UNANSWERED} (non-termination)"},
        {**base, "question_id": "Q3", "predicted_letter": NO_ANSWER_LETTER,
         "error": f"{SCORED_UNANSWERED} (backend)"},
        # A transient failure is a real gap and must still be retried.
        {**base, "question_id": "Q4", "predicted_letter": "", "error": "connection reset"},
    ]
    path = tmp_path / "raw_results.csv"
    _write_csv(path, rows)

    done = completed_keys(path)
    key = lambda q: (base["document_id"], q, base["model_label"], base["retrieval_condition"])
    assert key("Q1") in done
    assert key("Q2") in done, "adjudicated non-termination must not be re-attempted"
    assert key("Q3") in done, "adjudicated backend fault must not be re-attempted"
    assert key("Q4") not in done, "a transient error is still a gap to refill"


def test_score_unanswered_shares_run_benchmark_markers() -> None:
    """The writer and the reader of these rows must use one definition of the markers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import run_benchmark
    import score_unanswered

    assert score_unanswered.NO_ANSWER_LETTER is run_benchmark.NO_ANSWER_LETTER
    assert score_unanswered.SCORED_UNANSWERED is run_benchmark.SCORED_UNANSWERED
