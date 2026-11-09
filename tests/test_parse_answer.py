"""Regression tests for MCQ answer parsing.

Every case marked *regression* is a verbatim (or minimally trimmed) response taken
from ``results/*/data/raw_results.csv`` that the pre-2026-08-30 parser scored wrongly.
That parser uppercased the response and took the *first* ``[A-F]`` word match, so the
English article "a" was read as option A — which both rewrote verbose answers to A and
masked truncated responses that never contained an answer at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_benchmark import (
    RETRY_PROMPT_PREFIX,
    RETRY_PROMPT_SUFFIX,
    answer_region,
    parse_answer,
    strip_instruction_echo,
)


QUESTION = {
    "OptionA": "Tier 1 (0-4 employees)",
    "OptionB": "Tier 2 (5-19 employees)",
    "OptionC": "Tier 3 (20-99 employees)",
    "OptionD": "Tier 4 (100+ employees)",
    "OptionE": "No fee is charged",
    "OptionF": "Not stated in the document",
}


class TestBareLetter:
    @pytest.mark.parametrize("raw", ["A", "a", " D ", "D.", "**B**", "(C)", "E\n"])
    def test_bare_letter_forms(self, raw):
        assert parse_answer(raw) == raw.strip().strip("*(). \n").upper()


class TestArticleIsNotAnAnswer:
    """The core regression: prose "a" must never be read as option A."""

    def test_trailing_letter_wins_over_leading_article(self):
        # regression: gemma4_12b_thinking_off, scored A, actually answered E
        raw = (
            'The document states that the "combined rate" for sales tax is 7.473%. '
            "For a purchase of $1,000, the calculation is $1,000 x 0.07473 = $74.73.\n\nE"
        )
        assert parse_answer(raw) == "E"

    def test_article_only_response_is_unanswered(self):
        assert parse_answer("The rate is a fraction of the assessed value.") == ""

    def test_lowercase_a_never_matches_in_prose(self):
        assert parse_answer("a business with a permit needs a fee") == ""

    def test_regression_tier_answer(self):
        # regression: gemma4_12b_thinking_off, scored A, actually answered D
        raw = (
            "The Business License Fee Schedule is as follows: Tier 1 (0-4 employees) "
            "is $312; Tier 2 (5-19 employees) is $874. A business with 5 employees "
            "falls into Tier 2.\n\nD"
        )
        assert parse_answer(raw, question=QUESTION) == "D"


class TestLastMentionWins:
    def test_declaration_beats_earlier_option_mentions(self):
        raw = "Option B looks plausible and option C is close, but the answer is D."
        assert parse_answer(raw) == "D"

    def test_final_answer_phrasing(self):
        assert parse_answer("Reasoning...\nFinal answer: **F**") == "F"

    def test_last_of_several_declarations(self):
        assert parse_answer("Answer: B\nOn reflection, the correct answer is E") == "E"

    def test_deepseek_style_trailing_letter(self):
        # regression: deepseek_r1_7b_thinking_on, scored A, actually answered C
        raw = (
            "The answer cannot be determined with the given information as there's no "
            "context. However, a mid-range figure like C ($14,882,300) is plausible.\n\nC"
        )
        assert parse_answer(raw) == "C"


class TestTruncatedReasoningIsUnanswered:
    """An unclosed <think> block means generation stopped before any answer."""

    def test_unclosed_think_returns_empty(self):
        # regression: phi4_mini_reasoning_thinking_off, scored A, answered nothing
        raw = (
            "<think>\nOkay, let's see here. The question is about the daytime maximum "
            "noise level at M-1 property lines. I need to find a value that"
        )
        assert parse_answer(raw, question=QUESTION) == ""

    def test_closed_think_uses_only_the_text_after_it(self):
        raw = "<think>Option A seems right at a glance, but no.</think>\n\nC"
        assert parse_answer(raw) == "C"

    def test_empty_after_closed_think_is_unanswered(self):
        assert parse_answer("<think>A lot of reasoning about a fee.</think>   ") == ""

    def test_answer_region_signals_truncation(self):
        assert answer_region("<think>cut off mid thought") is None
        assert answer_region("<think>done</think> B").strip() == "B"
        assert answer_region("no think block").strip() == "no think block"


class TestCodesAreNotLetters:
    """District and ordinance codes must not be mistaken for answer letters."""

    @pytest.mark.parametrize(
        "raw",
        [
            "The C-3 district permits mixed use.",
            "Ordinance T-2009-003 governs this.",
            "Zoned M-1 and R-3 respectively.",
            "Measured in dBA at the property line.",
        ],
    )
    def test_codes_do_not_parse(self, raw):
        assert parse_answer(raw) == ""

    def test_code_then_real_answer(self):
        assert parse_answer("The C-3 district applies.\n\nB") == "B"


class TestOptionTextFallback:
    def test_exact_option_text_match(self):
        raw = "Based on the schedule, the business falls under Tier 4 (100+ employees)."
        assert parse_answer(raw, question=QUESTION) == "D"

    def test_no_question_means_no_text_fallback(self):
        assert parse_answer("Tier 4 (100+ employees)") == ""


class TestUnanswered:
    @pytest.mark.parametrize("raw", ["", "   ", "\n\n", "I cannot determine this."])
    def test_returns_empty_string(self, raw):
        assert parse_answer(raw) == ""


class TestInstructionEcho:
    """A response is evidence about the model only where it is the model's own words.

    Retries prepend/append an escalation instruction that ends "...A, B, C, D, E, or F.".
    Reasoning models sometimes repeat it verbatim, and the bare-capital rule then scored
    that trailing F as the answer. Six scored rows in the medium tier were decided this
    way -- every one of them predicted F -- so the echo is removed before matching.
    """

    def test_regression_echo_only_is_unanswered(self):
        raw = ("Without specific rates provided, it's impossible to determine the "
               "correct answer from the given options. " + RETRY_PROMPT_SUFFIX)
        assert parse_answer(raw) == ""

    def test_prefix_echo_is_unanswered(self):
        assert parse_answer(RETRY_PROMPT_PREFIX) == ""

    def test_real_answer_survives_a_trailing_echo(self):
        assert parse_answer("The answer is C. " + RETRY_PROMPT_SUFFIX) == "C"

    def test_answer_after_echo_still_parses(self):
        assert parse_answer(RETRY_PROMPT_PREFIX + "\n\nMy answer: E") == "E"

    @pytest.mark.parametrize("echo", [
        "Final answer must be exactly one character only: A, B, C, D, E, or F.",
        "Reply with exactly one character only: A B C D E F",
        "final answer must be exactly one character only: a, b, c, d, e, or f",
    ])
    def test_echo_variants_are_stripped(self, echo):
        assert "F" not in strip_instruction_echo(echo).upper()

    def test_a_genuine_option_list_is_not_stripped(self):
        raw = "The options are A, B, C, D, E, or F but I choose D"
        assert parse_answer(raw) == "D"

