"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    # Field order matters: callers may build EvalResult positionally, so the
    # required fields come first and the optional ones keep their defaults.
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


# A result passes only when every answer-side score reaches PASS_THRESHOLD.
# A score below FAILURE_TYPE_THRESHOLD names the failure type (Task 2).
PASS_THRESHOLD: float = 0.5
FAILURE_TYPE_THRESHOLD: float = 0.3


def _clamp(score: float) -> float:
    """Keep a score inside [0.0, 1.0]."""
    return max(0.0, min(1.0, score))


def _token_coverage(target: set[str], source: set[str]) -> float:
    """Fraction of ``target`` tokens that also appear in ``source``.

    Shared by every overlap metric: each one asks "how much of X is covered
    by Y" and only differs in which text is the denominator. An empty target
    has nothing to cover, so it scores 1.0 (and avoids dividing by zero).
    """
    if not target:
        return 1.0
    return _clamp(len(target & source) / len(target))


def _mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 for an empty list so empty benchmarks do not crash."""
    return sum(values) / len(values) if values else 0.0


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        return _token_coverage(_tokenize(answer), _tokenize(context))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _token_coverage(_tokenize(question), _tokenize(answer))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _token_coverage(_tokenize(expected), _tokenize(answer))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        return _token_coverage(_tokenize(expected), union_tokens)

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevant_seen = 0
        precision_sum = 0.0
        for rank, chunk in enumerate(contexts, start=1):
            chunk_coverage = _token_coverage(expected_tokens, _tokenize(chunk))
            if chunk_coverage >= relevance_threshold:
                relevant_seen += 1
                # Precision@k counted only at ranks that hold a relevant chunk.
                precision_sum += relevant_seen / rank

        if relevant_seen == 0:
            return 0.0
        return _clamp(precision_sum / relevant_seen)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)
        passed = min(faithfulness, relevance, completeness) >= PASS_THRESHOLD

        failure_type: str | None = None
        if not passed:
            if faithfulness < FAILURE_TYPE_THRESHOLD:
                failure_type = "hallucination"
            elif relevance < FAILURE_TYPE_THRESHOLD:
                failure_type = "irrelevant"
            elif completeness < FAILURE_TYPE_THRESHOLD:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
                retrieved_contexts=list(contexts or []),
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    query_tokens = _tokenize(query)
    # sorted() is stable, so chunks with equal overlap keep the retriever order.
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_SCORE: float = 0.5
LENIENCY_THRESHOLD: float = 0.8
SEVERITY_THRESHOLD: float = 0.3
# The first response must beat the average of the others by more than this
# margin before the batch is flagged for positional bias.
POSITIONAL_BIAS_MARGIN: float = 0.1


class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    @staticmethod
    def _build_prompt(question: str, answer: str, rubric: dict[str, Any]) -> str:
        """Build a judge prompt that asks for one 0-1 score per rubric criterion."""
        criteria = "\n".join(
            f"- {name}: {description}" for name, description in rubric.items()
        )
        example = json.dumps({name: 0.0 for name in rubric})
        return (
            "You are an impartial evaluator of OrbitTech Store customer-support "
            "answers.\n"
            "Score the answer on each criterion from 0.0 (fails) to 1.0 (fully "
            "meets). Judge content only: do not reward length, style, or "
            "position, and penalise claims that are not supported.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Criteria:\n{criteria}\n\n"
            "Respond with only a JSON object mapping each criterion name to "
            f"its score, for example: {example}"
        )

    @staticmethod
    def _parse_scores(raw_response: str, rubric: dict[str, Any]) -> dict[str, float]:
        """Read criterion scores from the judge output.

        Accepts a bare JSON object or one embedded in surrounding text (LLMs
        often add prose or code fences). Any criterion that cannot be read
        falls back to DEFAULT_JUDGE_SCORE.
        """
        parsed: Any = None
        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", raw_response or "", re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict):
            return {criterion: DEFAULT_JUDGE_SCORE for criterion in rubric}

        scores: dict[str, float] = {}
        for criterion in rubric:
            value = parsed.get(criterion)
            # bool is a subclass of int, so exclude it explicitly.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[criterion] = _clamp(float(value))
            else:
                scores[criterion] = DEFAULT_JUDGE_SCORE
        return scores

    @staticmethod
    def _mean_item_score(item: dict[str, Any]) -> float | None:
        """Average criterion score of one judged response, or None if it has none."""
        values = list(item.get("scores", {}).values())
        return _mean(values) if values else None

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        prompt = self._build_prompt(question, answer, rubric)
        raw_response = self.judge_llm_fn(prompt)
        return {
            "scores": self._parse_scores(raw_response, rubric),
            "reasoning": raw_response,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        all_scores = [
            score
            for item in scores_batch
            for score in item.get("scores", {}).values()
        ]
        if not all_scores:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }
        average_score = _mean(all_scores)

        # Positional bias needs a first response plus at least one other to
        # compare against; a batch of one cannot show a position effect.
        item_means = [self._mean_item_score(item) for item in scores_batch]
        first_mean = item_means[0]
        other_means = [mean for mean in item_means[1:] if mean is not None]
        positional_bias = (
            first_mean is not None
            and bool(other_means)
            and first_mean - _mean(other_means) > POSITIONAL_BIAS_MARGIN
        )

        return {
            "positional_bias": positional_bias,
            "leniency_bias": average_score > LENIENCY_THRESHOLD,
            "severity_bias": average_score < SEVERITY_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

ANSWER_METRICS: tuple[str, ...] = ("faithfulness", "relevance", "completeness")
REGRESSION_THRESHOLD: float = 0.05
# Averages such as 0.9 - 0.85 come out as 0.05000000000000004 in floating
# point; the tolerance keeps an exact 0.05 drop from counting as "more than".
_FLOAT_TOLERANCE: float = 1e-9


def _count_failure_types(failures: list[EvalResult]) -> dict[str, int]:
    """Count failures per failure_type; untyped failures are 'unclassified'."""
    return dict(
        Counter(failure.failure_type or "unclassified" for failure in failures)
    )


class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            # Keep the caller's pair so metadata such as the record id survives.
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        recall_scores = [
            result.context_recall
            for result in results
            if result.context_recall is not None
        ]
        precision_scores = [
            result.context_precision
            for result in results
            if result.context_precision is not None
        ]
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_faithfulness": _mean([result.faithfulness for result in results]),
            "avg_relevance": _mean([result.relevance for result in results]),
            "avg_completeness": _mean([result.completeness for result in results]),
            "avg_context_recall": _mean(recall_scores) if recall_scores else None,
            "avg_context_precision": (
                _mean(precision_scores) if precision_scores else None
            ),
            "failure_types": _count_failure_types(
                [result for result in results if not result.passed]
            ),
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions
        """
        comparison: dict[str, Any] = {}
        regressions: list[str] = []
        for metric in ANSWER_METRICS:
            new_avg = _mean([getattr(result, metric) for result in new_results])
            baseline_avg = _mean(
                [getattr(result, metric) for result in baseline_results]
            )
            comparison[f"new_avg_{metric}"] = new_avg
            comparison[f"baseline_avg_{metric}"] = baseline_avg
            if baseline_avg - new_avg > REGRESSION_THRESHOLD + _FLOAT_TOLERANCE:
                regressions.append(metric)

        comparison["regressions"] = regressions
        comparison["passed"] = not regressions
        return comparison

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            result
            for result in results
            if min(result.faithfulness, result.relevance, result.completeness)
            < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

ROOT_CAUSE_RETRIEVAL = "Context is missing or irrelevant — improve retrieval"
ROOT_CAUSE_PROMPT = "Answer does not address the question — improve prompt clarity"
ROOT_CAUSE_GENERATION = (
    "Answer is missing key information — increase context window or improve generation"
)
ROOT_CAUSE_MULTIPLE = "Multiple issues detected — review full pipeline"

# One concrete fix per failure type, keyed in lowercase.
SUGGESTION_BY_FAILURE_TYPE: dict[str, str] = {
    "hallucination": (
        "Constrain the generation prompt to answer only from retrieved OrbitTech "
        "policy text and add a claim-level grounding check that rejects "
        "unsupported amounts, dates, or promises"
    ),
    "irrelevant": (
        "Add intent classification and query rewriting so the assistant answers "
        "the policy the customer asked about (e.g. warranty vs. returns)"
    ),
    "incomplete": (
        "Require answers to list every condition, fee, and exception in the "
        "retrieved policy (restocking fee, policy version, exclusions) and add "
        "few-shot examples of complete answers"
    ),
    "off_topic": (
        "Add hybrid retrieval (BM25 + embeddings) with a reranker so the "
        "generator sees the right policy chunk first"
    ),
    "refusal": (
        "Relax over-strict guardrails so the assistant refuses only requests "
        "that 00_system_scope.md marks as out of scope"
    ),
}
# Pipeline-wide fixes used to reach the minimum of three suggestions.
GENERAL_SUGGESTIONS: tuple[str, ...] = (
    "Increase top-k or tune chunk size/overlap so multi-document policy "
    "evidence is retrieved together (raises Context Recall)",
    "Rerank retrieved chunks so the most relevant policy text is ranked first "
    "(raises Context Precision)",
    "Add every failed case to the golden dataset and run run_regression() on "
    "each prompt, model, or retrieval change",
)
MIN_SUGGESTIONS = 3
MISSING_SUGGESTION = "Pending — analyse with 5 Whys before choosing a fix"


def _markdown_cell(text: str) -> str:
    """Make text safe for one Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        return _count_failure_types(failures)

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"

        Rule: when all three scores are below PASS_THRESHOLD no single stage
        explains the failure, so the whole pipeline needs review. Otherwise
        the lowest score points to the stage to fix; ties resolve in the
        same order as failure_type (faithfulness, relevance, completeness).
        """
        scores = {
            ROOT_CAUSE_RETRIEVAL: failure.faithfulness,
            ROOT_CAUSE_PROMPT: failure.relevance,
            ROOT_CAUSE_GENERATION: failure.completeness,
        }
        if all(score < PASS_THRESHOLD for score in scores.values()):
            return ROOT_CAUSE_MULTIPLE
        return min(scores, key=scores.__getitem__)

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".
        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures):
            failure_id = f"F{index + 1:03d}"
            record_id = (failure.qa_pair.metadata or {}).get("id")
            if record_id:
                failure_id = f"{failure_id} ({record_id})"
            if index < len(suggestions):
                suggestion = suggestions[index]
            else:
                # Shorter suggestion list: fall back to the fix for this type.
                suggestion = SUGGESTION_BY_FAILURE_TYPE.get(
                    (failure.failure_type or "").lower(), MISSING_SUGGESTION
                )
            cells = [
                failure_id,
                failure.failure_type or "unclassified",
                self.find_root_cause(failure),
                suggestion,
                "Open",
            ]
            lines.append(
                "| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |"
            )
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        # Most frequent failure type first: fixing it helps the most cases.
        type_counts = Counter(
            (failure.failure_type or "").lower() for failure in failures
        )
        suggestions: list[str] = []
        for failure_type, _ in type_counts.most_common():
            suggestion = SUGGESTION_BY_FAILURE_TYPE.get(failure_type)
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)

        for suggestion in GENERAL_SUGGESTIONS:
            if len(suggestions) >= MIN_SUGGESTIONS:
                break
            if suggestion not in suggestions:
                suggestions.append(suggestion)
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
