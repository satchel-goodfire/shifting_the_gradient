"""Portable judge backends for LLM-as-judge calibration.

Why this module exists (all claims verified empirically 2026-07-29): the
production judge, :class:`grad_shift.evaluation.judge.OpenAiJudge`, scores a
response by requesting a single completion token with ``top_logprobs=20`` and
taking a probability-weighted mean over the integers 0-100. That protocol does
not port to current models:

* ``gpt-5.5`` and ``gpt-5.6-*`` reject ``logprobs`` with a 400.
* ``gpt-5.4-mini`` accepts ``logprobs`` but caps ``top_logprobs`` at 5.
* The Anthropic Messages API exposes no logprobs at all.
* Of the open-weight candidates only Llama-3.3 tokenizes every integer 0-100 as
  a single token. Qwen3 and Mistral-Small tokenize multi-digit numbers one digit
  at a time (10/101 single-token), so the single-token trick is unavailable
  there and they would need the ``0_10`` scale instead.

A calibration study therefore has to vary *which model judges* and *how the
score is extracted* independently. Otherwise a cross-judge difference is
uninterpretable: it could be the judge or it could be the extraction method.
Every backend here exposes the same :meth:`score_many` interface so the two axes
can be crossed -- in particular by running one model under both extraction
methods as a method control.

:class:`OpenAiJudge` is deliberately left unmodified so scores already published
from it stay reproducible.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .judge import OpenAiJudge

logger = logging.getLogger(__name__)

# Reuse the production aggregation so the logprob path has a single definition.
_aggregate_0_100 = OpenAiJudge._aggregate_0_100_score

_INT_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class JudgeSpec:
    """Declarative description of one judge configuration.

    A "judge" in the calibration study is a (model, extraction method) pair, not
    just a model -- see the module docstring for why the two must be separable.

    Attributes:
        id: Short identifier used in output column names and filenames.
        backend: ``"openai"`` or ``"vllm"``.
        model: Model name for the backend.
        extraction: ``"logprob"`` (probability-weighted mean over 0-100) or
            ``"sampled"`` (mean of ``n_samples`` sampled integers).
        top_logprobs: Candidates to request. 20 matches production; ``gpt-5.4``
            models cap this at 5, which truncates the distribution's tail.
        n_samples: Samples per response for ``"sampled"``. Requested via the
            API's ``n`` parameter, so K samples cost one prompt, not K.
        temperature: Sampling temperature. ``None`` omits the parameter.
        reasoning_effort: OpenAI reasoning models only. ``"none"`` keeps the
            response to a bare integer (~4 completion tokens).
        token_param: ``"max_tokens"`` (GPT-4.x) or ``"max_completion_tokens"``
            (GPT-5.x, which rejects ``max_tokens``).
        max_tokens: Completion-token budget. 1 suffices for the logprob path;
            sampled judges need headroom for the integer's digits.
        seed: Passed through when supported; best-effort determinism only.
    """

    id: str
    backend: str
    model: str
    extraction: str
    top_logprobs: int = 20
    n_samples: int = 1
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    token_param: str = "max_tokens"
    max_tokens: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.backend not in ("openai", "vllm"):
            raise ValueError(f"backend must be 'openai' or 'vllm', got {self.backend!r}")
        if self.extraction not in ("logprob", "sampled"):
            raise ValueError(
                f"extraction must be 'logprob' or 'sampled', got {self.extraction!r}"
            )
        if self.extraction == "sampled" and self.n_samples < 1:
            raise ValueError("n_samples must be >= 1 for sampled extraction")


# Registry for the calibration study. The first two entries are the same model
# under both extraction methods: that pair is the method control that lets a
# cross-judge difference be attributed to the judge rather than to extraction.
JUDGE_SPECS: Dict[str, JudgeSpec] = {
    # Production judge, replayed exactly (pipeline.py default judge_model).
    "gpt41mini_logprob": JudgeSpec(
        id="gpt41mini_logprob",
        backend="openai",
        model="gpt-4.1-mini-2025-04-14",
        extraction="logprob",
        top_logprobs=20,
        temperature=0.0,
        max_tokens=1,
        seed=0,
    ),
    # Method control: same model, sampled extraction.
    "gpt41mini_sampled": JudgeSpec(
        id="gpt41mini_sampled",
        backend="openai",
        model="gpt-4.1-mini-2025-04-14",
        extraction="sampled",
        n_samples=1,
        temperature=0.0,
        max_tokens=4,
        seed=0,
    ),
    # Newer generation, still logprob-based but tail-truncated at 5 candidates.
    "gpt54mini_logprob5": JudgeSpec(
        id="gpt54mini_logprob5",
        backend="openai",
        model="gpt-5.4-mini",
        extraction="logprob",
        top_logprobs=5,
        reasoning_effort="none",
        token_param="max_completion_tokens",
        max_tokens=16,
    ),
    # Frontier OpenAI model; logprobs unavailable, so sampled only.
    "gpt55_sampled": JudgeSpec(
        id="gpt55_sampled",
        backend="openai",
        model="gpt-5.5",
        extraction="sampled",
        n_samples=1,
        reasoning_effort="none",
        token_param="max_completion_tokens",
        max_tokens=16,
    ),
    # Open weights, and the only judge that reproduces the production protocol
    # exactly: Llama-3.3 tokenizes all of 0-100 as single tokens.
    "llama33_70b_logprob": JudgeSpec(
        id="llama33_70b_logprob",
        backend="vllm",
        model="meta-llama/Llama-3.3-70B-Instruct",
        extraction="logprob",
        top_logprobs=20,
        temperature=0.0,
        max_tokens=1,
    ),
}


def variant(spec_id: str, new_id: str, **overrides) -> JudgeSpec:
    """Derive a JudgeSpec from a registered one.

    Used for the repeat-sampling arm of the study, where the same judge is run
    at temperature 1 with ``n_samples>1`` to measure its own self-consistency
    ceiling -- between-judge agreement cannot exceed that ceiling.
    """
    base = JUDGE_SPECS[spec_id]
    return JudgeSpec(**{**base.__dict__, "id": new_id, **overrides})


def _parse_sampled_scores(texts: Sequence[Optional[str]]) -> List[int]:
    """Integers in ``[0, 100]`` parsed from sampled completions, in order."""
    vals = []
    for text in texts:
        if not text:
            continue
        match = _INT_RE.search(text)
        if match is None:
            continue
        value = int(match.group())
        if 0 <= value <= 100:
            vals.append(value)
    return vals


def _mean_sampled_score(texts: Sequence[Optional[str]]) -> Optional[float]:
    """Mean of the integers parsed from sampled completions.

    Returns None when no completion yields an integer in ``[0, 100]``, mirroring
    the production judge's convention that an unusable score is None rather than
    a silently wrong number.
    """
    vals = _parse_sampled_scores(texts)
    if not vals:
        return None
    return sum(vals) / len(vals)


class OpenAiBackendJudge:
    """Judge backed by the OpenAI chat-completions API.

    Args:
        spec: Judge configuration.
        prompt_template: Rubric with ``{question}``/``{answer}`` placeholders.
            Must be the same rubric the stored scores used, or the comparison
            measures prompt differences rather than judge differences.
    """

    def __init__(self, spec: JudgeSpec, prompt_template: str):
        if spec.backend != "openai":
            raise ValueError(f"{spec.id} is not an openai judge")
        self.spec = spec
        self.prompt_template = prompt_template

    def _request_kwargs(self) -> dict:
        spec = self.spec
        kwargs = {
            "model": spec.model,
            spec.token_param: spec.max_tokens,
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        if spec.seed is not None:
            kwargs["seed"] = spec.seed
        if spec.extraction == "logprob":
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = spec.top_logprobs
        elif spec.n_samples > 1:
            # One prompt, K completions: K samples at 1x the input-token cost.
            kwargs["n"] = spec.n_samples
        return kwargs

    async def _create_with_retry(self, client, **kwargs):
        """chat.completions.create with bounded backoff on transient errors.

        Same rationale as :meth:`OpenAiJudge._create_with_retry`: a single
        upstream 500 out of thousands of calls should not abort a sweep.
        """
        import openai

        delays = (1, 4, 16, None)
        for delay in delays:
            try:
                return await client.chat.completions.create(**kwargs)
            except (
                openai.InternalServerError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
            ) as exc:
                if delay is None:
                    raise
                logger.warning(
                    "Judge %s transient error (%s); retrying in %ss",
                    self.spec.id, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)

    async def _score_one(self, client, question: str, answer: str):
        """Return ``(score, per_sample_scores)``.

        ``per_sample_scores`` is empty for the logprob path (there is only one
        deterministic aggregate) and holds the individual parsed integers for the
        sampled path. Retaining them is what makes a within-judge reliability
        estimate possible: between-judge agreement cannot be interpreted without
        knowing how well a judge agrees with *itself*.
        """
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(question=question, answer=answer),
            }
        ]
        completion = await self._create_with_retry(
            client, messages=messages, **self._request_kwargs()
        )
        if self.spec.extraction == "logprob":
            try:
                top = completion.choices[0].logprobs.content[0].top_logprobs
            except (IndexError, AttributeError):
                return None, []
            score = _aggregate_0_100(
                {el.token: float(math.exp(el.logprob)) for el in top}
            )
            return score, []
        samples = _parse_sampled_scores(
            [choice.message.content for choice in completion.choices]
        )
        score = (sum(samples) / len(samples)) if samples else None
        return score, samples

    async def score_many(
        self,
        pairs: Sequence[Tuple[str, str]],
        max_concurrent: int = 40,
        progress: bool = True,
        with_samples: bool = False,
    ):
        """Score ``(question, answer)`` pairs, preserving input order.

        Returns a list of scores, or ``(scores, per_sample_scores)`` when
        ``with_samples`` is set.
        """
        client = OpenAiJudge._get_client()
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[Optional[float]] = [None] * len(pairs)
        samples: List[List[int]] = [[] for _ in pairs]

        async def run(idx: int, question: str, answer: str):
            async with semaphore:
                try:
                    results[idx], samples[idx] = await self._score_one(
                        client, question, answer
                    )
                except Exception as exc:  # noqa: BLE001 - one bad row must not kill a sweep
                    # Left as None and counted by the caller rather than
                    # silenced: a None is visible in the output as a gap.
                    logger.error("Judge %s failed on row %d: %s", self.spec.id, idx, exc)

        tasks = [run(i, q, a) for i, (q, a) in enumerate(pairs)]
        if progress:
            from tqdm import tqdm

            with tqdm(total=len(tasks), desc=f"judge:{self.spec.id}") as bar:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
        else:
            await asyncio.gather(*tasks)
        return (results, samples) if with_samples else results


class VllmLogprobJudge:
    """Local vLLM judge replicating the production logprob protocol.

    Only valid for models whose tokenizer maps every integer 0-100 to a single
    token (verified for Llama-3.3; false for Qwen3 and Mistral-Small). The
    constructor asserts this rather than silently producing scores aggregated
    over first-digits-only, which would look plausible and be wrong.
    """

    def __init__(
        self,
        spec: JudgeSpec,
        prompt_template: str,
        tensor_parallel_size: int = 4,
        max_model_len: int = 4096,
        download_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
    ):
        if spec.backend != "vllm":
            raise ValueError(f"{spec.id} is not a vllm judge")
        self.spec = spec
        self.prompt_template = prompt_template

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model)
        multi = [
            n for n in range(101)
            if len(self.tokenizer.encode(str(n), add_special_tokens=False)) != 1
        ]
        if multi:
            raise ValueError(
                f"{spec.model} does not tokenize 0-100 as single tokens "
                f"({len(multi)} multi-token, e.g. {multi[:5]}). The single-token "
                f"logprob protocol cannot be replicated on this model; use the "
                f"'0_10' scale or sampled extraction instead."
            )

        from vllm import LLM

        llm_kwargs = dict(
            model=spec.model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )
        if download_dir:
            llm_kwargs["download_dir"] = download_dir
        self.llm = LLM(**llm_kwargs)

    def score_many(
        self, pairs: Sequence[Tuple[str, str]], **_ignored
    ) -> List[Optional[float]]:
        """Score all pairs in one batched vLLM call, preserving input order."""
        from vllm import SamplingParams

        prompts = [
            self.tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": self.prompt_template.format(question=q, answer=a),
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q, a in pairs
        ]
        params = SamplingParams(
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature or 0.0,
            logprobs=self.spec.top_logprobs,
        )
        outputs = self.llm.generate(prompts, params)

        scores: List[Optional[float]] = []
        for out in outputs:
            step_logprobs = out.outputs[0].logprobs
            if not step_logprobs:
                scores.append(None)
                continue
            # vLLM keys by token id; decoded_token gives the surface string.
            dist = {
                (lp.decoded_token or "").strip(): float(math.exp(lp.logprob))
                for lp in step_logprobs[0].values()
            }
            scores.append(_aggregate_0_100(dist))
        return scores


def build_judge(spec: JudgeSpec, prompt_template: str, **backend_kwargs):
    """Instantiate the backend implementing ``spec``."""
    if spec.backend == "openai":
        return OpenAiBackendJudge(spec, prompt_template)
    return VllmLogprobJudge(spec, prompt_template, **backend_kwargs)
