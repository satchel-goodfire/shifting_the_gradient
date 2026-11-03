"""OpenAI-based judge for scoring model responses via logprob aggregation.

Replicates the scoring logic from persona_vectors_repo/judge.py. Uses the
OpenAI API to score responses on a 0-100 scale by requesting a single
completion token with logprobs and aggregating the probability-weighted scores.
"""

import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    """(key, value) of one ``.env`` line, or None for blanks/comments/others.

    Accepts the shell form ``export KEY=value`` as well as ``KEY=value``, so
    ``.env`` may be a file that a shell also sources.
    Surrounding quotes on the value are removed.
    """
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    if not key or " " in key:
        return None
    return key, value.strip().strip("'\"")


class OpenAiJudge:
    """Score model responses using OpenAI logprob aggregation.

    OpenAI models tokenize all integers from 0-100 as single tokens, so we can
    request exactly one completion token with ``top_logprobs=20`` and aggregate
    the probability-weighted scores.

    Args:
        model: OpenAI model name (e.g., ``"gpt-4.1-mini-2025-04-14"``).
        prompt_template: Template string with ``{question}`` and ``{answer}``
            placeholders.
        eval_type: Scoring method. One of ``"0_100"``, ``"0_10"``,
            ``"binary"``, ``"binary_text"``.
    """

    _client = None

    def __init__(
        self,
        model: str,
        prompt_template: str,
        eval_type: str = "0_100",
    ):
        valid_types = ("0_100", "0_10", "binary", "binary_text")
        if eval_type not in valid_types:
            raise ValueError(
                f"eval_type must be one of {valid_types}, got '{eval_type}'"
            )

        self.model = model
        self.prompt_template = prompt_template
        self.eval_type = eval_type

        self._aggregators = {
            "0_100": self._aggregate_0_100_score,
            "0_10": self._aggregate_0_10_score,
            "binary": self._aggregate_binary_score,
            "binary_text": self._aggregate_binary_text_score,
        }

    @classmethod
    def _get_client(cls):
        """Lazily create the AsyncOpenAI client.

        Loads ``.env`` from the project root if ``OPENAI_API_KEY`` is not
        already in the environment.
        """
        if cls._client is None:
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                cls._load_dotenv()
            from openai import AsyncOpenAI
            cls._client = AsyncOpenAI()
        return cls._client

    @staticmethod
    def _load_dotenv():
        """Load .env file from project root into os.environ."""
        import os
        from pathlib import Path
        # Walk up from this file to find .env
        current = Path(__file__).resolve().parent
        for _ in range(10):
            env_file = current / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        parsed = parse_dotenv_line(line)
                        if parsed is not None:
                            os.environ[parsed[0]] = parsed[1]
                return
            current = current.parent

    async def judge(self, **kwargs) -> Optional[float]:
        """Score a response.

        Args:
            **kwargs: Values to format into ``prompt_template`` (typically
                ``question`` and ``answer``).

        Returns:
            Score (float) or None if the judge refused / was uncertain.
        """
        messages = [
            {"role": "user", "content": self.prompt_template.format(**kwargs)}
        ]

        if self.eval_type == "binary_text":
            response_text = await self._query_full_text(messages)
            return self._aggregators[self.eval_type](response_text)
        else:
            logprobs = await self._logprob_probs(messages)
            return self._aggregators[self.eval_type](logprobs)

    async def _create_with_retry(self, client, **kwargs):
        """chat.completions.create with bounded backoff on transient errors.

        One raw 500/timeout out of the ~400 judge calls per eval otherwise
        aborts the whole checkpoint eval (seen 2026-07-28: repeated OpenAI
        server_error bursts killed eval shards). Non-transient errors and
        the final failed attempt still raise.
        """
        import asyncio
        import openai

        delays = (1, 4, 16, None)
        for delay in delays:
            try:
                return await client.chat.completions.create(**kwargs)
            except (openai.InternalServerError, openai.APITimeoutError,
                    openai.APIConnectionError, openai.RateLimitError) as e:
                if delay is None:
                    raise
                logger.warning(
                    f"Judge API transient error ({type(e).__name__}); "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)

    async def _logprob_probs(self, messages: list) -> dict:
        """Request a single token with logprobs and return {token: prob}."""
        client = self._get_client()
        completion = await self._create_with_retry(
            client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError):
            return {}

        return {el.token: float(math.exp(el.logprob)) for el in logprobs}

    async def _query_full_text(self, messages: list) -> str:
        """Request a full text completion (for binary_text eval_type)."""
        client = self._get_client()
        completion = await self._create_with_retry(
            client,
            model=self.model,
            messages=messages,
            temperature=0,
            seed=0,
        )
        try:
            return completion.choices[0].message.content
        except (IndexError, AttributeError):
            return ""

    # ------------------------------------------------------------------
    # Score aggregation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_0_100_score(score: dict) -> Optional[float]:
        """Aggregate logprob scores on the 0-100 integer scale."""
        total = 0.0
        weighted_sum = 0.0
        for key, prob in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 100:
                weighted_sum += int_key * prob
                total += prob

        if total < 0.25:
            return None
        return weighted_sum / total

    @staticmethod
    def _aggregate_0_10_score(score: dict) -> Optional[float]:
        """Aggregate logprob scores on the 0-9 integer scale."""
        refusal_prob = score.get("REFUSAL", 0)
        if refusal_prob > max(score.get(str(i), 0) for i in range(10)):
            return None

        total = 0.0
        weighted_sum = 0.0
        for key, prob in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if 0 <= int_key <= 9:
                weighted_sum += int_key * prob
                total += prob

        if total < 0.25:
            return None
        return weighted_sum / total

    @staticmethod
    def _aggregate_binary_score(score: dict) -> Optional[float]:
        """Aggregate P(YES) / (P(YES) + P(NO))."""
        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None

        denominator = yes_prob + no_prob
        if denominator < 0.25:
            return None
        return yes_prob / denominator

    @staticmethod
    def _aggregate_binary_text_score(response_text: str) -> Optional[float]:
        """Extract binary score from full text response."""
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0.0
        elif "<answer>YES</answer>" in response_text:
            return 1.0
        return None

    async def __call__(self, **kwargs) -> Optional[float]:
        return await self.judge(**kwargs)
