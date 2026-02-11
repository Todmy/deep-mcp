"""LLM-as-judge — evaluate predicted answers against gold standard."""

from __future__ import annotations

import json
import logging

from rlm_research.llm import LLMClient

log = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, a gold (correct) answer, and a predicted answer, \
determine whether the predicted answer is semantically equivalent to the gold answer.

Rules:
- The predicted answer does NOT need to match word-for-word.
- Minor formatting differences (e.g. "42" vs "42 years") are acceptable.
- Partial answers that contain the gold answer are correct.
- If the predicted answer contains the correct information among other text, it is correct.
- If the predicted answer is factually wrong or contradicts the gold answer, it is incorrect.
- If the predicted answer says "I don't know" or equivalent, it is incorrect.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Respond with ONLY a JSON object (no markdown fences):
{{"correct": true/false, "reasoning": "one sentence explanation"}}"""


async def judge_answer(
    predicted: str,
    gold: str,
    question: str,
    llm: LLMClient,
) -> dict:
    """Use LLM to judge whether predicted answer matches gold answer.

    Returns dict with keys: correct (bool), reasoning (str).
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        predicted=predicted[:2000],  # Truncate to avoid token bloat
    )

    response = await llm.generate(
        messages=[{"role": "user", "content": prompt}],
        depth=1,  # Use sub_model (cheap) for judging
    )

    try:
        # Strip markdown fences if LLM wraps response
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return {
            "correct": bool(result.get("correct", False)),
            "reasoning": str(result.get("reasoning", "")),
        }
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        log.warning("Judge parse error: %s — response: %s", exc, response.content[:200])
        return {"correct": False, "reasoning": f"Judge parse error: {exc}"}
