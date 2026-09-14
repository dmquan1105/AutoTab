"""Strict parsing for QA model responses."""

from __future__ import annotations

import re

from .types import Action, FinalAnswer


class ResponseFormatError(ValueError):
    """Raised when a model response violates the QA response contract."""


_ACTION = re.compile(
    r"\AThought:\s*(?P<thought>.+?)\s*\nAction:\s*```python\s*\n" r"(?P<code>.*?)\n```\s*\Z",
    re.DOTALL,
)
_FINAL = re.compile(
    r"\AThought:\s*(?P<thought>.+?)\s*\nFinal Answer:\s*(?P<answer>.+?)\s*\Z",
    re.DOTALL,
)


def parse_response(response: str) -> Action | FinalAnswer:
    """Parse exactly one action or final answer from a model response.

    Args:
        response: Raw text returned by the model.

    Returns:
        The parsed action or final answer.

    Raises:
        ResponseFormatError: If the response does not match exactly one allowed form.
    """
    value = response.strip()
    action = _ACTION.fullmatch(value)
    final = _FINAL.fullmatch(value)
    if action:
        if value.count("```") != 2 or "Final Answer:" in value:
            raise ResponseFormatError("An action must contain exactly one code block")
        code = action.group("code").strip()
        if not code:
            raise ResponseFormatError("The Python action cannot be empty")
        return Action(action.group("thought").strip(), code)
    if final:
        if "Action:" in value or "```" in value:
            raise ResponseFormatError("A final answer cannot contain an action")
        return FinalAnswer(final.group("thought").strip(), final.group("answer").strip())
    raise ResponseFormatError(
        "Use exactly 'Thought:' followed by either one 'Action:' Python block or 'Final Answer:'"
    )
