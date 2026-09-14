"""Versioned prompts for workbook QA."""

from __future__ import annotations

QA_PROMPT_VERSION = "qa.v1"


def system_prompt(workbook_description: str) -> str:
    """Build the QA system prompt.

    Args:
        workbook_description: Safe worksheet names and dimensions.

    Returns:
        The strict QA instructions supplied to the model.
    """
    return f"""You are a workbook QA agent. Ground every answer in tool observations.
Workbook worksheets: {workbook_description}

Only these read-only tools exist:
- load_dataframe(sheet_name=None): load one worksheet using its first row as DataFrame columns.
- inspect_range(range_ref, sheet_name=None): return values from one A1 range as rows.

Python runs in a restricted sandbox. Imports, files, network, processes, dynamic execution,
workbook mutation, and non-allowlisted calls are forbidden. A bare tool call produces no
observation, so wrap every tool result needed for analysis in print(...). Use exact worksheet
names.

Return exactly one form and no other text:
Thought: <reasoning>
Action: ```python
<one Python code block>
```

or:
Thought: <reasoning>
Final Answer: <answer grounded in observations>"""


def initial_prompt(question: str, evidence: str | None = None) -> str:
    """Build the initial user turn.

    Args:
        question: The user's workbook question.
        evidence: Optional exploration evidence.

    Returns:
        A prompt containing the untrusted question and optional evidence.
    """
    evidence_text = evidence if evidence else "None provided."
    return f"Question:\n{question}\n\nOptional exploration evidence:\n{evidence_text}"


def conversation_prompt(system: str, turns: list[tuple[str, str]]) -> str:
    """Flatten the QA conversation for the existing completion client."""
    sections = [f"System:\n{system}"]
    sections.extend(f"{role}:\n{content}" for role, content in turns)
    return "\n\n".join(sections)
