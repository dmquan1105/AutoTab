import json
from pathlib import Path
from types import SimpleNamespace

from autotab.config import load_config
from autotab.qa.pipeline import QAPipeline
from autotab.qa.sheetbrain.core.agent import SheetBrain
from autotab.qa.sheetbrain.modules.execution import ExecutionModule
from autotab.qa.sheetbrain.modules.validation import ValidationModule


def test_sheetbrain_runs_without_understanding_context() -> None:
    captured: dict[str, str] = {}

    class FakeExecution:
        def run(self, user_question: str) -> dict:
            captured["user_question"] = user_question
            return {
                "success": True,
                "answer": "Revenue is 10.",
                "total_turns": 1,
                "execution_summary": {},
                "conversation_history": [],
            }

    agent = SheetBrain.__new__(SheetBrain)
    agent.config = SimpleNamespace(
        max_turns=1,
        enable_validation=False,
        include_diagnostics=True,
    )
    agent.execution_module = FakeExecution()
    agent.validation_module = None

    result = agent.run("What is revenue?", enable_validation=False)

    assert captured["user_question"] == "What is revenue?"
    assert "understanding_output" not in result


def test_qa_pipeline_runs_sheetbrain_without_exploration(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSheetBrain:
        def __init__(self, **kwargs) -> None:
            captured["agent_kwargs"] = kwargs

        def run(self, query: str, **kwargs) -> dict:
            captured["run_kwargs"] = kwargs
            return {"success": True, "answer": "Revenue is 10."}

    monkeypatch.setattr("autotab.qa.pipeline.SheetBrain", FakeSheetBrain)
    config = load_config("config.example.yaml")
    config["runtime"]["artifact_root"] = str(tmp_path)

    output = QAPipeline(config).run("book.xlsx", "What is revenue?")

    assert captured["agent_kwargs"]["excel_path"] == "book.xlsx"
    assert captured["agent_kwargs"]["config"].extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert captured["run_kwargs"] == {"enable_validation": True}
    assert json.loads(output.read_text(encoding="utf-8"))["answer"] == "Revenue is 10."


def test_sheetbrain_prompts_omit_preloaded_workbook_context() -> None:
    client = SimpleNamespace()
    execution = ExecutionModule(client, "qwen", {}, {})
    validation = ValidationModule(client, "qwen")

    execution_prompt = execution._create_initial_user_prompt("What is revenue?")["content"]
    validation_prompt = validation._create_validation_prompt(
        {"success": True, "answer": "10", "conversation_history": []},
        "What is revenue?",
    )[0]["content"]

    assert "Sheet Content" not in execution_prompt
    assert "Understanding Context" not in execution_prompt
    assert "EXCEL DATA CONTEXT" not in validation_prompt


def test_old_qa_config_is_removed() -> None:
    assert "qa" not in load_config("config.example.yaml")


def test_sheetbrain_model_calls_disable_thinking() -> None:
    calls: list[dict] = []
    message = SimpleNamespace(content="**Thought:** done\n\nFinal Answer: 250")

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    execution = ExecutionModule(client, "qwen", {}, {}, extra_body)
    validation = ValidationModule(client, "qwen", extra_body)

    execution._get_llm_response(max_retries=1)
    validation._get_llm_response([], max_retries=1)

    assert [call["extra_body"] for call in calls] == [extra_body, extra_body]
