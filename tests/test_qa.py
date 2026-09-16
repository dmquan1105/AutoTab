import json
from pathlib import Path
from types import SimpleNamespace

from autotab.config import load_config
from autotab.qa.pipeline import QAPipeline
from autotab.qa.sheetbrain.core.agent import SheetBrain
from autotab.qa.sheetbrain.modules.execution import ExecutionModule
from autotab.qa.sheetbrain.modules.validation import ValidationModule


def test_sheetbrain_uses_exploration_as_understanding_output(tmp_path: Path) -> None:
    exploration = tmp_path / "exploration.md"
    exploration_text = "# Exploration\n\n## revenue\n\n- ranges: `Data!B2`\n"
    exploration.write_text(exploration_text, encoding="utf-8")
    captured: dict[str, str] = {}

    class FakeExecution:
        def run(self, understanding_output: str, user_question: str) -> dict:
            captured["understanding_output"] = understanding_output
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
        enable_understanding=True,
        include_diagnostics=True,
    )
    agent.exploration_path = exploration
    agent.execution_module = FakeExecution()
    agent.validation_module = None

    result = agent.run("What is revenue?", enable_validation=False)

    assert captured["understanding_output"] == exploration_text
    assert result["understanding_output"] == exploration_text


def test_qa_pipeline_runs_exploration_then_sheetbrain(tmp_path: Path, monkeypatch) -> None:
    exploration = tmp_path / "run" / "exploration" / "exploration.md"
    exploration.parent.mkdir(parents=True)
    exploration.write_text("# Exploration\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_explore(self, workbooks: list[str], query: str) -> Path:
        captured["workbooks"] = workbooks
        captured["query"] = query
        return exploration

    class FakeSheetBrain:
        def __init__(self, **kwargs) -> None:
            captured["agent_kwargs"] = kwargs

        def run(self, query: str, **kwargs) -> dict:
            captured["run_kwargs"] = kwargs
            return {"success": True, "answer": "Revenue is 10."}

    monkeypatch.setattr("autotab.qa.pipeline.ExplorationPipeline.run", fake_explore)
    monkeypatch.setattr("autotab.qa.pipeline.SheetBrain", FakeSheetBrain)
    config = load_config("config.example.yaml")

    output = QAPipeline(config).run("book.xlsx", "What is revenue?")

    assert captured["workbooks"] == ["book.xlsx"]
    assert captured["query"] == "What is revenue?"
    assert captured["agent_kwargs"]["exploration_path"] == str(exploration)
    assert captured["agent_kwargs"]["config"].extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert captured["run_kwargs"] == {
        "enable_validation": True,
        "enable_understanding": True,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["answer"] == "Revenue is 10."


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
    execution = ExecutionModule(client, "qwen", {}, {}, "", extra_body)
    validation = ValidationModule(client, "qwen", "", extra_body)

    execution._get_llm_response(max_retries=1)
    validation._get_llm_response([], max_retries=1)

    assert [call["extra_body"] for call in calls] == [extra_body, extra_body]
