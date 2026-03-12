"""Tests for workflow stage orchestration order and failure handling."""

from __future__ import annotations

from pathlib import Path

from orchestration.workflow import WorkflowStage, run_stage_sequence


def test_run_stage_sequence_executes_stages_in_order(tmp_path: Path) -> None:
    execution_order: list[str] = []

    def _make_stage(name: str) -> WorkflowStage:
        def _runner() -> dict[str, str]:
            execution_order.append(name)
            return {"marker": f"done_{name}"}

        return WorkflowStage(name=name, description=f"stage {name}", runner=_runner)

    stages = (_make_stage("one"), _make_stage("two"), _make_stage("three"))
    result = run_stage_sequence(
        stages=stages,
        log_dir=tmp_path / "logs",
        workflow_log_path=tmp_path / "logs" / "workflow.log",
    )

    assert result.success is True
    assert result.failed_stage_name is None
    assert execution_order == ["one", "two", "three"]
    assert [record.name for record in result.records] == ["one", "two", "three"]
    assert all(record.status == "ok" for record in result.records)
    assert (tmp_path / "logs" / "01_one.log").exists()
    assert (tmp_path / "logs" / "02_two.log").exists()
    assert (tmp_path / "logs" / "03_three.log").exists()


def test_run_stage_sequence_stops_after_failure(tmp_path: Path) -> None:
    execution_order: list[str] = []

    def _ok_stage() -> dict[str, str]:
        execution_order.append("ok")
        return {"status": "ok"}

    def _failing_stage() -> dict[str, str]:
        execution_order.append("fail")
        raise RuntimeError("intentional failure")

    def _never_stage() -> dict[str, str]:
        execution_order.append("never")
        return {"status": "unexpected"}

    stages = (
        WorkflowStage(name="ok", description="first", runner=_ok_stage),
        WorkflowStage(name="fail", description="second", runner=_failing_stage),
        WorkflowStage(name="never", description="third", runner=_never_stage),
    )
    result = run_stage_sequence(
        stages=stages,
        log_dir=tmp_path / "logs",
        workflow_log_path=tmp_path / "logs" / "workflow.log",
    )

    assert result.success is False
    assert result.failed_stage_name == "fail"
    assert execution_order == ["ok", "fail"]
    assert len(result.records) == 2
    assert result.records[0].status == "ok"
    assert result.records[1].status == "failed"
    assert result.records[1].details["error_type"] == "RuntimeError"
    assert "intentional failure" in result.records[1].details["error_message"]

    failing_stage_log = (tmp_path / "logs" / "02_fail.log").read_text(encoding="utf-8")
    assert "status=failed" in failing_stage_log
    assert "error_type=RuntimeError" in failing_stage_log
