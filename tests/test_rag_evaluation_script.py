from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/evaluate_rag.py")

spec = importlib.util.spec_from_file_location("evaluate_rag", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
evaluate_rag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate_rag
spec.loader.exec_module(evaluate_rag)


def test_rag_evaluation_regression_gate_writes_artifacts(tmp_path: Path) -> None:
    summary, cases = evaluate_rag.run_evaluation(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        chunks_path=tmp_path / "chunks.jsonl",
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
    )
    output_dir = tmp_path / "evaluation"
    evaluate_rag.write_artifacts(summary, cases, output_dir)

    persisted_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary.passed
    assert persisted_summary["passed"] is True
    assert persisted_summary["case_count"] == len(cases)
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "cases.csv").exists()
    assert (output_dir / "cases.json").exists()
