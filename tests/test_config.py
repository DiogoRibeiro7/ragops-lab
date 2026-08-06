from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from ragops_lab.config import ProjectPaths, RuntimeSettings


def test_runtime_settings_defaults_are_valid() -> None:
    settings = RuntimeSettings()

    assert settings.environment == "local"
    assert settings.random_seed >= 0
    assert isinstance(settings.paths, ProjectPaths)


def test_project_paths_accept_path_objects() -> None:
    paths = ProjectPaths(
        data_dir=Path("data"), artifact_dir=Path("artifacts"), model_dir=Path("models")
    )

    assert paths.data_dir == Path("data")
    assert paths.artifact_dir == Path("artifacts")
    assert paths.model_dir == Path("models")


def test_runtime_settings_can_load_environment_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RAGOPS_ENVIRONMENT", "ci")
    monkeypatch.setenv("RAGOPS_CHUNK_PATH", "custom/chunks.jsonl")
    monkeypatch.setenv("RAGOPS_TRACE_PATH", "custom/traces.jsonl")
    monkeypatch.setenv("RAGOPS_API_MAX_REQUEST_BYTES", "2048")
    monkeypatch.setenv("RAGOPS_API_MAX_TOP_K", "7")
    monkeypatch.setenv("RAGOPS_API_MAX_QUERY_CHARS", "120")

    settings = RuntimeSettings.from_env()

    assert settings.environment == "ci"
    assert settings.paths.chunk_path == Path("custom/chunks.jsonl")
    assert settings.paths.trace_path == Path("custom/traces.jsonl")
    assert settings.api_max_request_bytes == 2048
    assert settings.api_max_top_k == 7
    assert settings.api_max_query_chars == 120


def test_runtime_settings_reject_invalid_environment_integers(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGOPS_API_MAX_TOP_K", "many")

    try:
        RuntimeSettings.from_env()
    except ValueError as exc:
        assert "RAGOPS_API_MAX_TOP_K must be an integer" in str(exc)
    else:
        raise AssertionError("Expected invalid integer environment variable to fail.")


def test_runtime_settings_reject_invalid_environment_booleans(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGOPS_DEBUG", "sometimes")

    try:
        RuntimeSettings.from_env()
    except ValueError as exc:
        assert "RAGOPS_DEBUG must be a boolean value" in str(exc)
    else:
        raise AssertionError("Expected invalid boolean environment variable to fail.")
