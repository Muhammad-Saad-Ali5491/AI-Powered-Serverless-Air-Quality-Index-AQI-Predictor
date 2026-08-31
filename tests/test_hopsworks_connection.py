"""
Tests for src/features/feature_store.py's Hopsworks connection logic.

These guard against a real production bug: hopsworks.login() raises
"ExternalClientError: host cannot be of type NoneType, host is a
non-optional argument to connect to hopsworks from an external
environment" specifically when the HOPSWORKS_HOST environment variable is
PRESENT BUT EMPTY — which is exactly what GitHub Actions' `env:` blocks
produce when a repo variable was never configured (the var is still set,
just to an empty string, rather than being fully absent from os.environ).

The fix: always pass an explicit, non-empty `host` to hopsworks.login()
ourselves rather than relying on its internal env-var resolution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch, MagicMock

import pytest


def test_hopsworks_login_receives_explicit_host_when_config_host_is_empty(monkeypatch):
    """
    This is THE regression test: with HOPSWORKS_HOST unset (empty string,
    matching config.py's os.getenv(..., "") default), the login call must
    still receive a real, non-empty host string — never None and never "".
    """
    from src.features import feature_store as fs_module

    monkeypatch.setattr(fs_module.config, "HOPSWORKS_API_KEY", "fake_key")
    monkeypatch.setattr(fs_module.config, "HOPSWORKS_HOST", "")  # the exact GitHub Actions scenario
    monkeypatch.setattr(fs_module.config, "HOPSWORKS_PROJECT_NAME", "")

    mock_hopsworks = MagicMock()
    mock_project = MagicMock()
    mock_project.name = "test_project"
    mock_project.get_feature_store.return_value = MagicMock()
    mock_hopsworks.login.return_value = mock_project

    with patch.dict(sys.modules, {"hopsworks": mock_hopsworks}):
        store = fs_module.HopsworksFeatureStore()

    mock_hopsworks.login.assert_called_once()
    call_kwargs = mock_hopsworks.login.call_args.kwargs
    assert "host" in call_kwargs
    assert call_kwargs["host"], "host must never be empty/None — this is the exact bug that caused ExternalClientError"
    assert call_kwargs["host"] == "c.app.hopsworks.ai"
    assert call_kwargs["api_key_value"] == "fake_key"
    assert "project" not in call_kwargs  # not set when HOPSWORKS_PROJECT_NAME is empty


def test_hopsworks_login_respects_explicit_custom_host(monkeypatch):
    """Self-managed Hopsworks clusters must still be able to override the host."""
    from src.features import feature_store as fs_module

    monkeypatch.setattr(fs_module.config, "HOPSWORKS_API_KEY", "fake_key")
    monkeypatch.setattr(fs_module.config, "HOPSWORKS_HOST", "my.custom.hopsworks.server")
    monkeypatch.setattr(fs_module.config, "HOPSWORKS_PROJECT_NAME", "my_project")

    mock_hopsworks = MagicMock()
    mock_project = MagicMock()
    mock_project.name = "my_project"
    mock_project.get_feature_store.return_value = MagicMock()
    mock_hopsworks.login.return_value = mock_project

    with patch.dict(sys.modules, {"hopsworks": mock_hopsworks}):
        fs_module.HopsworksFeatureStore()

    call_kwargs = mock_hopsworks.login.call_args.kwargs
    assert call_kwargs["host"] == "my.custom.hopsworks.server"
    assert call_kwargs["project"] == "my_project"


def test_hopsworks_init_raises_without_api_key(monkeypatch):
    from src.features import feature_store as fs_module

    monkeypatch.setattr(fs_module.config, "HOPSWORKS_API_KEY", "")

    with pytest.raises(ValueError, match="HOPSWORKS_API_KEY"):
        fs_module.HopsworksFeatureStore()


def test_get_feature_store_falls_back_to_local_on_connection_failure(monkeypatch):
    """
    Even if the host is correct, a real network/auth failure must not
    crash the pipeline — it should fall back to the local store, exactly
    like the ExternalClientError case used to (before this fix) and
    exactly like it still should for any other connection problem.
    """
    from src.features import feature_store as fs_module

    monkeypatch.setattr(fs_module.config, "USE_HOPSWORKS", True)
    monkeypatch.setattr(fs_module.config, "HOPSWORKS_API_KEY", "fake_key")

    with patch.object(fs_module, "HopsworksFeatureStore", side_effect=RuntimeError("simulated connection failure")):
        store = fs_module.get_feature_store()

    assert isinstance(store, fs_module.LocalFeatureStore)
