"""Tests for the public package and command names."""

from __future__ import annotations

import importlib.util

from typer.testing import CliRunner

import genslm_embeddings
from genslm_embeddings.cli import app


def test_import_package_uses_new_name() -> None:
    """Expose only the renamed Python import package."""
    assert genslm_embeddings.__name__ == 'genslm_embeddings'
    assert importlib.util.find_spec('protein_search_evals') is None


def test_cli_help() -> None:
    """Expose the command-line application through its entry point."""
    result = CliRunner().invoke(app, ['--help'])

    assert result.exit_code == 0
    assert 'merge' in result.stdout
