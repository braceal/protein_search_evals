"""Tests for the public package and command names."""

from __future__ import annotations

from typer.testing import CliRunner

from genslm_embeddings.cli import app


def test_cli_help() -> None:
    """Expose the command-line application through its entry point."""
    result = CliRunner().invoke(app, ['--help'])

    assert result.exit_code == 0
    assert 'merge' in result.stdout
