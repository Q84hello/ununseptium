"""Tests for CLI module."""

from __future__ import annotations

from click.testing import CliRunner

from ununseptium.cli.main import cli


class TestCLI:
    """Test CLI commands."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Ununseptium" in result.output

    def test_info_command(self):
        """Test info command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Ununseptium" in result.output

    def test_doctor_command(self):
        """Test doctor diagnostics command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])

        assert result.exit_code == 0
        assert "Diagnostic" in result.output or "Library import" in result.output

    def test_debug_mode(self):
        """Test debug mode flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--debug", "info"])

        # Should work even with debug
        assert result.exit_code == 0


class TestAuditCommands:
    """Test audit CLI commands."""

    def test_audit_help(self):
        """Test audit command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["audit", "--help"])

        assert result.exit_code == 0
        assert "audit" in result.output.lower()


class TestVerifyCommands:
    """Test verify CLI commands."""

    def test_verify_help(self):
        """Test verify command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])

        assert result.exit_code == 0
        assert "verify" in result.output.lower() or "Verification" in result.output


class TestScreenCommands:
    """Test screen CLI commands."""

    def test_screen_help(self):
        """Test screen command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["screen", "--help"])

        assert result.exit_code == 0
        assert "screen" in result.output.lower()


class TestModelCommands:
    """Test model CLI commands."""

    def test_model_help(self):
        """Test model command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "--help"])

        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_model_list(self):
        """Test model list command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
