"""Tests for the CLI commands."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline.main import cli


class TestCLIHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MetaPipe" in result.output or "metapipe" in result.output.lower()

    def test_run_help_contains_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--metadata" in result.output
        assert "--config" in result.output
        assert "--threads" in result.output
        assert "--dry-run" in result.output
        assert "--verbose" in result.output

    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestCheckTools:
    def test_check_tools_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["check-tools"])
        assert result.exit_code == 0
        # Should show tool names
        assert "fastqc" in result.output or "Available" in result.output or "Not found" in result.output


class TestGenerateConfig:
    def test_generates_yaml_file(self, tmp_path: Path):
        runner = CliRunner()
        out_file = tmp_path / "test_config.yaml"
        result = runner.invoke(cli, ["generate-config", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "pipeline" in content

    def test_default_output_path(self, tmp_path: Path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["generate-config"])
            assert result.exit_code == 0
            assert Path("pipeline_config.yaml").exists()


class TestRunValidation:
    def test_run_requires_input(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--output", "/tmp/out"])
        assert result.exit_code != 0
        assert "input" in result.output.lower() or "missing" in result.output.lower()

    def test_run_requires_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--input", "/tmp"])
        assert result.exit_code != 0

    def test_run_nonexistent_input_fails(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--input", str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_dry_run_with_valid_input(self, synthetic_fastq_dir: Path, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--input", str(synthetic_fastq_dir),
            "--output", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "3" in result.output
