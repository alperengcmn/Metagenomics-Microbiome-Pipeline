"""Command-line interface for the metagenomics pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="MetaPipe")
def cli() -> None:
    """MetaPipe — End-to-end metagenomics and microbiome analysis pipeline."""


@cli.command("run")
@click.option("--input", "-i", "input_dir", required=True, type=click.Path(exists=True),
              help="Input directory containing FASTQ files.")
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(),
              help="Output directory for results.")
@click.option("--metadata", "-m", "metadata_path", type=click.Path(exists=True), default=None,
              help="Sample metadata TSV file (sample_id column required).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="YAML configuration file (overrides defaults).")
@click.option("--threads", "-t", default=4, show_default=True,
              help="Number of CPU threads.")
@click.option("--no-host-filter", is_flag=True, default=False,
              help="Skip host DNA filtering step.")
@click.option("--no-functional", is_flag=True, default=False,
              help="Skip functional profiling.")
@click.option("--skip-modules", multiple=True,
              type=click.Choice(
                  ["qc", "preprocessing", "host-filter", "taxonomy",
                   "diversity", "statistics", "functional", "visualization"]
              ),
              help="Skip specific modules.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Validate inputs and configuration without running analysis.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable verbose logging.")
def run(
    input_dir: str,
    output_dir: str,
    metadata_path: Optional[str],
    config_path: Optional[str],
    threads: int,
    no_host_filter: bool,
    no_functional: bool,
    skip_modules: Tuple[str, ...],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run the full metagenomics analysis pipeline."""
    from pipeline.core.config import PipelineConfig
    from pipeline.core.logging_setup import setup_logging
    from pipeline.core.pipeline import Pipeline, PipelineStatus
    from pipeline.modules.input_detection import SampleDetector
    from pipeline.reporting.report_generator import ReportGenerator

    out_path = Path(output_dir)
    logger = setup_logging(out_path, verbose=verbose)

    # Load and merge config
    cfg_path = Path(config_path) if config_path else None
    config = PipelineConfig.merge(cfg_path)
    config.pipeline.threads = threads
    config.pipeline.verbose = verbose
    config.pipeline.dry_run = dry_run

    skip = set(skip_modules)
    if no_host_filter:
        skip.add("host-filter")
    if no_functional:
        skip.add("functional")

    logger.info("MetaPipe v1.0.0 starting")
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Config hash: %s", config.config_hash())

    # ── Pre-flight environment validation ────────────────────────────────────
    pipeline = Pipeline(config, out_path, logger)
    env_check = pipeline.validate_environment(skip_modules=list(skip))

    _print_preflight_table(env_check["checks"])

    if env_check["errors"]:
        console.print()
        for err in env_check["errors"]:
            console.print(f"  [bold red]✗[/] {err}")

    if not env_check["all_critical_pass"] and not dry_run:
        console.print()
        console.print(
            "[bold red]Pre-flight check failed.[/] "
            "Fix the errors above before running the pipeline."
        )
        sys.exit(1)

    if env_check["warnings"]:
        console.print()
        for w in env_check["warnings"]:
            console.print(f"  [yellow]⚠[/] {w}")

    # Detect samples
    detector = SampleDetector(Path(input_dir), logger)
    samples = detector.detect()

    # Load metadata
    metadata = None
    if metadata_path:
        metadata = detector.load_metadata(Path(metadata_path))

    if dry_run:
        console.print(
            f"\n[bold green]Dry run complete.[/] Found {len(samples)} samples. "
            "Configuration is valid. No analysis performed."
        )
        return

    # ── Run pipeline ─────────────────────────────────────────────────────────
    results = pipeline.run(samples, metadata, skip_modules=list(skip))

    # ── Generate report ───────────────────────────────────────────────────────
    reporter = ReportGenerator(logger)
    report_path = reporter.generate(results, out_path)

    # ── Module status table ───────────────────────────────────────────────────
    console.print()
    _print_module_status_table(pipeline.module_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    pipeline_status: PipelineStatus = results.get("pipeline_status", PipelineStatus.COMPLETED)
    _print_final_status(pipeline_status, len(samples), report_path, config)


def _print_preflight_table(checks) -> None:
    """Render the pre-flight environment check table.

    Each check is a tuple of (module, tool, available, version, note).
    """
    table = Table(title="Pre-flight Check", show_header=True, header_style="bold blue")
    table.add_column("Module", style="bold")
    table.add_column("Tool")
    table.add_column("Status", justify="center")
    table.add_column("Version")
    table.add_column("Note")

    for module, tool, available, version, note in checks:
        status_str = "[green]✓[/]" if available else "[red]✗[/]"
        table.add_row(module, tool, status_str, version or "—", note or "")

    console.print()
    console.print(table)


def _print_module_status_table(module_results) -> None:
    """Render the per-module execution status table."""
    from pipeline.core.pipeline import ModuleStatus

    table = Table(title="Module Execution Summary", show_header=True, header_style="bold blue")
    table.add_column("Module", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Notes")

    status_styles = {
        ModuleStatus.SUCCESS: "[green]✓ SUCCESS[/]",
        ModuleStatus.FAILED:  "[red]✗ FAILED[/]",
        ModuleStatus.SKIPPED: "[dim]— SKIPPED[/]",
        ModuleStatus.WARNING: "[yellow]⚠ WARNING[/]",
    }

    for mr in module_results:
        status_str = status_styles.get(mr.status, str(mr.status))
        note = mr.error if mr.status == ModuleStatus.FAILED else mr.note
        table.add_row(mr.name, status_str, note or "")

    console.print(table)


def _print_final_status(pipeline_status, n_samples, report_path, config) -> None:
    """Print the final pipeline completion banner with accurate status."""
    from pipeline.core.pipeline import PipelineStatus

    status_color = {
        PipelineStatus.COMPLETED: "green",
        PipelineStatus.COMPLETED_WITH_WARNINGS: "yellow",
        PipelineStatus.FAILED: "red",
    }.get(pipeline_status, "white")

    console.print()
    console.rule(f"[bold {status_color}]Pipeline {pipeline_status.value}[/]")
    console.print(f"  Samples processed : {n_samples}")
    if report_path and report_path.exists():
        console.print(f"  Report            : {report_path}")
    console.print(f"  Config hash       : {config.config_hash()}")
    console.print()

    if pipeline_status == PipelineStatus.FAILED:
        sys.exit(1)


@cli.command("check-tools")
def check_tools() -> None:
    """Check availability of all external bioinformatics tools."""
    from pipeline.core.utils import check_required_tools

    tools = [
        "fastqc", "multiqc", "fastp", "trimmomatic",
        "bowtie2", "bwa", "samtools",
        "kraken2", "bracken", "metaphlan",
        "humann",
    ]
    results = check_required_tools(tools)

    table = Table(title="Tool Availability", show_header=True)
    table.add_column("Tool", style="bold")
    table.add_column("Status")
    table.add_column("Version")

    for tool, (available, version) in results.items():
        status = "[green]Available[/]" if available else "[red]Not found[/]"
        table.add_row(tool, status, version or "—")

    console.print(table)


@cli.command("generate-config")
@click.argument("output_path", default="pipeline_config.yaml")
def generate_config(output_path: str) -> None:
    """Write the default configuration to OUTPUT_PATH (default: pipeline_config.yaml)."""
    from pipeline.core.config import PipelineConfig

    config = PipelineConfig.from_defaults()
    path = Path(output_path)
    config.to_yaml(path)
    console.print(f"[green]Default configuration written to:[/] {path}")


if __name__ == "__main__":
    cli()
