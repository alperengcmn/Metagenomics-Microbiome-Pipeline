"""General utility functions for the metagenomics pipeline."""
from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Union


def run_command(
    cmd: list[str],
    logger: logging.Logger,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run an external command as a subprocess.

    Args:
        cmd: Command and arguments as a list of strings.
        logger: Logger instance for command output.
        check: Raise CalledProcessError if the command returns non-zero exit code.
        capture: Capture stdout/stderr instead of streaming to console.

    Returns:
        CompletedProcess instance.
    """
    logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
    )
    if capture and result.stdout:
        logger.debug("stdout: %s", result.stdout.strip())
    if capture and result.stderr:
        logger.debug("stderr: %s", result.stderr.strip())
    return result


def check_tool(tool_name: str) -> tuple[bool, str]:
    """Check whether an external tool is available on PATH.

    Args:
        tool_name: Name of the executable to check.

    Returns:
        Tuple of (available: bool, version_string: str).
        version_string is empty when the tool is unavailable.
    """
    executable = shutil.which(tool_name)
    if executable is None:
        return False, ""

    version_flags = ["--version", "-version", "version"]
    for flag in version_flags:
        try:
            result = subprocess.run(
                [tool_name, flag],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                return True, output.splitlines()[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    return True, "(version unknown)"


def check_required_tools(tools: list[str]) -> dict[str, tuple[bool, str]]:
    """Check availability of multiple tools.

    Args:
        tools: List of tool names to check.

    Returns:
        Dict mapping tool name to (available, version_string).
    """
    return {tool: check_tool(tool) for tool in tools}


def count_reads(fastq_path: Path) -> int:
    """Count the number of reads in a FASTQ file.

    Works with plain and gzip-compressed files. Each read occupies 4 lines.

    Args:
        fastq_path: Path to the FASTQ file.

    Returns:
        Number of reads.
    """
    opener = gzip.open if str(fastq_path).endswith(".gz") else open
    line_count = 0
    with opener(fastq_path, "rb") as fh:  # type: ignore[call-overload]
        for _ in fh:
            line_count += 1
    return line_count // 4


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numerator by denominator, returning default when denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator == 0.

    Returns:
        Result of division or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_number(n: Union[int, float]) -> str:
    """Format a number in a human-readable abbreviated form.

    Args:
        n: Number to format.

    Returns:
        String like '1.2M', '450K', '999', etc.
    """
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}"


def create_output_dir(path: Path) -> Path:
    """Create an output directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
