"""Read preprocessing using fastp or trimmomatic."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.core.exceptions import AnalysisError, ToolNotFoundError
from pipeline.core.utils import check_tool, run_command

logger = logging.getLogger(__name__)


class Preprocessor:
    """Trim and filter reads using fastp or trimmomatic.

    Args:
        config: Preprocessing configuration section.
        output_dir: Root output directory for trimmed reads.
        logger: Logger instance.
        threads: Number of CPU threads to pass to the trimming tool.
    """

    def __init__(
        self,
        config,
        output_dir: Path,
        logger: logging.Logger,
        threads: int = 4,
    ) -> None:
        self.config = config
        self.output_dir = output_dir / "preprocessed"
        self.log = logger
        self.threads = threads
        self._stats: List[Dict] = []

    def run(self, samples) -> list:
        """Preprocess all samples.

        Args:
            samples: List of Sample objects.

        Returns:
            List of Sample objects pointing to trimmed FASTQ files.

        Raises:
            ToolNotFoundError: If no supported trimming tool is available.
        """
        tool = self._select_tool()
        self.log.info("Preprocessing reads with %s", tool)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trimmed_samples = []
        for sample in samples:
            if tool == "fastp":
                trimmed = self._run_fastp(sample)
            else:
                trimmed = self._run_trimmomatic(sample)
            trimmed_samples.append(trimmed)

        return trimmed_samples

    def _select_tool(self) -> str:
        """Return the first available trimming tool."""
        preferred = getattr(self.config, "tool", "fastp")
        available, _ = check_tool(preferred)
        if available:
            return preferred

        for tool in ["fastp", "trimmomatic"]:
            avail, _ = check_tool(tool)
            if avail:
                return tool

        raise ToolNotFoundError(
            "fastp",
            extra_message="Neither fastp nor trimmomatic is available. "
            "Install at least one to run preprocessing.",
        )

    def _run_fastp(self, sample) -> object:
        """Run fastp on a single sample.

        Args:
            sample: Sample object.

        Returns:
            Updated Sample pointing to trimmed output files.
        """
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        out_r1 = sample_dir / f"{sample.sample_id}_R1_trimmed.fastq.gz"
        json_report = sample_dir / f"{sample.sample_id}_fastp.json"
        html_report = sample_dir / f"{sample.sample_id}_fastp.html"

        cmd = [
            "fastp",
            "--in1", str(sample.forward),
            "--out1", str(out_r1),
            "--json", str(json_report),
            "--html", str(html_report),
            "--qualified_quality_phred", str(self.config.min_quality),
            "--length_required", str(self.config.min_length),
            "--n_base_limit", str(int(self.config.max_n_rate * 100)),
            "--thread", str(self.threads),
        ]

        out_r2: Optional[Path] = None
        if sample.is_paired and sample.reverse is not None:
            out_r2 = sample_dir / f"{sample.sample_id}_R2_trimmed.fastq.gz"
            cmd += ["--in2", str(sample.reverse), "--out2", str(out_r2)]

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("preprocessing", f"fastp failed for {sample.sample_id}: {exc}") from exc

        stats = self._parse_fastp_json(json_report, sample.sample_id)
        self._stats.append(stats)

        # Return updated sample
        from pipeline.modules.input_detection import Sample
        return Sample(
            sample_id=sample.sample_id,
            forward=out_r1,
            reverse=out_r2,
            is_paired=out_r2 is not None,
        )

    def _parse_fastp_json(self, json_path: Path, sample_id: str) -> Dict:
        """Parse fastp JSON report for statistics.

        Logs a warning when the JSON cannot be read rather than silently
        returning zeros, so unexpected failures are visible in the log.
        """
        try:
            with open(json_path) as fh:
                data = json.load(fh)
            summary = data.get("summary", {})
            before = summary.get("before_filtering", {})
            after = summary.get("after_filtering", {})
            in_reads = before.get("total_reads", 0)
            out_reads = after.get("total_reads", 0)
            removed = in_reads - out_reads
            pct_removed = (removed / in_reads * 100) if in_reads > 0 else 0.0
            return {
                "sample_id": sample_id,
                "input_reads": in_reads,
                "output_reads": out_reads,
                "removed_reads": removed,
                "pct_removed": round(pct_removed, 2),
            }
        except Exception as exc:
            self.log.warning(
                "Could not parse fastp JSON report for %s (%s: %s). "
                "Trimming statistics will be unavailable for this sample.",
                sample_id,
                type(exc).__name__,
                exc,
            )
            return {
                "sample_id": sample_id,
                "input_reads": 0,
                "output_reads": 0,
                "removed_reads": 0,
                "pct_removed": 0.0,
            }

    def _run_trimmomatic(self, sample) -> object:
        """Run trimmomatic on a single sample."""
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        out_r1 = sample_dir / f"{sample.sample_id}_R1_trimmed.fastq.gz"
        cmd = ["trimmomatic"]

        if sample.is_paired and sample.reverse is not None:
            out_r2 = sample_dir / f"{sample.sample_id}_R2_trimmed.fastq.gz"
            unpaired_r1 = sample_dir / f"{sample.sample_id}_R1_unpaired.fastq.gz"
            unpaired_r2 = sample_dir / f"{sample.sample_id}_R2_unpaired.fastq.gz"
            cmd += [
                "PE",
                "-threads", str(self.threads),
                str(sample.forward), str(sample.reverse),
                str(out_r1), str(unpaired_r1),
                str(out_r2), str(unpaired_r2),
            ]
        else:
            out_r2 = None
            cmd += [
                "SE",
                "-threads", str(self.threads),
                str(sample.forward), str(out_r1),
            ]

        cmd += [
            f"SLIDINGWINDOW:{self.config.window_size}:{self.config.min_quality}",
            f"MINLEN:{self.config.min_length}",
        ]

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("preprocessing", f"trimmomatic failed for {sample.sample_id}: {exc}") from exc

        self._stats.append({
            "sample_id": sample.sample_id,
            "input_reads": 0,
            "output_reads": 0,
            "removed_reads": 0,
            "pct_removed": 0.0,
        })

        from pipeline.modules.input_detection import Sample
        return Sample(
            sample_id=sample.sample_id,
            forward=out_r1,
            reverse=out_r2,
            is_paired=out_r2 is not None,
        )

    def get_statistics(self) -> pd.DataFrame:
        """Return trimming statistics for all processed samples.

        Returns:
            DataFrame with columns: input_reads, output_reads, removed_reads, pct_removed.
        """
        if not self._stats:
            return pd.DataFrame()
        df = pd.DataFrame(self._stats).set_index("sample_id")
        df.index.name = "sample_id"
        return df
