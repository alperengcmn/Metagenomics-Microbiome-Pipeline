"""Quality control using FastQC and MultiQC."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.core.exceptions import AnalysisError, ToolNotFoundError
from pipeline.core.utils import check_tool, run_command

logger = logging.getLogger(__name__)


class QualityController:
    """Run FastQC/MultiQC quality control on FASTQ files.

    Args:
        config: QC configuration section.
        logger: Logger instance.
        threads: Number of CPU threads to pass to FastQC.
    """

    def __init__(
        self,
        config,
        logger: logging.Logger,
        threads: int = 4,
    ) -> None:
        self.config = config
        self.log = logger
        self.threads = threads

    def run_fastqc(self, samples, output_dir: Path) -> Path:
        """Run FastQC on all samples.

        Args:
            samples: List of Sample objects.
            output_dir: Directory for FastQC output.

        Returns:
            Path to the FastQC output directory.

        Raises:
            ToolNotFoundError: If fastqc is not available.
            AnalysisError: If FastQC exits with non-zero status.
        """
        available, version = check_tool("fastqc")
        if not available:
            raise ToolNotFoundError("fastqc")
        self.log.info("FastQC version: %s", version)

        output_dir.mkdir(parents=True, exist_ok=True)
        all_reads: List[str] = []
        for sample in samples:
            all_reads.extend(str(p) for p in sample.reads)

        cmd = [
            "fastqc",
            "--outdir", str(output_dir),
            "--threads", str(self.threads),
            "--quiet",
        ] + all_reads

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("quality_control", f"FastQC failed: {exc}") from exc

        self.log.info("FastQC complete. Results in %s", output_dir)
        return output_dir

    def run_multiqc(self, fastqc_dir: Path, output_dir: Path) -> Path:
        """Aggregate FastQC results with MultiQC.

        Args:
            fastqc_dir: Directory containing FastQC output files.
            output_dir: Directory for MultiQC output.

        Returns:
            Path to the MultiQC report.

        Raises:
            ToolNotFoundError: If multiqc is not available.
        """
        available, version = check_tool("multiqc")
        if not available:
            raise ToolNotFoundError("multiqc")
        self.log.info("MultiQC version: %s", version)

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "multiqc",
            str(fastqc_dir),
            "--outdir", str(output_dir),
            "--quiet",
            "--force",
        ]
        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("quality_control", f"MultiQC failed: {exc}") from exc

        report = output_dir / "multiqc_report.html"
        self.log.info("MultiQC report: %s", report)
        return report

    def parse_fastqc_summary(self, fastqc_zip_path: Path) -> Dict[str, object]:
        """Parse a FastQC ZIP archive to extract key QC metrics.

        Args:
            fastqc_zip_path: Path to a FastQC ``*_fastqc.zip`` file.

        Returns:
            Dictionary with keys: total_sequences, poor_quality_sequences,
            avg_quality, pct_gc, sequence_length.
        """
        metrics: Dict[str, object] = {
            "total_sequences": 0,
            "poor_quality_sequences": 0,
            "avg_quality": 0.0,
            "pct_gc": 0.0,
            "sequence_length": "",
        }

        try:
            with zipfile.ZipFile(fastqc_zip_path, "r") as zf:
                data_file = next(
                    n for n in zf.namelist() if n.endswith("fastqc_data.txt")
                )
                with zf.open(data_file) as fh:
                    content = fh.read().decode("utf-8")
        except Exception as exc:
            self.log.warning("Could not parse FastQC zip %s: %s", fastqc_zip_path, exc)
            return metrics

        for line in content.splitlines():
            if line.startswith("Total Sequences"):
                metrics["total_sequences"] = int(line.split("\t")[1])
            elif line.startswith("Sequences flagged as poor quality"):
                metrics["poor_quality_sequences"] = int(line.split("\t")[1])
            elif line.startswith("%GC"):
                metrics["pct_gc"] = float(line.split("\t")[1])
            elif line.startswith("Sequence length"):
                metrics["sequence_length"] = line.split("\t")[1]

        return metrics

    def generate_qc_report(
        self,
        pre_stats: Dict[str, Dict],
        post_stats: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Compare pre- and post-QC statistics.

        Args:
            pre_stats: Dict mapping sample_id to pre-QC metrics dict.
            post_stats: Dict mapping sample_id to post-QC metrics dict.

        Returns:
            DataFrame with one row per sample and columns for before/after comparison.
        """
        rows = []
        for sample_id in pre_stats:
            pre = pre_stats[sample_id]
            post = post_stats.get(sample_id, {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "pre_total_sequences": pre.get("total_sequences", 0),
                    "post_total_sequences": post.get("total_sequences", 0),
                    "pre_avg_quality": pre.get("avg_quality", 0.0),
                    "post_avg_quality": post.get("avg_quality", 0.0),
                    "pre_pct_gc": pre.get("pct_gc", 0.0),
                    "post_pct_gc": post.get("pct_gc", 0.0),
                }
            )
        df = pd.DataFrame(rows).set_index("sample_id")
        return df
