"""Host DNA filtering using bowtie2 or bwa."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.core.exceptions import AnalysisError, ConfigurationError, ToolNotFoundError
from pipeline.core.utils import check_tool, run_command


class HostFilter:
    """Remove host-derived reads by mapping against a reference genome.

    Args:
        config: Host filtering configuration section.
        output_dir: Root output directory for filtered reads.
        logger: Logger instance.
        threads: Number of CPU threads to pass to the alignment tool.
    """

    def __init__(
        self,
        config,
        output_dir: Path,
        logger: logging.Logger,
        threads: int = 4,
    ) -> None:
        self.config = config
        self.output_dir = output_dir / "host_filtered"
        self.log = logger
        self.threads = threads
        self._stats: List[Dict] = []

    def run(self, samples) -> list:
        """Filter host reads from all samples.

        Args:
            samples: List of Sample objects.

        Returns:
            List of Sample objects pointing to host-filtered FASTQ files.

        Raises:
            ConfigurationError: If reference_index is not configured.
            ToolNotFoundError: If the alignment tool is not available.
        """
        if not self.config.reference_index:
            raise ConfigurationError(
                "host_filtering.reference_index is not set. "
                "Provide the path to a bowtie2 (or bwa) index prefix in your config."
            )

        tool = self._select_tool()
        self.log.info("Host filtering with %s against %s", tool, self.config.reference_index)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filtered = []
        for sample in samples:
            if tool == "bowtie2":
                filtered.append(self._run_bowtie2(sample))
            else:
                filtered.append(self._run_bwa(sample))
        return filtered

    def _select_tool(self) -> str:
        preferred = getattr(self.config, "tool", "bowtie2")
        avail, _ = check_tool(preferred)
        if avail:
            return preferred
        for tool in ["bowtie2", "bwa"]:
            avail, _ = check_tool(tool)
            if avail:
                return tool
        raise ToolNotFoundError("bowtie2")

    def _run_bowtie2(self, sample) -> object:
        """Map reads with bowtie2 and extract unmapped (microbial) reads."""
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        sam_file = sample_dir / f"{sample.sample_id}.sam"

        cmd = [
            "bowtie2",
            "-x", self.config.reference_index,
            self.config.sensitivity,
            "--threads", str(self.threads),
            "-S", str(sam_file),
        ]

        out_r2: Optional[Path] = None
        if sample.is_paired and sample.reverse is not None:
            out_r1 = sample_dir / f"{sample.sample_id}_filtered1.fastq.gz"
            out_r2 = sample_dir / f"{sample.sample_id}_filtered2.fastq.gz"
            cmd += [
                "-1", str(sample.forward),
                "-2", str(sample.reverse),
                "--un-conc-gz", str(sample_dir / f"{sample.sample_id}_filtered%.fastq.gz"),
            ]
        else:
            out_r1 = sample_dir / f"{sample.sample_id}_R1_filtered.fastq.gz"
            cmd += [
                "-U", str(sample.forward),
                "--un-gz", str(out_r1),
            ]

        try:
            result = run_command(cmd, self.log, check=True, capture=True)
            stats = self._parse_bowtie2_stats(result.stderr, sample.sample_id)
        except Exception as exc:
            raise AnalysisError("host_filtering", f"bowtie2 failed for {sample.sample_id}: {exc}") from exc

        self._stats.append(stats)

        from pipeline.modules.input_detection import Sample
        return Sample(
            sample_id=sample.sample_id,
            forward=out_r1,
            reverse=out_r2,
            is_paired=out_r2 is not None,
        )

    def _parse_bowtie2_stats(self, stderr: str, sample_id: str) -> Dict:
        """Extract alignment statistics from bowtie2 stderr output."""
        total = 0
        aligned = 0
        for line in (stderr or "").splitlines():
            m = re.search(r"(\d+) reads; of these:", line)
            if m:
                total = int(m.group(1))
            m = re.search(r"(\d+) \(.+\) aligned", line)
            if m:
                aligned += int(m.group(1))

        host_reads = aligned
        microbial = total - host_reads
        pct_host = (host_reads / total * 100) if total > 0 else 0.0
        return {
            "sample_id": sample_id,
            "total_reads": total,
            "host_reads": host_reads,
            "pct_host": round(pct_host, 2),
            "microbial_reads": microbial,
        }

    def _run_bwa(self, sample) -> object:
        """Map reads with bwa mem and extract unmapped (microbial) reads.

        Uses a subprocess pipeline:
            bwa mem → samtools view (unmapped) → samtools fastq
        For paired-end data the unmapped filter is ``-f 12 -F 256`` (both mates
        unmapped, exclude secondary), so only concordantly unmapped pairs are
        written.  For single-end data ``-f 4`` selects unmapped reads.

        Raises:
            ToolNotFoundError: If samtools is not available.
            AnalysisError: If any step in the pipeline fails.
        """
        # samtools is required to extract unmapped reads from the SAM stream
        samtools_avail, _ = check_tool("samtools")
        if not samtools_avail:
            raise ToolNotFoundError(
                "samtools",
                extra_message="samtools is required alongside bwa to extract unmapped reads.",
            )

        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        out_r1 = sample_dir / f"{sample.sample_id}_R1_filtered.fastq.gz"
        out_r2: Optional[Path] = None

        # ── bwa mem command ──────────────────────────────────────────────────
        bwa_cmd = [
            "bwa", "mem",
            "-t", str(self.threads),
            self.config.reference_index,
            str(sample.forward),
        ]
        if sample.is_paired and sample.reverse:
            bwa_cmd.append(str(sample.reverse))
            out_r2 = sample_dir / f"{sample.sample_id}_R2_filtered.fastq.gz"

        # ── samtools view: keep only unmapped reads ──────────────────────────
        if sample.is_paired and sample.reverse:
            # -f 12  → both mates unmapped
            # -F 256 → exclude secondary alignments
            view_cmd = ["samtools", "view", "-b", "-f", "12", "-F", "256"]
        else:
            # -f 4 → read unmapped
            view_cmd = ["samtools", "view", "-b", "-f", "4"]

        # ── samtools fastq: convert BAM to FASTQ ─────────────────────────────
        if sample.is_paired and sample.reverse:
            fastq_cmd = [
                "samtools", "fastq",
                "-1", str(out_r1),
                "-2", str(out_r2),
                "-0", "/dev/null",   # discard singletons
                "-s", "/dev/null",
                "-",
            ]
        else:
            fastq_cmd = [
                "samtools", "fastq",
                "-0", str(out_r1),   # single-end output
                "-",
            ]

        try:
            p_bwa = subprocess.Popen(
                bwa_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            p_view = subprocess.Popen(
                view_cmd,
                stdin=p_bwa.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Allow p_bwa to receive SIGPIPE if p_view exits early
            if p_bwa.stdout:
                p_bwa.stdout.close()

            p_fastq = subprocess.Popen(
                fastq_cmd,
                stdin=p_view.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if p_view.stdout:
                p_view.stdout.close()

            _, fastq_stderr = p_fastq.communicate()
            p_view.wait()
            p_bwa.wait()

            for proc, name in [(p_bwa, "bwa"), (p_view, "samtools view"), (p_fastq, "samtools fastq")]:
                if proc.returncode not in (0, None):
                    raise subprocess.CalledProcessError(proc.returncode, name)

        except subprocess.CalledProcessError as exc:
            raise AnalysisError(
                "host_filtering",
                f"bwa/samtools pipeline failed for {sample.sample_id}: {exc}",
            ) from exc
        except Exception as exc:
            raise AnalysisError(
                "host_filtering",
                f"bwa pipeline error for {sample.sample_id}: {exc}",
            ) from exc

        self._stats.append({
            "sample_id": sample.sample_id,
            "total_reads": 0,      # bwa stderr parsing not implemented
            "host_reads": 0,
            "pct_host": 0.0,
            "microbial_reads": 0,
        })

        from pipeline.modules.input_detection import Sample
        return Sample(
            sample_id=sample.sample_id,
            forward=out_r1,
            reverse=out_r2,
            is_paired=out_r2 is not None,
        )

    def get_statistics(self) -> pd.DataFrame:
        """Return host filtering statistics for all samples.

        Returns:
            DataFrame with columns: total_reads, host_reads, pct_host, microbial_reads.
        """
        if not self._stats:
            return pd.DataFrame()
        return pd.DataFrame(self._stats).set_index("sample_id")
