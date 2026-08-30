"""Taxonomic profiling using Kraken2 or MetaPhlAn."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.core.exceptions import AnalysisError, ConfigurationError, ToolNotFoundError
from pipeline.core.utils import check_tool, run_command

logger = logging.getLogger(__name__)

KRAKEN2_RANK_MAP = {
    "U": "unclassified", "R": "root", "D": "domain", "K": "kingdom",
    "P": "phylum", "C": "class", "O": "order", "F": "family",
    "G": "genus", "S": "species",
}


class TaxonomicProfiler:
    """Profile sample taxonomy using Kraken2 or MetaPhlAn.

    Args:
        config: Taxonomic profiling configuration section.
        output_dir: Root output directory.
        logger: Logger instance.
        threads: Number of CPU threads to pass to the profiling tool.
    """

    def __init__(
        self,
        config,
        output_dir: Path,
        logger: logging.Logger,
        threads: int = 4,
    ) -> None:
        self.config = config
        self.output_dir = output_dir / "taxonomy"
        self.log = logger
        self.threads = threads
        # sample_id -> {level -> relative_abundance_series}
        self._profiles: Dict[str, Dict[str, pd.Series]] = {}

    def run(self, samples) -> Dict:
        """Run taxonomic profiling on all samples.

        Returns:
            Dict containing abundance matrices at each taxonomic level.
        """
        if not self.config.database:
            raise ConfigurationError(
                "taxonomic_profiling.database is not configured. "
                "Set the path to your Kraken2 or MetaPhlAn database."
            )

        tool = self._select_tool()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            self.log.info("Profiling %s with %s", sample.sample_id, tool)
            if tool == "kraken2":
                self._run_kraken2(sample)
            else:
                self._run_metaphlan(sample)

        return self._compile_results()

    def _select_tool(self) -> str:
        preferred = getattr(self.config, "tool", "kraken2")
        avail, _ = check_tool(preferred)
        if avail:
            return preferred
        for tool in ["kraken2", "metaphlan"]:
            avail, _ = check_tool(tool)
            if avail:
                return tool
        raise ToolNotFoundError("kraken2")

    def _run_kraken2(self, sample) -> None:
        """Run Kraken2 and parse the report."""
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        report = sample_dir / f"{sample.sample_id}.kraken2.report"
        output = sample_dir / f"{sample.sample_id}.kraken2.out"

        cmd = [
            "kraken2",
            "--db", self.config.database,
            "--report", str(report),
            "--output", str(output),
            "--confidence", str(self.config.confidence),
            "--threads", str(self.threads),
        ]
        if sample.is_paired:
            cmd += ["--paired", str(sample.forward), str(sample.reverse)]
        else:
            cmd.append(str(sample.forward))

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("taxonomy", f"Kraken2 failed for {sample.sample_id}: {exc}") from exc

        self._profiles[sample.sample_id] = self._parse_kraken2_report(report)

    def _parse_kraken2_report(self, report_path: Path) -> Dict[str, pd.Series]:
        """Parse a Kraken2 report file into per-level abundance Series.

        Kraken2 report columns: pct, clade_reads, taxon_reads, rank, taxid, name
        """
        level_data: Dict[str, Dict[str, float]] = {lv: {} for lv in self.config.levels}

        try:
            with open(report_path) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 6:
                        continue
                    pct = float(parts[0].strip())
                    rank_code = parts[3].strip()
                    name = parts[5].strip()
                    rank = KRAKEN2_RANK_MAP.get(rank_code, "").lower()
                    if rank in level_data:
                        level_data[rank][name] = pct / 100.0
        except Exception as exc:
            self.log.warning("Could not parse Kraken2 report %s: %s", report_path, exc)

        return {lv: pd.Series(data, dtype=float) for lv, data in level_data.items()}

    def _run_metaphlan(self, sample) -> None:
        """Run MetaPhlAn and parse output."""
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        output = sample_dir / f"{sample.sample_id}.metaphlan.txt"

        reads = [str(sample.forward)]
        if sample.is_paired and sample.reverse:
            reads.append(str(sample.reverse))

        cmd = [
            "metaphlan",
            ",".join(reads),
            "--input_type", "fastq",
            "--output_file", str(output),
            "--nproc", str(self.threads),
        ]
        if self.config.database:
            cmd += ["--bowtie2db", self.config.database]

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("taxonomy", f"MetaPhlAn failed for {sample.sample_id}: {exc}") from exc

        self._profiles[sample.sample_id] = self._parse_metaphlan_output(output)

    def _parse_metaphlan_output(self, output_path: Path) -> Dict[str, pd.Series]:
        """Parse MetaPhlAn output into per-level abundance Series."""
        level_data: Dict[str, Dict[str, float]] = {lv: {} for lv in self.config.levels}
        level_markers = {
            "k__": "kingdom", "p__": "phylum", "c__": "class",
            "o__": "order", "f__": "family", "g__": "genus", "s__": "species",
        }
        try:
            with open(output_path) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    clade = parts[0]
                    abundance = float(parts[1]) / 100.0
                    taxon = clade.split("|")[-1]
                    for prefix, level in level_markers.items():
                        if taxon.startswith(prefix):
                            name = taxon[len(prefix):]
                            if level in level_data:
                                level_data[level][name] = abundance
                            break
        except Exception as exc:
            self.log.warning("Could not parse MetaPhlAn output %s: %s", output_path, exc)

        return {lv: pd.Series(data, dtype=float) for lv, data in level_data.items()}

    def _compile_results(self) -> Dict:
        """Build abundance matrices for each taxonomic level."""
        matrices = {}
        for level in self.config.levels:
            frames = {}
            for sample_id, profile in self._profiles.items():
                series = profile.get(level, pd.Series(dtype=float))
                frames[sample_id] = series
            if frames:
                mat = pd.DataFrame(frames).fillna(0.0)
                mat.index.name = "taxon"
                matrices[level] = mat
        return {"abundance_matrices": matrices, "levels": self.config.levels}

    def get_abundance_matrix(self, level: str) -> pd.DataFrame:
        """Return relative abundance matrix for a given taxonomic level.

        Args:
            level: Taxonomic level (e.g., 'species', 'genus').

        Returns:
            DataFrame with taxa as rows and samples as columns.
        """
        results = self._compile_results()
        return results["abundance_matrices"].get(level, pd.DataFrame())

    def get_dominant_taxa(self, level: str, n: int = 10) -> pd.DataFrame:
        """Return the top N most abundant taxa averaged across samples."""
        mat = self.get_abundance_matrix(level)
        if mat.empty:
            return pd.DataFrame()
        mean_abund = mat.mean(axis=1).sort_values(ascending=False)
        return mat.loc[mean_abund.head(n).index]

    def get_rare_taxa(self, level: str, threshold: float = 0.001) -> pd.DataFrame:
        """Return taxa with mean abundance below the threshold."""
        mat = self.get_abundance_matrix(level)
        if mat.empty:
            return pd.DataFrame()
        mean_abund = mat.mean(axis=1)
        return mat.loc[mean_abund[mean_abund < threshold].index]

    def get_core_microbiome(
        self, level: str, prevalence: float = 0.5, abundance: float = 0.001
    ) -> pd.DataFrame:
        """Return taxa present at >= abundance in >= prevalence fraction of samples."""
        mat = self.get_abundance_matrix(level)
        if mat.empty:
            return pd.DataFrame()
        n_samples = mat.shape[1]
        above_threshold = (mat >= abundance).sum(axis=1) / n_samples
        core_taxa = above_threshold[above_threshold >= prevalence].index
        return mat.loc[core_taxa]
