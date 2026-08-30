"""Functional profiling using HUMAnN3."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from pipeline.core.exceptions import AnalysisError, ToolNotFoundError
from pipeline.core.utils import check_tool, run_command

logger = logging.getLogger(__name__)


class FunctionalProfiler:
    """Perform functional profiling using HUMAnN3.

    If HUMAnN is not available, logs a clear warning and returns empty results
    so the pipeline continues gracefully.

    Args:
        config: Functional profiling configuration section.
        output_dir: Root output directory.
        logger: Logger instance.
        threads: Number of CPU threads to pass to HUMAnN.
    """

    def __init__(
        self,
        config,
        output_dir: Path,
        logger: logging.Logger,
        threads: int = 4,
    ) -> None:
        self.config = config
        self.output_dir = output_dir / "functional"
        self.log = logger
        self.threads = threads
        self._gene_families: Optional[pd.DataFrame] = None
        self._pathways: Optional[pd.DataFrame] = None

    def run(self, samples) -> Dict:
        """Run HUMAnN on all samples if available.

        Args:
            samples: List of Sample objects.

        Returns:
            Dict with gene_families and pathways DataFrames.
            Returns empty DataFrames if HUMAnN is not installed.
        """
        available, version = check_tool("humann")
        if not available:
            self.log.warning(
                "HUMAnN is not installed. Skipping functional profiling. "
                "Install with: conda install -c bioconda humann"
            )
            return self._empty_results()

        self.log.info("HUMAnN version: %s", version)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sample_gene_families = []
        sample_pathways = []

        for sample in samples:
            self.log.info("Functional profiling: %s", sample.sample_id)
            gf, pw = self._run_humann(sample)
            if gf is not None:
                sample_gene_families.append(gf)
            if pw is not None:
                sample_pathways.append(pw)

        if sample_gene_families:
            self._gene_families = pd.concat(sample_gene_families, axis=1)
            self._gene_families.index.name = "gene_family"

        if sample_pathways:
            self._pathways = pd.concat(sample_pathways, axis=1)
            self._pathways.index.name = "pathway"

        return {
            "gene_families": self._gene_families if self._gene_families is not None else pd.DataFrame(),
            "pathways": self._pathways if self._pathways is not None else pd.DataFrame(),
        }

    def _run_humann(self, sample) -> tuple:
        """Run HUMAnN on a single sample.

        Returns:
            Tuple of (gene_families_series, pathways_series).
        """
        sample_dir = self.output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "humann",
            "--input", str(sample.forward),
            "--output", str(sample_dir),
            "--threads", str(self.threads),
        ]
        if self.config.nucleotide_db:
            cmd += ["--nucleotide-database", self.config.nucleotide_db]
        if self.config.protein_db:
            cmd += ["--protein-database", self.config.protein_db]

        try:
            run_command(cmd, self.log, check=True, capture=True)
        except Exception as exc:
            raise AnalysisError("functional", f"HUMAnN failed for {sample.sample_id}: {exc}") from exc

        gf_file = next(sample_dir.glob("*_genefamilies.tsv"), None)
        pw_file = next(sample_dir.glob("*_pathabundance.tsv"), None)

        gf_series = None
        if gf_file and gf_file.exists():
            gf_series = self._parse_humann_table(gf_file, sample.sample_id)

        pw_series = None
        if pw_file and pw_file.exists():
            pw_series = self._parse_humann_table(pw_file, sample.sample_id)

        return gf_series, pw_series

    def _parse_humann_table(self, path: Path, sample_id: str) -> pd.Series:
        """Parse a HUMAnN output table into a Series.

        Args:
            path: Path to the HUMAnN TSV output file.
            sample_id: Sample identifier (becomes Series name).

        Returns:
            Pandas Series indexed by feature name.
        """
        try:
            df = pd.read_csv(path, sep="\t", comment="#", header=0)
            df.columns = ["feature", "abundance"]
            # Exclude stratified rows (contain '|')
            df = df[~df["feature"].str.contains(r"\|", na=False)]
            series = df.set_index("feature")["abundance"]
            series.name = sample_id
            return series
        except Exception as exc:
            self.log.warning("Could not parse HUMAnN table %s: %s", path, exc)
            return pd.Series(name=sample_id, dtype=float)

    def get_pathway_abundance(self) -> pd.DataFrame:
        """Return pathway abundance matrix (pathways × samples)."""
        return self._pathways if self._pathways is not None else pd.DataFrame()

    def get_gene_families(self) -> pd.DataFrame:
        """Return gene family abundance matrix (gene_families × samples)."""
        return self._gene_families if self._gene_families is not None else pd.DataFrame()

    def _empty_results(self) -> Dict:
        return {"gene_families": pd.DataFrame(), "pathways": pd.DataFrame()}
