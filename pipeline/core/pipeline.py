"""Pipeline orchestrator — status-tracked, output-validated end-to-end execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pipeline.core.config import PipelineConfig
from pipeline.core.exceptions import AnalysisError, ConfigurationError, ToolNotFoundError
from pipeline.modules.input_detection import Sample


# ---------------------------------------------------------------------------
# Status types
# ---------------------------------------------------------------------------

class ModuleStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WARNING = "WARNING"   # completed but with caveats (non-fatal)


class PipelineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED WITH WARNINGS"
    FAILED = "FAILED"


# Modules whose failure makes downstream results meaningless.
_CRITICAL_MODULES = {"taxonomic_profiling"}


@dataclass
class ModuleResult:
    """Execution record for a single pipeline stage.

    Attributes:
        name: Module identifier (e.g., 'taxonomic_profiling').
        status: Outcome of the execution attempt.
        error: Full error message when status is FAILED.
        note: Informational note (reason for skip, caveats, etc.).
    """

    name: str
    status: ModuleStatus
    error: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Coordinate the end-to-end metagenomics analysis.

    Args:
        config: Full pipeline configuration.
        output_dir: Root output directory for all results.
        logger: Logger instance.
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.log = logger
        self.threads: int = config.pipeline.threads
        self._module_results: List[ModuleResult] = []

    @property
    def module_results(self) -> List[ModuleResult]:
        """Return the list of per-module execution results."""
        return self._module_results

    # ------------------------------------------------------------------
    # Environment validation (pre-flight)
    # ------------------------------------------------------------------

    def validate_environment(
        self, skip_modules: Optional[List[str]] = None
    ) -> Dict:
        """Check all required tools and configuration before running the pipeline.

        Performs PATH checks, version detection, and configuration validation
        for every enabled module.

        Args:
            skip_modules: List of module names that will be skipped at runtime.

        Returns:
            Dict with:
              'checks': list of (module, tool, available, version, note) tuples
              'all_critical_pass': bool — False means the pipeline cannot produce
                  useful results and should not proceed.
              'warnings': list of non-fatal warning strings.
              'errors': list of critical error strings.
        """
        from pipeline.core.utils import check_tool

        skip = set(skip_modules or [])
        checks: List[Tuple] = []
        warnings: List[str] = []
        errors: List[str] = []

        def _chk(module: str, tool: str, note: str = "") -> Tuple[bool, str]:
            avail, ver = check_tool(tool)
            checks.append((module, tool, avail, ver, note))
            return avail, ver

        # --- Quality Control ---
        if "qc" not in skip and self.config.quality_control.enabled:
            avail, _ = _chk("quality_control", "fastqc")
            if not avail:
                warnings.append(
                    "FastQC not found — quality control will be skipped. "
                    "Install: conda install -c bioconda fastqc"
                )
            if self.config.quality_control.multiqc:
                avail, _ = _chk("quality_control", "multiqc")
                if not avail:
                    warnings.append(
                        "MultiQC not found — aggregate QC report will be skipped. "
                        "Install: pip install multiqc"
                    )

        # --- Preprocessing ---
        if "preprocessing" not in skip and self.config.preprocessing.enabled:
            preferred = self.config.preprocessing.tool
            avail, ver = check_tool(preferred)
            if avail:
                checks.append(("preprocessing", preferred, True, ver, "preferred"))
            else:
                fallback = "trimmomatic" if preferred == "fastp" else "fastp"
                checks.append(("preprocessing", preferred, False, "", f"not found; trying {fallback}"))
                avail2, ver2 = check_tool(fallback)
                checks.append(("preprocessing", fallback, avail2, ver2,
                               "fallback" if avail2 else "also not found"))
                if not avail2:
                    warnings.append(
                        f"Neither '{preferred}' nor '{fallback}' found — "
                        "preprocessing will be skipped and raw reads will be used. "
                        f"Install: conda install -c bioconda {preferred}"
                    )

        # --- Host Filtering ---
        if "host-filter" not in skip and self.config.host_filtering.enabled:
            preferred = self.config.host_filtering.tool
            avail, ver = check_tool(preferred)
            if avail:
                checks.append(("host_filtering", preferred, True, ver, "preferred"))
            else:
                fallback = "bwa" if preferred == "bowtie2" else "bowtie2"
                checks.append(("host_filtering", preferred, False, "", f"not found; trying {fallback}"))
                avail2, ver2 = check_tool(fallback)
                checks.append(("host_filtering", fallback, avail2, ver2,
                               "fallback" if avail2 else "also not found"))
                if not avail2:
                    warnings.append(
                        f"Neither '{preferred}' nor '{fallback}' found — "
                        "host filtering will be skipped. "
                        f"Install: conda install -c bioconda bowtie2"
                    )
            if preferred == "bwa" or (not avail and check_tool("bwa")[0]):
                bwa_needs_samtools, _ = check_tool("samtools")
                checks.append(("host_filtering", "samtools", bwa_needs_samtools, "", "required for bwa path"))
                if not bwa_needs_samtools:
                    warnings.append(
                        "samtools not found — required when using bwa for host filtering. "
                        "Install: conda install -c bioconda samtools"
                    )
            if not self.config.host_filtering.reference_index:
                errors.append(
                    "host_filtering.reference_index is not set. "
                    "Provide the bowtie2/bwa genome index prefix in your YAML config, "
                    "or disable host filtering with --no-host-filter."
                )

        # --- Taxonomic Profiling (critical) ---
        if "taxonomy" not in skip and self.config.taxonomic_profiling.enabled:
            preferred = self.config.taxonomic_profiling.tool
            avail, ver = check_tool(preferred)
            if avail:
                checks.append(("taxonomic_profiling", preferred, True, ver, "preferred"))
            else:
                fallback = "metaphlan" if preferred == "kraken2" else "kraken2"
                checks.append(("taxonomic_profiling", preferred, False, "", f"not found; trying {fallback}"))
                avail2, ver2 = check_tool(fallback)
                checks.append(("taxonomic_profiling", fallback, avail2, ver2,
                               "fallback" if avail2 else "also not found"))
                if not avail2:
                    errors.append(
                        f"CRITICAL: Neither '{preferred}' nor '{fallback}' is available. "
                        "Taxonomic profiling is required for diversity and statistical analysis. "
                        "Install: conda install -c bioconda kraken2 bracken  "
                        "OR  conda install -c bioconda metaphlan"
                    )
            if not self.config.taxonomic_profiling.database:
                errors.append(
                    "CRITICAL: taxonomic_profiling.database is not set. "
                    "Provide the path to your Kraken2 or MetaPhlAn database in the YAML config."
                )

        # --- Functional Profiling ---
        if "functional" not in skip and self.config.functional_profiling.enabled:
            avail, _ = _chk("functional_profiling", "humann")
            if not avail:
                warnings.append(
                    "HUMAnN not found — functional profiling will be skipped. "
                    "Install: conda install -c bioconda humann"
                )

        return {
            "checks": checks,
            "all_critical_pass": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        samples: List[Sample],
        metadata: Optional[pd.DataFrame] = None,
        skip_modules: Optional[List[str]] = None,
    ) -> Dict:
        """Execute the full analysis pipeline on the detected samples.

        Args:
            samples: Detected Sample objects from input detection.
            metadata: Optional sample metadata DataFrame indexed by sample_id.
            skip_modules: Module names to skip (e.g., ['host-filter', 'functional']).

        Returns:
            Dict containing results from all executed modules, a list of
            ModuleResult records, and the final PipelineStatus.
        """
        skip = set(skip_modules or [])
        self._module_results = []

        results: Dict = {
            "samples": samples,
            "config": self.config,
            "config_hash": self.config.config_hash(),
            "figures": {},
            "module_results": self._module_results,
        }

        active_samples = samples
        taxonomy_data_available = False

        # ---- Stage 1: Pre-QC ----
        if "qc" not in skip and self.config.quality_control.enabled:
            self.log.info("--- Stage: Quality Control (pre-processing) ---")
            qc_pre, mr = self._run_qc(active_samples, "pre")
            results["qc_pre"] = qc_pre
            self._module_results.append(mr)

        # ---- Stage 2: Preprocessing ----
        if "preprocessing" not in skip and self.config.preprocessing.enabled:
            self.log.info("--- Stage: Preprocessing ---")
            active_samples, pre_stats, mr = self._run_preprocessing(active_samples)
            results["preprocessing_stats"] = pre_stats
            self._module_results.append(mr)

        # ---- Stage 3: Post-QC ----
        if "qc" not in skip and self.config.quality_control.enabled:
            self.log.info("--- Stage: Quality Control (post-processing) ---")
            qc_post, mr = self._run_qc(active_samples, "post")
            results["qc_post"] = qc_post
            # Don't append again — merge into the pre-QC result note

        # ---- Stage 4: Host Filtering ----
        if "host-filter" not in skip and self.config.host_filtering.enabled:
            self.log.info("--- Stage: Host Filtering ---")
            active_samples, hf_stats, mr = self._run_host_filtering(active_samples)
            results["host_filter_stats"] = hf_stats
            self._module_results.append(mr)

        # ---- Stage 5: Taxonomic Profiling ----
        tax_results: Dict = {"abundance_matrices": {}, "levels": []}
        if "taxonomy" not in skip and self.config.taxonomic_profiling.enabled:
            self.log.info("--- Stage: Taxonomic Profiling ---")
            tax_results, mr = self._run_taxonomic_profiling(active_samples)
            results["taxonomy"] = tax_results
            self._module_results.append(mr)
            self._save_intermediate_results("taxonomy", tax_results)
            taxonomy_data_available = (
                mr.status == ModuleStatus.SUCCESS
                and bool(tax_results.get("abundance_matrices"))
                and any(not m.empty for m in tax_results["abundance_matrices"].values())
            )
            if not taxonomy_data_available and mr.status != ModuleStatus.FAILED:
                mr.status = ModuleStatus.FAILED
                mr.error = "Taxonomic profiling produced no abundance data."
                mr.note = "Check database path and read classification rate."
        else:
            if "taxonomy" in skip:
                self._module_results.append(
                    ModuleResult(
                        "taxonomic_profiling", ModuleStatus.SKIPPED,
                        note="Explicitly skipped by user (--skip-modules taxonomy).",
                    )
                )

        # ---- Stage 6: Diversity Analysis ----
        if "diversity" not in skip and self.config.diversity.enabled:
            self.log.info("--- Stage: Diversity Analysis ---")
            if not taxonomy_data_available:
                mr = ModuleResult(
                    "diversity_analysis", ModuleStatus.SKIPPED,
                    note="Skipped: no taxonomic abundance data available from previous stage.",
                )
            else:
                div_results, mr = self._run_diversity_analysis(tax_results, metadata)
                results["diversity"] = {
                    "alpha_diversity": div_results.alpha_diversity,
                    "beta_diversity": div_results.beta_diversity,
                    "pcoa_coords": div_results.pcoa_coords,
                    "pcoa_variance": div_results.pcoa_variance,
                    "nmds_coords": div_results.nmds_coords,
                }
            self._module_results.append(mr)

        # ---- Stage 7: Statistical Analysis ----
        if "statistics" not in skip and self.config.statistics.enabled:
            self.log.info("--- Stage: Statistical Analysis ---")
            if not taxonomy_data_available:
                mr = ModuleResult(
                    "statistical_analysis", ModuleStatus.SKIPPED,
                    note="Skipped: no taxonomic abundance data available.",
                )
            elif metadata is None:
                mr = ModuleResult(
                    "statistical_analysis", ModuleStatus.SKIPPED,
                    note="Skipped: no metadata provided (--metadata).",
                )
            elif "group" not in metadata.columns:
                mr = ModuleResult(
                    "statistical_analysis", ModuleStatus.SKIPPED,
                    note="Skipped: metadata has no 'group' column for group comparisons.",
                )
            else:
                stat_results, mr = self._run_statistical_analysis(
                    tax_results, results.get("diversity", {}), metadata
                )
                results["statistics"] = {
                    "alpha_diversity_tests": stat_results.alpha_diversity_tests,
                    "permanova_results": stat_results.permanova_results,
                    "differential_abundance": stat_results.differential_abundance,
                }
            self._module_results.append(mr)

        # ---- Stage 8: Functional Profiling ----
        if "functional" not in skip and self.config.functional_profiling.enabled:
            self.log.info("--- Stage: Functional Profiling ---")
            func_results, mr = self._run_functional_profiling(active_samples)
            results["functional"] = func_results
            self._module_results.append(mr)

        # ---- Stage 9: Visualization ----
        if "visualization" not in skip and self.config.visualization.enabled:
            self.log.info("--- Stage: Visualization ---")
            figures, mr = self._run_visualization(results, metadata)
            results["figures"] = figures
            self._module_results.append(mr)

        # ---- Stage 10: Biological Interpretation ----
        self.log.info("--- Stage: Biological Interpretation ---")
        from pipeline.modules.biological_interpretation import BiologicalInterpreter
        interp = BiologicalInterpreter(self.config.statistics, self.log)
        results["interpretation"] = interp.generate(results)

        # ---- Final status ----
        results["pipeline_status"] = self._determine_pipeline_status()
        return results

    def _determine_pipeline_status(self) -> PipelineStatus:
        """Derive overall pipeline status from per-module results.

        Rules:
        - Any critical module (e.g. taxonomic_profiling) FAILED → FAILED
        - Any non-critical module FAILED or any module WARNING → COMPLETED WITH WARNINGS
        - All modules SUCCESS or SKIPPED → COMPLETED
        """
        failed = [m for m in self._module_results if m.status == ModuleStatus.FAILED]
        warned = [m for m in self._module_results if m.status == ModuleStatus.WARNING]

        if failed:
            critical_failed = [m for m in failed if m.name in _CRITICAL_MODULES]
            if critical_failed:
                return PipelineStatus.FAILED
            return PipelineStatus.COMPLETED_WITH_WARNINGS

        if warned:
            return PipelineStatus.COMPLETED_WITH_WARNINGS

        return PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Module runners — each returns (data, ModuleResult)
    # ------------------------------------------------------------------

    def _run_qc(
        self, samples: List[Sample], stage: str
    ) -> Tuple[Dict, ModuleResult]:
        from pipeline.modules.quality_control import QualityController

        module_name = f"quality_control_{stage}"
        qc_dir = self.output_dir / "qc" / stage
        controller = QualityController(self.config.quality_control, self.log, threads=self.threads)
        try:
            controller.run_fastqc(samples, qc_dir)
            if self.config.quality_control.multiqc:
                try:
                    controller.run_multiqc(qc_dir, qc_dir / "multiqc")
                except ToolNotFoundError as exc:
                    self.log.warning("MultiQC skipped: %s", exc)
            # Output validation: check that at least one fastqc result exists
            zips = list(qc_dir.glob("*_fastqc.zip")) if qc_dir.exists() else []
            if not zips:
                return {}, ModuleResult(
                    module_name, ModuleStatus.WARNING,
                    note="FastQC ran but produced no output ZIP files.",
                )
            return {}, ModuleResult(module_name, ModuleStatus.SUCCESS,
                                   note=f"{len(zips)} FastQC report(s) generated.")
        except ToolNotFoundError as exc:
            self.log.warning("[QC-%s] Skipped: %s", stage, exc)
            return {}, ModuleResult(
                module_name, ModuleStatus.WARNING, error=str(exc),
                note="FastQC not available; QC stage skipped.",
            )
        except AnalysisError as exc:
            self.log.error("[QC-%s] Failed: %s", stage, exc)
            return {}, ModuleResult(module_name, ModuleStatus.FAILED, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            self.log.error("[QC-%s] Unexpected error: %s", stage, exc, exc_info=True)
            return {}, ModuleResult(module_name, ModuleStatus.FAILED, error=str(exc))

    def _run_preprocessing(
        self, samples: List[Sample]
    ) -> Tuple[List[Sample], pd.DataFrame, ModuleResult]:
        from pipeline.modules.preprocessing import Preprocessor

        try:
            preprocessor = Preprocessor(
                self.config.preprocessing, self.output_dir, self.log, threads=self.threads
            )
            trimmed = preprocessor.run(samples)
            stats = preprocessor.get_statistics()
            # Output validation
            missing = [s for s in trimmed if not s.forward.exists()]
            if missing:
                ids = ", ".join(s.sample_id for s in missing)
                raise AnalysisError(
                    "preprocessing",
                    f"Trimmed output files not found for: {ids}. "
                    "Check that the trimming tool completed successfully.",
                )
            return trimmed, stats, ModuleResult(
                "preprocessing", ModuleStatus.SUCCESS,
                note=f"Trimmed {len(trimmed)} sample(s).",
            )
        except ToolNotFoundError as exc:
            self.log.warning("[Preprocessing] Skipped: %s", exc)
            return samples, pd.DataFrame(), ModuleResult(
                "preprocessing", ModuleStatus.WARNING, error=str(exc),
                note="No trimming tool available; raw reads passed to downstream stages.",
            )
        except AnalysisError as exc:
            self.log.error("[Preprocessing] Failed: %s", exc)
            return samples, pd.DataFrame(), ModuleResult(
                "preprocessing", ModuleStatus.FAILED, error=str(exc),
                note="Falling back to raw reads for downstream stages.",
            )

    def _run_host_filtering(
        self, samples: List[Sample]
    ) -> Tuple[List[Sample], pd.DataFrame, ModuleResult]:
        from pipeline.modules.host_filtering import HostFilter

        try:
            hf = HostFilter(
                self.config.host_filtering, self.output_dir, self.log, threads=self.threads
            )
            filtered = hf.run(samples)
            stats = hf.get_statistics()
            # Output validation
            missing = [s for s in filtered if not s.forward.exists()]
            if missing:
                ids = ", ".join(s.sample_id for s in missing)
                raise AnalysisError(
                    "host_filtering",
                    f"Filtered output FASTQ files not found for: {ids}.",
                )
            return filtered, stats, ModuleResult(
                "host_filtering", ModuleStatus.SUCCESS,
                note=f"Host reads removed from {len(filtered)} sample(s).",
            )
        except (ConfigurationError, ToolNotFoundError) as exc:
            self.log.warning("[Host Filtering] Skipped: %s", exc)
            return samples, pd.DataFrame(), ModuleResult(
                "host_filtering", ModuleStatus.WARNING, error=str(exc),
                note="Host filtering skipped; microbial reads may include host contamination.",
            )
        except AnalysisError as exc:
            self.log.error("[Host Filtering] Failed: %s", exc)
            return samples, pd.DataFrame(), ModuleResult(
                "host_filtering", ModuleStatus.FAILED, error=str(exc),
                note="Using unfiltered reads for downstream stages.",
            )

    def _run_taxonomic_profiling(
        self, samples: List[Sample]
    ) -> Tuple[Dict, ModuleResult]:
        from pipeline.modules.taxonomic_profiling import TaxonomicProfiler

        empty = {"abundance_matrices": {}, "levels": []}
        try:
            profiler = TaxonomicProfiler(
                self.config.taxonomic_profiling, self.output_dir, self.log, threads=self.threads
            )
            result = profiler.run(samples)
            matrices = result.get("abundance_matrices", {})
            if not matrices or all(m.empty for m in matrices.values()):
                return empty, ModuleResult(
                    "taxonomic_profiling", ModuleStatus.FAILED,
                    error="Taxonomic profiling produced no abundance data.",
                    note="Verify the database path and that reads are being classified.",
                )
            levels_with_data = [lv for lv, m in matrices.items() if not m.empty]
            return result, ModuleResult(
                "taxonomic_profiling", ModuleStatus.SUCCESS,
                note=f"Abundance matrices produced at: {', '.join(levels_with_data)}.",
            )
        except (ToolNotFoundError, ConfigurationError) as exc:
            self.log.error("[Taxonomy] FAILED: %s", exc)
            return empty, ModuleResult(
                "taxonomic_profiling", ModuleStatus.FAILED, error=str(exc),
                note="Diversity, statistical, and functional analyses cannot proceed without taxonomy data.",
            )
        except AnalysisError as exc:
            self.log.error("[Taxonomy] FAILED: %s", exc)
            return empty, ModuleResult(
                "taxonomic_profiling", ModuleStatus.FAILED, error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("[Taxonomy] Unexpected error: %s", exc, exc_info=True)
            return empty, ModuleResult(
                "taxonomic_profiling", ModuleStatus.FAILED, error=str(exc),
                note="Unexpected error during taxonomic profiling.",
            )

    def _run_diversity_analysis(
        self, tax_results: Dict, metadata: Optional[pd.DataFrame]
    ) -> Tuple[object, ModuleResult]:
        from pipeline.modules.diversity_analysis import DiversityAnalyzer, DiversityResults

        matrices = tax_results.get("abundance_matrices", {})
        matrix = None
        for level in ["species", "genus", "family", "order", "phylum"]:
            if level in matrices and not matrices[level].empty:
                matrix = matrices[level]
                break
        if matrix is None and matrices:
            matrix = next(m for m in matrices.values() if not m.empty)
        if matrix is None:
            return DiversityResults(), ModuleResult(
                "diversity_analysis", ModuleStatus.SKIPPED,
                note="No non-empty abundance matrix available.",
            )

        try:
            analyzer = DiversityAnalyzer(self.config.diversity, self.log)
            result = analyzer.run(matrix, metadata)
            if result.alpha_diversity.empty:
                return result, ModuleResult(
                    "diversity_analysis", ModuleStatus.WARNING,
                    note="Alpha diversity computation returned an empty result.",
                )
            return result, ModuleResult("diversity_analysis", ModuleStatus.SUCCESS)
        except Exception as exc:
            self.log.error("[Diversity] Failed: %s", exc)
            from pipeline.modules.diversity_analysis import DiversityResults
            return DiversityResults(), ModuleResult(
                "diversity_analysis", ModuleStatus.FAILED, error=str(exc),
            )

    def _run_statistical_analysis(
        self, tax_results: Dict, div_results: Dict, metadata: pd.DataFrame
    ) -> Tuple[object, ModuleResult]:
        from pipeline.modules.statistical_analysis import StatisticalAnalyzer, StatisticalResults

        matrices = tax_results.get("abundance_matrices", {})
        alpha_df = div_results.get("alpha_diversity", pd.DataFrame())
        beta_dict = div_results.get("beta_diversity", {})

        abundance = None
        for level in ["species", "genus", "family"]:
            if level in matrices and not matrices[level].empty:
                abundance = matrices[level]
                break
        if abundance is None and matrices:
            abundance = next((m for m in matrices.values() if not m.empty), None)
        if abundance is None:
            return StatisticalResults(), ModuleResult(
                "statistical_analysis", ModuleStatus.SKIPPED,
                note="No abundance data available.",
            )

        try:
            analyzer = StatisticalAnalyzer(self.config.statistics, self.log)
            result = analyzer.run(abundance, alpha_df, metadata, beta_dict)
            return result, ModuleResult("statistical_analysis", ModuleStatus.SUCCESS)
        except Exception as exc:
            self.log.error("[Statistics] Failed: %s", exc)
            return StatisticalResults(), ModuleResult(
                "statistical_analysis", ModuleStatus.FAILED, error=str(exc),
            )

    def _run_functional_profiling(
        self, samples: List[Sample]
    ) -> Tuple[Dict, ModuleResult]:
        from pipeline.modules.functional_profiling import FunctionalProfiler

        empty = {"gene_families": pd.DataFrame(), "pathways": pd.DataFrame()}
        try:
            profiler = FunctionalProfiler(
                self.config.functional_profiling, self.output_dir, self.log, threads=self.threads
            )
            result = profiler.run(samples)
            gf = result.get("gene_families", pd.DataFrame())
            pw = result.get("pathways", pd.DataFrame())
            if (isinstance(gf, pd.DataFrame) and not gf.empty) or (
                isinstance(pw, pd.DataFrame) and not pw.empty
            ):
                return result, ModuleResult("functional_profiling", ModuleStatus.SUCCESS)
            # HUMAnN was not available — profiler returned empty gracefully
            return result, ModuleResult(
                "functional_profiling", ModuleStatus.WARNING,
                note="HUMAnN not available or produced no output; functional profiling skipped.",
            )
        except AnalysisError as exc:
            self.log.error("[Functional] Failed: %s", exc)
            return empty, ModuleResult(
                "functional_profiling", ModuleStatus.FAILED, error=str(exc),
            )

    def _run_visualization(
        self, all_results: Dict, metadata: Optional[pd.DataFrame]
    ) -> Tuple[Dict, ModuleResult]:
        from pipeline.modules.visualization import Visualizer

        try:
            viz = Visualizer(self.config.visualization, self.output_dir, self.log)
            figures: Dict = {}
            tax = all_results.get("taxonomy", {})
            matrices = tax.get("abundance_matrices", {})

            for level, mat in list(matrices.items())[:2]:
                if not mat.empty:
                    p = viz.plot_taxonomic_composition(mat, level, metadata)
                    if p:
                        figures[f"composition_{level}"] = p
                    p = viz.plot_relative_abundance_heatmap(mat, level)
                    if p:
                        figures[f"heatmap_{level}"] = p

            div = all_results.get("diversity", {})
            alpha_df = div.get("alpha_diversity", pd.DataFrame())
            for metric in ["shannon", "simpson", "observed"]:
                if isinstance(alpha_df, pd.DataFrame) and metric in alpha_df.columns:
                    p = viz.plot_alpha_diversity(alpha_df, metadata, metric)
                    if p:
                        figures[f"alpha_{metric}"] = p

            if div.get("pcoa_coords") is not None:
                p = viz.plot_pcoa(div["pcoa_coords"], metadata, div.get("pcoa_variance"))
                if p:
                    figures["pcoa"] = p

            if div.get("nmds_coords") is not None:
                p = viz.plot_nmds(div["nmds_coords"], metadata)
                if p:
                    figures["nmds"] = p

            stat = all_results.get("statistics", {})
            da_df = stat.get("differential_abundance", pd.DataFrame())
            if isinstance(da_df, pd.DataFrame) and not da_df.empty:
                p = viz.plot_differential_abundance(da_df)
                if p:
                    figures["volcano"] = p

            func = all_results.get("functional", {})
            pw = func.get("pathways", pd.DataFrame())
            if isinstance(pw, pd.DataFrame) and not pw.empty:
                p = viz.plot_functional_heatmap(pw)
                if p:
                    figures["functional_heatmap"] = p

            if matrices:
                first_mat = next((m for m in matrices.values() if not m.empty), None)
                if first_mat is not None:
                    p = viz.plot_rarefaction_curve(first_mat)
                    if p:
                        figures["rarefaction"] = p

            status = ModuleStatus.SUCCESS if figures else ModuleStatus.WARNING
            note = (
                f"Generated {len(figures)} figure(s)."
                if figures
                else "No figures produced (no upstream data available)."
            )
            return figures, ModuleResult("visualization", status, note=note)

        except Exception as exc:
            self.log.error("[Visualization] Failed: %s", exc)
            return {}, ModuleResult("visualization", ModuleStatus.FAILED, error=str(exc))

    def _save_intermediate_results(self, module_name: str, results: Dict) -> None:
        """Save abundance matrices as TSV files for reproducibility."""
        module_dir = self.output_dir / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        for level, mat in results.get("abundance_matrices", {}).items():
            if isinstance(mat, pd.DataFrame) and not mat.empty:
                out = module_dir / f"{level}_abundance.tsv"
                mat.to_csv(out, sep="\t")
                self.log.debug("Saved %s abundance table → %s", level, out)
