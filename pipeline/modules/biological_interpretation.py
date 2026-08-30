"""Biological interpretation of pipeline results."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InterpretationReport:
    """Structured biological interpretation of pipeline results.

    Attributes:
        taxonomic_summary: High-level description of microbial composition.
        diversity_findings: Alpha and beta diversity interpretation.
        group_differences: PERMANOVA and between-group comparison summary.
        differential_taxa: Enriched and depleted taxa interpretation.
        functional_changes: Key pathway changes if data is available.
        cross_module_insights: Connections between taxonomy and function.
        all_findings: Combined ordered list of findings for report rendering.
    """

    taxonomic_summary: Dict = field(default_factory=dict)
    diversity_findings: Dict = field(default_factory=dict)
    group_differences: Dict = field(default_factory=dict)
    differential_taxa: Dict = field(default_factory=dict)
    functional_changes: Dict = field(default_factory=dict)
    cross_module_insights: List[str] = field(default_factory=list)
    all_findings: List[str] = field(default_factory=list)


class BiologicalInterpreter:
    """Generate structured biological interpretations from analysis results.

    Interpretations are intentionally conservative: associations are reported
    as observations, not causal claims.

    Args:
        config: Statistics configuration for significance thresholds.
        logger: Logger instance.
    """

    def __init__(self, config, logger: logging.Logger) -> None:
        self.config = config
        self.log = logger

    def generate(self, all_results: Dict) -> InterpretationReport:
        """Generate a biological interpretation report from all pipeline results.

        Args:
            all_results: Dict containing results from all pipeline modules,
                including keys: 'taxonomy', 'diversity', 'statistics', 'functional'.

        Returns:
            InterpretationReport with structured findings.
        """
        tax_summary = self._interpret_taxonomy(all_results.get("taxonomy", {}))
        div_findings = self._interpret_diversity(all_results.get("diversity", {}))
        grp_diffs = self._interpret_group_differences(all_results.get("statistics", {}))
        diff_taxa = self._interpret_differential_taxa(all_results.get("statistics", {}))
        func_changes = self._interpret_functional(all_results.get("functional", {}))
        cross_insights = self._cross_module_insights(all_results)

        all_findings: List[str] = []
        for section_dict in [tax_summary, div_findings, grp_diffs, diff_taxa, func_changes]:
            for key, value in section_dict.items():
                if isinstance(value, str) and value:
                    all_findings.append(value)
                elif isinstance(value, list):
                    all_findings.extend(str(v) for v in value if v)
        all_findings.extend(cross_insights)

        return InterpretationReport(
            taxonomic_summary=tax_summary,
            diversity_findings=div_findings,
            group_differences=grp_diffs,
            differential_taxa=diff_taxa,
            functional_changes=func_changes,
            cross_module_insights=cross_insights,
            all_findings=all_findings,
        )

    def _interpret_taxonomy(self, tax_results: Dict) -> Dict:
        """Summarise dominant taxa, rare taxa, and core microbiome."""
        out: Dict = {}
        matrices = tax_results.get("abundance_matrices", {})
        if not matrices:
            out["summary"] = "No taxonomic profiling results available."
            return out

        for level, mat in matrices.items():
            if mat.empty:
                continue
            mean_abund = mat.mean(axis=1).sort_values(ascending=False)
            top5 = mean_abund.head(5)
            top_names = ", ".join(
                f"{name} ({abund*100:.1f}%)" for name, abund in top5.items()
            )
            out[f"dominant_{level}"] = (
                f"At the {level} level, the five most abundant taxa observed across all "
                f"samples were: {top_names}."
            )

            rare = mean_abund[mean_abund < 0.001]
            if len(rare) > 0:
                out[f"rare_{level}"] = (
                    f"{len(rare)} {level}-level taxa had mean relative abundance "
                    f"below 0.1% and are considered rare in this dataset."
                )

            n_samples = mat.shape[1]
            core = (mat > 0.001).sum(axis=1) / n_samples
            core_count = (core >= 0.5).sum()
            out[f"core_{level}"] = (
                f"{core_count} {level}-level taxa were present at >0.1% abundance "
                f"in at least 50% of samples (core microbiome)."
            )
            break  # Summarise the most specific level available

        return out

    def _interpret_diversity(self, div_results: Dict) -> Dict:
        """Interpret alpha and beta diversity levels."""
        out: Dict = {}
        alpha_df = div_results.get("alpha_diversity")
        if alpha_df is None or (hasattr(alpha_df, "empty") and alpha_df.empty):
            out["alpha"] = "Alpha diversity metrics were not computed."
            return out

        if "shannon" in alpha_df.columns:
            shannon_vals = alpha_df["shannon"].dropna()
            mean_h = shannon_vals.mean()
            std_h = shannon_vals.std()
            if mean_h > 3.5:
                level = "high"
            elif mean_h > 2.0:
                level = "moderate"
            else:
                level = "low"
            out["alpha_level"] = (
                f"Mean Shannon diversity across samples was {mean_h:.2f} ± {std_h:.2f} "
                f"(nats), indicating {level} overall community diversity."
            )

        if "observed" in alpha_df.columns:
            obs = alpha_df["observed"].dropna()
            out["richness"] = (
                f"Observed species richness ranged from {obs.min():.0f} to {obs.max():.0f} "
                f"taxa per sample (mean {obs.mean():.0f})."
            )

        return out

    def _interpret_group_differences(self, stat_results: Dict) -> Dict:
        """Interpret PERMANOVA and alpha diversity test results."""
        out: Dict = {}
        permanova = stat_results.get("permanova_results", {})
        alpha_tests = stat_results.get("alpha_diversity_tests")
        alpha = self.config.alpha

        for metric, res in permanova.items():
            if "error" in res:
                continue
            sig_text = "statistically significant" if res.get("significant") else "not statistically significant"
            out[f"permanova_{metric}"] = (
                f"PERMANOVA on {metric} distances showed {sig_text} community-level "
                f"differences between groups (F = {res.get('F_statistic', 'n/a')}, "
                f"R² = {res.get('R_squared', 'n/a')}, p = {res.get('p_value', 'n/a')}, "
                f"{res.get('permutations', 999)} permutations)."
            )

        if alpha_tests is not None and not (hasattr(alpha_tests, "empty") and alpha_tests.empty):
            sig_metrics = []
            if hasattr(alpha_tests, "iterrows"):
                for metric, row in alpha_tests.iterrows():
                    if row.get("significant"):
                        sig_metrics.append(metric)
            if sig_metrics:
                out["alpha_diff"] = (
                    f"Significant differences in alpha diversity were observed for: "
                    f"{', '.join(sig_metrics)} (q < {alpha})."
                )
            else:
                out["alpha_diff"] = (
                    f"No significant differences in alpha diversity were detected "
                    f"between groups at q < {alpha}."
                )

        return out

    def _interpret_differential_taxa(self, stat_results: Dict) -> Dict:
        """Interpret differential abundance findings."""
        out: Dict = {}
        da_df = stat_results.get("differential_abundance")
        if da_df is None or (hasattr(da_df, "empty") and da_df.empty):
            out["da"] = "Differential abundance analysis was not performed."
            return out

        sig = da_df[da_df.get("significant", False) == True] if hasattr(da_df, "__getitem__") else pd.DataFrame()
        if hasattr(sig, "empty") and sig.empty:
            out["da"] = "No taxa showed statistically significant differential abundance between groups."
            return out

        enriched = sig[sig["log2FC"] > 0] if "log2FC" in sig.columns else pd.DataFrame()
        depleted = sig[sig["log2FC"] < 0] if "log2FC" in sig.columns else pd.DataFrame()

        if not (hasattr(enriched, "empty") and enriched.empty):
            top_enriched = ", ".join(enriched.index[:5].tolist())
            out["enriched"] = (
                f"{len(enriched)} taxa showed observed increases in the treatment group, "
                f"including: {top_enriched}."
            )
        if not (hasattr(depleted, "empty") and depleted.empty):
            top_depleted = ", ".join(depleted.index[:5].tolist())
            out["depleted"] = (
                f"{len(depleted)} taxa showed observed decreases in the treatment group, "
                f"including: {top_depleted}."
            )

        return out

    def _interpret_functional(self, func_results: Dict) -> Dict:
        """Interpret functional profiling pathway results."""
        out: Dict = {}
        pathways = func_results.get("pathways")
        if pathways is None or (hasattr(pathways, "empty") and pathways.empty):
            out["functional"] = "Functional profiling was not performed or returned no results."
            return out

        n_pathways = len(pathways)
        out["pathway_count"] = (
            f"{n_pathways} metabolic pathways were detected across samples. "
            "Functional diversity reflects the metabolic potential of the community."
        )
        return out

    def _cross_module_insights(self, all_results: Dict) -> List[str]:
        """Identify connections between taxonomy and functional results."""
        insights: List[str] = []
        tax = all_results.get("taxonomy", {})
        func = all_results.get("functional", {})

        if tax.get("abundance_matrices") and func.get("pathways") is not None:
            if not (hasattr(func["pathways"], "empty") and func["pathways"].empty):
                insights.append(
                    "Both taxonomic composition and functional pathway data are available. "
                    "Observed shifts in microbial taxa may correspond to changes in "
                    "community metabolic capacity."
                )

        stat = all_results.get("statistics", {})
        permanova = stat.get("permanova_results", {})
        da_df = stat.get("differential_abundance")

        if permanova and da_df is not None and not (hasattr(da_df, "empty") and da_df.empty):
            sig_perm = any(v.get("significant") for v in permanova.values() if "error" not in v)
            sig_da = hasattr(da_df, "__getitem__") and "significant" in da_df.columns and da_df["significant"].any()
            if sig_perm and sig_da:
                insights.append(
                    "Community-level differences detected by PERMANOVA are supported by "
                    "taxon-level differential abundance findings, suggesting consistent "
                    "compositional shifts between groups."
                )

        return insights
