"""Publication-quality visualizations for the metagenomics pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


class Visualizer:
    """Generate figures for the metagenomics pipeline analysis report.

    Args:
        config: Visualization configuration section.
        output_dir: Root output directory; figures written to output_dir/figures/.
        logger: Logger instance.
    """

    def __init__(self, config, output_dir: Path, logger: logging.Logger) -> None:
        self.config = config
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger
        plt.rcParams.update(_PUBLICATION_STYLE)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def plot_taxonomic_composition(
        self,
        abundance_matrix: pd.DataFrame,
        level: str = "genus",
        metadata: Optional[pd.DataFrame] = None,
    ) -> Optional[Path]:
        """Stacked bar chart of taxonomic composition per sample.

        Args:
            abundance_matrix: Taxa × samples relative abundance DataFrame.
            level: Taxonomic level label for axis titles.
            metadata: Optional metadata for sample ordering/grouping.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if abundance_matrix.empty:
            self.log.warning("Skipping composition plot: empty abundance matrix.")
            return None
        try:
            top_n = min(self.config.top_taxa, len(abundance_matrix))
            mean_abund = abundance_matrix.mean(axis=1).sort_values(ascending=False)
            top_taxa = mean_abund.head(top_n).index
            other_abund = abundance_matrix.loc[~abundance_matrix.index.isin(top_taxa)].sum(axis=0)
            plot_mat = abundance_matrix.loc[top_taxa].copy()
            if other_abund.sum() > 0:
                plot_mat.loc["Other"] = other_abund

            fig, ax = plt.subplots(figsize=(max(8, len(plot_mat.columns) * 0.6), 5))
            cmap = plt.get_cmap(self.config.color_palette)
            colors = [cmap(i / len(plot_mat)) for i in range(len(plot_mat))]
            plot_mat.T.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Relative Abundance")
            ax.set_title(f"Taxonomic Composition — {level.capitalize()}")
            ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8,
                title=level.capitalize(), frameon=False,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylim(0, 1)
            return self._save(fig, f"composition_{level}")
        except Exception as exc:
            self.log.warning("Composition plot failed: %s", exc)
            return None

    def plot_relative_abundance_heatmap(
        self, abundance_matrix: pd.DataFrame, level: str = "genus"
    ) -> Optional[Path]:
        """Clustered heatmap of relative abundances.

        Args:
            abundance_matrix: Taxa × samples relative abundance DataFrame.
            level: Taxonomic level label.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if abundance_matrix.empty:
            return None
        try:
            top_n = min(self.config.top_taxa, len(abundance_matrix))
            mean_abund = abundance_matrix.mean(axis=1).sort_values(ascending=False)
            mat = abundance_matrix.loc[mean_abund.head(top_n).index]
            log_mat = np.log10(mat + 1e-5)
            fig = sns.clustermap(
                log_mat,
                cmap="YlOrRd",
                figsize=(max(8, len(mat.columns) * 0.6), max(6, top_n * 0.3)),
                yticklabels=True,
                xticklabels=True,
                cbar_kws={"label": "log10(Rel. Abund.)"},
            )
            fig.ax_heatmap.set_xlabel("Sample")
            fig.ax_heatmap.set_ylabel(level.capitalize())
            return self._save_clustermap(fig, f"heatmap_{level}")
        except Exception as exc:
            self.log.warning("Heatmap failed: %s", exc)
            return None

    def plot_alpha_diversity(
        self,
        alpha_df: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        metric: str = "shannon",
    ) -> Optional[Path]:
        """Boxplot with jitter of alpha diversity by group.

        Args:
            alpha_df: Samples × metrics alpha diversity DataFrame.
            metadata: Optional metadata DataFrame with group column.
            metric: Which metric column to plot.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if alpha_df.empty or metric not in alpha_df.columns:
            return None
        try:
            plot_df = alpha_df[[metric]].copy()
            if metadata is not None and "group" in metadata.columns:
                common = plot_df.index.intersection(metadata.index)
                plot_df = plot_df.loc[common]
                plot_df["group"] = metadata.loc[common, "group"]
                x_col = "group"
            else:
                plot_df["group"] = "All"
                x_col = "group"

            fig, ax = plt.subplots(figsize=(5, 4))
            palette = sns.color_palette("Set2")
            sns.boxplot(data=plot_df, x=x_col, y=metric, ax=ax, palette=palette, width=0.5)
            sns.stripplot(data=plot_df, x=x_col, y=metric, ax=ax, color="black", size=4, alpha=0.6, jitter=True)
            ax.set_xlabel("Group")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Alpha Diversity — {metric.capitalize()}")
            return self._save(fig, f"alpha_{metric}")
        except Exception as exc:
            self.log.warning("Alpha diversity plot failed: %s", exc)
            return None

    def plot_pcoa(
        self,
        pcoa_coords: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        variance_explained: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Scatter plot of PCoA ordination.

        Args:
            pcoa_coords: Samples × PC coordinates DataFrame.
            metadata: Optional metadata for group colouring.
            variance_explained: Array of variance fractions per axis.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if pcoa_coords is None or pcoa_coords.empty:
            return None
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            x_col = pcoa_coords.columns[0]
            y_col = pcoa_coords.columns[1] if pcoa_coords.shape[1] > 1 else x_col

            if metadata is not None and "group" in metadata.columns:
                common = pcoa_coords.index.intersection(metadata.index)
                coords = pcoa_coords.loc[common]
                groups = metadata.loc[common, "group"]
                palette = dict(zip(groups.unique(), sns.color_palette("Set1", len(groups.unique()))))
                for grp, grp_df in coords.groupby(groups):
                    ax.scatter(grp_df[x_col], grp_df[y_col], label=grp, color=palette[grp], s=60, alpha=0.8)
                ax.legend(title="Group", frameon=False)
            else:
                ax.scatter(pcoa_coords[x_col], pcoa_coords[y_col], s=60, alpha=0.8)

            xe = f" ({variance_explained[0]*100:.1f}%)" if variance_explained is not None and len(variance_explained) > 0 else ""
            ye = f" ({variance_explained[1]*100:.1f}%)" if variance_explained is not None and len(variance_explained) > 1 else ""
            ax.set_xlabel(f"PC1{xe}")
            ax.set_ylabel(f"PC2{ye}")
            ax.set_title("PCoA — Bray-Curtis Dissimilarity")
            return self._save(fig, "pcoa")
        except Exception as exc:
            self.log.warning("PCoA plot failed: %s", exc)
            return None

    def plot_nmds(
        self, nmds_coords: pd.DataFrame, metadata: Optional[pd.DataFrame] = None
    ) -> Optional[Path]:
        """NMDS ordination scatter plot.

        Args:
            nmds_coords: Samples × NMDS axes DataFrame.
            metadata: Optional metadata for group colouring.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if nmds_coords is None or nmds_coords.empty:
            return None
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            x_col, y_col = nmds_coords.columns[0], nmds_coords.columns[1]

            if metadata is not None and "group" in metadata.columns:
                common = nmds_coords.index.intersection(metadata.index)
                coords = nmds_coords.loc[common]
                groups = metadata.loc[common, "group"]
                palette = dict(zip(groups.unique(), sns.color_palette("Set1", len(groups.unique()))))
                for grp, grp_df in coords.groupby(groups):
                    ax.scatter(grp_df[x_col], grp_df[y_col], label=grp, color=palette[grp], s=60, alpha=0.8)
                ax.legend(title="Group", frameon=False)
            else:
                ax.scatter(nmds_coords[x_col], nmds_coords[y_col], s=60, alpha=0.8)

            ax.set_xlabel("NMDS1")
            ax.set_ylabel("NMDS2")
            ax.set_title("NMDS Ordination")
            return self._save(fig, "nmds")
        except Exception as exc:
            self.log.warning("NMDS plot failed: %s", exc)
            return None

    def plot_differential_abundance(self, da_df: pd.DataFrame) -> Optional[Path]:
        """Volcano plot of differential abundance results.

        Args:
            da_df: DataFrame with columns log2FC, q_value, and significant index.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if da_df is None or da_df.empty or "log2FC" not in da_df.columns or "q_value" not in da_df.columns:
            return None
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            log2fc = da_df["log2FC"].fillna(0)
            neg_log_q = -np.log10(da_df["q_value"].clip(1e-300, 1))
            colors = ["#d73027" if s else "#4575b4" for s in da_df.get("significant", [False] * len(da_df))]
            ax.scatter(log2fc, neg_log_q, c=colors, s=30, alpha=0.7)
            ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="grey", linestyle="-", linewidth=0.5)
            ax.set_xlabel("log₂ Fold Change")
            ax.set_ylabel("−log₁₀(q-value)")
            ax.set_title("Differential Abundance — Volcano Plot")
            sig_n = da_df.get("significant", pd.Series(dtype=bool)).sum()
            ax.text(0.98, 0.98, f"Significant: {sig_n}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color="grey")
            return self._save(fig, "volcano")
        except Exception as exc:
            self.log.warning("Volcano plot failed: %s", exc)
            return None

    def plot_functional_heatmap(self, pathway_df: pd.DataFrame) -> Optional[Path]:
        """Heatmap of top functional pathway abundances.

        Args:
            pathway_df: Pathways × samples abundance DataFrame.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if pathway_df is None or pathway_df.empty:
            return None
        try:
            top = pathway_df.mean(axis=1).sort_values(ascending=False).head(30)
            mat = pathway_df.loc[top.index]
            log_mat = np.log10(mat + 1e-8)
            fig = sns.clustermap(
                log_mat, cmap="viridis", figsize=(10, 8),
                yticklabels=True, xticklabels=True,
                cbar_kws={"label": "log10(Abundance)"},
            )
            return self._save_clustermap(fig, "functional_heatmap")
        except Exception as exc:
            self.log.warning("Functional heatmap failed: %s", exc)
            return None

    def plot_rarefaction_curve(self, abundance_matrix: pd.DataFrame) -> Optional[Path]:
        """Rarefaction curves per sample.

        Args:
            abundance_matrix: Taxa × samples count or abundance DataFrame.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if abundance_matrix.empty:
            return None
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = sns.color_palette("tab10", len(abundance_matrix.columns))
            rng = np.random.default_rng(42)

            for i, sample in enumerate(abundance_matrix.columns):
                counts = abundance_matrix[sample].values
                total = int(counts.sum())
                if total == 0:
                    continue
                # Simulate rarefaction by subsampling
                steps = np.linspace(1, total, min(20, total), dtype=int)
                richness = []
                for n in steps:
                    frac = n / total
                    sampled = np.random.binomial(np.round(counts).astype(int), min(frac, 1.0))
                    richness.append(int(np.sum(sampled > 0)))
                ax.plot(steps, richness, label=sample, color=palette[i], alpha=0.8)

            ax.set_xlabel("Sequencing Depth (reads)")
            ax.set_ylabel("Observed Taxa")
            ax.set_title("Rarefaction Curves")
            ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
            return self._save(fig, "rarefaction")
        except Exception as exc:
            self.log.warning("Rarefaction curve failed: %s", exc)
            return None

    def plot_qc_summary(self, qc_stats_df: pd.DataFrame) -> Optional[Path]:
        """Bar plot of read counts before and after QC.

        Args:
            qc_stats_df: DataFrame with sample_id index and pre/post read count columns.

        Returns:
            Path to the saved figure, or None on failure.
        """
        if qc_stats_df is None or qc_stats_df.empty:
            return None
        try:
            fig, ax = plt.subplots(figsize=(max(6, len(qc_stats_df) * 0.8), 4))
            x = np.arange(len(qc_stats_df))
            width = 0.35
            pre_col = "pre_total_sequences" if "pre_total_sequences" in qc_stats_df.columns else qc_stats_df.columns[0]
            post_col = "post_total_sequences" if "post_total_sequences" in qc_stats_df.columns else qc_stats_df.columns[1]
            ax.bar(x - width / 2, qc_stats_df[pre_col], width, label="Before QC", color="#4393c3")
            ax.bar(x + width / 2, qc_stats_df[post_col], width, label="After QC", color="#d6604d")
            ax.set_xticks(x)
            ax.set_xticklabels(qc_stats_df.index, rotation=45, ha="right")
            ax.set_ylabel("Read Count")
            ax.set_title("Read Counts Before and After QC")
            ax.legend(frameon=False)
            return self._save(fig, "qc_summary")
        except Exception as exc:
            self.log.warning("QC summary plot failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, name: str) -> Path:
        """Save a matplotlib figure to the figures directory."""
        path = self.figures_dir / f"{name}.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        if self.config.format != "svg":
            svg_path = self.figures_dir / f"{name}.svg"
            fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        self.log.debug("Saved figure: %s", path)
        return path

    def _save_clustermap(self, clustermap_fig, name: str) -> Path:
        """Save a seaborn ClusterGrid figure."""
        path = self.figures_dir / f"{name}.{self.config.format}"
        clustermap_fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()
        self.log.debug("Saved figure: %s", path)
        return path
