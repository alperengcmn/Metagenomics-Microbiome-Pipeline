"""Statistical analysis: alpha diversity tests, PERMANOVA, differential abundance."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResults:
    """Container for all statistical analysis outputs.

    Attributes:
        alpha_diversity_tests: DataFrame with test results per metric.
        permanova_results: Dict with PERMANOVA statistics per beta metric.
        differential_abundance: DataFrame of differentially abundant taxa.
    """

    alpha_diversity_tests: pd.DataFrame = field(default_factory=pd.DataFrame)
    permanova_results: Dict[str, Dict] = field(default_factory=dict)
    differential_abundance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation as effect size for Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - (2 * U) / (n1 * n2))


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    nx, ny = len(x), len(y)
    if nx + ny < 3:
        return float("nan")
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def _clr_transform(matrix: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Centered log-ratio transformation for compositional data.

    Args:
        matrix: Taxa × samples abundance DataFrame.
        pseudocount: Small value added to zeros before log transformation.

    Returns:
        CLR-transformed DataFrame with same shape.
    """
    mat = matrix.values.astype(float) + pseudocount
    log_mat = np.log(mat)
    clr = log_mat - log_mat.mean(axis=0, keepdims=True)
    return pd.DataFrame(clr, index=matrix.index, columns=matrix.columns)


class StatisticalAnalyzer:
    """Perform statistical comparisons on diversity and abundance data.

    Args:
        config: Statistics configuration section.
        logger: Logger instance.
    """

    def __init__(self, config, logger: logging.Logger) -> None:
        self.config = config
        self.log = logger

    def run(
        self,
        abundance_matrix: pd.DataFrame,
        alpha_diversity: pd.DataFrame,
        metadata: pd.DataFrame,
        beta_diversity: Optional[Dict[str, pd.DataFrame]] = None,
        group_column: str = "group",
    ) -> StatisticalResults:
        """Run all configured statistical tests.

        Args:
            abundance_matrix: Taxa × samples DataFrame.
            alpha_diversity: Samples × metrics alpha diversity DataFrame.
            metadata: Sample metadata DataFrame (indexed by sample_id).
            beta_diversity: Optional dict of beta distance matrices.
            group_column: Column in metadata defining comparison groups.

        Returns:
            StatisticalResults dataclass.
        """
        if group_column not in metadata.columns:
            self.log.warning(
                "Column '%s' not found in metadata. Skipping group-level statistics.",
                group_column,
            )
            return StatisticalResults()

        groups = metadata[group_column]
        alpha_tests = self._test_alpha_diversity(alpha_diversity, groups)

        permanova = {}
        if beta_diversity:
            for metric, dist_mat in beta_diversity.items():
                self.log.info("Running PERMANOVA on %s distances", metric)
                permanova[metric] = self._permanova(dist_mat, groups)

        da = self._differential_abundance(abundance_matrix, groups)

        return StatisticalResults(
            alpha_diversity_tests=alpha_tests,
            permanova_results=permanova,
            differential_abundance=da,
        )

    def _test_alpha_diversity(
        self, alpha_df: pd.DataFrame, groups: pd.Series
    ) -> pd.DataFrame:
        """Test for differences in alpha diversity between groups.

        Uses Mann-Whitney U for 2 groups, Kruskal-Wallis for 3+.
        """
        if alpha_df.empty:
            return pd.DataFrame()

        common_samples = alpha_df.index.intersection(groups.index)
        if len(common_samples) == 0:
            return pd.DataFrame()

        alpha_df = alpha_df.loc[common_samples]
        groups_aligned = groups.loc[common_samples]
        unique_groups = groups_aligned.unique()
        n_groups = len(unique_groups)

        rows = []
        for metric in alpha_df.columns:
            values_by_group = {
                g: alpha_df.loc[groups_aligned == g, metric].dropna().values
                for g in unique_groups
            }

            if n_groups == 2:
                g1, g2 = unique_groups
                x, y = values_by_group[g1], values_by_group[g2]
                if len(x) < 2 or len(y) < 2:
                    continue
                stat, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                effect = _rank_biserial_r(x, y)
                test_name = "Mann-Whitney U"
            else:
                group_arrays = [v for v in values_by_group.values() if len(v) >= 2]
                if len(group_arrays) < 2:
                    continue
                stat, pval = stats.kruskal(*group_arrays)
                effect = float("nan")
                test_name = "Kruskal-Wallis"

            rows.append({
                "metric": metric,
                "test": test_name,
                "statistic": round(float(stat), 4),
                "p_value": float(pval),
                "effect_size": round(float(effect), 4) if not np.isnan(effect) else float("nan"),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        _, qvals, _, _ = multipletests(
            df["p_value"].values, method=self.config.correction_method
        )
        df["q_value"] = qvals
        df["significant"] = df["q_value"] < self.config.alpha
        return df.set_index("metric")

    def _permanova(self, dist_matrix: pd.DataFrame, groups: pd.Series) -> Dict:
        """Permutation-based MANOVA (PERMANOVA) test.

        Args:
            dist_matrix: Square symmetric distance DataFrame.
            groups: Series mapping sample_id to group label.

        Returns:
            Dict with keys: F_statistic, R_squared, p_value, permutations.
        """
        common = dist_matrix.index.intersection(groups.index)
        if len(common) < 4:
            return {"error": "Insufficient samples for PERMANOVA", "n_samples": len(common)}

        D = dist_matrix.loc[common, common].values.astype(float)
        grp = groups.loc[common].values

        def _ss_total(D: np.ndarray) -> float:
            """SS_total = sum of all squared distances / N (Anderson 2001)."""
            n = D.shape[0]
            return float(np.sum(D ** 2) / n)

        def _ss_within(D: np.ndarray, grp: np.ndarray) -> float:
            """SS_within = sum over groups of (sum of squared within-group distances / n_g)."""
            total = 0.0
            for g in np.unique(grp):
                mask = grp == g
                n_g = mask.sum()
                if n_g < 2:
                    continue
                D_g = D[np.ix_(mask, mask)]
                total += float(np.sum(D_g ** 2) / n_g)
            return total

        def _f_stat(D: np.ndarray, grp: np.ndarray) -> float:
            n = len(grp)
            k = len(np.unique(grp))
            if k < 2:
                return 0.0
            ss_t = _ss_total(D)
            ss_w = _ss_within(D, grp)
            ss_b = ss_t - ss_w
            df_between = k - 1
            df_within = n - k
            if df_within <= 0:
                return 0.0
            if ss_w <= 0:
                return float("inf") if ss_b > 0 else 0.0
            return (ss_b / df_between) / (ss_w / df_within)

        obs_f = _f_stat(D, grp)
        n_perms = self.config.permanova_permutations
        perm_f = []
        rng = np.random.default_rng(42)
        for _ in range(n_perms):
            shuffled = rng.permutation(grp)
            perm_f.append(_f_stat(D, shuffled))

        perm_f_arr = np.array(perm_f)
        p_value = float((np.sum(perm_f_arr >= obs_f) + 1) / (n_perms + 1))

        # R² = SS_between / SS_total
        ss_t = _ss_total(D)
        ss_w = _ss_within(D, grp)
        r2 = float((ss_t - ss_w) / ss_t) if ss_t != 0 else 0.0

        return {
            "F_statistic": round(obs_f, 4),
            "R_squared": round(r2, 4),
            "p_value": round(p_value, 4),
            "permutations": n_perms,
            "significant": p_value < self.config.alpha,
        }

    def _differential_abundance(
        self, abundance_matrix: pd.DataFrame, groups: pd.Series
    ) -> pd.DataFrame:
        """Identify differentially abundant taxa using CLR + Kruskal-Wallis / t-test.

        Only operates when exactly 2 groups are present for log2FC computation.
        """
        common = abundance_matrix.columns.intersection(groups.index)
        if len(common) < 4:
            self.log.warning("Too few samples for differential abundance analysis.")
            return pd.DataFrame()

        mat = abundance_matrix[common]
        grp = groups.loc[common]

        # Filter by prevalence and abundance
        n_samples = mat.shape[1]
        prevalence = (mat > 0).sum(axis=1) / n_samples
        mean_abund = mat.mean(axis=1)
        keep = (prevalence >= self.config.min_prevalence) & (mean_abund >= self.config.min_abundance)
        mat = mat.loc[keep]

        if mat.empty:
            self.log.warning("No taxa pass prevalence/abundance filters for DA analysis.")
            return pd.DataFrame()

        clr = _clr_transform(mat)
        unique_groups = grp.unique()
        n_groups = len(unique_groups)

        rows = []
        for taxon in clr.index:
            values_by_group = {
                g: clr.loc[taxon, grp == g].values for g in unique_groups
            }
            arrays = [v for v in values_by_group.values() if len(v) >= 2]
            if len(arrays) < 2:
                continue

            if n_groups == 2:
                g1, g2 = unique_groups
                x, y = values_by_group[g1], values_by_group[g2]
                _, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                log2fc = float(np.mean(y) - np.mean(x)) / np.log(2) if np.log(2) != 0 else 0.0
            else:
                _, pval = stats.kruskal(*arrays)
                log2fc = float("nan")

            rows.append({"taxon": taxon, "log2FC": log2fc, "p_value": float(pval)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        _, qvals, _, _ = multipletests(
            df["p_value"].values, method=self.config.correction_method
        )
        df["q_value"] = qvals
        df["significant"] = df["q_value"] < self.config.alpha
        return df.set_index("taxon").sort_values("q_value")
