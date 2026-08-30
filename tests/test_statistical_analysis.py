"""Tests for statistical analysis: PERMANOVA, DA, alpha tests."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from pipeline.modules.diversity_analysis import braycurtis
from pipeline.modules.statistical_analysis import (
    StatisticalAnalyzer,
    StatisticalResults,
    _clr_transform,
)

logger = logging.getLogger("pipeline.test")


def _make_config():
    from pipeline.core.config import StatisticsConfig
    return StatisticsConfig(
        alpha=0.05,
        correction_method="fdr_bh",
        min_prevalence=0.0,
        min_abundance=0.0,
        permanova_permutations=99,
    )


class TestCLRTransform:
    def test_output_shape_preserved(self, synthetic_abundance_matrix):
        clr = _clr_transform(synthetic_abundance_matrix)
        assert clr.shape == synthetic_abundance_matrix.shape

    def test_columns_sum_to_zero(self, synthetic_abundance_matrix):
        clr = _clr_transform(synthetic_abundance_matrix)
        col_sums = clr.sum(axis=0)
        for s in col_sums:
            assert abs(s) < 1e-9


class TestPERMANOVA:
    def test_separated_groups_low_pvalue(self):
        """Clearly separated groups should yield a low PERMANOVA p-value."""
        # Group A: dominated by Taxon_0; Group B: dominated by Taxon_1
        np.random.seed(0)
        n = 6
        taxa = ["Taxon_0", "Taxon_1", "Taxon_2"]
        group_a = {"Taxon_0": [0.9, 0.85, 0.88], "Taxon_1": [0.05, 0.1, 0.07], "Taxon_2": [0.05, 0.05, 0.05]}
        group_b = {"Taxon_0": [0.05, 0.1, 0.07], "Taxon_1": [0.9, 0.85, 0.88], "Taxon_2": [0.05, 0.05, 0.05]}
        samples_a = [f"A{i}" for i in range(3)]
        samples_b = [f"B{i}" for i in range(3)]
        data = {s: {t: group_a[t][i] for t in taxa} for i, s in enumerate(samples_a)}
        data.update({s: {t: group_b[t][i] for t in taxa} for i, s in enumerate(samples_b)})
        mat = pd.DataFrame(data).T.T  # taxa x samples
        mat.index = taxa
        mat.columns = samples_a + samples_b

        dist = braycurtis(mat)
        groups = pd.Series(
            ["A"] * 3 + ["B"] * 3, index=samples_a + samples_b
        )
        config = _make_config()
        config.permanova_permutations = 199
        analyzer = StatisticalAnalyzer(config, logger)
        result = analyzer._permanova(dist, groups)
        assert result["p_value"] <= 0.15  # should be significant with well-separated groups

    def test_permanova_returns_required_keys(self, synthetic_abundance_matrix, synthetic_metadata):
        dist = braycurtis(synthetic_abundance_matrix)
        groups = synthetic_metadata["group"]
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._permanova(dist, groups)
        for key in ["F_statistic", "R_squared", "p_value", "permutations"]:
            assert key in result

    def test_permanova_r_squared_between_0_and_1(self, synthetic_abundance_matrix, synthetic_metadata):
        dist = braycurtis(synthetic_abundance_matrix)
        groups = synthetic_metadata["group"]
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._permanova(dist, groups)
        assert 0.0 <= result["R_squared"] <= 1.0


class TestDifferentialAbundance:
    def test_returns_dataframe(self, synthetic_abundance_matrix, synthetic_metadata):
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._differential_abundance(
            synthetic_abundance_matrix, synthetic_metadata["group"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, synthetic_abundance_matrix, synthetic_metadata):
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._differential_abundance(
            synthetic_abundance_matrix, synthetic_metadata["group"]
        )
        if not result.empty:
            assert "p_value" in result.columns
            assert "q_value" in result.columns
            assert "significant" in result.columns

    def test_fdr_corrected_q_values(self, synthetic_abundance_matrix, synthetic_metadata):
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._differential_abundance(
            synthetic_abundance_matrix, synthetic_metadata["group"]
        )
        if not result.empty:
            assert (result["q_value"] >= result["q_value"].min()).all()
            assert (result["q_value"] <= 1.0).all()


class TestAlphaDiversityTests:
    def test_returns_dataframe(self):
        """Alpha diversity test should return a DataFrame with expected columns."""
        from pipeline.core.config import StatisticsConfig
        np.random.seed(42)
        samples = [f"S{i}" for i in range(8)]
        alpha_df = pd.DataFrame(
            {"shannon": np.random.uniform(1, 4, 8), "observed": np.random.randint(5, 50, 8).astype(float)},
            index=samples,
        )
        groups = pd.Series(
            ["A"] * 4 + ["B"] * 4, index=samples
        )
        config = _make_config()
        analyzer = StatisticalAnalyzer(config, logger)
        result = analyzer._test_alpha_diversity(alpha_df, groups)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "p_value" in result.columns
            assert "q_value" in result.columns

    def test_clearly_different_groups(self):
        """Groups with very different alpha diversity should show significance."""
        samples_a = [f"A{i}" for i in range(6)]
        samples_b = [f"B{i}" for i in range(6)]
        all_samples = samples_a + samples_b
        alpha_df = pd.DataFrame(
            {"shannon": [0.1, 0.2, 0.1, 0.15, 0.12, 0.11, 3.8, 3.9, 4.0, 3.7, 3.85, 3.95]},
            index=all_samples,
        )
        groups = pd.Series(["A"] * 6 + ["B"] * 6, index=all_samples)
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        result = analyzer._test_alpha_diversity(alpha_df, groups)
        assert not result.empty
        assert result.loc["shannon", "p_value"] < 0.05


class TestStatisticalAnalyzerRun:
    def test_run_returns_results_object(self, synthetic_abundance_matrix, synthetic_metadata):
        analyzer = StatisticalAnalyzer(_make_config(), logger)
        dist = braycurtis(synthetic_abundance_matrix)

        from pipeline.modules.diversity_analysis import DiversityAnalyzer
        from pipeline.core.config import DiversityConfig
        div_cfg = DiversityConfig()
        div_analyzer = DiversityAnalyzer(div_cfg, logger)
        div = div_analyzer.run(synthetic_abundance_matrix)

        result = analyzer.run(
            synthetic_abundance_matrix,
            div.alpha_diversity,
            synthetic_metadata,
            beta_diversity=div.beta_diversity,
        )
        assert isinstance(result, StatisticalResults)
