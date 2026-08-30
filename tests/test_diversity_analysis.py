"""Tests for diversity analysis algorithms."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from pipeline.modules.diversity_analysis import (
    DiversityAnalyzer,
    braycurtis,
    chao1,
    evenness,
    jaccard,
    nmds,
    observed,
    pcoa,
    shannon,
    simpson,
)

logger = logging.getLogger("pipeline.test")


class TestShannonEntropy:
    def test_uniform_distribution(self):
        """Shannon of uniform n species = log(n)."""
        n = 5
        counts = np.ones(n)
        h = shannon(counts)
        assert abs(h - np.log(n)) < 1e-9

    def test_single_species(self):
        """Shannon of a community with one species = 0."""
        counts = np.array([100.0, 0.0, 0.0])
        assert shannon(counts) == pytest.approx(0.0)

    def test_all_zeros(self):
        assert shannon(np.zeros(5)) == pytest.approx(0.0)

    def test_two_equal_species(self):
        h = shannon(np.array([1.0, 1.0]))
        assert h == pytest.approx(np.log(2))

    def test_returns_float(self):
        assert isinstance(shannon(np.array([1, 2, 3])), float)


class TestSimpson:
    def test_single_species_zero_diversity(self):
        """Simpson of one dominant species = 0."""
        assert simpson(np.array([100.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_two_equal_species(self):
        s = simpson(np.array([1.0, 1.0]))
        assert s == pytest.approx(0.5)

    def test_all_zeros(self):
        assert simpson(np.zeros(5)) == pytest.approx(0.0)

    def test_bounded_0_to_1(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            counts = rng.integers(0, 100, size=10).astype(float)
            s = simpson(counts)
            assert 0.0 <= s <= 1.0


class TestChao1:
    def test_known_values(self):
        """Chao1 with 10 observed, 5 singletons, 2 doubletons."""
        counts = np.array([1, 1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=float)
        s_obs = 10
        f1, f2 = 5, 2
        expected = s_obs + f1 ** 2 / (2 * f2)
        assert chao1(counts) == pytest.approx(expected)

    def test_no_doubletons(self):
        """When f2=0, uses bias-corrected form f1*(f1-1)/2."""
        counts = np.array([1, 1, 1, 3, 5], dtype=float)
        s_obs = 5
        f1 = 3
        expected = s_obs + f1 * (f1 - 1) / 2
        assert chao1(counts) == pytest.approx(expected)

    def test_all_abundant(self):
        """No singletons or doubletons => Chao1 = observed."""
        counts = np.array([10, 20, 30], dtype=float)
        assert chao1(counts) == pytest.approx(3.0)


class TestObserved:
    def test_counts_nonzero(self):
        assert observed(np.array([0, 1, 0, 5, 0])) == pytest.approx(2.0)

    def test_all_zero(self):
        assert observed(np.zeros(10)) == pytest.approx(0.0)

    def test_all_present(self):
        assert observed(np.ones(7)) == pytest.approx(7.0)


class TestEvenness:
    def test_single_species_returns_zero(self):
        assert evenness(np.array([10.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_uniform_evenness_is_one(self):
        counts = np.ones(5)
        j = evenness(counts)
        assert j == pytest.approx(1.0, abs=1e-9)


class TestBrayCurtis:
    def test_self_distance_is_zero(self, synthetic_abundance_matrix):
        bc = braycurtis(synthetic_abundance_matrix)
        for s in synthetic_abundance_matrix.columns:
            assert bc.loc[s, s] == pytest.approx(0.0)

    def test_symmetry(self, synthetic_abundance_matrix):
        bc = braycurtis(synthetic_abundance_matrix)
        samples = synthetic_abundance_matrix.columns.tolist()
        for i, s1 in enumerate(samples):
            for j, s2 in enumerate(samples):
                assert bc.loc[s1, s2] == pytest.approx(bc.loc[s2, s1], abs=1e-10)

    def test_bounded_0_to_1(self, synthetic_abundance_matrix):
        bc = braycurtis(synthetic_abundance_matrix)
        assert (bc.values >= 0).all()
        assert (bc.values <= 1 + 1e-9).all()

    def test_identical_samples_distance_zero(self):
        data = pd.DataFrame(
            {"S1": [0.5, 0.3, 0.2], "S2": [0.5, 0.3, 0.2]},
            index=["A", "B", "C"],
        )
        bc = braycurtis(data)
        assert bc.loc["S1", "S2"] == pytest.approx(0.0)

    def test_completely_different_samples_max_distance(self):
        data = pd.DataFrame(
            {"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
            index=["A", "B"],
        )
        bc = braycurtis(data)
        assert bc.loc["S1", "S2"] == pytest.approx(1.0)


class TestJaccard:
    def test_self_distance_zero(self, synthetic_abundance_matrix):
        jac = jaccard(synthetic_abundance_matrix)
        for s in synthetic_abundance_matrix.columns:
            assert jac.loc[s, s] == pytest.approx(0.0)

    def test_symmetry(self, synthetic_abundance_matrix):
        jac = jaccard(synthetic_abundance_matrix)
        samples = synthetic_abundance_matrix.columns.tolist()
        assert jac.loc[samples[0], samples[1]] == pytest.approx(
            jac.loc[samples[1], samples[0]], abs=1e-10
        )


class TestPCoA:
    def test_output_dimensions(self, synthetic_abundance_matrix):
        dist = braycurtis(synthetic_abundance_matrix)
        coords, var = pcoa(dist)
        assert coords.shape[0] == dist.shape[0]
        assert coords.shape[1] >= 1

    def test_variance_explained_sums_to_one(self, synthetic_abundance_matrix):
        dist = braycurtis(synthetic_abundance_matrix)
        _, var = pcoa(dist)
        assert abs(var.sum() - 1.0) < 1e-9

    def test_index_matches_samples(self, synthetic_abundance_matrix):
        dist = braycurtis(synthetic_abundance_matrix)
        coords, _ = pcoa(dist)
        assert set(coords.index) == set(dist.index)


class TestNMDS:
    def test_output_shape(self, synthetic_abundance_matrix):
        dist = braycurtis(synthetic_abundance_matrix)
        coords = nmds(dist)
        assert coords.shape == (len(synthetic_abundance_matrix.columns), 2)


class TestDiversityAnalyzer:
    def _make_config(self):
        from pipeline.core.config import DiversityConfig
        return DiversityConfig(
            alpha_metrics=["shannon", "simpson", "chao1", "observed"],
            beta_metrics=["braycurtis", "jaccard"],
            ordination=["pcoa", "nmds"],
        )

    def test_run_returns_results(self, synthetic_abundance_matrix):
        from pipeline.modules.diversity_analysis import DiversityResults
        analyzer = DiversityAnalyzer(self._make_config(), logger)
        results = analyzer.run(synthetic_abundance_matrix)
        assert isinstance(results, DiversityResults)

    def test_alpha_diversity_shape(self, synthetic_abundance_matrix):
        analyzer = DiversityAnalyzer(self._make_config(), logger)
        results = analyzer.run(synthetic_abundance_matrix)
        alpha = results.alpha_diversity
        assert alpha.shape[0] == synthetic_abundance_matrix.shape[1]
        assert "shannon" in alpha.columns

    def test_beta_diversity_computed(self, synthetic_abundance_matrix):
        analyzer = DiversityAnalyzer(self._make_config(), logger)
        results = analyzer.run(synthetic_abundance_matrix)
        assert "braycurtis" in results.beta_diversity

    def test_pcoa_computed(self, synthetic_abundance_matrix):
        analyzer = DiversityAnalyzer(self._make_config(), logger)
        results = analyzer.run(synthetic_abundance_matrix)
        assert results.pcoa_coords is not None
