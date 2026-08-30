"""Alpha and beta diversity analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import spatial, stats
from packaging.version import Version
from sklearn.manifold import MDS
import sklearn as _sklearn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpha diversity functions
# ---------------------------------------------------------------------------

def shannon(counts: np.ndarray) -> float:
    """Compute Shannon entropy: -sum(p * log(p)) for p > 0.

    Args:
        counts: Array of raw counts or relative abundances.

    Returns:
        Shannon entropy value (nats).
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def simpson(counts: np.ndarray) -> float:
    """Compute Simpson's diversity index: 1 - sum(p^2).

    Args:
        counts: Array of raw counts or relative abundances.

    Returns:
        Simpson's diversity (0 = no diversity, 1 = max diversity).
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p ** 2))


def chao1(counts: np.ndarray) -> float:
    """Compute Chao1 richness estimate.

    Chao1 = S_obs + f1^2 / (2 * f2), where f1 = singletons, f2 = doubletons.
    When f2 = 0, uses the bias-corrected form: S_obs + f1*(f1-1)/2.

    Args:
        counts: Array of raw integer counts (not relative abundances).

    Returns:
        Chao1 richness estimate.
    """
    counts = np.asarray(counts, dtype=float)
    s_obs = float(np.sum(counts > 0))
    f1 = float(np.sum(counts == 1))
    f2 = float(np.sum(counts == 2))
    if f2 > 0:
        return s_obs + (f1 ** 2) / (2.0 * f2)
    else:
        return s_obs + f1 * (f1 - 1) / 2.0


def observed(counts: np.ndarray) -> float:
    """Count the number of observed species (abundance > 0).

    Args:
        counts: Array of raw counts or relative abundances.

    Returns:
        Number of observed taxa.
    """
    return float(np.sum(np.asarray(counts, dtype=float) > 0))


def evenness(counts: np.ndarray) -> float:
    """Compute Pielou's evenness: Shannon / log(S_obs).

    Args:
        counts: Array of raw counts or relative abundances.

    Returns:
        Pielou's J evenness (0–1). Returns 0 when fewer than 2 taxa observed.
    """
    s_obs = observed(counts)
    if s_obs < 2:
        return 0.0
    h = shannon(counts)
    return h / np.log(s_obs)


# ---------------------------------------------------------------------------
# Beta diversity functions
# ---------------------------------------------------------------------------

def braycurtis(abundance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Bray-Curtis dissimilarity matrix.

    Args:
        abundance_matrix: DataFrame with taxa as rows and samples as columns.

    Returns:
        Square dissimilarity DataFrame (samples × samples).
    """
    samples = abundance_matrix.columns.tolist()
    X = abundance_matrix.values.T  # shape: (n_samples, n_taxa)
    n = len(samples)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = spatial.distance.braycurtis(X[i], X[j])
            dist[i, j] = d
            dist[j, i] = d
    return pd.DataFrame(dist, index=samples, columns=samples)


def jaccard(abundance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Jaccard distance matrix (presence/absence).

    Args:
        abundance_matrix: DataFrame with taxa as rows and samples as columns.

    Returns:
        Square distance DataFrame (samples × samples).
    """
    samples = abundance_matrix.columns.tolist()
    X = (abundance_matrix.values.T > 0).astype(float)
    n = len(samples)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = spatial.distance.jaccard(X[i], X[j])
            dist[i, j] = d
            dist[j, i] = d
    return pd.DataFrame(dist, index=samples, columns=samples)


# ---------------------------------------------------------------------------
# Ordination
# ---------------------------------------------------------------------------

def pcoa(distance_matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Principal Coordinates Analysis via eigen decomposition.

    Args:
        distance_matrix: Square symmetric distance DataFrame.

    Returns:
        Tuple of (coordinates DataFrame, variance_explained array).
    """
    D = distance_matrix.values.astype(float)
    n = D.shape[0]
    # Double-centring
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (D ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Keep positive eigenvalues only
    pos_mask = eigenvalues > 1e-10
    pos_vals = eigenvalues[pos_mask]
    pos_vecs = eigenvectors[:, pos_mask]
    coords = pos_vecs * np.sqrt(pos_vals)
    variance_explained = pos_vals / pos_vals.sum()
    samples = distance_matrix.index.tolist()
    col_names = [f"PC{i+1}" for i in range(coords.shape[1])]
    return pd.DataFrame(coords, index=samples, columns=col_names), variance_explained


def nmds(distance_matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Non-metric multidimensional scaling using sklearn MDS.

    Args:
        distance_matrix: Square symmetric distance DataFrame.
        n_components: Number of dimensions for the ordination.

    Returns:
        DataFrame of NMDS coordinates (samples × dimensions).
    """
    if Version(_sklearn.__version__) >= Version("1.8"):
        mds = MDS(
            n_components=n_components,
            metric_mds=False,
            metric="precomputed",
            random_state=42,
            max_iter=500,
            n_init=4,
            init="random",
        )
    else:
        mds = MDS(
            n_components=n_components,
            metric=False,
            dissimilarity="precomputed",
            random_state=42,
            max_iter=500,
            n_init=4,
        )
    coords = mds.fit_transform(distance_matrix.values.astype(float))
    samples = distance_matrix.index.tolist()
    col_names = [f"NMDS{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=samples, columns=col_names)


# ---------------------------------------------------------------------------
# Results container and main class
# ---------------------------------------------------------------------------

@dataclass
class DiversityResults:
    """Container for all diversity analysis outputs.

    Attributes:
        alpha_diversity: DataFrame with samples as rows and metrics as columns.
        beta_diversity: Dict mapping metric name to distance matrix DataFrame.
        pcoa_coords: PCoA coordinate DataFrame or None.
        pcoa_variance: Array of variance explained per axis, or None.
        nmds_coords: NMDS coordinate DataFrame or None.
    """

    alpha_diversity: pd.DataFrame = field(default_factory=pd.DataFrame)
    beta_diversity: Dict[str, pd.DataFrame] = field(default_factory=dict)
    pcoa_coords: Optional[pd.DataFrame] = None
    pcoa_variance: Optional[np.ndarray] = None
    nmds_coords: Optional[pd.DataFrame] = None


class DiversityAnalyzer:
    """Compute alpha and beta diversity metrics from an abundance matrix.

    Args:
        config: Diversity configuration section.
        logger: Logger instance.
    """

    def __init__(self, config, logger: logging.Logger) -> None:
        self.config = config
        self.log = logger

    def run(
        self,
        abundance_matrix: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
    ) -> DiversityResults:
        """Compute all configured diversity metrics.

        Args:
            abundance_matrix: Taxa × samples relative abundance DataFrame.
            metadata: Optional sample metadata DataFrame (indexed by sample_id).

        Returns:
            DiversityResults dataclass with all computed metrics.
        """
        self.log.info("Computing alpha diversity metrics")
        alpha_df = self._compute_alpha(abundance_matrix)

        self.log.info("Computing beta diversity metrics")
        beta_dict = self._compute_beta(abundance_matrix)

        pcoa_coords, pcoa_var = None, None
        nmds_coords = None

        if beta_dict:
            # Use braycurtis for ordination if available, else first metric
            ref_metric = "braycurtis" if "braycurtis" in beta_dict else next(iter(beta_dict))
            dist_mat = beta_dict[ref_metric]

            if "pcoa" in (self.config.ordination or []):
                self.log.info("Running PCoA")
                try:
                    pcoa_coords, pcoa_var = pcoa(dist_mat)
                except Exception as exc:
                    self.log.warning("PCoA failed: %s", exc)

            if "nmds" in (self.config.ordination or []):
                self.log.info("Running NMDS")
                try:
                    nmds_coords = nmds(dist_mat)
                except Exception as exc:
                    self.log.warning("NMDS failed: %s", exc)

        return DiversityResults(
            alpha_diversity=alpha_df,
            beta_diversity=beta_dict,
            pcoa_coords=pcoa_coords,
            pcoa_variance=pcoa_var,
            nmds_coords=nmds_coords,
        )

    def _compute_alpha(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured alpha diversity metrics."""
        metrics_map = {
            "shannon": shannon,
            "simpson": simpson,
            "chao1": chao1,
            "observed": observed,
            "evenness": evenness,
        }
        configured = [m for m in (self.config.alpha_metrics or []) if m in metrics_map]
        if not configured:
            configured = ["shannon", "simpson", "observed"]

        rows = {}
        for sample in matrix.columns:
            counts = matrix[sample].values
            row = {}
            for metric in configured:
                fn = metrics_map[metric]
                try:
                    row[metric] = fn(counts)
                except Exception as exc:
                    self.log.warning("Alpha metric %s failed for %s: %s", metric, sample, exc)
                    row[metric] = float("nan")
            rows[sample] = row

        df = pd.DataFrame(rows).T
        df.index.name = "sample_id"
        return df

    def _compute_beta(self, matrix: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute all configured beta diversity distance matrices."""
        metrics_map = {"braycurtis": braycurtis, "jaccard": jaccard}
        result = {}
        for metric in (self.config.beta_metrics or []):
            fn = metrics_map.get(metric)
            if fn is None:
                self.log.debug("Beta metric '%s' not implemented natively; skipping.", metric)
                continue
            try:
                result[metric] = fn(matrix)
            except Exception as exc:
                self.log.warning("Beta metric %s failed: %s", metric, exc)
        return result
