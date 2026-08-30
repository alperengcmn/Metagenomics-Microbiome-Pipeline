"""Shared pytest fixtures for the metagenomics pipeline test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def synthetic_abundance_matrix() -> pd.DataFrame:
    """10 taxa x 6 samples relative abundance matrix (rows sum to 1)."""
    np.random.seed(42)
    data = np.random.dirichlet(np.ones(10), size=6).T  # 10x6
    taxa = [f"Taxon_{i}" for i in range(10)]
    samples = [f"Sample_{i}" for i in range(6)]
    return pd.DataFrame(data, index=taxa, columns=samples)


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    """Metadata DataFrame with 'group' column, indexed by sample_id."""
    return pd.DataFrame(
        {
            "sample_id": [f"Sample_{i}" for i in range(6)],
            "group": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
        }
    ).set_index("sample_id")


@pytest.fixture
def synthetic_fastq_dir(tmp_path: Path) -> Path:
    """Fake paired-end FASTQ files for 3 samples."""
    fq_dir = tmp_path / "fastq"
    fq_dir.mkdir()
    for sample in ["SRR001", "SRR002", "SRR003"]:
        (fq_dir / f"{sample}_R1.fastq.gz").write_bytes(b"fake")
        (fq_dir / f"{sample}_R2.fastq.gz").write_bytes(b"fake")
    return fq_dir
