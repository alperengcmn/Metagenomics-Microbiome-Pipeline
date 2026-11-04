"""Tests for FASTQ input detection."""
from __future__ import annotations

import logging
import pytest
from pathlib import Path

from pipeline.core.exceptions import InputDetectionError
from pipeline.modules.input_detection import Sample, SampleDetector

logger = logging.getLogger("pipeline.test")


class TestSampleDetection:
    def test_detects_paired_end(self, synthetic_fastq_dir: Path):
        detector = SampleDetector(synthetic_fastq_dir, logger)
        samples = detector.detect()
        assert len(samples) == 3
        paired = [s for s in samples if s.is_paired]
        assert len(paired) == 3

    def test_sample_ids_correct(self, synthetic_fastq_dir: Path):
        detector = SampleDetector(synthetic_fastq_dir, logger)
        samples = detector.detect()
        ids = {s.sample_id for s in samples}
        assert ids == {"SRR001", "SRR002", "SRR003"}

    def test_sample_reads_property(self, synthetic_fastq_dir: Path):
        detector = SampleDetector(synthetic_fastq_dir, logger)
        samples = detector.detect()
        for s in samples:
            assert len(s.reads) == 2  # paired
            assert all(isinstance(p, Path) for p in s.reads)

    def test_empty_dir_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(InputDetectionError):
            SampleDetector(empty_dir, logger).detect()

    def test_single_end_detection(self, tmp_path: Path):
        se_dir = tmp_path / "se"
        se_dir.mkdir()
        (se_dir / "sampleA.fastq.gz").write_bytes(b"fake")
        (se_dir / "sampleB.fastq.gz").write_bytes(b"fake")
        detector = SampleDetector(se_dir, logger)
        samples = detector.detect()
        assert len(samples) == 2
        assert all(not s.is_paired for s in samples)

    def test_no_duplicate_sample_ids(self, tmp_path: Path):
        fq_dir = tmp_path / "fq"
        fq_dir.mkdir()
        (fq_dir / "SRR001_R1.fastq.gz").write_bytes(b"fake")
        (fq_dir / "SRR001_R2.fastq.gz").write_bytes(b"fake")
        detector = SampleDetector(fq_dir, logger)
        samples = detector.detect()
        ids = [s.sample_id for s in samples]
        assert len(ids) == len(set(ids))

    def test_load_metadata(self, tmp_path: Path):
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("sample_id\tgroup\nS1\tControl\nS2\tTreatment\n")
        detector = SampleDetector(tmp_path, logger)
        df = detector.load_metadata(tsv)
        assert df.index.name == "sample_id"
        assert "group" in df.columns
        assert len(df) == 2

    def test_load_metadata_missing_column(self, tmp_path: Path):
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("id\tgroup\nS1\tControl\n")
        detector = SampleDetector(tmp_path, logger)
        with pytest.raises(InputDetectionError):
            detector.load_metadata(tsv)


class TestSampleDataclass:
    def test_paired_reads_property(self, tmp_path: Path):
        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        r1.write_bytes(b"")
        r2.write_bytes(b"")
        s = Sample(sample_id="test", forward=r1, reverse=r2, is_paired=True)
        assert s.reads == [r1, r2]

    def test_single_end_reads_property(self, tmp_path: Path):
        r1 = tmp_path / "r1.fastq.gz"
        r1.write_bytes(b"")
        s = Sample(sample_id="test", forward=r1, is_paired=False)
        assert s.reads == [r1]

@pytest.mark.parametrize("filenames", [
    ["sample_R1.fastq", "sample_R1.fastq.gz", "sample_R2.fastq.gz"],
    ["sample_R1.fastq.gz", "sample_R2.fastq", "sample_R2.fastq.gz"],
    ["sample.fastq", "sample.fastq.gz"],
])
def test_duplicate_input_files_raise(tmp_path, filenames):
    """Reject colliding inputs before dictionary assignment drops a file."""
    for filename in filenames:
        (tmp_path / filename).touch()
    with pytest.raises(InputDetectionError, match="Duplicate sample ID"):
        SampleDetector(tmp_path, logger).detect()
