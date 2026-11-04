"""Input FASTQ detection and sample metadata loading."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline.core.exceptions import InputDetectionError

logger = logging.getLogger(__name__)

FASTQ_EXTENSIONS = {".fastq", ".fastq.gz", ".fq", ".fq.gz"}
# Patterns that indicate paired-end R1 files; companion R2 derived by substitution
_R1_PATTERNS = [
    (re.compile(r"_R1(?=\.|_|$)"), "_R2"),
    (re.compile(r"_1(?=\.|_|$)"), "_2"),
    (re.compile(r"_r1(?=\.|_|$)"), "_r2"),
    (re.compile(r"\.R1(?=\.)"), ".R2"),
]


@dataclass
class Sample:
    """Represents a single biological sample with one or two FASTQ files.

    Attributes:
        sample_id: Unique identifier for the sample.
        forward: Path to the forward (R1) FASTQ file.
        reverse: Path to the reverse (R2) FASTQ file, or None for single-end.
        is_paired: True when a paired reverse read file is present.
    """

    sample_id: str
    forward: Path
    reverse: Optional[Path] = None
    is_paired: bool = False

    @property
    def reads(self) -> List[Path]:
        """Return all read files associated with this sample."""
        if self.reverse is not None:
            return [self.forward, self.reverse]
        return [self.forward]


class SampleDetector:
    """Detect and validate FASTQ samples in an input directory.

    Args:
        input_dir: Directory containing FASTQ files.
        logger: Logger instance for status messages.
    """

    def __init__(self, input_dir: Path, logger: logging.Logger) -> None:
        self.input_dir = input_dir
        self.log = logger

    def detect(self) -> List[Sample]:
        """Auto-detect FASTQ files and pair them into Sample objects.

        Returns:
            Sorted list of Sample objects.

        Raises:
            InputDetectionError: If no files are found, pairing fails, or
                duplicate sample IDs are detected.
        """
        fastq_files = self._collect_fastq_files()
        if not fastq_files:
            raise InputDetectionError(
                f"No FASTQ files found in '{self.input_dir}'. "
                "Expected files with extensions: .fastq, .fastq.gz, .fq, .fq.gz"
            )

        self.log.info("Found %d FASTQ file(s) in %s", len(fastq_files), self.input_dir)

        forward: Dict[str, Path] = {}
        reverse: Dict[str, Path] = {}
        unpaired: Dict[str, Path] = {}

        for path in fastq_files:
            r1_sample, r2_sample = self._classify_file(path)
            if r1_sample is not None:
                sample_id, bucket = r1_sample, forward
            elif r2_sample is not None:
                sample_id, bucket = r2_sample, reverse
            else:
                sample_id, bucket = self._infer_sample_id(path), unpaired
            if sample_id in bucket:
                raise InputDetectionError(
                    f"Duplicate sample ID detected: '{sample_id}' in files "
                    f"'{bucket[sample_id].name}' and '{path.name}'"
                )
            bucket[sample_id] = path

        # Check for orphaned R1 / R2 files
        if forward or reverse:
            self._validate_pairs(forward, reverse)

        samples: List[Sample] = []
        seen_ids: set[str] = set()

        # Paired-end samples
        for sample_id, fwd_path in sorted(forward.items()):
            if sample_id in seen_ids:
                raise InputDetectionError(f"Duplicate sample ID detected: '{sample_id}'")
            seen_ids.add(sample_id)
            rev_path = reverse.get(sample_id)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    forward=fwd_path,
                    reverse=rev_path,
                    is_paired=rev_path is not None,
                )
            )

        # Single-end samples
        for sample_id, path in sorted(unpaired.items()):
            if sample_id in seen_ids:
                raise InputDetectionError(f"Duplicate sample ID detected: '{sample_id}'")
            seen_ids.add(sample_id)
            samples.append(Sample(sample_id=sample_id, forward=path, is_paired=False))

        paired = sum(1 for s in samples if s.is_paired)
        self.log.info(
            "Detected %d sample(s): %d paired-end, %d single-end",
            len(samples),
            paired,
            len(samples) - paired,
        )
        return samples

    def _collect_fastq_files(self) -> List[Path]:
        """Return all FASTQ files in input_dir (non-recursive)."""
        files: List[Path] = []
        for f in sorted(self.input_dir.iterdir()):
            if f.is_file():
                name = f.name
                for ext in FASTQ_EXTENSIONS:
                    if name.endswith(ext):
                        files.append(f)
                        break
        return files

    def _classify_file(self, path: Path) -> tuple[Optional[str], Optional[str]]:
        """Return (r1_sample_id, r2_sample_id) — at most one is set."""
        stem = path.name
        for ext in sorted(FASTQ_EXTENSIONS, key=len, reverse=True):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        for pattern, r2_suffix in _R1_PATTERNS:
            if pattern.search(stem):
                sample_id = pattern.sub("", stem)
                return sample_id, None

        for _, r2_suffix in _R1_PATTERNS:
            r2_pattern = re.compile(r2_suffix.replace("_", "_") + r"(?=\.|_|$)")
            if r2_pattern.search(stem):
                sample_id = r2_pattern.sub("", stem)
                return None, sample_id

        return None, None

    def _infer_sample_id(self, path: Path) -> str:
        """Strip read suffix and FASTQ extensions to derive a sample ID."""
        stem = path.name
        for ext in sorted(FASTQ_EXTENSIONS, key=len, reverse=True):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        # Strip any trailing read designators
        stem = re.sub(r"[_.]?[Rr]?[12]$", "", stem)
        return stem

    def _validate_pairs(
        self, forward: Dict[str, Path], reverse: Dict[str, Path]
    ) -> None:
        """Raise InputDetectionError for any unpaired R1 or R2 files."""
        r1_only = set(forward.keys()) - set(reverse.keys())
        r2_only = set(reverse.keys()) - set(forward.keys())

        messages: List[str] = []
        if r1_only:
            messages.append(
                "R1 files without matching R2: " + ", ".join(sorted(r1_only))
            )
        if r2_only:
            messages.append(
                "R2 files without matching R1: " + ", ".join(sorted(r2_only))
            )
        if messages:
            raise InputDetectionError(
                "Unpaired FASTQ files detected:\n" + "\n".join(messages)
            )

    def load_metadata(self, metadata_path: Path) -> pd.DataFrame:
        """Load sample metadata from a TSV file.

        Args:
            metadata_path: Path to TSV file with a 'sample_id' column.

        Returns:
            DataFrame indexed by sample_id.

        Raises:
            InputDetectionError: If 'sample_id' column is missing.
        """
        df = pd.read_csv(metadata_path, sep="\t")
        if "sample_id" not in df.columns:
            raise InputDetectionError(
                f"Metadata file '{metadata_path}' must contain a 'sample_id' column. "
                f"Found columns: {list(df.columns)}"
            )
        df = df.set_index("sample_id")
        self.log.info("Loaded metadata for %d samples from %s", len(df), metadata_path)
        return df
