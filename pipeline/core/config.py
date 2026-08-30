"""Pydantic v2 configuration models for the metagenomics pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineSettings(BaseModel):
    threads: int = Field(4, ge=1, description="Number of CPU threads to use.")
    verbose: bool = False
    dry_run: bool = False


class QCConfig(BaseModel):
    enabled: bool = True
    tool: str = "fastqc"
    multiqc: bool = True


class PreprocessingConfig(BaseModel):
    enabled: bool = True
    tool: str = "fastp"
    adapter_trimming: bool = True
    quality_trimming: bool = True
    min_quality: int = Field(20, ge=0, le=40)
    min_length: int = Field(50, ge=1)
    window_size: int = Field(4, ge=1)
    max_n_rate: float = Field(0.05, ge=0.0, le=1.0)


class HostFilteringConfig(BaseModel):
    enabled: bool = True
    tool: str = "bowtie2"
    reference_index: Optional[str] = None
    sensitivity: str = "--very-sensitive"
    remove_unmapped: bool = False


class TaxonomicConfig(BaseModel):
    enabled: bool = True
    tool: str = "kraken2"
    database: Optional[str] = None
    confidence: float = Field(0.1, ge=0.0, le=1.0)
    bracken_enabled: bool = True
    bracken_level: str = "S"
    levels: List[str] = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


class DiversityConfig(BaseModel):
    enabled: bool = True
    alpha_metrics: List[str] = ["shannon", "simpson", "chao1", "observed"]
    beta_metrics: List[str] = ["braycurtis", "jaccard"]
    ordination: List[str] = ["pcoa", "nmds"]
    rarefaction_depth: Optional[int] = None

    @field_validator("rarefaction_depth")
    @classmethod
    def validate_rarefaction(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("rarefaction_depth must be >= 1")
        return v


class StatisticsConfig(BaseModel):
    enabled: bool = True
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    correction_method: str = "fdr_bh"
    min_prevalence: float = Field(0.1, ge=0.0, le=1.0)
    min_abundance: float = Field(0.001, ge=0.0)
    permanova_permutations: int = Field(999, ge=9)
    differential_abundance_method: str = "ancom-bc"


class FunctionalConfig(BaseModel):
    enabled: bool = True
    tool: str = "humann"
    database: Optional[str] = None
    nucleotide_db: Optional[str] = None
    protein_db: Optional[str] = None
    uniref: str = "uniref90"
    pathways: str = "metacyc"


class VisualizationConfig(BaseModel):
    enabled: bool = True
    format: str = "png"
    dpi: int = Field(300, ge=72)
    style: str = "publication"
    top_taxa: int = Field(20, ge=1)
    color_palette: str = "tab20"


class PipelineConfig(BaseModel):
    """Top-level configuration for the metagenomics pipeline."""

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    quality_control: QCConfig = Field(default_factory=QCConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    host_filtering: HostFilteringConfig = Field(default_factory=HostFilteringConfig)
    taxonomic_profiling: TaxonomicConfig = Field(default_factory=TaxonomicConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    functional_profiling: FunctionalConfig = Field(default_factory=FunctionalConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @classmethod
    def from_defaults(cls) -> "PipelineConfig":
        """Create a PipelineConfig with all default values."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            PipelineConfig instance populated from the file.

        Raises:
            ConfigurationError: If the file cannot be parsed.
        """
        from pipeline.core.exceptions import ConfigurationError

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return cls.model_validate(data)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse config file {path}: {exc}") from exc
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration in {path}: {exc}") from exc

    @classmethod
    def merge(cls, user_config_path: Optional[Path] = None) -> "PipelineConfig":
        """Load default configuration and override with user values.

        Args:
            user_config_path: Optional path to a user YAML config.

        Returns:
            Merged PipelineConfig.
        """
        defaults = cls.from_defaults().model_dump()
        if user_config_path is not None:
            with open(user_config_path, "r", encoding="utf-8") as fh:
                user_data = yaml.safe_load(fh) or {}
            # Deep merge: user values override defaults
            for section, values in user_data.items():
                if section in defaults and isinstance(values, dict):
                    defaults[section].update(values)
                else:
                    defaults[section] = values
        return cls.model_validate(defaults)

    def to_yaml(self, path: Path) -> None:
        """Serialize this configuration to a YAML file.

        Args:
            path: Destination path for the YAML file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)

    def config_hash(self) -> str:
        """Return a short SHA-256 hash of the serialised config for reproducibility."""
        data = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:12]
