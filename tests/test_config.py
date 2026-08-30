"""Tests for pipeline configuration loading and validation."""
from __future__ import annotations

import pytest
from pathlib import Path

from pipeline.core.config import (
    PipelineConfig,
    PipelineSettings,
    DiversityConfig,
)
from pipeline.core.exceptions import ConfigurationError


class TestPipelineConfigDefaults:
    def test_from_defaults_returns_config(self):
        cfg = PipelineConfig.from_defaults()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.pipeline.threads == 4
        assert cfg.quality_control.enabled is True
        assert cfg.preprocessing.min_quality == 20

    def test_config_hash_is_string(self):
        cfg = PipelineConfig.from_defaults()
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 12

    def test_config_hash_is_deterministic(self):
        h1 = PipelineConfig.from_defaults().config_hash()
        h2 = PipelineConfig.from_defaults().config_hash()
        assert h1 == h2

    def test_different_configs_have_different_hashes(self):
        c1 = PipelineConfig.from_defaults()
        c2 = PipelineConfig.from_defaults()
        c2.pipeline.threads = 8
        assert c1.config_hash() != c2.config_hash()


class TestPipelineConfigValidation:
    def test_threads_must_be_positive(self):
        with pytest.raises(Exception):
            PipelineSettings(threads=0)

    def test_min_quality_range(self):
        from pipeline.core.config import PreprocessingConfig
        with pytest.raises(Exception):
            PreprocessingConfig(min_quality=41)
        with pytest.raises(Exception):
            PreprocessingConfig(min_quality=-1)

    def test_confidence_range(self):
        from pipeline.core.config import TaxonomicConfig
        with pytest.raises(Exception):
            TaxonomicConfig(confidence=1.5)

    def test_rarefaction_depth_must_be_positive(self):
        with pytest.raises(Exception):
            DiversityConfig(rarefaction_depth=0)

    def test_rarefaction_depth_none_is_allowed(self):
        cfg = DiversityConfig(rarefaction_depth=None)
        assert cfg.rarefaction_depth is None


class TestPipelineConfigFromYaml:
    def test_from_yaml(self, tmp_path: Path):
        yaml_content = "pipeline:\n  threads: 8\n  verbose: true\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)
        cfg = PipelineConfig.from_yaml(config_file)
        assert cfg.pipeline.threads == 8
        assert cfg.pipeline.verbose is True

    def test_from_yaml_missing_file(self, tmp_path: Path):
        with pytest.raises(Exception):
            PipelineConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_to_yaml_round_trip(self, tmp_path: Path):
        cfg1 = PipelineConfig.from_defaults()
        cfg1.pipeline.threads = 12
        out = tmp_path / "out_config.yaml"
        cfg1.to_yaml(out)
        assert out.exists()
        cfg2 = PipelineConfig.from_yaml(out)
        assert cfg2.pipeline.threads == 12


class TestPipelineConfigMerge:
    def test_merge_no_user_config(self):
        cfg = PipelineConfig.merge(None)
        assert cfg.pipeline.threads == 4

    def test_merge_overrides_defaults(self, tmp_path: Path):
        yaml_content = "pipeline:\n  threads: 16\n"
        config_file = tmp_path / "override.yaml"
        config_file.write_text(yaml_content)
        cfg = PipelineConfig.merge(config_file)
        assert cfg.pipeline.threads == 16
        # Other defaults preserved
        assert cfg.preprocessing.min_quality == 20
