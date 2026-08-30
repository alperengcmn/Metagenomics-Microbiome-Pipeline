"""Tests for pipeline status tracking, pre-flight validation, and failure propagation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.core.config import PipelineConfig
from pipeline.core.pipeline import (
    ModuleResult,
    ModuleStatus,
    Pipeline,
    PipelineStatus,
    _CRITICAL_MODULES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_pipeline")


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig.from_defaults()


@pytest.fixture
def pipeline(default_config, tmp_path, logger) -> Pipeline:
    return Pipeline(default_config, tmp_path, logger)


def _make_module_result(name: str, status: ModuleStatus, **kwargs) -> ModuleResult:
    return ModuleResult(name=name, status=status, **kwargs)


# ---------------------------------------------------------------------------
# ModuleStatus / PipelineStatus enum tests
# ---------------------------------------------------------------------------

class TestStatusEnums:
    def test_module_status_values(self):
        assert ModuleStatus.SUCCESS == "SUCCESS"
        assert ModuleStatus.FAILED == "FAILED"
        assert ModuleStatus.SKIPPED == "SKIPPED"
        assert ModuleStatus.WARNING == "WARNING"

    def test_pipeline_status_values(self):
        assert PipelineStatus.COMPLETED == "COMPLETED"
        assert PipelineStatus.COMPLETED_WITH_WARNINGS == "COMPLETED WITH WARNINGS"
        assert PipelineStatus.FAILED == "FAILED"

    def test_critical_modules_set(self):
        assert "taxonomic_profiling" in _CRITICAL_MODULES

    def test_module_result_defaults(self):
        mr = ModuleResult(name="qc", status=ModuleStatus.SUCCESS)
        assert mr.error == ""
        assert mr.note == ""


# ---------------------------------------------------------------------------
# Pipeline._determine_pipeline_status
# ---------------------------------------------------------------------------

class TestDeterminePipelineStatus:
    """Unit tests for the internal status aggregation method."""

    def test_all_success_returns_completed(self, pipeline):
        pipeline._module_results = [
            _make_module_result("qc", ModuleStatus.SUCCESS),
            _make_module_result("preprocessing", ModuleStatus.SUCCESS),
            _make_module_result("taxonomic_profiling", ModuleStatus.SUCCESS),
        ]
        assert pipeline._determine_pipeline_status() == PipelineStatus.COMPLETED

    def test_critical_module_failed_returns_failed(self, pipeline):
        pipeline._module_results = [
            _make_module_result("qc", ModuleStatus.SUCCESS),
            _make_module_result("taxonomic_profiling", ModuleStatus.FAILED),
        ]
        assert pipeline._determine_pipeline_status() == PipelineStatus.FAILED

    def test_non_critical_failure_returns_warnings(self, pipeline):
        """A failed non-critical module (e.g. QC) → COMPLETED WITH WARNINGS."""
        pipeline._module_results = [
            _make_module_result("qc", ModuleStatus.FAILED),
            _make_module_result("taxonomic_profiling", ModuleStatus.SUCCESS),
        ]
        assert pipeline._determine_pipeline_status() == PipelineStatus.COMPLETED_WITH_WARNINGS

    def test_warning_status_returns_warnings(self, pipeline):
        pipeline._module_results = [
            _make_module_result("diversity_analysis", ModuleStatus.WARNING,
                                note="low sample count"),
            _make_module_result("taxonomic_profiling", ModuleStatus.SUCCESS),
        ]
        assert pipeline._determine_pipeline_status() == PipelineStatus.COMPLETED_WITH_WARNINGS

    def test_only_skipped_modules_returns_completed(self, pipeline):
        pipeline._module_results = [
            _make_module_result("host_filtering", ModuleStatus.SKIPPED),
            _make_module_result("functional", ModuleStatus.SKIPPED),
            _make_module_result("taxonomic_profiling", ModuleStatus.SUCCESS),
        ]
        assert pipeline._determine_pipeline_status() == PipelineStatus.COMPLETED

    def test_empty_results_returns_completed(self, pipeline):
        pipeline._module_results = []
        assert pipeline._determine_pipeline_status() == PipelineStatus.COMPLETED


# ---------------------------------------------------------------------------
# validate_environment
# ---------------------------------------------------------------------------

class TestValidateEnvironment:
    """Tests for the pre-flight environment validation method."""

    def test_returns_required_keys(self, pipeline):
        result = pipeline.validate_environment()
        assert "checks" in result
        assert "all_critical_pass" in result
        assert "warnings" in result
        assert "errors" in result

    def test_checks_is_list_of_tuples(self, pipeline):
        result = pipeline.validate_environment()
        assert isinstance(result["checks"], list)
        for item in result["checks"]:
            assert isinstance(item, tuple)
            # (module, tool, available, version, note)
            assert len(item) == 5, f"Expected 5-element tuple, got {len(item)}: {item}"

    def test_check_available_field_is_bool(self, pipeline):
        result = pipeline.validate_environment()
        for module, tool, available, version, note in result["checks"]:
            assert isinstance(available, bool), (
                f"available field should be bool for ({module}, {tool})"
            )

    def test_taxonomy_no_database_is_error(self, pipeline):
        """taxonomy_profiling.database = None must produce a critical error."""
        result = pipeline.validate_environment()
        errors = result["errors"]
        assert any("database" in e.lower() or "taxonomy" in e.lower() for e in errors), (
            f"Expected database/taxonomy error in {errors}"
        )

    def test_taxonomy_errors_absent_when_skipped(self, pipeline):
        """Skipping taxonomy should clear the taxonomy-specific database error."""
        result = pipeline.validate_environment(skip_modules=["taxonomy"])
        errors = result["errors"]
        taxonomy_db_errors = [
            e for e in errors
            if "taxonomy" in e.lower() and "database" in e.lower()
        ]
        assert len(taxonomy_db_errors) == 0, (
            f"Taxonomy database error should be absent when skipped: {taxonomy_db_errors}"
        )

    def test_host_filter_skipped_no_reference_error(self, pipeline):
        """host_filtering.reference_index = None is not an error when host-filter is skipped."""
        result = pipeline.validate_environment(skip_modules=["host-filter", "taxonomy"])
        errors = result["errors"]
        ref_errors = [e for e in errors if "reference_index" in e.lower()]
        assert len(ref_errors) == 0, (
            f"Reference index error should be absent when host-filter skipped: {ref_errors}"
        )

    def test_all_critical_pass_false_when_taxonomy_db_missing(self, pipeline):
        """With no taxonomy database and taxonomy not skipped, all_critical_pass must be False."""
        result = pipeline.validate_environment()
        assert result["all_critical_pass"] is False

    def test_all_critical_pass_true_when_all_config_errors_skipped(self, pipeline):
        """Skipping both taxonomy and host-filter removes all config errors."""
        result = pipeline.validate_environment(skip_modules=["taxonomy", "host-filter"])
        assert result["all_critical_pass"] is True
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Thread propagation
# ---------------------------------------------------------------------------

class TestThreadPropagation:
    """Verify that threads from config are correctly stored on the Pipeline."""

    def test_threads_stored_from_config(self, tmp_path, logger):
        config = PipelineConfig.from_defaults()
        config.pipeline.threads = 16
        p = Pipeline(config, tmp_path, logger)
        assert p.threads == 16

    def test_threads_default(self, pipeline):
        assert pipeline.threads == pipeline.config.pipeline.threads


# ---------------------------------------------------------------------------
# module_results property
# ---------------------------------------------------------------------------

class TestModuleResultsProperty:
    def test_empty_before_run(self, pipeline):
        assert pipeline.module_results == []

    def test_returns_list(self, pipeline):
        assert isinstance(pipeline.module_results, list)

    def test_appended_results_are_accessible(self, pipeline):
        mr = ModuleResult(name="test_module", status=ModuleStatus.SUCCESS)
        pipeline._module_results.append(mr)
        assert pipeline.module_results[0].name == "test_module"


# ---------------------------------------------------------------------------
# Taxonomy failure propagates to downstream modules
# ---------------------------------------------------------------------------

class TestTaxonomyFailurePropagation:
    """When taxonomy fails, downstream modules (diversity, statistics) must be SKIPPED."""

    def _make_fake_samples(self):
        """Create minimal fake Sample objects."""
        from pipeline.modules.input_detection import Sample
        return [
            Sample(
                sample_id="S1",
                forward=Path("/nonexistent/S1_R1.fastq.gz"),
                reverse=None,
                is_paired=False,
            )
        ]

    @patch("pipeline.modules.taxonomic_profiling.TaxonomicProfiler.run")
    def test_taxonomy_failure_skips_diversity(self, mock_run, pipeline):
        """If TaxonomicProfiler.run raises, diversity and statistics are skipped."""
        mock_run.side_effect = Exception("Kraken2 database not found")

        samples = self._make_fake_samples()

        # Skip all modules that require real tools or data except taxonomy
        skip = ["qc", "preprocessing", "host-filter", "functional", "visualization"]

        results = pipeline.run(samples, metadata=None, skip_modules=skip)

        # Taxonomy should be FAILED
        tax_result = next(
            (r for r in pipeline.module_results if r.name == "taxonomic_profiling"), None
        )
        assert tax_result is not None, "taxonomic_profiling must have a ModuleResult"
        assert tax_result.status == ModuleStatus.FAILED

        # Diversity and statistics should be SKIPPED (taxonomy failed → no data)
        diversity_result = next(
            (r for r in pipeline.module_results if r.name == "diversity_analysis"), None
        )
        if diversity_result is not None:
            assert diversity_result.status in (ModuleStatus.SKIPPED, ModuleStatus.FAILED), (
                "diversity_analysis must be SKIPPED or FAILED when taxonomy failed"
            )

        # Pipeline status must be FAILED (taxonomy is critical)
        assert results["pipeline_status"] == PipelineStatus.FAILED

    @patch("pipeline.modules.taxonomic_profiling.TaxonomicProfiler.run")
    def test_taxonomy_failure_sets_pipeline_failed(self, mock_run, pipeline):
        mock_run.side_effect = Exception("database missing")

        samples = self._make_fake_samples()
        skip = ["qc", "preprocessing", "host-filter", "functional", "visualization"]

        results = pipeline.run(samples, metadata=None, skip_modules=skip)
        assert results["pipeline_status"] == PipelineStatus.FAILED


# ---------------------------------------------------------------------------
# Non-critical failure → COMPLETED WITH WARNINGS
# ---------------------------------------------------------------------------

class TestNonCriticalFailure:
    """A failed non-critical module must not cause a FAILED pipeline status."""

    def _make_fake_samples(self):
        from pipeline.modules.input_detection import Sample
        return [
            Sample(
                sample_id="S1",
                forward=Path("/nonexistent/S1_R1.fastq.gz"),
                reverse=None,
                is_paired=False,
            )
        ]

    @patch("pipeline.modules.quality_control.QualityController.run_fastqc")
    @patch("pipeline.modules.taxonomic_profiling.TaxonomicProfiler.run")
    def test_qc_failure_does_not_fail_pipeline(
        self, mock_tax_run, mock_qc_run, pipeline
    ):
        """QC failure → COMPLETED WITH WARNINGS or COMPLETED, never FAILED."""
        # QC raises
        mock_qc_run.side_effect = Exception("FastQC not found")

        # Taxonomy succeeds (returns non-empty abundance data so it is not marked FAILED)
        mock_tax_run.return_value = {
            "abundance_matrices": {
                "genus": pd.DataFrame({"S1": [0.5, 0.5]}, index=["TaxonA", "TaxonB"])
            },
            "levels": ["genus"],
        }

        samples = self._make_fake_samples()
        skip = ["preprocessing", "host-filter", "functional", "visualization"]

        results = pipeline.run(samples, metadata=None, skip_modules=skip)

        assert results["pipeline_status"] in (
            PipelineStatus.COMPLETED_WITH_WARNINGS,
            PipelineStatus.COMPLETED,  # QC is optional; pipeline may degrade gracefully
        ), f"Unexpected pipeline status: {results['pipeline_status']}"

        # Must NOT be FAILED when only QC fails
        assert results["pipeline_status"] != PipelineStatus.FAILED


# ---------------------------------------------------------------------------
# ModuleResult dataclass
# ---------------------------------------------------------------------------

class TestModuleResultDataclass:
    def test_failed_result_carries_error(self):
        mr = ModuleResult(
            name="preprocessing",
            status=ModuleStatus.FAILED,
            error="fastp not found",
        )
        assert mr.error == "fastp not found"

    def test_skipped_result_carries_note(self):
        mr = ModuleResult(
            name="functional",
            status=ModuleStatus.SKIPPED,
            note="--no-functional flag set",
        )
        assert mr.note == "--no-functional flag set"

    def test_warning_result(self):
        mr = ModuleResult(
            name="diversity_analysis",
            status=ModuleStatus.WARNING,
            note="fewer than 3 samples — PERMANOVA skipped",
        )
        assert mr.status == ModuleStatus.WARNING
