![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

# Metagenomics & Microbiome Pipeline

An end-to-end Python pipeline for metagenomics and 16S/shotgun microbiome analysis. Starting from raw FASTQ files, the pipeline runs quality control, read trimming, host filtering, taxonomic profiling, diversity analysis, statistical testing, functional annotation, and report generation — all from a single command.

---

## Pipeline Flow

```
Raw FASTQ
    │
    ▼
Quality Control (FastQC + MultiQC)
    │
    ▼
Preprocessing (fastp / trimmomatic)
    │
    ▼
Host Filtering (bowtie2 / bwa)
    │
    ▼
Taxonomic Profiling (Kraken2 / MetaPhlAn)   ← required
    │
    ├──▶ Diversity Analysis (Shannon, Simpson, Chao1, Bray-Curtis, PCoA, NMDS)
    │
    ├──▶ Statistical Analysis (PERMANOVA, Mann-Whitney, Kruskal-Wallis, DA)
    │
    └──▶ Functional Profiling (HUMAnN3)
              │
              ▼
    Biological Interpretation
              │
              ▼
    HTML + Markdown Report
```

---

## Key Capabilities

- Auto-detects paired-end and single-end FASTQ files from any naming convention (`_R1/_R2`, `_1/_2`)
- Supports Kraken2 + Bracken or MetaPhlAn for taxonomic profiling
- All alpha diversity metrics implemented natively (no bioinformatics tool dependencies for analysis)
- PERMANOVA with configurable permutations using the correct SS-decomposition formula (Anderson 2001)
- CLR-transformed differential abundance testing with FDR correction
- PCoA via Gower's double-centring; NMDS via scikit-learn (API-compatible with sklearn ≥ 1.8)
- HUMAnN3 functional profiling (optional — pipeline continues if HUMAnN is not installed)
- Publication-quality figures: composition stacked bars, heatmaps, volcano plots, rarefaction curves, ordination
- Pre-flight environment validation table before each run
- Accurate pipeline completion status: `COMPLETED`, `COMPLETED WITH WARNINGS`, or `FAILED`
- Per-module status table in the terminal output after each run
- HTML report with embedded figures and a reproducibility hash for every run
- Full Pydantic v2 configuration with YAML merge (defaults overridden by user config)

---

## Installation

```bash
git clone <repo>
cd "Metagenomics & Microbiome Pipeline"
pip install -e .
```

External tools — install the ones needed for the stages you want to run:

```bash
# Taxonomic profiling (required for diversity, statistics, and interpretation)
conda install -c bioconda kraken2 bracken
# or: conda install -c bioconda metaphlan

# QC
conda install -c bioconda fastqc multiqc

# Preprocessing
conda install -c bioconda fastp
# or: conda install -c bioconda trimmomatic

# Host filtering
conda install -c bioconda bowtie2 samtools
# or: conda install -c bioconda bwa samtools

# Functional profiling (optional)
conda install -c bioconda humann
```

> **Note**: Taxonomic profiling is not optional. If it is skipped or fails, downstream modules (diversity analysis, statistical analysis, biological interpretation) cannot run. All other stages can be skipped individually with `--skip-modules`.

---

## Quick Start

```bash
# Check which external tools are available
metapipe check-tools

# Generate a default config to customise
metapipe generate-config my_config.yaml

# Run the full pipeline
metapipe run \
  --input  /path/to/fastq/ \
  --output /path/to/results/ \
  --metadata config/example_metadata.tsv \
  --config  my_config.yaml \
  --threads 8

# Validate inputs without running (dry run)
metapipe run --input /path/to/fastq/ --output /tmp/out --dry-run

# Skip specific modules
metapipe run --input ... --output ... --skip-modules host-filter --skip-modules functional
```

---

## Configuration

Override any default by providing a YAML file with only the keys you want to change:

```yaml
pipeline:
  threads: 16

taxonomic_profiling:
  tool: kraken2
  database: /data/kraken2_db
  confidence: 0.1

diversity:
  alpha_metrics: [shannon, simpson, chao1, observed]
  beta_metrics: [braycurtis, jaccard]

statistics:
  alpha: 0.05
  correction_method: fdr_bh
  permanova_permutations: 999

visualization:
  format: png
  dpi: 300
  top_taxa: 20
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline.threads` | 4 | CPU threads passed to every external tool |
| `taxonomic_profiling.database` | null | Path to Kraken2 or MetaPhlAn database **(required)** |
| `host_filtering.reference_index` | null | Path to bowtie2 (or bwa) host genome index prefix |
| `statistics.correction_method` | `fdr_bh` | Multiple testing correction (`fdr_bh`, `bonferroni`, `holm`) |
| `diversity.rarefaction_depth` | null | Auto (10th percentile of read counts) |

---

## Analysis Modules

| Module | Description | External Tools | Required? |
|--------|-------------|----------------|-----------|
| `quality_control` | Per-sample and aggregate QC reports | FastQC, MultiQC | Optional |
| `preprocessing` | Adapter trimming, quality filtering | fastp or trimmomatic | Optional |
| `host_filtering` | Remove host genome reads | bowtie2 + samtools, or bwa + samtools | Optional |
| `taxonomic_profiling` | Abundance matrices at all taxonomic levels | Kraken2+Bracken **or** MetaPhlAn | **Required** |
| `diversity_analysis` | Alpha (Shannon, Simpson, Chao1) and beta diversity (Bray-Curtis, Jaccard, PCoA, NMDS) | None (pure Python) | Depends on taxonomy |
| `statistical_analysis` | PERMANOVA, group alpha tests, differential abundance (CLR+FDR) | None (pure Python) | Depends on taxonomy |
| `functional_profiling` | Pathway and gene family abundance | HUMAnN3 | Optional |
| `biological_interpretation` | Conservative, structured textual interpretation of all results | None | Depends on taxonomy |
| `visualization` | Stacked bars, heatmaps, volcano plot, rarefaction curves, ordination | matplotlib, seaborn | Optional |
| `report_generator` | HTML report + Markdown summary | None | Always runs |

---

## Pipeline Status

After every run the terminal displays a per-module status table and a final status banner:

| Status | Meaning |
|--------|---------|
| `COMPLETED` | All enabled modules succeeded |
| `COMPLETED WITH WARNINGS` | One or more non-critical modules failed or produced caveats; results are still available |
| `FAILED` | A critical module (`taxonomic_profiling`) failed; downstream results are not meaningful |

A non-zero exit code is returned only when the pipeline status is `FAILED`.

---

## Output Structure

```
results/
├── pipeline.log                  # Full run log
├── pipeline_report.html          # Interactive HTML report
├── summary.md                    # Concise Markdown summary
├── figures/
│   ├── composition_genus.png
│   ├── heatmap_genus.png
│   ├── alpha_shannon.png
│   ├── pcoa.png
│   ├── nmds.png
│   ├── volcano.png
│   └── rarefaction.png
├── qc/
│   ├── pre/                      # Pre-trimming FastQC output
│   └── post/                     # Post-trimming FastQC output
├── preprocessed/                 # Trimmed FASTQ files
├── host_filtered/                # Non-host FASTQ files
├── taxonomy/
│   ├── species_abundance.tsv
│   ├── genus_abundance.tsv
│   └── ...
└── functional/                   # HUMAnN3 output
```

---

## Metadata Format

A TSV file with a required `sample_id` column matching FASTQ file prefixes:

```
sample_id	group	subject	timepoint	age	sex
SRR001	Control	S01	baseline	32	F
SRR004	Treatment	S04	baseline	37	M
```

See `config/example_metadata.tsv` for a complete example.

---

## Reproducibility

Every run produces a `config_hash` — a 12-character SHA-256 hash of the full configuration. This hash is embedded in the HTML report and in the log, allowing exact parameter sets to be identified and shared.

To reproduce a run:
1. Use the same `--config` YAML file (verify the hash matches)
2. Use the same reference databases (Kraken2/MetaPhlAn)
3. Use the same input FASTQ files

---

## Technical Highlights

- **Type-safe configuration** via Pydantic v2 with field-level validators
- **Correct PERMANOVA** using the SS-decomposition approach from Anderson (2001), fully permutation-based
- **CLR transformation** for compositional data before differential abundance testing
- **Accurate status reporting** — pipeline exits with an error code and marks status as `FAILED` when a critical module fails; non-critical failures yield `COMPLETED WITH WARNINGS`
- **Thread propagation** — `pipeline.threads` is forwarded to every external tool; no hardcoded thread count
- **Real host-read extraction** — the bwa path runs a `bwa mem | samtools view | samtools fastq` pipeline to actually extract unmapped reads (not a placeholder)
- **Visible parse errors** — `fastp` JSON parse failures are logged as warnings instead of silently returning zeros
- **No data directory** committed to the repository; all outputs are user-specified at runtime
- **Production logging** via `rich.logging.RichHandler` to console and plain-text file handler

---

## Project Structure

```
Metagenomics & Microbiome Pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── default_config.yaml
│   └── example_metadata.tsv
├── pipeline/
│   ├── main.py                        # Click CLI entry point
│   ├── core/
│   │   ├── config.py                  # Pydantic v2 configuration models
│   │   ├── exceptions.py              # Custom exception hierarchy
│   │   ├── logging_setup.py           # Rich + file logging
│   │   ├── pipeline.py                # Orchestrator with status tracking
│   │   └── utils.py                   # Subprocess, file, formatting utilities
│   ├── modules/
│   │   ├── input_detection.py         # FASTQ auto-detection
│   │   ├── quality_control.py         # FastQC / MultiQC wrapper
│   │   ├── preprocessing.py           # fastp / trimmomatic wrapper
│   │   ├── host_filtering.py          # bowtie2 / bwa+samtools wrapper
│   │   ├── taxonomic_profiling.py     # Kraken2 / MetaPhlAn wrapper + parsers
│   │   ├── diversity_analysis.py      # Native alpha/beta diversity + ordination
│   │   ├── statistical_analysis.py    # PERMANOVA, DA, alpha tests
│   │   ├── functional_profiling.py    # HUMAnN3 wrapper
│   │   ├── biological_interpretation.py
│   │   └── visualization.py           # matplotlib / seaborn figures
│   └── reporting/
│       └── report_generator.py        # HTML + Markdown report
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_input_detection.py
    ├── test_diversity_analysis.py
    ├── test_statistical_analysis.py
    ├── test_cli.py
    └── test_pipeline_status.py
```
