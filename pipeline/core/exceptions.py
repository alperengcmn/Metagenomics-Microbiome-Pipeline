"""Custom exception hierarchy for the metagenomics pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class ConfigurationError(PipelineError):
    """Raised when the pipeline configuration is invalid or incomplete."""


class InputDetectionError(PipelineError):
    """Raised when input FASTQ files cannot be detected or are malformed."""


class ToolNotFoundError(PipelineError):
    """Raised when an external bioinformatics tool is not available on PATH.

    Attributes
    ----------
    tool_name : str
        The name of the missing tool.
    """

    INSTALL_HINTS: dict = {
        "fastqc": "conda install -c bioconda fastqc  OR  brew install fastqc",
        "multiqc": "pip install multiqc",
        "fastp": "conda install -c bioconda fastp",
        "trimmomatic": "conda install -c bioconda trimmomatic",
        "bowtie2": "conda install -c bioconda bowtie2",
        "bwa": "conda install -c bioconda bwa",
        "kraken2": "conda install -c bioconda kraken2",
        "metaphlan": "conda install -c bioconda metaphlan",
        "bracken": "conda install -c bioconda bracken",
        "humann": "conda install -c bioconda humann",
        "samtools": "conda install -c bioconda samtools",
    }

    def __init__(self, tool_name: str, extra_message: str = "") -> None:
        self.tool_name = tool_name
        hint = self.INSTALL_HINTS.get(
            tool_name, f"Please install {tool_name} and ensure it is on your PATH."
        )
        msg = (
            f"Required tool '{tool_name}' was not found on PATH.\n"
            f"Install with: {hint}"
        )
        if extra_message:
            msg += f"\n{extra_message}"
        super().__init__(msg)


class AnalysisError(PipelineError):
    """Raised when an analysis step fails during execution.

    Attributes
    ----------
    module : str
        The pipeline module where the error occurred.
    """

    def __init__(self, module: str, message: str) -> None:
        self.module = module
        super().__init__(f"[{module}] {message}")


class InsufficientDataError(PipelineError):
    """Raised when there is not enough data to perform an analysis."""


class OutputValidationError(PipelineError):
    """Raised when an output file fails validation checks."""
