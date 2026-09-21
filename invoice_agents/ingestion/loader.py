"""Turn an invoice file into text the extractor can read.

Every format reduces to raw text and goes to the model the same way. The temptation is
to parse the structured formats natively -- JSON and XML are right there -- but the CSVs
alone come in two incompatible shapes (INV-1006 is key/value with repeated keys that
clobber under DictReader; INV-1007 is tabular with trailing summary rows), and a real
inbox contains formats nobody enumerated. Uniform text in, structured data out, with
Python verifying the numbers afterwards, beats a parser per vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".txt", ".json", ".csv", ".xml", ".pdf"})

from ..thresholds import MAX_DOCUMENT_CHARS as MAX_CHARS


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be turned into text."""


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    source_format: str
    raw_text: str

    @property
    def is_empty(self) -> bool:
        return not self.raw_text.strip()


def _load_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DocumentLoadError(
            f"Reading {path.name} needs pdfplumber. Install it with: "
            'pip install -e ".[parsers]"'
        ) from exc

    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF {path.name}: {exc}") from exc

    return "\n".join(pages)


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Legacy exports from the vendor side are not reliably UTF-8.
        return path.read_text(encoding="latin-1")


def load_document(path: str | Path) -> LoadedDocument:
    """Read an invoice file into text.

    Raises:
        DocumentLoadError: missing file, unsupported suffix, or unreadable content.
    """
    path = Path(path)

    if not path.exists():
        raise DocumentLoadError(f"No such invoice file: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentLoadError(
            f"Unsupported format {suffix!r} for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    raw_text = _load_pdf(path) if suffix == ".pdf" else _load_text(path)

    if len(raw_text) > MAX_CHARS:
        raw_text = raw_text[:MAX_CHARS] + "\n\n[truncated]"

    document = LoadedDocument(
        path=path, source_format=suffix.lstrip("."), raw_text=raw_text
    )

    if document.is_empty:
        raise DocumentLoadError(
            f"{path.name} produced no text. If it is a scanned PDF it needs OCR, "
            "which this pipeline does not perform."
        )

    return document
