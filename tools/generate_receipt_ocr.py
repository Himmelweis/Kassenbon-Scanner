# -*- coding: utf-8 -*-
"""Erzeugt rohe OCR-Referenztexte fuer die Receipt-Regressionstests.

Verarbeitung:
- Bilder werden mit Tesseract erkannt.
- PDFs liefern zuerst eingebetteten Text, sofern ausreichend vorhanden.
- Scan-PDFs werden seitenweise mit PyMuPDF gerendert und per OCR erkannt.
- Vorhandene, nicht-leere OCR-Dateien werden standardmaessig nicht ersetzt.

Beispiele:
    python tools/generate_receipt_ocr.py
    python tools/generate_receipt_ocr.py --overwrite
    python tools/generate_receipt_ocr.py --language deu+eng
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - verständliche Laufzeitmeldung unten
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = ROOT / "tests" / "receipts" / "images"
DEFAULT_OCR_DIR = ROOT / "tests" / "receipts" / "ocr"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def normalize_raw_text(text: str) -> str:
    """Normalisiert nur technische Zeilenumbrueche, nicht den OCR-Inhalt."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def require_dependencies() -> None:
    missing: list[str] = []
    if Image is None:
        missing.append("Pillow")
    if pytesseract is None:
        missing.append("pytesseract")
    if fitz is None:
        missing.append("PyMuPDF")
    if missing:
        raise RuntimeError(
            "Fehlende Python-Pakete: " + ", ".join(missing) +
            ". Installiere sie mit: pip install pillow pytesseract pymupdf"
        )


def prepare_image(image: "Image.Image") -> "Image.Image":
    """Behutsame Vorbereitung ohne inhaltliche Textbereinigung."""
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def ocr_image(image: "Image.Image", language: str, config: str) -> str:
    return pytesseract.image_to_string(prepare_image(image), lang=language, config=config)


def extract_image_text(path: Path, language: str, config: str) -> str:
    with Image.open(path) as image:
        pages: list[str] = []
        frame_count = getattr(image, "n_frames", 1)
        for frame_index in range(frame_count):
            if frame_count > 1:
                image.seek(frame_index)
            pages.append(ocr_image(image.copy(), language, config))
    return "\n\n".join(pages)


def embedded_pdf_text(document: "fitz.Document") -> str:
    pages = [page.get_text("text") for page in document]
    return "\n\n".join(pages)


def has_useful_embedded_text(text: str, minimum_characters: int) -> bool:
    useful = "".join(character for character in text if character.isalnum())
    return len(useful) >= minimum_characters


def render_pdf_page(page: "fitz.Page", dpi: int) -> "Image.Image":
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    mode = "RGB" if pixmap.n >= 3 else "L"
    return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)


def extract_pdf_text(
    path: Path,
    language: str,
    config: str,
    dpi: int,
    minimum_embedded_characters: int,
) -> tuple[str, str]:
    with fitz.open(path) as document:
        embedded = embedded_pdf_text(document)
        if has_useful_embedded_text(embedded, minimum_embedded_characters):
            return embedded, "PDF-Text"

        pages: list[str] = []
        for page in document:
            with render_pdf_page(page, dpi) as image:
                pages.append(ocr_image(image, language, config))
        return "\n\n".join(pages), "PDF-OCR"


def receipt_files(images_dir: Path) -> Iterable[Path]:
    return sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR-Texte fuer Receipt-Testfixtures erzeugen.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Auch nicht-leere OCR-Dateien ersetzen.")
    parser.add_argument("--language", default="deu+eng", help="Tesseract-Sprachen, Standard: deu+eng")
    parser.add_argument("--tesseract-config", default="--psm 6", help="Zusätzliche Tesseract-Optionen.")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="Render-Auflösung fuer Scan-PDFs.")
    parser.add_argument(
        "--minimum-embedded-characters",
        type=int,
        default=30,
        help="Mindestzahl alphanumerischer Zeichen fuer eingebetteten PDF-Text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        require_dependencies()
    except RuntimeError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    images_dir = args.images_dir.resolve()
    ocr_dir = args.ocr_dir.resolve()

    if not images_dir.is_dir():
        print(f"Fehler: Bildverzeichnis nicht gefunden: {images_dir}", file=sys.stderr)
        return 2

    ocr_dir.mkdir(parents=True, exist_ok=True)
    files = list(receipt_files(images_dir))
    if not files:
        print(f"Keine unterstützten Belege gefunden in: {images_dir}")
        return 0

    written = skipped = failed = empty = 0

    for index, source in enumerate(files, start=1):
        target = ocr_dir / f"{source.stem}.txt"
        if target.exists() and target.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"[{index:>2}/{len(files)}] ÜBERSPRUNGEN  {source.name}")
            continue

        try:
            if source.suffix.lower() == ".pdf":
                text, method = extract_pdf_text(
                    source,
                    args.language,
                    args.tesseract_config,
                    args.pdf_dpi,
                    args.minimum_embedded_characters,
                )
            else:
                text = extract_image_text(source, args.language, args.tesseract_config)
                method = "Bild-OCR"

            normalized = normalize_raw_text(text)
            target.write_text(normalized, encoding="utf-8")
            written += 1

            if not normalized.strip():
                empty += 1
                print(f"[{index:>2}/{len(files)}] LEER          {source.name} ({method})")
            else:
                line_count = len(normalized.splitlines())
                print(f"[{index:>2}/{len(files)}] GESCHRIEBEN   {source.name} ({method}, {line_count} Zeilen)")
        except Exception as exc:  # Einzelne Problemdatei soll den Batch nicht stoppen.
            failed += 1
            print(f"[{index:>2}/{len(files)}] FEHLER        {source.name}: {exc}", file=sys.stderr)

    print(
        "\nFertig: "
        f"{written} geschrieben, {skipped} übersprungen, "
        f"{failed} fehlgeschlagen, {empty} ohne erkannten Text."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
