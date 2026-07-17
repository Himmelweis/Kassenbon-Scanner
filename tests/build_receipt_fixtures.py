# -*- coding: utf-8 -*-
"""Bereitet Testdateien fuer vorhandene Belegbilder und PDFs vor.

Das Skript veraendert oder benennt Originalbelege nicht um. Fuer jede Datei in
``tests/receipts/images`` werden anhand des Basisnamens fehlende OCR- und
Expected-Dateien angelegt.

Aufruf im Projektordner:

    python tests/build_receipt_fixtures.py

Vorhandene TXT- und JSON-Dateien werden standardmaessig nicht ueberschrieben.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}
ROOT = Path(__file__).resolve().parent
RECEIPTS_DIR = ROOT / "receipts"
IMAGES_DIR = RECEIPTS_DIR / "images"
OCR_DIR = RECEIPTS_DIR / "ocr"
EXPECTED_DIR = RECEIPTS_DIR / "expected"


def expected_template(case_id: str, source_suffix: str) -> dict[str, object]:
    return {
        "metadata": {
            "case_id": case_id,
            "source": source_suffix.lstrip(".").lower(),
            "description": "",
            "known_issue": "",
            "verified": False,
            "active": True,
        },
        "expected": {
            "store": None,
            "receipt_type": None,
            "total": None,
            "payment": None,
        },
    }


def prepare(overwrite: bool = False) -> tuple[int, int, int]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    created_ocr = 0
    created_expected = 0

    for image_path in images:
        case_id = image_path.stem
        ocr_path = OCR_DIR / f"{case_id}.txt"
        expected_path = EXPECTED_DIR / f"{case_id}.expected.json"

        if overwrite or not ocr_path.exists():
            ocr_path.write_text(
                "", encoding="utf-8"
            )
            created_ocr += 1

        if overwrite or not expected_path.exists():
            expected_path.write_text(
                json.dumps(
                    expected_template(case_id, image_path.suffix),
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            created_expected += 1

    return len(images), created_ocr, created_expected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Erzeugt fehlende OCR- und Expected-Vorlagen fuer Testbelege."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vorhandene TXT- und JSON-Dateien ueberschreiben.",
    )
    args = parser.parse_args()

    total, created_ocr, created_expected = prepare(overwrite=args.overwrite)
    print(f"{total} Belegdatei(en) gefunden")
    print(f"{created_ocr} OCR-Datei(en) angelegt")
    print(f"{created_expected} Expected-Datei(en) angelegt")

    if total == 0:
        print(f"Hinweis: Belege nach {IMAGES_DIR} kopieren.")
    else:
        print("Naechster Schritt: OCR-Texte einfuegen und Sollwerte verifizieren.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
