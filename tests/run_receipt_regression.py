# -*- coding: utf-8 -*-
"""Fuehrt Parser-Regressionstests gegen gespeicherte OCR-Texte aus.

Aufruf im Projektordner:

    python tests/run_receipt_regression.py

Nur aktive, verifizierte Testfaelle mit nichtleerem OCR-Text werden bewertet.
Fehlende oder noch nicht gepflegte Dateien erscheinen separat als SKIP.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from receipt_pipeline import finalize_parsed_receipt  # noqa: E402

RECEIPTS_DIR = Path(__file__).resolve().parent / "receipts"
OCR_DIR = RECEIPTS_DIR / "ocr"
EXPECTED_DIR = RECEIPTS_DIR / "expected"

FIELD_MAP = {
    "store": "Laden",
    "receipt_type": "Belegtyp",
    "total": "Betrag (€)",
    "payment": "Zahlung",
}


@dataclass
class CaseResult:
    case_id: str
    status: str
    differences: list[str] = field(default_factory=list)
    note: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON-Wurzel muss ein Objekt sein")
    return value


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _matches(expected: Any, actual: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(expected), float(actual), abs_tol=0.011)
        except (TypeError, ValueError):
            return False
    return _normalise_text(expected) == _normalise_text(actual)


def run_case(expected_path: Path) -> CaseResult:
    case_id = expected_path.name.removesuffix(".expected.json")
    ocr_path = OCR_DIR / f"{case_id}.txt"

    try:
        document = _load_json(expected_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return CaseResult(case_id, "ERROR", note=f"Expected-Datei ungueltig: {exc}")

    metadata = document.get("metadata") or {}
    expected = document.get("expected") or {}

    if metadata.get("active") is False:
        return CaseResult(case_id, "SKIP", note="Testfall deaktiviert")
    if metadata.get("verified") is not True:
        return CaseResult(case_id, "SKIP", note="Sollwerte noch nicht verifiziert")
    if not ocr_path.exists():
        return CaseResult(case_id, "SKIP", note="OCR-Datei fehlt")

    try:
        raw = ocr_path.read_text(encoding="utf-8")
    except OSError as exc:
        return CaseResult(case_id, "ERROR", note=f"OCR-Datei nicht lesbar: {exc}")
    if not raw.strip():
        return CaseResult(case_id, "SKIP", note="OCR-Datei ist leer")

    initial: dict[str, Any] = {}
    if expected.get("receipt_type") is not None:
        initial["Belegtyp"] = expected["receipt_type"]

    actual = finalize_parsed_receipt(initial, raw)
    differences: list[str] = []

    for expected_key, actual_key in FIELD_MAP.items():
        wanted = expected.get(expected_key)
        if wanted is None:
            continue
        found = actual.get(actual_key)
        if not _matches(wanted, found):
            differences.append(
                f"{expected_key}: erwartet={wanted!r}, gefunden={found!r}"
            )

    if differences:
        return CaseResult(case_id, "FAIL", differences=differences)
    return CaseResult(case_id, "PASS")


def main() -> int:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    expected_files = sorted(EXPECTED_DIR.glob("*.expected.json"))

    if not expected_files:
        print("Keine Expected-Dateien gefunden.")
        print("Zuerst ausfuehren: python tests/build_receipt_fixtures.py")
        return 2

    results = [run_case(path) for path in expected_files]
    icons = {"PASS": "OK", "FAIL": "FEHLER", "SKIP": "SKIP", "ERROR": "ERROR"}

    for result in results:
        print(f"[{icons[result.status]}] {result.case_id}")
        if result.note:
            print(f"       {result.note}")
        for difference in result.differences:
            print(f"       {difference}")

    counts = {status: sum(r.status == status for r in results) for status in icons}
    evaluated = counts["PASS"] + counts["FAIL"]
    print("\nZusammenfassung")
    print("---------------")
    print(f"Testfaelle insgesamt: {len(results)}")
    print(f"Bewertet:            {evaluated}")
    print(f"Bestanden:            {counts['PASS']}")
    print(f"Fehlgeschlagen:       {counts['FAIL']}")
    print(f"Uebersprungen:        {counts['SKIP']}")
    print(f"Dateifehler:          {counts['ERROR']}")

    if evaluated:
        rate = counts["PASS"] / evaluated * 100
        print(f"Trefferquote:         {rate:.1f} %")

    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
