# -*- coding: utf-8 -*-
"""Modulare Detektoren auf Basis allgemeiner Receipt-Heuristiken."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from detector_base import BaseDetector, DetectionResult, ReceiptDocument
from receipt_pipeline import (
    extract_payment_values,
    extract_store_candidate,
    extract_total_candidate,
)


_MONEY_TOKEN = r"(\d{1,4}[.,]\d{2})"
_DATE_TOKEN = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b")


def _line_number(document: ReceiptDocument, source_text: str | None) -> int | None:
    if not source_text:
        return None
    needle = source_text.casefold()
    for index, line in enumerate(document.text.splitlines(), start=1):
        if needle in line.casefold() or line.casefold() in needle:
            return index
    return None


def _source_line(document: ReceiptDocument, position: int) -> tuple[str | None, int | None]:
    offset = 0
    for index, line in enumerate(document.text.splitlines(keepends=True), start=1):
        if offset <= position < offset + len(line):
            return line.strip(), index
        offset += len(line)
    return None, None


def _confidence_from_score(score: int, maximum: int = 110) -> float:
    return max(0.0, min(1.0, score / maximum))


def _money(value: str) -> float:
    return float(value.replace(".", "").replace(",", "."))


def _labeled_money(text: str, labels: str) -> float | None:
    patterns = (
        rf"\b(?:{labels})\b[^\d\r\n-]{{0,25}}-?{_MONEY_TOKEN}",
        rf"{_MONEY_TOKEN}[ \t]*(?:EUR|EURO|€)?[ \t]*(?:{labels})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return abs(_money(match.group(1)))
    return None


class StoreDetector(BaseDetector[str]):
    field = "store"

    def detect(self, document: ReceiptDocument) -> DetectionResult[str]:
        value, score, source = extract_store_candidate(document.text, document.receipt_type)
        warnings: tuple[str, ...] = ()
        if value is None:
            warnings = ("Kein plausibler Haendler im Kopfbereich gefunden",)
        elif score < 65:
            warnings = ("Haendlerkandidat liegt unter der Uebernahmeschwelle",)
        return DetectionResult(
            field=self.field,
            value=value,
            confidence=_confidence_from_score(score, maximum=107),
            detector=self.name,
            source_text=value,
            line_number=_line_number(document, value),
            reasoning=(source,) if source else (),
            warnings=warnings,
            metadata={"raw_score": score},
        )


class AmountDetector(BaseDetector[float]):
    field = "total"

    def detect(self, document: ReceiptDocument) -> DetectionResult[float]:
        value, source, score = extract_total_candidate(document.text, document.receipt_type)
        warnings: tuple[str, ...] = ()
        if value is None:
            warnings = ("Kein Gesamtbetrag gefunden",)
        elif score < 80:
            warnings = ("Gesamtbetrag ist nur schwach belegt",)
        return DetectionResult(
            field=self.field,
            value=value,
            confidence=_confidence_from_score(score),
            detector=self.name,
            reasoning=(source,) if source else (),
            warnings=warnings,
            metadata={"raw_score": score, "source_rule": source},
        )


class PaymentDetector(BaseDetector[str]):
    field = "payment"

    def detect(self, document: ReceiptDocument) -> DetectionResult[str]:
        values = extract_payment_values(document.text)
        given = _labeled_money(document.text, r"GEGEBEN(?:ER[ \t]+BETRAG)?")
        change = _labeled_money(document.text, r"ZURÜCK|ZURUECK|RÜCKGELD|RUECKGELD")
        if given is not None:
            values["Gegeben (€)"] = given
        if change is not None:
            values["Wechselgeld (€)"] = change
        if given is not None and change is not None and given >= change:
            values["Betrag aus Bararithmetik (€)"] = round(given - change, 2)

        value = values.get("Zahlung")
        reasoning: list[str] = []
        if value == "Karte":
            reasoning.append("Kartenmerkmal im OCR-Text gefunden")
        elif value == "Bar":
            reasoning.append("Barzahlungsmerkmal im OCR-Text gefunden")
        return DetectionResult(
            field=self.field,
            value=value,
            confidence=0.95 if value else 0.0,
            detector=self.name,
            reasoning=tuple(reasoning),
            warnings=() if value else ("Keine Zahlungsart erkannt",),
            metadata={
                "given": values.get("Gegeben (€)"),
                "change": values.get("Wechselgeld (€)"),
                "cash_total": values.get("Betrag aus Bararithmetik (€)"),
            },
        )


class DateDetector(BaseDetector[str]):
    field = "date"

    def detect(self, document: ReceiptDocument) -> DetectionResult[str]:
        candidates: list[tuple[int, int, str, str]] = []
        for match in _DATE_TOKEN.finditer(document.text):
            day, month, year = (int(part) for part in match.groups())
            if year < 100:
                year += 2000
            try:
                normalized = date(year, month, day).isoformat()
            except ValueError:
                continue
            source_line, _ = _source_line(document, match.start())
            labeled = bool(source_line and re.search(r"(?i)\b(DATUM|DATE|BELEGDATUM)\b", source_line))
            score = 100 if labeled else 80
            candidates.append((score, -match.start(), normalized, source_line or match.group(0)))

        if not candidates:
            return DetectionResult(
                field=self.field,
                value=None,
                confidence=0.0,
                detector=self.name,
                warnings=("Kein gueltiges Datum gefunden",),
            )

        candidates.sort(reverse=True)
        score, _, value, source_text = candidates[0]
        return DetectionResult(
            field=self.field,
            value=value,
            confidence=_confidence_from_score(score, maximum=100),
            detector=self.name,
            source_text=source_text,
            line_number=_line_number(document, source_text),
            reasoning=("Datumsbezeichner in derselben Zeile" if score == 100 else "Plausibles Datumsformat im OCR-Text",),
            metadata={"raw_score": score},
        )


class VATDetector(BaseDetector[list[float]]):
    field = "vat"

    def detect(self, document: ReceiptDocument) -> DetectionResult[list[float]]:
        rates: set[float] = set()
        source_lines: list[str] = []
        for line in document.text.splitlines():
            if not re.search(r"(?i)\b(MWST|MWST\.|UST|VAT|STEUER)\b", line):
                continue
            for match in re.finditer(r"(?<!\d)(\d{1,2}(?:[.,]\d+)?)\s*%", line):
                rate = float(match.group(1).replace(",", "."))
                if 0 < rate <= 30:
                    rates.add(rate)
                    source_lines.append(line.strip())

        value = sorted(rates)
        if not value:
            return DetectionResult(
                field=self.field,
                value=[],
                confidence=0.0,
                detector=self.name,
                warnings=("Kein MwSt.-Satz mit Steuerkontext gefunden",),
            )

        source_text = source_lines[0] if source_lines else None
        return DetectionResult(
            field=self.field,
            value=value,
            confidence=0.95,
            detector=self.name,
            source_text=source_text,
            line_number=_line_number(document, source_text),
            reasoning=("Prozentsatz in einer Steuerzeile gefunden",),
            metadata={"rate_count": len(value)},
        )


class ReceiptTypeDetector(BaseDetector[str]):
    field = "receipt_type"

    def detect(self, document: ReceiptDocument) -> DetectionResult[str]:
        text = document.text.upper()
        rules = (
            ("card", 100, r"\b(KUNDENBELEG|HÄNDLERBELEG|HAENDLERBELEG|GIROCARD|KARTENZAHLUNG)\b", "Merkmal eines Kartenbelegs"),
            ("invoice", 95, r"\b(RECHNUNG|RECHNUNGSNUMMER|INVOICE)\b", "Rechnungsmerkmal"),
            ("receipt", 80, r"\b(KASSENBON|QUITTUNG|BONNUMMER|BELEGNUMMER)\b", "Kassenbonmerkmal"),
        )
        for value, score, pattern, reason in rules:
            match = re.search(pattern, text)
            if match:
                source_text, line_number = _source_line(document, match.start())
                return DetectionResult(
                    field=self.field,
                    value=value,
                    confidence=_confidence_from_score(score, maximum=100),
                    detector=self.name,
                    source_text=source_text,
                    line_number=line_number,
                    reasoning=(reason,),
                    metadata={"raw_score": score},
                )
        return DetectionResult(
            field=self.field,
            value="generic",
            confidence=0.35,
            detector=self.name,
            reasoning=("Kein eindeutiges Belegtyp-Merkmal gefunden",),
            warnings=("Belegtyp nur generisch klassifiziert",),
        )


class DetectorPipeline:
    """Fuehrt registrierte Detektoren generisch und in stabiler Reihenfolge aus."""

    def __init__(self, detectors: Iterable[BaseDetector[Any]] | None = None) -> None:
        self._detectors: list[BaseDetector[Any]] = []
        selected = (
            (
                StoreDetector(),
                AmountDetector(),
                PaymentDetector(),
                DateDetector(),
                VATDetector(),
                ReceiptTypeDetector(),
            )
            if detectors is None
            else detectors
        )
        for detector in selected:
            self.register(detector)

    @property
    def detectors(self) -> tuple[BaseDetector[Any], ...]:
        return tuple(self._detectors)

    def register(self, detector: BaseDetector[Any]) -> None:
        if any(item.field == detector.field for item in self._detectors):
            raise ValueError(f"Detektor fuer Feld {detector.field!r} ist bereits registriert")
        self._detectors.append(detector)

    def detect(self, document: ReceiptDocument) -> dict[str, DetectionResult[Any]]:
        return {detector.field: detector.detect(document) for detector in self._detectors}

    def detect_text(
        self,
        text: str,
        receipt_type: str = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, DetectionResult[Any]]:
        return self.detect(ReceiptDocument(text=text, receipt_type=receipt_type, metadata=metadata or {}))
