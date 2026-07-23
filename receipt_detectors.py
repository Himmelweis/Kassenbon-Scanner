# -*- coding: utf-8 -*-
"""Modulare Detektoren auf Basis der bestehenden Receipt-Heuristiken.

Die bisherigen Funktionen in ``receipt_pipeline`` bleiben bewusst erhalten.
Dadurch kann die neue Schnittstelle schrittweise eingefuehrt werden, ohne
bestehende Aufrufer zu brechen.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from detector_base import BaseDetector, DetectionResult, ReceiptDocument
from receipt_pipeline import (
    extract_payment_values,
    extract_store_candidate,
    extract_total_candidate,
)


def _line_number(document: ReceiptDocument, source_text: str | None) -> int | None:
    if not source_text:
        return None
    needle = source_text.casefold()
    for index, line in enumerate(document.text.splitlines(), start=1):
        if needle in line.casefold() or line.casefold() in needle:
            return index
    return None


def _confidence_from_score(score: int, maximum: int = 110) -> float:
    return max(0.0, min(1.0, score / maximum))


class StoreDetector(BaseDetector[str]):
    field = "store"

    def detect(self, document: ReceiptDocument) -> DetectionResult[str]:
        value, score, source = extract_store_candidate(
            document.text,
            document.receipt_type,
        )
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
        value, source, score = extract_total_candidate(
            document.text,
            document.receipt_type,
        )
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
        value = values.get("Zahlung")

        reasoning: list[str] = []
        if value == "Karte":
            reasoning.append("Kartenmerkmal im OCR-Text gefunden")
        elif value == "Bar":
            reasoning.append("Barzahlungsmerkmal im OCR-Text gefunden")

        confidence = 0.95 if value else 0.0
        warnings = () if value else ("Keine Zahlungsart erkannt",)

        return DetectionResult(
            field=self.field,
            value=value,
            confidence=confidence,
            detector=self.name,
            reasoning=tuple(reasoning),
            warnings=warnings,
            metadata={
                "given": values.get("Gegeben (€)"),
                "change": values.get("Wechselgeld (€)"),
                "cash_total": values.get("Betrag aus Bararithmetik (€)"),
            },
        )


class DetectorPipeline:
    """Fuehrt registrierte Detektoren generisch und in stabiler Reihenfolge aus."""

    def __init__(self, detectors: Iterable[BaseDetector[Any]] | None = None) -> None:
        self._detectors: list[BaseDetector[Any]] = []
        selected = (
            (StoreDetector(), AmountDetector(), PaymentDetector())
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
        return {
            detector.field: detector.detect(document)
            for detector in self._detectors
        }

    def detect_text(
        self,
        text: str,
        receipt_type: str = "generic",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, DetectionResult[Any]]:
        return self.detect(
            ReceiptDocument(
                text=text,
                receipt_type=receipt_type,
                metadata=metadata or {},
            )
        )
