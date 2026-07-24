# -*- coding: utf-8 -*-

import pytest

from detector_base import BaseDetector, DetectionResult, ReceiptDocument
from receipt_detectors import (
    AmountDetector,
    DateDetector,
    DetectorPipeline,
    PaymentDetector,
    ReceiptTypeDetector,
    StoreDetector,
    VATDetector,
)


def test_detection_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        DetectionResult(field="total", value=10.0, confidence=1.1, detector="TestDetector")


def test_default_pipeline_returns_explainable_results() -> None:
    text = """Beispiel Markt GmbH
Musterstrasse 1
12345 Musterstadt
DATUM 24.07.2026
GESAMT 12,34 EUR
MWST 19 % 1,97
KARTENZAHLUNG
"""
    results = DetectorPipeline().detect_text(text)

    assert set(results) == {"store", "total", "payment", "date", "vat", "receipt_type"}
    assert results["store"].value == "Beispiel Markt GmbH"
    assert results["total"].value == 12.34
    assert results["payment"].value == "Karte"
    assert results["date"].value == "2026-07-24"
    assert results["vat"].value == [19.0]
    assert results["receipt_type"].value == "card"

    for result in results.values():
        assert isinstance(result, DetectionResult)
        assert result.detector
        assert 0.0 <= result.confidence <= 1.0


def test_store_result_contains_source_line() -> None:
    result = StoreDetector().detect(ReceiptDocument("Testladen GmbH\nAdresse\nSUMME 4,20"))
    assert result.value == "Testladen GmbH"
    assert result.line_number == 1
    assert result.source_text == "Testladen GmbH"


def test_amount_result_exposes_rule_metadata() -> None:
    result = AmountDetector().detect(ReceiptDocument("Artikel 1,00\nGESAMTBETRAG 9,99"))
    assert result.value == 9.99
    assert result.metadata["source_rule"] == "label-before-strong"
    assert result.metadata["raw_score"] >= 100


def test_payment_result_keeps_cash_details_in_metadata() -> None:
    result = PaymentDetector().detect(
        ReceiptDocument("SUMME 8,00\nGEGEBEN 10,00\nRUECKGELD 2,00\nBAR")
    )
    assert result.value == "Bar"
    assert result.metadata["given"] == 10.0
    assert result.metadata["change"] == 2.0
    assert result.metadata["cash_total"] == 8.0


def test_date_detector_prefers_labeled_valid_date() -> None:
    result = DateDetector().detect(
        ReceiptDocument("Referenz 01.02.2024\nDATUM 24.07.2026\nSUMME 4,20")
    )
    assert result.value == "2026-07-24"
    assert result.line_number == 2
    assert result.confidence == 1.0


def test_date_detector_rejects_invalid_calendar_date() -> None:
    result = DateDetector().detect(ReceiptDocument("DATUM 31.02.2026"))
    assert result.value is None
    assert result.confidence == 0.0
    assert result.warnings


def test_vat_detector_collects_unique_rates_only_in_tax_context() -> None:
    result = VATDetector().detect(
        ReceiptDocument("Rabatt 50 %\nMWST 7 % 0,70\nSteuer 19% 1,90\nMWST 7% 0,70")
    )
    assert result.value == [7.0, 19.0]
    assert result.metadata["rate_count"] == 2
    assert result.line_number == 2


def test_receipt_type_detector_uses_structural_keywords() -> None:
    card = ReceiptTypeDetector().detect(ReceiptDocument("KUNDENBELEG\nGIROCARD\nEUR 12,34"))
    invoice = ReceiptTypeDetector().detect(ReceiptDocument("RECHNUNG\nRechnungsnummer 42"))
    generic = ReceiptTypeDetector().detect(ReceiptDocument("Beispiel Markt\nSUMME 4,20"))

    assert card.value == "card"
    assert invoice.value == "invoice"
    assert generic.value == "generic"
    assert generic.warnings


def test_pipeline_rejects_duplicate_fields() -> None:
    pipeline = DetectorPipeline([])
    pipeline.register(StoreDetector())
    with pytest.raises(ValueError):
        pipeline.register(StoreDetector())


def test_detector_base_class_requires_detect() -> None:
    class BrokenDetector(BaseDetector[str]):
        field = "broken"

    with pytest.raises(TypeError):
        BrokenDetector()
