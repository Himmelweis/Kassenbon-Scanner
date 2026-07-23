# -*- coding: utf-8 -*-

import pytest

from detector_base import BaseDetector, DetectionResult, ReceiptDocument
from receipt_detectors import AmountDetector, DetectorPipeline, PaymentDetector, StoreDetector


def test_detection_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        DetectionResult(
            field="total",
            value=10.0,
            confidence=1.1,
            detector="TestDetector",
        )


def test_default_pipeline_returns_explainable_results() -> None:
    text = """Beispiel Markt GmbH
Musterstrasse 1
12345 Musterstadt
GESAMT 12,34 EUR
KARTENZAHLUNG
"""

    results = DetectorPipeline().detect_text(text)

    assert set(results) == {"store", "total", "payment"}
    assert results["store"].value == "Beispiel Markt GmbH"
    assert results["total"].value == 12.34
    assert results["payment"].value == "Karte"

    for result in results.values():
        assert isinstance(result, DetectionResult)
        assert result.detector
        assert 0.0 <= result.confidence <= 1.0


def test_store_result_contains_source_line() -> None:
    document = ReceiptDocument("Testladen GmbH\nAdresse\nSUMME 4,20")

    result = StoreDetector().detect(document)

    assert result.value == "Testladen GmbH"
    assert result.line_number == 1
    assert result.source_text == "Testladen GmbH"


def test_amount_result_exposes_rule_metadata() -> None:
    document = ReceiptDocument("Artikel 1,00\nGESAMTBETRAG 9,99")

    result = AmountDetector().detect(document)

    assert result.value == 9.99
    assert result.metadata["source_rule"] == "label-before-strong"
    assert result.metadata["raw_score"] >= 100


def test_payment_result_keeps_cash_details_in_metadata() -> None:
    document = ReceiptDocument("SUMME 8,00\nGEGEBEN 10,00\nRUECKGELD 2,00\nBAR")

    result = PaymentDetector().detect(document)

    assert result.value == "Bar"
    assert result.metadata["given"] == 10.0
    assert result.metadata["change"] == 2.0
    assert result.metadata["cash_total"] == 8.0


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
