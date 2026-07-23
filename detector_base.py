# -*- coding: utf-8 -*-
"""Gemeinsame Schnittstellen und Datentraeger fuer Receipt-Detektoren."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ReceiptDocument:
    """Normalisierte Eingabe fuer alle Detektoren.

    Weitere strukturierte OCR-Daten koennen spaeter ergaenzt werden, ohne die
    Signatur der Detektoren zu veraendern.
    """

    text: str
    receipt_type: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lines(self) -> list[str]:
        return [line.strip() for line in self.text.splitlines() if line.strip()]


@dataclass(frozen=True)
class DetectionAlternative(Generic[T]):
    value: T
    confidence: float
    source_text: str | None = None


@dataclass(frozen=True)
class DetectionResult(Generic[T]):
    """Ein Detektionsergebnis inklusive nachvollziehbarer Herleitung."""

    field: str
    value: T | None
    confidence: float
    detector: str
    source_text: str | None = None
    line_number: int | None = None
    reasoning: tuple[str, ...] = ()
    alternatives: tuple[DetectionAlternative[T], ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence muss zwischen 0.0 und 1.0 liegen")
        if self.line_number is not None and self.line_number < 1:
            raise ValueError("line_number ist 1-basiert und muss positiv sein")

    @property
    def found(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert das Ergebnis fuer GUI, Debug-Ausgabe oder JSON."""
        return asdict(self)


class BaseDetector(ABC, Generic[T]):
    """Einheitliche Basisklasse fuer austauschbare Receipt-Detektoren."""

    field: str

    @abstractmethod
    def detect(self, document: ReceiptDocument) -> DetectionResult[T]:
        """Analysiert ein Dokument und liefert stets ein DetectionResult."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
