# -*- coding: utf-8 -*-
"""Zentrale, isolierte Nachbearbeitung fuer erkannte Kassenbons.

Diese Datei veraendert den bestehenden Scanner noch nicht. Sie sammelt die
allgemeinen Regeln fuer Haendler, Gesamtbetrag, Zahlung und Qualitaetspruefung,
damit die Logik spaeter von Bild-, PDF- und Mehrteiler-Scans gemeinsam genutzt
werden kann.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MONEY_RE = r"(?<!\d)(\d{1,4}[.,]\d{2})(?!\d)"
DATE_RE = re.compile(r"\b\d{1,2}[.,]\d{1,2}[.,]\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


@dataclass(frozen=True)
class TotalCandidate:
    value: float
    score: int
    source: str
    position: int


def _money(value: str) -> float:
    return float(value.replace(".", "").replace(",", "."))


def _clean_lines(raw: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in (raw or "").splitlines() if line.strip()]


def extract_store_candidate(raw: str, receipt_type: str | None = None) -> tuple[str | None, int, str | None]:
    """Ermittelt einen plausiblen Firmennamen aus dem Kopfbereich.

    Keine Datenbank einzelner Geschaefte: bewertet werden nur Position,
    Adress-/Personen-/Belegwoerter und typische Unternehmensstrukturen.
    """
    lines = _clean_lines(raw)[:15]
    if not lines:
        return None, 0, None

    upper = [line.upper() for line in lines]

    # Struktur APOTHEKE / ROSEN -> Rosen Apotheke.
    for i, line in enumerate(upper):
        if "APOTHEKE" not in line:
            continue
        if len(lines[i].split()) > 1:
            return lines[i].title(), 95, "pharmacy-header"
        neighbours = []
        if i > 0:
            neighbours.append(lines[i - 1])
        if i + 1 < len(lines):
            neighbours.append(lines[i + 1])
        for candidate in neighbours:
            if re.fullmatch(r"[A-Za-zÄÖÜäöüß\- ]{3,30}", candidate):
                if not re.search(r"(?i)\b(HEIKE|HERR|FRAU|INHABER|INHABERIN|STRASSE|STR\.|WEG|PLATZ|TEL|FAX)\b", candidate):
                    return f"{candidate.title()} Apotheke", 95, "pharmacy-split-header"
        return "Apotheke", 70, "pharmacy-generic"

    invalid = re.compile(
        r"(?i)\b(QUITTUNG|RECHNUNG|BELEG|KUNDENBELEG|KARTENZAHLUNG|GIROCARD|"
        r"SUMME|TOTAL|GESAMT|BETRAG|DATUM|UHRZEIT|KUNDENNUMMER|BONNUMMER|"
        r"STEUER|MWST|NETTO|BRUTTO|TEL|FAX|EMAIL)\b"
    )
    address = re.compile(r"(?i)(\b\d{5}\b|STRASSE|STRAßE|STR\.|WEG|PLATZ|ALLEE|GASSE|HÖHE|HOEHE)")
    person_hint = re.compile(r"(?i)\b(HEIKE|SVEN|JUTTA|HERR|FRAU|BEDIENER|KELLNER|INHABER|INHABERIN)\b")

    candidates: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        if len(line) < 3 or len(line) > 60:
            continue
        if invalid.search(line) or address.search(line):
            continue
        if re.fullmatch(r"[\d\s.,€#*\-/]+", line):
            continue

        score = 80 - idx * 4
        if idx == 0:
            score += 15
        if any(token in line.upper() for token in ("GMBH", "KG", "AG", "OHG", "APOTHEKE", "RISTORANTE", "LANDHAUS", "TANKSTELLE")):
            score += 12
        if person_hint.search(line):
            score -= 25
        letters = sum(ch.isalpha() for ch in line)
        if letters < 4:
            score -= 30
        candidates.append((score, line, "header-score"))

    if not candidates:
        return None, 0, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, name, source = candidates[0]
    return name, max(0, score), source


def extract_total_candidate(raw: str, receipt_type: str | None = None) -> tuple[float | None, str | None, int]:
    """Waehlt den Gesamtbetrag anhand bewerteter Kontexttreffer.

    Unterstuetzt sowohl `SUMME 8,46` als auch `8,46 / zu zahlen` und lehnt
    Datums-, Uhrzeit-, Netto- und Steuerzeilen als starke Kandidaten ab.
    """
    text = raw or ""
    candidates: list[TotalCandidate] = []

    def add(pattern: str, score: int, source: str, value_group: int = 1) -> None:
        for match in re.finditer(pattern, text, re.I | re.S):
            try:
                value = _money(match.group(value_group))
            except (ValueError, IndexError):
                continue
            window = text[max(0, match.start() - 50): min(len(text), match.end() + 50)]
            if DATE_RE.search(window) and source not in {"cash-arithmetic"}:
                # Datum im unmittelbaren Trefferfenster ist ein Warnsignal.
                score_adjusted = score - 45
            else:
                score_adjusted = score
            if re.search(r"(?i)\b(NETTO|MWST|STEUER|ZUZ\.|ZUZahlung)\b", window):
                score_adjusted -= 35
            if value > 0:
                candidates.append(TotalCandidate(value, score_adjusted, source, match.start()))

    # Betrag nach Bezeichner.
    add(rf"\b(?:BAR[- ]?TOTAL|RECHNUNGSBETRAG|GESAMTBETRAG|ENDBETRAG)\b[^\d]{{0,45}}{MONEY_RE}", 110, "label-before-strong")
    add(rf"\b(?:TOTAL|SUMME|GESAMT|ZU\s+ZAHLEN)\b[^\d]{{0,45}}{MONEY_RE}", 100, "label-before")

    # Betrag vor Bezeichner.
    add(rf"{MONEY_RE}\s*(?:EUR|EURO|€)?\s*(?:\n\s*)?(?:ZU\s+ZAHLEN|BAR[- ]?TOTAL|SUMME|TOTAL|KARTENZAHLUNG)", 105, "label-after")

    # Kartenbeleg: EUR 365,15 oder Betrag 122,08.
    add(rf"\b(?:EUR|EURO|€)\s*{MONEY_RE}\b", 88, "currency-prefix")
    add(rf"\bBETRAG\b[^\d]{{0,25}}{MONEY_RE}", 85, "payment-amount")

    if not candidates:
        return None, None, 0

    candidates.sort(key=lambda item: (item.score, item.position), reverse=True)
    best = candidates[0]
    return round(best.value, 2), best.source, best.score


def extract_payment_values(raw: str) -> dict[str, Any]:
    text = raw or ""
    result: dict[str, Any] = {}

    if re.search(r"(?i)\b(GIROCARD|KARTENZAHLUNG|VISA|MASTERCARD|EC[- ]?KARTE|KONTAKTLOS)\b", text):
        result["Zahlung"] = "Karte"
    elif re.search(r"(?i)\b(BAR|BARZAHLUNG|GEGEBEN\s+BAR)\b", text):
        result["Zahlung"] = "Bar"

    given_patterns = [
        rf"{MONEY_RE}\s*(?:EUR|€)?\s*(?:\n\s*)?GEGEBEN\s+BAR",
        rf"\bGEGEBEN(?:ER\s+BETRAG)?\b[^\d]{{0,25}}{MONEY_RE}",
        rf"{MONEY_RE}\s*(?:EUR|€)?\s*(?:\n\s*)?BAR\b",
    ]
    for pattern in given_patterns:
        match = re.search(pattern, text, re.I | re.S)
        if match:
            groups = [g for g in match.groups() if g]
            result["Gegeben (€)"] = _money(groups[-1])
            break

    change_patterns = [
        rf"{MONEY_RE}\s*(?:EUR|€)?\s*(?:\n\s*)?(?:ZURÜCK|ZURUECK|RÜCKGELD|RUECKGELD)",
        rf"\b(?:ZURÜCK|ZURUECK|RÜCKGELD|RUECKGELD)\b[^\d-]{{0,25}}-?{MONEY_RE}",
    ]
    for pattern in change_patterns:
        match = re.search(pattern, text, re.I | re.S)
        if match:
            groups = [g for g in match.groups() if g]
            result["Wechselgeld (€)"] = abs(_money(groups[-1]))
            break

    given = result.get("Gegeben (€)")
    change = result.get("Wechselgeld (€)")
    if isinstance(given, (int, float)) and isinstance(change, (int, float)):
        calculated = round(given - change, 2)
        if calculated > 0:
            result["Betrag aus Bararithmetik (€)"] = calculated

    return result


def validate_final_result(best: dict[str, Any], raw: str) -> dict[str, Any]:
    reasons: list[str] = []

    if not best.get("Datum"):
        reasons.append("Datum fehlt")
    if not best.get("Uhrzeit"):
        reasons.append("Uhrzeit fehlt")

    store = str(best.get("Laden") or "").strip()
    if not store:
        reasons.append("Laden fehlt")
    elif len(store) < 3 or re.fullmatch(r"[\d\s.,€#*\-/]+", store):
        reasons.append("Laden unplausibel")

    try:
        total = float(best.get("Betrag (€)"))
        if total <= 0 or total > 10000:
            reasons.append("Betrag unplausibel")
    except (TypeError, ValueError):
        reasons.append("Betrag fehlt/ungueltig")
        total = None

    source_score = int(best.get("_total_score") or 0)
    if source_score < 80:
        reasons.append("Gesamtbetrag nicht sicher")

    if total is not None and best.get("Datum"):
        try:
            _, month, day = str(best["Datum"]).split("-")
            date_like = float(f"{int(day)}.{int(month):02d}")
            if abs(total - date_like) < 0.001:
                reasons.append("Betrag entspricht vermutlich dem Datum")
        except (ValueError, TypeError):
            pass

    given = best.get("Gegeben (€)")
    change = best.get("Wechselgeld (€)")
    if isinstance(given, (int, float)) and isinstance(change, (int, float)) and total is not None:
        if abs(round(given - change, 2) - total) > 0.02:
            reasons.append("Barzahlung rechnerisch unplausibel")

    best["Prüfstatus"] = "✅ OK" if not reasons else "🔎 Prüfen: " + ", ".join(dict.fromkeys(reasons))
    return best


def finalize_parsed_receipt(best: dict[str, Any], raw: str) -> dict[str, Any]:
    """Zentrale Nachbearbeitung ohne Speichern oder OCR."""
    best = dict(best or {})
    best["Rohtext"] = raw or best.get("Rohtext", "")

    receipt_type = str(best.get("Belegtyp") or "generic")

    store, store_score, store_source = extract_store_candidate(raw, receipt_type)
    if store and store_score >= 65:
        best["Laden"] = store
        best["_store_score"] = store_score
        best["_store_source"] = store_source

    payment = extract_payment_values(raw)
    for key in ("Zahlung", "Gegeben (€)", "Wechselgeld (€)"):
        if key in payment:
            best[key] = payment[key]

    total, total_source, total_score = extract_total_candidate(raw, receipt_type)
    cash_total = payment.get("Betrag aus Bararithmetik (€)")
    if isinstance(cash_total, (int, float)):
        if total is None or total_score < 100 or abs(float(total) - cash_total) <= 0.02:
            total = cash_total
            total_source = "cash-arithmetic"
            total_score = 105

    if total is not None:
        best["Betrag (€)"] = total
    best["_total_source"] = total_source
    best["_total_score"] = total_score

    return validate_final_result(best, raw)
