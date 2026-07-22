# -*- coding: utf-8 -*-
"""Grafischer Editor fuer Kassenbon-Regressionsdaten.

Aufruf im Projektordner:

    python tools/receipt_verifier.py

Das Werkzeug liest die Belege aus ``tests/receipts/images``, den OCR-Text aus
``tests/receipts/ocr`` und schreibt die Sollwerte nach
``tests/receipts/expected/<name>.expected.json``.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

try:
    from PIL import Image, ImageTk
except ImportError:  # Bildvorschau bleibt optional.
    Image = None
    ImageTk = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from receipt_pipeline import (
        extract_payment_values,
        extract_store_candidate,
        extract_total_candidate,
    )
except ImportError:
    extract_payment_values = None
    extract_store_candidate = None
    extract_total_candidate = None

RECEIPTS_DIR = PROJECT_ROOT / "tests" / "receipts"
IMAGES_DIR = RECEIPTS_DIR / "images"
OCR_DIR = RECEIPTS_DIR / "ocr"
EXPECTED_DIR = RECEIPTS_DIR / "expected"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}
RECEIPT_TYPES = (
    "",
    "Lebensmittel",
    "Tankstelle",
    "Restaurant",
    "Apotheke",
    "Baumarkt",
    "Hotel",
    "Sonstige",
)
PAYMENT_TYPES = ("", "Bar", "Karte", "Gemischt", "Unbekannt")


def empty_fixture(case_id: str, source: str) -> dict[str, Any]:
    return {
        "metadata": {
            "case_id": case_id,
            "source": source.lstrip(".").lower(),
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


class ReceiptVerifier(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Kassenbon-Testfälle verifizieren")
        self.geometry("1280x820")
        self.minsize(980, 650)

        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        OCR_DIR.mkdir(parents=True, exist_ok=True)

        self.cases = self._discover_cases()
        self.index = self._first_unverified_index()
        self.current_image: Any = None
        self.current_fixture: dict[str, Any] = {}

        self.store_var = tk.StringVar()
        self.receipt_type_var = tk.StringVar()
        self.total_var = tk.StringVar()
        self.payment_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.issue_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.suggestion_var = tk.StringVar()

        self._build_ui()
        self.bind("<Control-s>", lambda _event: self.save_and_next())
        self.bind("<Control-Right>", lambda _event: self.skip())
        self.bind("<Control-Left>", lambda _event: self.previous())

        if self.cases:
            self.load_case(self.index)
        else:
            self._show_empty_state()

    def _discover_cases(self) -> list[Path]:
        if not IMAGES_DIR.exists():
            return []
        return sorted(
            path
            for path in IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _expected_path(self, image_path: Path) -> Path:
        return EXPECTED_DIR / f"{image_path.stem}.expected.json"

    def _ocr_path(self, image_path: Path) -> Path:
        return OCR_DIR / f"{image_path.stem}.txt"

    def _read_fixture(self, image_path: Path) -> dict[str, Any]:
        path = self._expected_path(image_path)
        if not path.exists():
            return empty_fixture(image_path.stem, image_path.suffix)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Ungültige JSON-Datei", f"{path.name}\n\n{exc}")
            return empty_fixture(image_path.stem, image_path.suffix)

        template = empty_fixture(image_path.stem, image_path.suffix)
        template["metadata"].update(data.get("metadata") or {})
        template["expected"].update(data.get("expected") or {})
        return template

    def _first_unverified_index(self) -> int:
        for index, image_path in enumerate(self.cases):
            fixture = self._read_fixture(image_path)
            if not bool(fixture["metadata"].get("verified")):
                return index
        return 0

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(12, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Kassenbon-Verifizierer", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=12)

        preview_frame = ttk.Labelframe(body, text="Original", padding=8)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, anchor="center", justify="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        body.add(preview_frame, weight=2)

        center = ttk.Panedwindow(body, orient=tk.VERTICAL)
        ocr_frame = ttk.Labelframe(center, text="OCR-Text", padding=8)
        ocr_frame.columnconfigure(0, weight=1)
        ocr_frame.rowconfigure(0, weight=1)
        self.ocr_text = tk.Text(ocr_frame, wrap="word", font=("Consolas", 10), undo=False)
        ocr_scroll = ttk.Scrollbar(ocr_frame, orient="vertical", command=self.ocr_text.yview)
        self.ocr_text.configure(yscrollcommand=ocr_scroll.set)
        self.ocr_text.grid(row=0, column=0, sticky="nsew")
        ocr_scroll.grid(row=0, column=1, sticky="ns")
        center.add(ocr_frame, weight=3)

        suggestion_frame = ttk.Labelframe(center, text="Vorschlag der aktuellen Pipeline", padding=8)
        ttk.Label(suggestion_frame, textvariable=self.suggestion_var, justify="left").pack(
            fill="both", expand=True
        )
        ttk.Button(
            suggestion_frame,
            text="Vorschläge in leere Felder übernehmen",
            command=self.apply_suggestions,
        ).pack(anchor="e", pady=(8, 0))
        center.add(suggestion_frame, weight=1)
        body.add(center, weight=3)

        form = ttk.Labelframe(body, text="Verifizierte Sollwerte", padding=12)
        form.columnconfigure(1, weight=1)
        row = 0
        row = self._entry_row(form, row, "Händler", self.store_var)

        ttk.Label(form, text="Belegtyp").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.receipt_type_var,
            values=RECEIPT_TYPES,
            state="normal",
        ).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1

        row = self._entry_row(form, row, "Gesamtbetrag", self.total_var)

        ttk.Label(form, text="Zahlung").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.payment_var,
            values=PAYMENT_TYPES,
            state="normal",
        ).grid(row=row, column=1, sticky="ew", pady=5)
        row += 1

        row = self._entry_row(form, row, "Beschreibung", self.description_var)
        row = self._entry_row(form, row, "Bekanntes Problem", self.issue_var)

        ttk.Separator(form).grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        row += 1
        ttk.Label(
            form,
            text="Speichern setzt verified=true.\nLeere Sollwerte werden als null gespeichert.",
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        body.add(form, weight=2)

        footer = ttk.Frame(self, padding=12)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(2, weight=1)
        ttk.Button(footer, text="← Zurück", command=self.previous).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(footer, text="Überspringen", command=self.skip).grid(row=0, column=1)
        ttk.Button(footer, text="Speichern und weiter →", command=self.save_and_next).grid(
            row=0, column=3, padx=(8, 0)
        )

    @staticmethod
    def _entry_row(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 8))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=5)
        return row + 1

    def _show_empty_state(self) -> None:
        self.status_var.set("Keine Belege gefunden")
        self.preview_label.configure(
            text=f"Keine unterstützten Dateien in\n{IMAGES_DIR}", image=""
        )

    def load_case(self, index: int) -> None:
        if not self.cases:
            return
        self.index = index % len(self.cases)
        image_path = self.cases[self.index]
        self.current_fixture = self._read_fixture(image_path)
        metadata = self.current_fixture["metadata"]
        expected = self.current_fixture["expected"]

        verified_count = sum(
            1 for path in self.cases if bool(self._read_fixture(path)["metadata"].get("verified"))
        )
        state = "verifiziert" if metadata.get("verified") else "offen"
        self.status_var.set(
            f"{self.index + 1} / {len(self.cases)} · {verified_count} verifiziert · {image_path.name} · {state}"
        )

        ocr_path = self._ocr_path(image_path)
        try:
            ocr = ocr_path.read_text(encoding="utf-8") if ocr_path.exists() else ""
        except OSError as exc:
            ocr = f"OCR-Datei konnte nicht gelesen werden: {exc}"
        self.ocr_text.configure(state="normal")
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert("1.0", ocr)
        self.ocr_text.configure(state="disabled")

        self.store_var.set(self._display(expected.get("store")))
        self.receipt_type_var.set(self._display(expected.get("receipt_type")))
        self.total_var.set(self._display_total(expected.get("total")))
        self.payment_var.set(self._display(expected.get("payment")))
        self.description_var.set(self._display(metadata.get("description")))
        self.issue_var.set(self._display(metadata.get("known_issue")))

        self._load_preview(image_path)
        self.suggestions = self._pipeline_suggestions(ocr)
        self.suggestion_var.set(
            "\n".join(
                (
                    f"Händler: {self.suggestions.get('store') or '—'}",
                    f"Betrag: {self._display_total(self.suggestions.get('total')) or '—'}",
                    f"Zahlung: {self.suggestions.get('payment') or '—'}",
                    f"Quellen: {self.suggestions.get('sources') or '—'}",
                )
            )
        )

    @staticmethod
    def _display(value: Any) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _display_total(value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return f"{float(value):.2f}".replace(".", ",")
        except (TypeError, ValueError):
            return str(value)

    def _load_preview(self, image_path: Path) -> None:
        self.current_image = None
        if image_path.suffix.lower() == ".pdf":
            self.preview_label.configure(
                image="", text=f"PDF-Vorschau wird in Version 1 nicht gerendert.\n\n{image_path.name}"
            )
            return
        if Image is None or ImageTk is None:
            self.preview_label.configure(
                image="",
                text="Für JPG/TIFF-Vorschauen wird Pillow benötigt.\n\npip install pillow",
            )
            return
        try:
            image = Image.open(image_path)
            image.thumbnail((430, 650))
            self.current_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.current_image, text="")
        except Exception as exc:  # Pillow kann viele formatbezogene Fehler liefern.
            self.preview_label.configure(image="", text=f"Bild konnte nicht geladen werden:\n{exc}")

    def _pipeline_suggestions(self, ocr: str) -> dict[str, Any]:
        suggestions: dict[str, Any] = {"store": None, "total": None, "payment": None, "sources": ""}
        sources: list[str] = []
        if not ocr.strip():
            suggestions["sources"] = "kein OCR-Text"
            return suggestions

        try:
            if extract_store_candidate is not None:
                store, score, source = extract_store_candidate(ocr)
                suggestions["store"] = store
                sources.append(f"store={source}/{score}")
            if extract_total_candidate is not None:
                total, source, score = extract_total_candidate(ocr)
                suggestions["total"] = total
                sources.append(f"total={source}/{score}")
            if extract_payment_values is not None:
                payment = extract_payment_values(ocr)
                suggestions["payment"] = payment.get("Zahlung")
                sources.append("payment=receipt_pipeline")
        except Exception as exc:
            sources.append(f"Pipelinefehler: {exc}")
        suggestions["sources"] = ", ".join(sources) or "Pipeline nicht importierbar"
        return suggestions

    def apply_suggestions(self) -> None:
        if not self.store_var.get().strip() and self.suggestions.get("store"):
            self.store_var.set(str(self.suggestions["store"]))
        if not self.total_var.get().strip() and self.suggestions.get("total") is not None:
            self.total_var.set(self._display_total(self.suggestions["total"]))
        if not self.payment_var.get().strip() and self.suggestions.get("payment"):
            self.payment_var.set(str(self.suggestions["payment"]))

    @staticmethod
    def _nullable(value: str) -> str | None:
        cleaned = value.strip()
        return cleaned or None

    def _parse_total(self) -> float | None:
        raw = self.total_var.get().strip().replace("€", "").replace(" ", "")
        if not raw:
            return None
        if "," in raw and "." in raw:
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", ".")
        try:
            value = round(float(raw), 2)
        except ValueError as exc:
            raise ValueError("Der Gesamtbetrag muss eine Zahl sein, z. B. 12,34.") from exc
        if value < 0:
            raise ValueError("Der Gesamtbetrag darf nicht negativ sein.")
        return value

    def save_and_next(self) -> None:
        if not self.cases:
            return
        try:
            total = self._parse_total()
        except ValueError as exc:
            messagebox.showwarning("Ungültiger Betrag", str(exc))
            return

        image_path = self.cases[self.index]
        fixture = self.current_fixture
        fixture["metadata"].update(
            {
                "case_id": image_path.stem,
                "source": image_path.suffix.lstrip(".").lower(),
                "description": self.description_var.get().strip(),
                "known_issue": self.issue_var.get().strip(),
                "verified": True,
                "active": bool(fixture["metadata"].get("active", True)),
            }
        )
        fixture["expected"].update(
            {
                "store": self._nullable(self.store_var.get()),
                "receipt_type": self._nullable(self.receipt_type_var.get()),
                "total": total,
                "payment": self._nullable(self.payment_var.get()),
            }
        )

        path = self._expected_path(image_path)
        try:
            path.write_text(
                json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror("Speichern fehlgeschlagen", f"{path}\n\n{exc}")
            return

        next_index = self._next_unverified_index(self.index)
        self.load_case(next_index)

    def _next_unverified_index(self, current: int) -> int:
        for offset in range(1, len(self.cases) + 1):
            candidate = (current + offset) % len(self.cases)
            fixture = self._read_fixture(self.cases[candidate])
            if not bool(fixture["metadata"].get("verified")):
                return candidate
        messagebox.showinfo("Fertig", "Alle Testfälle sind verifiziert.")
        return (current + 1) % len(self.cases)

    def skip(self) -> None:
        if self.cases:
            self.load_case((self.index + 1) % len(self.cases))

    def previous(self) -> None:
        if self.cases:
            self.load_case((self.index - 1) % len(self.cases))


def main() -> int:
    app = ReceiptVerifier()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
