# Kassenbon Scanner

OCR-basierter Scanner für Kassenbons und Zahlungsbelege.

## Features

- OCR mit PaddleOCR
- PDF + Bilder
- Excel-Export
- Duplikaterkennung
- Kategorien
- Tankstellen / Supermärkte / Einzelhandel / Kartenbelege

## Installation

```bash
pip install -r requirements.txt

## Start 

python gui.py

Unterstützte Belegtypen
fuel
grocery
retail
pharmacy
card_receipt
Bekannte Einschränkungen
unscharfe Bilder können fehlerhaft sein
gemischte MwSt wird nicht immer vollständig getrennt
Kartenbelege enthalten oft keine Artikeldaten