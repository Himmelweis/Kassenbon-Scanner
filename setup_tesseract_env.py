# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:59:16 2025

@author: ONeum
"""

import os
import sys
import shutil
import pytesseract
from PIL import Image
import subprocess

print("="*60)
print("🔧 Automatisches Setup & Diagnose für Tesseract + Spyder")
print("="*60)

# -----------------------------
# 1️⃣ Aktuelles Environment prüfen
# -----------------------------
expected_env = "HaushaltsBuch"
python_exe = sys.executable

print(f"\n🐍 Aktiver Python-Interpreter: {python_exe}")
if expected_env.lower() not in python_exe.lower():
    print(f"⚠️ WARNUNG: Du bist NICHT im '{expected_env}'-Environment!")
    print("   → Bitte Spyder über Anaconda aus dem richtigen Environment starten.")
else:
    print("✅ Spyder läuft im richtigen Environment.")

# -----------------------------
# 2️⃣ Tesseract EXE prüfen
# -----------------------------
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(tesseract_path):
    print(f"❌ Tesseract nicht gefunden unter:\n   {tesseract_path}")
    print("➡ Bitte prüfe die Installation oder passe den Pfad an.")
    sys.exit(1)
else:
    print(f"✅ Tesseract gefunden: {tesseract_path}")

pytesseract.pytesseract.tesseract_cmd = tesseract_path

# -----------------------------
# 3️⃣ TESSDATA_PREFIX setzen
# -----------------------------
tessdata_dir = r"C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
print(f"📂 TESSDATA_PREFIX gesetzt auf: {tessdata_dir}")

# -----------------------------
# 4️⃣ Verfügbare Sprachdateien prüfen
# -----------------------------
tessdata_path = os.path.join(tessdata_dir, "tessdata")
if not os.path.exists(tessdata_path):
    print(f"❌ Tessdata-Ordner fehlt: {tessdata_path}")
    sys.exit(1)

langs = [f for f in os.listdir(tessdata_path) if f.endswith(".traineddata")]
print(f"📄 Gefundene Sprachdateien: {', '.join(langs) if langs else 'Keine gefunden!'}")

# -----------------------------
# 5️⃣ Tesseract-Version prüfen
# -----------------------------
try:
    version = pytesseract.get_tesseract_version()
    print(f"📌 Tesseract-Version: {version}")
except Exception as e:
    print(f"❌ Fehler beim Laden von Tesseract: {e}")
    sys.exit(1)

# -----------------------------
# 6️⃣ Testlauf: Liste der Sprachen direkt abfragen
# -----------------------------
try:
    print("\n🔍 Prüfe direkt verfügbare Sprachen...")
    langs = pytesseract.get_languages(config='')
    print(f"✅ Verfügbare Sprachen (Tesseract): {langs}")
except Exception as e:
    print(f"❌ Fehler beim Laden der Sprachliste: {e}")

# -----------------------------
# 7️⃣ OCR-Test mit Musterbild
# -----------------------------
test_image = "bon1.jpg"
if os.path.exists(test_image):
    print(f"\n📷 Starte OCR-Test mit '{test_image}' ...")
    try:
        text = pytesseract.image_to_string(Image.open(test_image), lang="deu")
        print("\n=== OCR-Ausgabe ===\n")
        print(text if text.strip() else "⚠️ Kein Text erkannt!")
        print("\n===================")
    except pytesseract.TesseractError as e:
        print(f"❌ OCR-Fehler: {e}")
else:
    print(f"\n⚠️ Kein Testbild '{test_image}' gefunden → Überspringe OCR-Test.")

print("\n✅ Setup & Diagnose abgeschlossen.")
print("="*60)
