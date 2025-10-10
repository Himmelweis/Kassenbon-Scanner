# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:42:08 2025

@author: ONeum
"""

import os
import sys
import pytesseract
from PIL import Image
import subprocess

print("="*60)
print("🔍 Tesseract & Environment Diagnose für Spyder")
print("="*60)

# 1️⃣ Aktuelles Python-Environment prüfen
print(f"\n🟢 Aktives Environment: {sys.prefix}")
if "Haushaltsbuch" not in sys.prefix:
    print("⚠️ WARNUNG: Spyder läuft NICHT im 'Haushaltsbuch'-Environment!")
    print("   -> Bitte Spyder über Anaconda starten oder Interpreter wechseln.")
else:
    print("✅ Spyder läuft im Haushaltsbuch-Environment.")

# 2️⃣ Prüfen, ob Tesseract installiert ist
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(tesseract_path):
    print(f"❌ Tesseract nicht gefunden: {tesseract_path}")
    print("Bitte Tesseract installieren oder den Pfad anpassen.")
    sys.exit(1)
else:
    print(f"✅ Tesseract gefunden unter: {tesseract_path}")

# 3️⃣ Tesseract-Pfad in pytesseract setzen
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 4️⃣ TESSDATA_PREFIX richtig setzen
tessdata_dir = r"C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = tessdata_dir

# 5️⃣ Prüfen, ob Sprachdateien vorhanden sind
tess_lang_path = os.path.join(tessdata_dir, "tessdata")
print(f"\n📂 Sprachdateien-Ordner: {tess_lang_path}")

if os.path.exists(tess_lang_path):
    langs = [f for f in os.listdir(tess_lang_path) if f.endswith(".traineddata")]
    if langs:
        print("✅ Gefundene Sprachdateien:", ", ".join(langs))
    else:
        print("❌ Keine Sprachdateien gefunden! Bitte deu.traineddata herunterladen.")
else:
    print("❌ tessdata-Ordner existiert nicht!")
    sys.exit(1)

# 6️⃣ Tesseract selbst fragen, welche Sprachen verfügbar sind
print("\n🔄 Prüfe verfügbare Sprachen...")
try:
    result = subprocess.run(
        [tesseract_path, "--list-langs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
except Exception as e:
    print("❌ Fehler beim Abruf der Sprachen:", e)

# 7️⃣ Tesseract-Version ausgeben
print("📌 Tesseract-Version:", pytesseract.get_tesseract_version())

# 8️⃣ Test-OCR mit bon1.jpg
print("\n🔎 Teste OCR mit bon1.jpg ...")
test_image = "bon1.jpg"
if os.path.exists(test_image):
    try:
        text = pytesseract.image_to_string(Image.open(test_image), lang="deu")
        print("\n✅ OCR erfolgreich! Ausgabetext:")
        print("-"*50)
        print(text)
        print("-"*50)
    except pytesseract.TesseractError as e:
        print("❌ OCR-Fehler:", e)
else:
    print(f"⚠️ Kein Testbild gefunden. Lege bitte {test_image} in deinen Projektordner.")

print("\n✅ Diagnose abgeschlossen.")
print("="*60)
