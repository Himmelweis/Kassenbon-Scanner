# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:34:56 2025

@author: ONeum
"""

# test_single_receipt.py
# Minimaltest für einen einzelnen Kassenbon – ohne Excel, mit viel Debug-Ausgabe.

import os, re, cv2, pytesseract
from datetime import datetime
import numpy as np

# ---- Pfad zu Tesseract (ggf. anpassen, falls anders installiert) ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# ---- robuster Bildlader (funktioniert auch mit Umlauten im Pfad) ----
def imread_safe(path, flags=cv2.IMREAD_COLOR):
    try:
        # unicode-sicher: über numpy.fromfile + cv2.imdecode
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        # Fallback
        return cv2.imread(path, flags)

# ---- einfache & alternative Vorverarbeitung ----
def preprocess_basic(img):
    scale = 1.5
    img2 = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # leicht normalisieren (hilft bei Vergilbung)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # entrauschen
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # Kontrast pushen
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # binär
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def preprocess_adaptive(img):
    scale = 1.5
    img2 = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 31, 8)
    return bw

# ---- OCR-Helfer ----
CFG1 = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
CFG2 = r'--oem 3 --psm 4'

def fix_text(s: str) -> str:
    s = s.replace("€", " EUR")
    s = s.replace("O", "0").replace("I", "1")
    # US->DE Dezimal, wenn EUR dahinter
    s = re.sub(r'(\d+)\.(\d{2})\s*(EUR|€)', r'\1,\2 \3', s)
    return s

def ocr_text_variants(img):
    texts = []
    # Variante A: basic + psm6
    t1 = pytesseract.image_to_string(preprocess_basic(img), lang="deu+eng", config=CFG1)
    texts.append(fix_text(t1))
    # Variante B: adaptive + psm6
    t2 = pytesseract.image_to_string(preprocess_adaptive(img), lang="deu+eng", config=CFG1)
    texts.append(fix_text(t2))
    # Variante C: adaptive + psm4
    t3 = pytesseract.image_to_string(preprocess_adaptive(img), lang="deu+eng", config=CFG2)
    texts.append(fix_text(t3))
    return texts

# ---- Beträge sammeln & Total wählen ----
def parse_money_de(num: str) -> float | None:
    if not num: return None
    s = num.replace(" ", "").replace(".", "").replace(",", ".")
    try: return float(s)
    except: return None

def all_amounts(text: str):
    vals = []
    for m in re.finditer(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})(?:\s*(?:EUR|€))?", text):
        v = parse_money_de(m.group(1))
        if v is not None:
            vals.append(v)
    # eindeutige, absteigend
    out = []
    for v in sorted(vals, reverse=True):
        if not any(abs(v - u) < 1e-3 for u in out):
            out.append(v)
    return out

def pick_total(text: str, liter: float | None = None, price_per_l: float | None = None) -> float | None:
    # ROI um TOTAL/SUMME/BAR/EC
    lines = text.splitlines()
    roi = []
    for i, ln in enumerate(lines):
        if re.search(r"(TOTAL|SUMME|GESAMT|ZU\s*ZAHLEN|BAR|EC)", ln, re.IGNORECASE):
            roi.extend(lines[max(0,i-3):min(len(lines), i+4)])
    roi_text = "\n".join(roi) if roi else text

    roi_vals = all_amounts(roi_text)
    all_vals = all_amounts(text)

    def plausible(xs): return [x for x in xs if 5.0 <= x <= 1000.0]  # Grenzen bei Bedarf anpassen
    roi_pl, all_pl = plausible(roi_vals), plausible(all_vals)

    # Wenn Liter & €/L vorhanden: beste Übereinstimmung (±4 %)
    if liter and price_per_l:
        est = liter * price_per_l
        best, best_rel = None, 1e9
        for v in roi_pl + all_pl:
            rel = abs(v - est) / max(est, 1e-6)
            if rel < best_rel:
                best, best_rel = v, rel
        if best is not None and best_rel <= 0.04:
            return best

    # sonst: größter plausibler Betrag, bevorzugt aus ROI
    if roi_pl: return max(roi_pl)
    if all_pl: return max(all_pl)
    return max(all_vals) if all_vals else None

# ---- Datum/Uhrzeit/Händler grob mit Plausibilität ----
def extract_core(text: str):
    data = {}

    # Datum
    dm = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b", text) or re.search(r"\b(\d{2}\.\d{2}\.\d{2})\b", text) or re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if dm:
        ds = dm.group(1)
        for fmt in ("%d.%m.%Y","%d.%m.%y","%Y-%m-%d"):
            try:
                dt = datetime.strptime(ds, fmt)
                if 2000 <= dt.year <= 2100:
                    data["Datum"] = dt.strftime("%d.%m.%Y")
                    break
            except: pass
        if "Datum" not in data:
            data["Datum"] = ds  # unparsed, zur Sichtprüfung

    # Uhrzeit
    tm = re.search(r"\b([01]\d|2[0-3]):([0-5]\d)\b", text)
    if tm:
        data["Uhrzeit"] = f"{tm.group(1)}:{tm.group(2)}"

    # Händler (erste sinnvolle Kopfzeile mit Keyword-Hilfe)
    MERCHANTS = ["bft","aral","shell","esso","jet","total","rewe","edeka","aldi","lidl","kaufland","penny","dm","rossmann","müller"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:12]:
        low = ln.lower()
        if any(k in low for k in MERCHANTS):
            data["Laden"] = ln
            break
    if "Laden" not in data:
        for ln in lines[:6]:
            if re.match(r"^(#|\d|mw?st|netto|brutto|total|summe|gesamt)", ln, re.IGNORECASE):
                continue
            if len(ln) >= 3:
                data["Laden"] = ln
                break

    # Liter / €/L (nur grob)
    m_liter = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*(?:l|liter)\b", text, re.IGNORECASE)
    if m_liter:
        data["Liter"] = parse_money_de(m_liter.group(1))
    m_pp = re.search(r"\b([0-9]+(?:[.,][0-9]+))\s*(?:EUR|€)\s*/\s*(?:l|liter)\b", text, re.IGNORECASE)
    if m_pp:
        data["€/L"] = parse_money_de(m_pp.group(1))

    # Betrag (mit Fallback)
    data["Betrag (€)"] = pick_total(text, data.get("Liter"), data.get("€/L"))

    return data

# ----------------- MAIN: hier Pfad zu deinem Testbild eintragen -----------------
if __name__ == "__main__":
    # Beispiel: lege dein Testbild hier fest (absoluter Pfad empfohlen)
    TEST_IMAGE = r"C:\Users\ONeum\Documents\ChatGPT\Kassenbons\ZG-Tankstelle.jpg"  # <-- anpassen!

    if not os.path.exists(TEST_IMAGE):
        print(f"❗ Bild nicht gefunden: {TEST_IMAGE}")
        raise SystemExit

    img = imread_safe(TEST_IMAGE)
    if img is None:
        print(f"⚠️ Bild konnte nicht geladen werden (Pfad/Umlaute?): {TEST_IMAGE}")
        raise SystemExit

    print(f"\n🖼️ Datei: {TEST_IMAGE}  |  Größe: {img.shape[1]}x{img.shape[0]}")

    texts = ocr_text_variants(img)
    for i, tx in enumerate(texts, 1):
        print(f"\n--- OCR Variante {i} ---")
        print(tx[:2000])  # die ersten 2000 Zeichen (sonst wird's lang)
        print("...")

    # nimm die „beste“ (hier einfach die letzte) zum Testen des Parsers
    best_text = texts[-1]
    print("\n🔎 Alle Geldbeträge (absteigend, eindeutig):")
    print(all_amounts(best_text))

    data = extract_core(best_text)
    print("\n✅ Extrahiert:")
    for k in ("Datum","Uhrzeit","Laden","Liter","€/L","Betrag (€)"):
        print(f"  {k:10}: {data.get(k)}")
