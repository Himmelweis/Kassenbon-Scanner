from pathlib import Path
from paddleocr import PaddleOCR
import re

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "Kassenbons_test"

# Hier den echten Dateinamen eintragen:
IMG_NAME = "Hagebau.jpg"
img = TEST_DIR / IMG_NAME

print("=== DATEI ===")
print(img)
print("Existiert:", img.exists())

if not img.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {img}")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

result = ocr.predict(str(img))
page = result[0]

texts = page.get("rec_texts", [])
scores = page.get("rec_scores", [])
boxes = page.get("rec_boxes", [])

lines = []

for i, text in enumerate(texts):
    score = float(scores[i]) if i < len(scores) else 0.0
    box = boxes[i] if i < len(boxes) else None

    if box is None or not text:
        continue

    # rec_boxes sind meist [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    lines.append({
        "text": text.strip(),
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
    })

# Nach vertikaler Position sortieren
lines.sort(key=lambda d: (d["cy"], d["x1"]))

print("\n=== OCR-ZEILEN (sortiert) ===")
for ln in lines:
    print(f'{ln["cy"]:>5} | {ln["score"]:.3f} | {ln["text"]}')

# --- grobe Blocktrennung nach Seitenhöhe ---
if not lines:
    raise RuntimeError("Keine OCR-Zeilen erkannt.")

max_y = max(ln["y2"] for ln in lines)

header_lines = [ln for ln in lines if ln["cy"] <= max_y * 0.22]
footer_lines = [ln for ln in lines if ln["cy"] >= max_y * 0.72]
middle_lines = [ln for ln in lines if ln not in header_lines and ln not in footer_lines]

print("\n=== HEADER ===")
for ln in header_lines:
    print(ln["text"])

print("\n=== MITTELTEIL ===")
for ln in middle_lines:
    print(ln["text"])

print("\n=== FOOTER / TOTALS ===")
for ln in footer_lines:
    print(ln["text"])

# --- einfache Feldextraktion ---
store = None
address = None
date_time = None
total = None
payment = None

# Header: erste sinnvolle Zeilen
for ln in header_lines:
    txt = ln["text"]
    if not store and any(c.isalpha() for c in txt) and len(txt) > 8:
        store = txt
        continue
    if not address and any(ch.isdigit() for ch in txt) and any(c.isalpha() for c in txt):
        address = txt

# Datum/Uhrzeit: überall suchen, aber gezielt
for ln in lines:
    txt = ln["text"]
    if not date_time and any(ch.isdigit() for ch in txt):
        if ":" in txt or "Uhr" in txt:
            date_time = txt

# Summe / Zahlung robuster erkennen
import re

total = None
payment = None
given_amount = None
change_amount = None

def _find_money_near(idx, window_before=2, window_after=2, allow_negative=True):
    vals = []
    start = max(0, idx - window_before)
    end = min(len(lines), idx + window_after + 1)

    for j in range(start, end):
        txt2 = lines[j]["text"]
        for m in re.findall(r"-?\d+[.,]\d{2}", txt2):
            try:
                val = float(m.replace(",", "."))
            except Exception:
                continue
            if not allow_negative and val < 0:
                continue
            vals.append((j, m, val, txt2))
    return vals

# 1) Zahlung + Gegeben + Rückgeld
for i, ln in enumerate(lines):
    txt = ln["text"].lower()

    if "ruckgeld" in txt or "rückgeld" in txt:
        payment = "Bar"
        nearby = _find_money_near(i, window_before=3, window_after=1, allow_negative=True)
        # Für Rückgeld bevorzugt negative Beträge in der Nähe
        negs = [x for x in nearby if x[2] < 0]
        if negs:
            change_amount = negs[-1][1]
        elif nearby:
            change_amount = nearby[-1][1]

    if "gegeben" in txt or ("bar" == txt.strip()) or ("barzahlung" in txt):
        payment = "Bar"
        nearby = _find_money_near(i, window_before=1, window_after=3, allow_negative=False)
        pos = [x for x in nearby if x[2] > 0]
        if pos:
            given_amount = pos[-1][1]

# Fallback: größter positiver Bar-Betrag unten = gegeben
if not given_amount and payment == "Bar":
    lower_part = [ln for ln in lines if ln["cy"] > max_y * 0.35]
    nums = []
    for ln in lower_part:
        for m in re.findall(r"\d+[.,]\d{2}", ln["text"]):
            try:
                val = float(m.replace(",", "."))
            except Exception:
                continue
            nums.append((val, m, ln["text"]))
    if nums:
        nums.sort(key=lambda x: x[0])
        given_amount = nums[-1][1]

# 2) Summe explizit suchen
TOTAL_KEYWORDS = ["gesamtsumme", "summe", "summi", "gesamt", "endbetrag", "zu zahlen", "zahlbetrag"]

for i, ln in enumerate(lines):
    txt = ln["text"].lower()

    if any(k in txt for k in TOTAL_KEYWORDS):
        nearby = _find_money_near(i, window_before=1, window_after=2, allow_negative=False)
        # kleine Werte wie 1,00 ignorieren
        pos = [x for x in nearby if x[2] > 2.0]
        if pos:
            # nimm den kleineren plausiblen Betrag, nicht den gegebenen 100,00
            pos.sort(key=lambda x: x[2])
            total = pos[0][1]
            break

# 3) Bei Barzahlung: berechnete Summe hat Vorrang
if payment == "Bar" and given_amount and change_amount:
    try:
        g = float(given_amount.replace(",", "."))
        c = abs(float(change_amount.replace(",", ".")))
        calc_total = round(g - c, 2)

        # Wenn noch keine Summe da ist ODER die gefundene Summe identisch mit "gegeben" ist,
        # dann die berechnete Summe verwenden.
        if not total:
            total = f"{calc_total:.2f}".replace(".", ",")
        else:
            try:
                found_total = float(total.replace(",", "."))
                given_val = float(given_amount.replace(",", "."))
                if abs(found_total - given_val) < 0.001:
                    total = f"{calc_total:.2f}".replace(".", ",")
            except Exception:
                total = f"{calc_total:.2f}".replace(".", ",")
    except Exception:
        pass

# 4) Letzter Fallback: Brutto aus Steuerblock
if not total:
    for i, ln in enumerate(lines):
        txt = ln["text"].lower()
        if "brutto" in txt:
            nearby = _find_money_near(i, window_before=0, window_after=3, allow_negative=False)
            pos = [x for x in nearby if x[2] > 2.0]
            if pos:
                total = pos[0][1]
                break

if date_time:
    m = re.search(r"(\d{2}\.\d{2}\.\d{2,4}).*?(\d{2}:\d{2})", date_time)
    if m:
        date_time = f"{m.group(1)} {m.group(2)}"

print("\n=== ERKANNTE FELDER ===")
print("Laden    :", store)
print("Adresse  :", address)
print("Datum/Zeit:", date_time)
print("Summe    :", total)
print("Zahlung  :", payment)
print("Gegeben  :", given_amount)
print("Rückgeld :", change_amount)