# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:57:37 2025

@author: ONeum
"""

# merge_scans.py
# Fügt mehrteilige Bon-Scans zusammen:
#   Bon001a.png + Bon001b.png  -> Bon001_merged.png
#   Quittung-12_a.jpg + Quittung-12_b.jpg + Quittung-12_c.jpg -> Quittung-12_merged.png
#
# Unterstützt: PNG/JPG/JPEG/TIF/TIFF/WEBP
# Optionen: automatisches Zuschneiden weißer Ränder, Überlappung in Pixeln entfernen, Originale verschieben.

import os, re, numpy as np, cv2
from typing import List, Tuple

# ======== KONFIGURATION ========
FOLDER = r"C:\Users\ONeum\Documents\ChatGPT\Kassenbons"  # <-- Ordner mit den gescannten Teilbildern
OUTPUT_SUFFIX = "_merged"     # Ergebnis-Dateiname: <basis>_merged.png
MOVE_PARTS_TO = "_parts"      # Originale nach <FOLDER>\_parts\<basis>\ verschieben (None, um zu deaktivieren)
TRIM_BORDERS = True           # Weiße Ränder automatisch abschneiden
TRIM_THRESHOLD = 245          # 0..255, höher = empfindlicher für "weiß"
OVERLAP_PX = 80               # so viele Pixel vom oberen Rand jeder Folgeseite wegschneiden (0 = aus)
PAD_TO_SAME_WIDTH = True      # Teilbilder auf gleiche Breite bringen (padding), statt zu skalieren
SCALE_TO_MAX_WIDTH = False    # Alternative: auf die größte Breite sanft skalieren (nur wenn PAD_TO_SAME_WIDTH=False)
OUTPUT_FORMAT = ".png"        # ".png" empfohlen
# ===============================


VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

# Unicode-sicheres Laden
def imread_safe(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img

def imwrite_safe(path, img):
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    data = cv2.imencode(ext, img, params)[1]
    data.tofile(path)

def trim_white_borders(img: np.ndarray, thresh=TRIM_THRESHOLD) -> np.ndarray:
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    # Maske: "nicht weiß"
    mask = gray < thresh
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return img[y0:y1+1, x0:x1+1]

# Gruppiert Dateien nach Basisname + Buchstabensuffix (a,b,c, …)
# Erlaubt optionalen Trenner: _, -, oder keiner
PATTERN = re.compile(r"""^(?P<base>.+?)(?:[_-]?) (?P<part>[A-Za-z])$""", re.X)

def split_base_part(filename_noext: str) -> Tuple[str, str | None]:
    m = PATTERN.match(filename_noext)
    if m:
        return m.group("base"), m.group("part").lower()
    # kein Suffix-Buchstabe → keine Gruppe
    return filename_noext, None

def find_groups(folder: str):
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VALID_EXTS]
    groups = {}  # base -> list of (part, fullpath)
    singles = []
    for f in files:
        base_noext, ext = os.path.splitext(f)
        base, part = split_base_part(base_noext)
        full = os.path.join(folder, f)
        if part is None:
            singles.append(full)
        else:
            groups.setdefault(base, []).append((part, full))
    # nur Gruppen mit mind. 2 Teilen
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    return groups, singles

def order_parts(parts: List[Tuple[str, str]]) -> List[str]:
    # sortiert a,b,c,…; falls gemischt (a/A), egal
    parts_sorted = sorted(parts, key=lambda t: t[0])
    return [p[1] for p in parts_sorted]

def ensure_same_width(images: List[np.ndarray]) -> List[np.ndarray]:
    widths = [im.shape[1] for im in images]
    max_w = max(widths)
    out = []
    for im in images:
        h, w = im.shape[:2]
        if w == max_w:
            out.append(im)
        else:
            # links/rechts weiß auffüllen
            pad = max_w - w
            left = pad // 2
            right = pad - left
            if len(im.shape) == 3:
                pad_color = [255, 255, 255]
            else:
                pad_color = 255
            out.append(cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT, value=pad_color))
    return out

def scale_to_max_width(images: List[np.ndarray]) -> List[np.ndarray]:
    widths = [im.shape[1] for im in images]
    max_w = max(widths)
    outs = []
    for im in images:
        h, w = im.shape[:2]
        if w == max_w:
            outs.append(im)
        else:
            scale = max_w / float(w)
            nh = int(round(h * scale))
            outs.append(cv2.resize(im, (max_w, nh), interpolation=cv2.INTER_CUBIC))
    return outs

def merge_vertical(paths: List[str]) -> np.ndarray:
    imgs = []
    for i, p in enumerate(paths):
        im = imread_safe(p, cv2.IMREAD_COLOR)
        if im is None:
            print(f"⚠️ Konnte nicht laden: {p}")
            continue
        if TRIM_BORDERS:
            im = trim_white_borders(im, TRIM_THRESHOLD)
        # Overlap bei Folgeseiten wegschneiden
        if OVERLAP_PX and i > 0 and im.shape[0] > OVERLAP_PX:
            im = im[OVERLAP_PX:, :]
        imgs.append(im)

    if not imgs:
        return None

    # gleiche Breite herstellen
    if PAD_TO_SAME_WIDTH:
        imgs = ensure_same_width(imgs)
    elif SCALE_TO_MAX_WIDTH:
        imgs = scale_to_max_width(imgs)
    else:
        # nichts tun: cv2.vconcat verlangt identische Breiten, daher notfalls harte Korrektur
        min_w = min(im.shape[1] for im in imgs)
        imgs = [im[:, :min_w] for im in imgs]

    merged = cv2.vconcat(imgs)
    return merged

def main():
    os.makedirs(FOLDER, exist_ok=True)
    groups, singles = find_groups(FOLDER)
    if not groups:
        print("ℹ️ Keine mehrteiligen Scans mit Suffix a/b/c gefunden.")
        print("   Beispiel: Bon001a.png, Bon001b.png  →  Bon001_merged.png")
        return

    print(f"📂 Ordner: {FOLDER}")
    print(f"🧩 Gefundene Gruppen: {len(groups)}\n")

    for base, parts in groups.items():
        ordered = order_parts(parts)
        print(f"→ Mergen: {base}  ({len(ordered)} Teile)")
        merged = merge_vertical(ordered)
        if merged is None:
            print("   ❌ Mergen fehlgeschlagen (Bilder fehlten?).")
            continue

        out_name = f"{base}{OUTPUT_SUFFIX}{OUTPUT_FORMAT}"
        out_path = os.path.join(FOLDER, out_name)
        imwrite_safe(out_path, merged)
        print(f"   ✅ Gespeichert: {out_path}  | Größe: {merged.shape[1]}x{merged.shape[0]}")

        # Originale wegsortieren
        if MOVE_PARTS_TO:
            sub = os.path.join(FOLDER, MOVE_PARTS_TO, base)
            os.makedirs(sub, exist_ok=True)
            for p in ordered:
                try:
                    os.replace(p, os.path.join(sub, os.path.basename(p)))
                except Exception as e:
                    print(f"   ⚠️ Verschieben fehlgeschlagen: {p} ({e})")

    print("\nFertig. Jetzt kannst du die *_merged.png Dateien mit deinem OCR-Skript verarbeiten.")

if __name__ == "__main__":
    main()
