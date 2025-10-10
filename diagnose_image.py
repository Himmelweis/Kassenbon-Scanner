# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 18:37:36 2025

@author: ONeum
"""

def diagnose_image(path):
    import os, cv2, numpy as np
    print("🧪 Diagnose:", path)
    if not os.path.exists(path):
        print("❌ Datei existiert nicht.")
        return
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        img = None
        print("❌ imdecode-Fehler:", e)

    if img is None:
        print("⚠️ imdecode lieferte None, versuche cv2.imread ...")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("❌ Konnte nicht laden (Pfad/Datei beschädigt?).")
        return

    h, w = img.shape[:2]
    ch = 1 if len(img.shape)==2 else img.shape[2]
    print(f"✅ geladen: {w}x{h}px, Kanäle: {ch}")
    if ch == 4:
        print("ℹ️ Alphakanal vorhanden – wird in ocr_text_multi jetzt entfernt.")
    if h > 8000:
        print("ℹ️ Sehr langes Bild – Kachelung wird verwendet.")
