import cv2
import pytesseract
from PIL import Image
import os

# Pfad zu Tesseract explizit setzen
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Bild laden
image_path = "bon1.jpg"
img = cv2.imread(image_path)

# In Graustufen umwandeln
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kontrast verstärken und Rauschen reduzieren
gray = cv2.medianBlur(gray, 3)

# Binarisierung für besseren Textkontrast
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR mit deutschem Sprachmodell
custom_config = r'--oem 3 --psm 6'  # Engine 3 = LSTM, Page Segmentation Mode 6 = "Block of Text"
text = pytesseract.image_to_string(thresh, lang="deu", config=custom_config)

print("\n=== OCR-ERGEBNIS ===")
print(text)
