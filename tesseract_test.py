from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR"

print("Tesseract-Version:", pytesseract.get_tesseract_version())
print("Verfügbare Sprachen:", pytesseract.get_languages(config=''))

img_file = "bon1.jpg"
if os.path.exists(img_file):
    text = pytesseract.image_to_string(Image.open(img_file), lang="deu")
    print("\nErkannter Text:\n", text)
else:
    print("Bitte lege 'bon1.jpg' in den Projektordner.")
