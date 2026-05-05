import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
import os

# 👉 deine Funktion importieren
from kassenbon_scanner import scan_kassenbon


def handle_files(paths):
    for path in paths:
        print(f"📄 Verarbeite: {path}")
        try:
            scan_kassenbon(path)
        except Exception as e:
            print(f"❌ Fehler: {e}")


def open_files():
    files = filedialog.askopenfilenames(
        title="Kassenbons auswählen",
        filetypes=[("Bilder/PDF", "*.jpg *.jpeg *.png *.pdf")]
    )
    if files:
        handle_files(files)


def drop_event(event):
    files = root.tk.splitlist(event.data)
    handle_files(files)


# GUI Setup
root = TkinterDnD.Tk()
root.title("Kassenbon Scanner")
root.geometry("400x200")

label = tk.Label(
    root,
    text="Kassenbons hier hineinziehen\noder Button klicken",
    font=("Arial", 12),
    justify="center"
)
label.pack(expand=True)

btn = tk.Button(root, text="Dateien auswählen", command=open_files)
btn.pack(pady=10)

# Drag & Drop aktivieren (Windows)
try:
    root.tk.call('tk', 'scaling', 1.0)
    root.drop_target_register(DND_FILES)
    root.dnd_bind("<<Drop>>", drop_event)
except Exception:
    print("⚠️ Drag & Drop nicht verfügbar, nutze Button")

root.mainloop()