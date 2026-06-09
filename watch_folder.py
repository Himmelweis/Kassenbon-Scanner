import time
import shutil
from pathlib import Path
from kassenbon_scanner import scan_kassenbon, scan_pdf_receipt


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OK_DIR = BASE_DIR / "_ok"
ERROR_DIR = BASE_DIR / "_error"

SUPPORTED = {".jpg", ".jpeg", ".png", ".pdf"}


def ensure_dirs():
    INPUT_DIR.mkdir(exist_ok=True)
    OK_DIR.mkdir(exist_ok=True)
    ERROR_DIR.mkdir(exist_ok=True)


def wait_until_file_ready(path: Path, timeout: int = 30) -> bool:
    last_size = -1

    for _ in range(timeout):
        if not path.exists():
            return False

        size = path.stat().st_size

        if size == last_size and size > 0:
            return True

        last_size = size
        time.sleep(1)

    return False


def move_safe(src: Path, target_dir: Path):
    target = target_dir / src.name

    if target.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1

        while target.exists():
            target = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    for attempt in range(10):
        try:
            shutil.move(str(src), str(target))
            return
        except PermissionError:
            time.sleep(1)

    raise PermissionError(f"Datei konnte nicht verschoben werden: {src}")

def process_file(path: Path):
    print(f"\n📄 Neue Datei erkannt: {path.name}")

    if path.suffix.lower() not in SUPPORTED:
        print("⏭ Nicht unterstützter Dateityp.")
        return

    if not wait_until_file_ready(path):
        print("❌ Datei war nicht stabil lesbar.")
        move_safe(path, ERROR_DIR)
        return

    try:
        if path.suffix.lower() == ".pdf":
            result = scan_pdf_receipt(str(path))
        else:
            result = scan_kassenbon(str(path))

        if result:
            print("✅ Erfolgreich verarbeitet.")
            move_safe(path, OK_DIR)
        else:
            print("❗ Konnte nicht verarbeitet werden.")
            move_safe(path, ERROR_DIR)

    except Exception as e:
        print(f"❌ Fehler: {e}")
        move_safe(path, ERROR_DIR)


def process_existing_files():
    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            process_file(path)


if __name__ == "__main__":
    ensure_dirs()

    print(f"👀 Überwache Ordner: {INPUT_DIR}")
    print("Dateien hier ablegen. Beenden mit STRG+C.")

    seen = set()

    try:
        while True:
            for path in sorted(INPUT_DIR.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in SUPPORTED:
                    continue

                if path in seen:
                    continue

                seen.add(path)
                process_file(path)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Überwachung beendet.")