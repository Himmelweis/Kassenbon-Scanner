import pandas as pd
import matplotlib.pyplot as plt

EXCEL_PATH = "kassenbons.xlsx"

df = pd.read_excel(EXCEL_PATH)

# Nur gültige Datensätze
df = df[df["Betrag (€)"].notna()]
df = df[df["Datum"].notna()]

# Datum umwandeln
df["Datum"] = pd.to_datetime(df["Datum"])

# Monatsspalte erzeugen
df["Monat"] = df["Datum"].dt.strftime("%Y-%m")

# =========================
# Gesamtausgaben je Kategorie
# =========================
summary = (
    df.groupby("Kategorie")["Betrag (€)"]
    .sum()
    .sort_values(ascending=False)
)

print("\n📊 Ausgaben nach Kategorie:\n")

for cat, total in summary.items():
    print(f"{cat:<20} {total:>10.2f} €")


# =========================
# Monatsübersicht
# =========================
monthly = (
    df.groupby(["Monat", "Kategorie"])["Betrag (€)"]
    .sum()
    .reset_index()
)

print("\n📅 Monatsübersicht:\n")

for _, row in monthly.iterrows():
    print(
        f"{row['Monat']}  |  "
        f"{row['Kategorie']:<15}  |  "
        f"{row['Betrag (€)']:>8.2f} €"
    )

# =========================
# Diagramm: Ausgaben je Kategorie
# =========================
summary.plot(kind="bar")
plt.title("Ausgaben nach Kategorie")
plt.xlabel("Kategorie")
plt.ylabel("Betrag (€)")
plt.tight_layout()
plt.savefig("reports/auswertung_kategorien.png")
plt.show()


# =========================
# Diagramm: Monatsausgaben gesamt
# =========================
monthly_total = (
    df.groupby("Monat")["Betrag (€)"]
    .sum()
    .sort_index()
)

monthly_total.plot(kind="bar")
plt.title("Ausgaben pro Monat")
plt.xlabel("Monat")
plt.ylabel("Betrag (€)")
plt.tight_layout()
plt.savefig("reports/auswertung_monate.png")
plt.show()