import kassenbon_scanner
kassenbon_scanner.TEST_MODE = True

from kassenbon_scanner import extract_best_datetime, fix_truncated_day

def test_hagebau_date():
    raw = """
    hagebaumarkt
    Datum
    7.02.2607:55
    001191501270226401000021
    """

    date, time = extract_best_datetime(raw)
    date = fix_truncated_day(date, raw)

    print("DATE:", date)
    print("TIME:", time)

    assert date == "2026-02-27"
    assert time == "07:55"

def test_lidl_store():
    raw = """
    LIDL
    Melanchthonstr. 91
    Summe 192,33
    Datum:05.03.26 Zeit:17:24
    """

    date, time = extract_best_datetime(raw)

    assert date == "2026-03-05"
    assert time == "17:24"

def test_fuel_basic():
    best = {
        "Betrag (€)": 104.98,
        "MwSt %": 19
    }

    from kassenbon_scanner import enrich_fuel_data
    best = enrich_fuel_data(best)

    assert best["Netto (€)"] == 88.22
    assert best["MwSt (€)"] == 16.76

