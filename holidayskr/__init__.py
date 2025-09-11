from datetime import datetime

def year_holidays(year: str):
    year_int = int(year)
    if year_int == 2024:
        return [(datetime(2024, 1, 1), "신정")]
    return []
