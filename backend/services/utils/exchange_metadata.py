from zoneinfo import ZoneInfo

EXCHANGE_TIMEZONES = {
    "NMS": ZoneInfo("America/New_York"),   # NASDAQ
    "NYQ": ZoneInfo("America/New_York"),   # NYSE
    "PCX": ZoneInfo("America/New_York"),   # ARCA
    "CBOE": ZoneInfo("America/Chicago"),   # CBOE
    "LSE": ZoneInfo("Europe/London"),      # London
    "FRA": ZoneInfo("Europe/Berlin"),      # Frankfurt
}
