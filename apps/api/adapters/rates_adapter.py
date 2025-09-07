from zoneinfo import ZoneInfo
import pandas as pd
from fredapi import Fred
import os
from datetime import datetime

from api.data.metadata import TENOR_TO_ID
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FredApiKey = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FredApiKey)


class RatesAdapter:
    @staticmethod
    def generateSOFR():
        dataPart1 = fred.get_series("SOFR", observation_start="2000-01-01", observation_end="2023-01-01")
        dataPart2 = fred.get_series("SOFR", observation_start="2023-01-01")

        dfPart1 = dataPart1.reset_index().rename(columns={"index": "date", 0: "SOFR"})
        dfPart2 = dataPart2.reset_index().rename(columns={"index": "date", 0: "SOFR"})

        rates = pd.concat([dfPart1, dfPart2]).drop_duplicates(subset=["date"]).sort_values(by="date")
        rates.ffill(inplace=True)

        rates.to_csv(DATA_DIR / "sofr_data.csv", index=False)

    @staticmethod
    def generateTreasuryYields():
        dfs = []
        for tenor, seriesId in TENOR_TO_ID.items():
            dataPart1 = fred.get_series(seriesId, observation_start="2000-01-01", observation_end="2023-01-01")
            dataPart2 = fred.get_series(seriesId, observation_start="2023-01-01")

            dfPart1 = dataPart1.reset_index().rename(columns={"index": "date", 0: tenor})
            dfPart2 = dataPart2.reset_index().rename(columns={"index": "date", 0: tenor})

            df = pd.concat([dfPart1, dfPart2], ignore_index=True).drop_duplicates(subset=["date"]).sort_values(by="date")

            df["parYield"] = df[tenor]*0.01
            df["parYield"] = df["parYield"].round(4)
            df["date"] = pd.to_datetime(df["date"]).dt.date

            dfs.append(df[["date", "parYield"]].rename(columns={"parYield": tenor}))


        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=["date"], how="outer")
        df = df.set_index("date")

        merged = merged.sort_values("date").drop_duplicates("date")
        merged.ffill(inplace=True)
        merged.to_csv(DATA_DIR / "treasury_par_yields.csv", index=False)

    