import stat
from zoneinfo import ZoneInfo
import pandas as pd
from fredapi import Fred
import os
from datetime import datetime, timedelta

from sympy import series
from data.metadata import TENOR_TO_ID
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

        rates.to_csv(DATA_DIR / "sofr_data.csv", index=False)

    @staticmethod
    def getTodaySOFR() -> float:
        rates = pd.read_csv(DATA_DIR / 'sofr_data.csv', parse_dates=['date']).dropna(subset=['date'])
        todayRate = rates['SOFR'].iloc[-1] 
        return todayRate

    @staticmethod
    def generateTreasuryYields():
        dfs = []
        for tenor, seriesId in TENOR_TO_ID.items():
            dataPart1 = fred.get_series(seriesId, observation_start="2000-01-01", observation_end="2023-01-01")
            dataPart2 = fred.get_series(seriesId, observation_start="2023-01-01")

            dfPart1 = dataPart1.reset_index().rename(columns={"index": "date", 0: tenor})
            dfPart2 = dataPart2.reset_index().rename(columns={"index": "date", 0: tenor})

            df = pd.concat([dfPart1, dfPart2]).drop_duplicates(subset=["date"]).sort_values(by="date")
            df["tenor"] = tenor
            df["parYield"] = df[tenor]*0.01
            df["date"] = pd.to_datetime(df["date"]).dt.date

            dfs.append(df)

        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=["date"], how="outer")

        merged = merged.sort_values("date").drop_duplicates("date")
        merged.to_csv(DATA_DIR / "treasury_par_yields.csv", index=False)

    @staticmethod
    def updateTreasuryYields():
        parYields = pd.DataFrame()
        for tenor, seriesId in TENOR_TO_ID.items():
            parYields[tenor] = [None]
            parYields[tenor].iloc[0] = int((fred.get_series_latest_release(seriesId).iloc[-1]))*0.01
            parYields["date"].iloc[0] = [datetime.now(ZoneInfo("America/New_York")).date()]
        
        df = pd.read_csv(DATA_DIR / "treasury_par_yields.csv", parse_dates=['date'])
        if not

        
    
    @staticmethod
    def getTodayYields() -> dict:
        parYields = {}
        for tenor, seriesId in TENOR_TO_ID.items():
            parYields[tenor] = int((fred.get_series_latest_release(seriesId).iloc[-1]))*0.01
        return parYields

