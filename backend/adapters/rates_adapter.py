import pandas as pd
from fredapi import Fred
import os
from datetime import datetime, timedelta
from data.metadata import TENOR_TO_ID

FredApiKey = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FredApiKey)


class RatesAdapter:
    @staticmethod
    def generateSOFR():
        data_part1 = fred.get_series('SOFR', observation_start='2000-01-01', observation_end='2023-01-01')

        data_part2 = fred.get_series('SOFR', observation_start='2023-01-01')

        df_part1 = data_part1.reset_index().rename(columns={'index': 'date', 0: 'value'})
        df_part2 = data_part2.reset_index().rename(columns={'index': 'date', 0: 'value'})

        rates = pd.concat([df_part1, df_part2]).drop_duplicates(subset=['date']).sort_values(by='date')
        
        rates.to_csv('RateData.csv', index=False)

    @staticmethod
    def fetchTreasuryYields():
        data = {}
        for tenor, seriesId in TENOR_TO_ID.items():
            data[tenor] = fred.get_series_latest_release(seriesId).iloc[-1]
        return data

    @staticmethod
    def getTodayRate():
        rates = pd.read_csv('RateData.csv', parse_dates=['date']).dropna(subset=['date'])
        todayRate = rates['value'].iloc[-1] 
        return todayRate
