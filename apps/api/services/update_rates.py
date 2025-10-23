from api.adapters.rates_adapter import RatesAdapter
from api.engines.zero_rates import ZeroRatesEngine
import pandas as pd


RatesAdapter.generateTreasuryYields()
RatesAdapter.generateSOFR()

treasuryYields = pd.read_csv("data/treasury_par_yields.csv").set_index("date")

zeroRates = []
for date, row in (
    treasuryYields.index,
    treasuryYields[["0.5", "1", "2", "3", "5", "7", "10", "20", "30"]].to_numpy(),
):
    solved = ZeroRatesEngine.calcZeroRates(row)
    solved.insert(0, date)
    zeroRates.append(solved)

zeroRates = pd.DataFrame(
    zeroRates, columns=["date", "0.5", "1", "2", "3", "5", "7", "10", "20", "30"]
)
zeroRates.to_csv("data/treasury_zero_rates.csv", index=False)
