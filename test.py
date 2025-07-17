from time import sleep
from functools import lru_cache
from matplotlib import cm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import root_scalar
from scipy.interpolate import UnivariateSpline
from math import isfinite
from scipy.interpolate import griddata
from matplotlib import cm, dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import os


FredApiKey = os.getenv("FRED_API_KEY")

from fredapi import Fred
fred = Fred(api_key=FredApiKey)




class surface():
    def __init__(self):
        self._ticker = None
        self.stock = None
        self.spot = None
        self.history = None
        self.range = 7
        self.options = {}
        self.calls = {}
        self.puts = {}
        self.timeToExpiries = {}
        self.latestRate = None
        self.surface = {}
        self.dividendYield = None

    @property
    def ticker(self):
        return self._ticker
    @ticker.setter
    def ticker(self, value):
        self._ticker = value
        self.stock = yf.Ticker(self.ticker)
        self.spot = self.stock.history(period="1d")['Close'].iloc[-1]
        #self.history = self.stock.history(period="1mo")
        print(f"Spot price for {self.ticker} is ${self.spot:.2f}\n")
        self.getCalls()
        self.getPuts()
        self.getTimeToExpiry()
        self.getDividendYield()
        self.getRates()
        self.ivSurface()
        
        
        


    

    def getCalls(self):
        for i in range(0,self.range):
            sleep(0.5)  
            temp = self.stock.option_chain(self.stock.options[i]).calls
            prices = temp[['strike', 'lastPrice', 'bid', 'ask']]
            prices = prices[(prices['strike'] <= self.spot*1.2) & (prices['strike'] >= self.spot*0.8)].sort_values(by='strike')
            #print(temp['contractSymbol'].iloc[0])
            #print(len(self.ticker))
            self.calls[f"{temp['contractSymbol'].iloc[0][4+len(self.ticker):6+len(self.ticker)]}/{temp['contractSymbol'].iloc[0][2+len(self.ticker):4+len(self.ticker)]}"] = prices
                     
    def getPuts(self):
        for i in range(0,self.range):
            temp = self.stock.option_chain(self.stock.options[i]).puts
            #print(temp['contractSymbol'].iloc[0])
            prices = temp[['strike', 'lastPrice', 'bid', 'ask']]
            prices = prices[(prices['strike'] <= self.spot*1.2) & (prices['strike'] >= self.spot*0.8)].sort_values(by='strike')
            #print(temp['contractSymbol'].iloc[0])
            #print(len(self.ticker))
            self.puts[f"{temp['contractSymbol'].iloc[0][4+len(self.ticker):6+len(self.ticker)]}/{temp['contractSymbol'].iloc[0][2+len(self.ticker):4+len(self.ticker)]}"] = prices


    def printCalls(self):
        if not self.calls:
            self.getCalls()
        print(f"Options data for {self.ticker}:\n")
        for expiry, data in self.calls.items():
            print(f"Expiry: {expiry}")
            print(data.to_string(index=False))
            print("\n")
    
    def printPuts(self):
        if not self.puts:
            self.getPuts()
        print(f"Options data for {self.ticker}:\n")
        for expiry, data in self.puts.items():
            print(f"Expiry: {expiry}")
            print(data.to_string(index=False))
            print("\n")

    def printOptions(self):
        print(f"Options data for {self.ticker}:\n")
        all_expiries = sorted(set(self.calls) | set(self.puts))
        for expiry in all_expiries:

            calls_df = self.calls.get(expiry, pd.DataFrame(columns=['strike','lastPrice','bid','ask']))
            puts_df  = self.puts.get(expiry,  pd.DataFrame(columns=['strike','lastPrice','bid','ask']))

            calls_df = calls_df.rename(columns={
                'lastPrice':'call_last','bid':'call_bid','ask':'call_ask'
            })
            puts_df  = puts_df.rename(columns={
                'lastPrice':'put_last','bid':'put_bid','ask':'put_ask'
            })

            merged = pd.merge(calls_df, puts_df,
                            on='strike',
                            how='outer'
                            ).sort_values('strike').reset_index(drop=True)

            merged = merged[[
                'call_bid','call_ask','call_last',
                'strike',
                'put_last','put_bid','put_ask'
            ]]

            print(f"Expiry: {expiry}")
            print(merged.to_string(index=False))
            print("\n")
    
    def getTimeToExpiry(self):
        self.timeToExpiries = {}
        for expiry in self.calls:
            day, month = map(int, expiry.split('/'))
            today = datetime.today()
            year = today.year
            expiry_date = datetime(year, month, day)
            self.timeToExpiries[expiry] = (expiry_date - today).days / 365

    def printTimeToExpiry(self):
        if not self.timeToExpiries:
            self.getTimeToExpiry()
        print("Time to expiry (in years):")
        for expiry, tte in self.timeToExpiries.items():
            print(f"{expiry}: {tte:.4f} years")
        print("\n")

    def generateRates(self):
        data_part1 = fred.get_series('EFFR', observation_start='2000-01-01', observation_end='2023-01-01')

        data_part2 = fred.get_series('EFFR', observation_start='2023-01-01')

        df_part1 = data_part1.reset_index().rename(columns={'index': 'date', 0: 'value'})
        df_part2 = data_part2.reset_index().rename(columns={'index': 'date', 0: 'value'})

        rates = pd.concat([df_part1, df_part2]).drop_duplicates(subset=['date']).sort_values(by='date')
        
        rates.to_csv('RateData.csv', index=False)

    def getRates(self):
        try:
            rates = pd.read_csv('RateData.csv', parse_dates=['date']).dropna(subset=['date'])
            rates = rates[['date', 'value']]
        except:
            self.generateRates()
            rates = pd.read_csv('RateData.csv', parse_dates=['date']).dropna(subset=['date'])
            rates = rates[['date', 'value']]
        
        rates['value'] = rates['value'].ffill()
        
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        if rates['date'].iloc[-1] != yesterday:
        
            rates.to_csv('RateData.csv', index=False)
            update = fred.get_series('EFFR', observation_start=rates['date'].iloc[-1])
            rates = pd.concat([rates, update]).drop_duplicates(subset=['date']).dropna(subset=['date']).sort_values(by='date')
            rates = rates[['date', 'value']]
            rates.to_csv('RateData.csv', index=False)

        self.latestRate = rates['value'].iloc[-1]
        return rates

    def printRate(self):
        rates = self.getRates()
        self.latestRate = rates['value'].iloc[-1]
        print(f"Latest rate: {self.latestRate:.2f}%\n")
        return 

    def getDividendYield(self):
        self.dividendYield = self.stock.info.get('dividendYield', 0)
    
    def printInfo(self):
        info = self.stock.info
        print(f"Ticker: {self.ticker}")
        print(f"Company Name: {info.get('longName', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        print(f"Market Cap: {info.get('marketCap', 'N/A')}")
        print(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n")

    def printAll(self):
        self.printInfo()
        self.printRate()
        self.printOptions()

    def _blackScholesCall(self, sigma, K, T):
        r = self.latestRate / 100
        q = self.dividendYield
        S = self.spot
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    def impliedVolatility(self, K, T, C):

        def fsafe(sigma):
            pricediff = self._blackScholesCall(sigma, K, T) - C
            if not isfinite(pricediff):
                raise ValueError(f"f returned bad value for sigma={sigma} and K={K}, T={T}, C={C}")
            if pricediff > abs(C) * 50:
                raise ValueError(f"f returned too large value for sigma={sigma} and K={K}, T={T}, C={C}")
            return pricediff
        
        f = lambda sigma: self._blackScholesCall(sigma, K, T) - C
        
        try:
            return root_scalar(f, bracket=[1e-6, 5.0], method='brentq', x0=0.05, x1=0.5).root
        except:
            try:
                return root_scalar(f, bracket=[1e-2, 5.0], method='brentq', x0=0.2, x1=0.4).root
            except:
                return np.nan

    def ivSurface(self):
        for expiry, data in self.calls.items():
            print(self.latestRate)
            T = self.timeToExpiries[expiry]
            data = data.copy()
            #data = data[(data['bid'] > 0) & (data['ask'] > data['bid']) & (data['ask'] < self.spot * 2)]
            data['midPrice'] = (data['bid'] + data['ask']) / 2
            data['IV'] = data.apply(lambda row: self.impliedVolatility(C = (row['bid'] + row['ask'])/2, K = row['strike'], T = T), axis=1)
            print(data)
            #data['IV'] = data.apply(lambda row: self.impliedVolatility(C = row['lastPrice'], K = row['strike'], T = T), axis=1)
            data['smoothedIV'] = UnivariateSpline(data.dropna(subset=['IV'])['strike'], data.dropna(subset=['IV'])['IV'], s=0.8)(data['strike'])
            self.surface[expiry] = data[['strike', 'midPrice', 'IV', 'smoothedIV']].copy()
            #for exp, data in self.surface.items():
            #a    if max(data[exp]['IV']) > 2 and 
        return


    def printIVSurface(self):
        for exp, data in self.surface.items():
            print(f"IV Surface for {self.ticker} on {exp}:")
            print(data[['strike', 'midPrice', 'IV', 'smoothedIV']].to_string(index=False))
            print("\n")
        return
        
    def plot2dIVSurface(self):
        fig, ax = plt.subplots(figsize=(10,6), dpi=600)
        for exp, data in self.surface.items():
            ax.plot(data['strike'], data['smoothedIV'], label=f'{exp}')
        ax.axvline(self.spot, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'Implied Volatility Surface for {self.ticker}')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.legend()
        ax.grid()
        ax.show()
        return fig

    def plot3dIVSurface(self):
        # --- build the surface grid as before ---
        pts, ivs = [], []
        for exp_str, data in self.surface.items():
            day, month = map(int, exp_str.split('/'))
            dt = datetime(datetime.today().year, month, day)
            dt_num = mdates.date2num(dt)
            for K, iv in zip(data['strike'], data['smoothedIV']):
                pts.append((K, dt_num))
                ivs.append(iv)
        pts = np.array(pts); ivs = np.array(ivs)

        K_vals = np.linspace(pts[:,0].min(), pts[:,0].max(), 60)
        D_vals = np.linspace(pts[:,1].min(), pts[:,1].max(), 60)
        K_grid, D_grid = np.meshgrid(K_vals, D_vals)
        Z = griddata(pts, ivs, (K_grid, D_grid), method='cubic')

        # --- compute the empirical CDF of the valid Z values ---
        Z_flat = Z.flatten()
        valid = ~np.isnan(Z_flat)
        z_valid = Z_flat[valid]
        # get the rank of each value
        sorter = np.argsort(z_valid)
        ranks = np.empty_like(sorter)
        ranks[sorter] = np.arange(len(z_valid))
        cdf_vals = ranks / (len(z_valid) - 1)
        # build a full cdf array aligned with Z_flat
        cdf_flat = np.empty_like(Z_flat)
        cdf_flat.fill(np.nan)
        cdf_flat[valid] = cdf_vals
        CDF = cdf_flat.reshape(Z.shape)

        fig = plt.figure(figsize=(9,6), dpi=600)
        ax  = fig.add_subplot(111, projection='3d')

    
        facecolors = cm.inferno(CDF)

        surf = ax.plot_surface(
            K_grid, D_grid, Z,
            facecolors=facecolors,
            shade=False,
            rcount=Z.shape[0], ccount=Z.shape[1],
            antialiased=True
        )

        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Expiry Date')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'3D Implied Volatility Surface for {self.ticker}')
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        ax.yaxis.set_major_locator(mdates.AutoDateLocator())

        # Build a color‐bar that shows the corresponding IV quantiles
        # We'll sample a few quantiles for ticks
        quantiles = np.linspace(0, 1, 6)
        iv_ticks = np.quantile(z_valid, quantiles)
        # Create a mappable with the same colormap
        mappable = cm.ScalarMappable(cmap=cm.inferno)
        mappable.set_array(quantiles)  # use 0–1 range
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_ticks(quantiles)
        cbar.set_ticklabels([f"{q*100:.0f}%" for q in quantiles])
        cbar.set_label('Empirical Quantile of IV')

        # --- lock elevation rotation as before ---
        fixed_elev = ax.elev
        last_x = None

        def on_button_press(event):
            nonlocal last_x
            if event.inaxes == ax and event.button == 1:
                last_x = event.x

        def on_mouse_move(event):
            nonlocal last_x
            if last_x is None or event.button != 1 or event.inaxes != ax:
                return
            dx = event.x - last_x
            last_x = event.x
            new_azim = ax.azim - dx * 0.2
            ax.view_init(elev=fixed_elev, azim=new_azim)
            fig.canvas.draw_idle()

        def on_button_release(event):
            nonlocal last_x
            last_x = None

        fig.canvas.mpl_connect('button_press_event',   on_button_press)
        fig.canvas.mpl_connect('motion_notify_event',  on_mouse_move)
        fig.canvas.mpl_connect('button_release_event', on_button_release)

        plt.show()
        plt.tight_layout()
        return 
    

test = surface()
test.ticker = "MSFT"
test.printAll()
test.printCalls()
test.printIVSurface()
test.plot2dIVSurface()

