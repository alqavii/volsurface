from email import header
from time import sleep
from functools import lru_cache
from xmlrpc.client import ExpatParser
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
import streamlit as st
import pytz
import threading

FredApiKey = st.secrets["FRED_API_KEY"] or os.getenv("FRED_API_KEY")

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
        self.universe = {}
        self.autocompleteList = []

    @property
    def ticker(self):
        return self._ticker
    @ticker.setter
    def ticker(self, value):
        self._ticker = value
        self.stock = yf.Ticker(self.ticker)
        self.spot = self.stock.history(period="1d")['Close'].iloc[-1]
        self.history = self.stock.history(period="1mo")
        print(f"Spot price for {self.ticker} is ${self.spot:.2f}\n")
        self.backgroundTickerUniverse()
        self.getCalls()
        self.getPuts()
        self.getTimeToExpiry()
        self.getRates()
        self.getDividendYield()
        self.ivSurface()
        



    def getTickerUniverse(self, full=True):
        tickerUniverse = "tickerUniverse.csv"
        print("Fetching ticker universe...")

        if os.path.exists(tickerUniverse):
            modified = datetime.fromtimestamp(os.path.getmtime(tickerUniverse))
            age_seconds = (datetime.now() - modified).total_seconds()
            if age_seconds < 86400:  # 1 day
                print("Using cached ticker universe...")
                df = pd.read_csv(tickerUniverse)
                self.universe = df.set_index("ticker").to_dict(orient="index")
                self.autocompleteList = [
                    f"{ticker} - {info['name']}"
                    for ticker, info in sorted(self.universe.items(), key=lambda x: x[1]['mcap'], reverse=True)
                ]
                return
        print("Ticker universe is outdated, fetching new data...")
        tickers = pd.read_csv("nasdaqlisted.txt", sep="|").iloc[:-1]
        tickers = tickers[["Symbol"]].rename(columns={"Symbol": "ticker"})
        tickers = tickers.dropna(subset=["ticker"])
        tickers["ticker"] = tickers["ticker"].astype(str)

        rows = []
        for ticker in tickers['ticker']:
            try:
                sleep(0.1)  # Avoid hitting API limits
                print(f"Processing {ticker}...")
                stock = yf.Ticker(ticker)
                spot = stock.history(period="1d")['Close'].iloc[-1]
                name = stock.info.get('longName', ticker)
                mcap = stock.info.get('marketCap', 0)
                if (mcap >= 1e7) and (full or mcap > 1e9):
                    rows.append({"ticker": ticker, "name": name, "mcap": mcap})
            except Exception as e:
                print(f"Failed {ticker}: {e}")
                continue

        df = pd.DataFrame(rows)
        df.to_csv(tickerUniverse, index=False)
        self.universe = df.set_index("ticker").to_dict(orient="index")
        self.autocompleteList = [
            f"{ticker} - {info['name']}"
            for ticker, info in sorted(self.universe.items(), key=lambda x: x[1]['mcap'], reverse=True)
        ]
        return



    def backgroundTickerUniverse(self):
        if os.path.exists("tickerUniverse.csv"):
            df = pd.read_csv("tickerUniverse.csv")
            modified = datetime.fromtimestamp(os.path.getmtime("tickerUniverse.csv"))
            age_seconds = (datetime.now() - modified).total_seconds()
            if age_seconds > 86400: 
                threading.Thread(target=self.getTickerUniverse).start()
            if age_seconds > 3600 and age_seconds < 86400:
                threading.Thread(target=self.getTickerUniverse, kwargs={"full": False}).start()
        else:
            threading.Thread(target=self.getTickerUniverse).start()
        return

 
    def getCalls(self):
        for i in range(0,self.range):
            sleep(0.2)  
            temp = self.stock.option_chain(self.stock.options[i]).calls
            prices = temp[['strike', 'lastPrice', 'bid', 'ask']]
            prices = prices[(prices['strike'] <= self.spot*1.2) & (prices['strike'] >= self.spot*0.8)].sort_values(by='strike')
            #print(temp['contractSymbol'].iloc[0])
            #print(len(self.ticker))
            self.calls[f"{temp['contractSymbol'].iloc[0][4+len(self.ticker):6+len(self.ticker)]}/{temp['contractSymbol'].iloc[0][2+len(self.ticker):4+len(self.ticker)]}"] = prices

                  
    def getPuts(self):
        for i in range(0,self.range):
            sleep(0.2)
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
        eastern = pytz.timezone('US/Eastern')
        for expiry in self.calls:
            day, month = map(int, expiry.split('/'))
            now = datetime.now(eastern)
            year = now.year
            expiryDate = eastern.localize(datetime(year, month, day, 16, 0, 0))  
            self.timeToExpiries[expiry] = (expiryDate - now).total_seconds() / (365.25 * 24 * 3600)  

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
        return 

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
        self.printDividendYield()
        self.printOptions()


    def _blackScholesCall(self, sigma, K, T):
        r = self.latestRate / 100
        q = self.dividendYield /100
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
            T = self.timeToExpiries[expiry]
            if T * 365 * 24 < 6:
                continue
            data = data.copy()
            #data = data[(data['bid'] > 0) & (data['ask'] > data['bid']) & (data['ask'] < self.spot * 2)]
            data['midPrice'] = (data['bid'] + data['ask']) / 2
            data['IV'] = data.apply(lambda row: self.impliedVolatility(C = (row['bid'] + row['ask'])/2, K = row['strike'], T = T), axis=1)
            #data['IV'] = data.apply(lambda row: self.impliedVolatility(C = row['lastPrice'], K = row['strike'], T = T), axis=1)
            data['smoothedIV'] = UnivariateSpline(data.dropna(subset=['IV'])['strike'], data.dropna(subset=['IV'])['IV'], s=0.01)(data['strike'])
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


    def fig2dIVSurface(self):
        # explode your surface dict into a flat DataFrame
        rows = []
        for exp, df in self.surface.items():
            for K, iv in zip(df['strike'], df['smoothedIV']):
                rows.append({'Expiry': exp, 'Strike': K, 'IV': iv})
        pdf = pd.DataFrame(rows)

        fig = px.line(
            pdf,
            x='Strike',
            y='IV',
            color='Expiry',
            title=f'2D IV Surface for {self.ticker}'
        )
        fig.add_vline(x=self.spot, line_dash='dash', line_color='black')
        return fig
    


    def fig3dIVSurface(self):
        # collect points
        strikes, Ts, IVs = [], [], []
        for exp, df in self.surface.items():
            day, month = map(int, exp.split('/'))
            dt = datetime(datetime.today().year, month, day)
            t_num = mdates.date2num(dt)
            for K, iv in zip(df['strike'], df['smoothedIV']):
                strikes.append(K)
                Ts.append(t_num)
                IVs.append(iv)

        # make a meshgrid for surface
        grid_K = np.linspace(min(strikes), max(strikes), 60)
        grid_T = np.linspace(min(Ts), max(Ts), 60)
        K_mesh, T_mesh = np.meshgrid(grid_K, grid_T)
        Z = griddata(
            np.column_stack((strikes, Ts)), IVs,
            (K_mesh, T_mesh),
            method='cubic'
        )

        # convert numeric date back to datetime for axis
        T_dates = [mdates.num2date(x) for x in grid_T]

        fig = go.Figure(data=go.Surface(
            x=grid_K,
            y=T_dates,
            z=Z,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(title='IV')
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Expiry Date',
                zaxis_title='IV'
            ),
            title=f'3D IV Surface for {self.ticker}',
            height=600,
            autosize=True,
            
        )
        return fig




st.title("Volatility Surface Explorer")

surf = surface()


selection = st.sidebar.selectbox(
    "Select a Ticker",
    options=["Select a ticker"] + surf.autocompleteList
)



if selection != "Select a ticker":

    ticker = selection.split(" - ")[0].upper()    
    surf.ticker = ticker    
    
    st.subheader("Market Data") 
    st.write(f"• Spot price: ${surf.spot:.2f}")
    st.write(f"• Risk-free rate: {surf.latestRate:.2f}%")
    try:
        st.write(f"• Dividend yield: {surf.dividendYield:.2f}%")
    except:
        st.write(f"• Dividend yield: 0%")

    # 3D plot
    st.subheader("3D Volatility Surface")
    fig3d = surf.fig3dIVSurface()   # ditto
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D plot
    st.subheader("Impled Volatility Surface (2D)")
    fig2d = surf.fig2dIVSurface()   # your method must return the Figure
    st.plotly_chart(fig2d, use_container_width=True)

    