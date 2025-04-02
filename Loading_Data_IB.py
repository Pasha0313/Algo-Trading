import pandas as pd
import numpy as np
import time
import re
import pytz
from ib_insync import IB, util, Forex, Stock, Future

def normalize_bar_length_for_timedelta(bar_length):
    bar_length = bar_length.strip().lower()
    bar_length = re.sub(r"\bmins\b", "min", bar_length)
    bar_length = re.sub(r"\bsecs\b", "sec", bar_length)
    return bar_length

def get_safe_duration_str(bar_length):
    bar_length = bar_length.strip().lower()
    duration_map = {'1 sec': '30 M','5 sec': '1 H','10 sec': '2 H','30 sec': '1 D',
                    '1 min': '1 D' ,'5 min': '1 W','15 min': '2 W','30 min': '3 W',
                    '1 hour': '1 M','1 day': '1 Y'}
    
    bar_length = bar_length.replace("mins", "min").replace("secs", "sec")
    return duration_map.get(bar_length, '1 D')

def fetch_historical_data(ib, contract, bar_length, start, end=None, timezone="US/Eastern"):
    all_bars = []
    # Load selected timezone
    tz = pytz.timezone(timezone)

    # Apply timezone localization
    current_start = pd.to_datetime(start)
    current_start = tz.localize(current_start) if current_start.tzinfo is None else current_start.tz_convert(tz)

    if end:
        current_end = pd.to_datetime(end)
        current_end = tz.localize(current_end) if current_end.tzinfo is None else current_end.tz_convert(tz)
    else:
        current_end = pd.Timestamp.utcnow().tz_localize(tz)

    # Normalize bar length & calculate batch duration
    bar_length_td = normalize_bar_length_for_timedelta(bar_length)
    duration_per_batch = get_safe_duration_str(bar_length)

    while current_start < current_end:
        #print(f"Requesting data up to {current_end.strftime('%Y-%m-%d %H:%M:%S')}...")
        print(f"Requesting data up to {current_end.strftime('%Y-%m-%d %H:%M')}...")

        try:
            bars = ib.reqHistoricalData(
                contract,endDateTime=current_end.strftime('%Y%m%d-%H:%M:%S'),  
                durationStr=duration_per_batch,barSizeSetting=bar_length,
                whatToShow='MIDPOINT',useRTH=False,formatDate=1)

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            break

        if not bars:
            print("No more data returned.")
            break

        df = util.df(bars)
        df['date'] = df['date'].dt.tz_convert(timezone)  
        df = df[df['date'] >= current_start]

        if df.empty:
            print("Only one new candle collected, exiting loop.")
            break

        all_bars.insert(0, df)  # reverse order
        #current_end = df['date'].min() - pd.Timedelta(seconds=1)
        min_date = df['date'].min()
        if min_date >= current_end:
            print("⚠️ No older data found. Breaking to avoid infinite loop.")
            break

        current_end = min_date - pd.Timedelta(seconds=1)

        print(f"Collected {sum(len(x) for x in all_bars)} candles so far...")
        time.sleep(1)

    if not all_bars:
        print("❌ No data collected.")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "returns", "Complete"])

    data = pd.concat(all_bars)
    data.set_index('date', inplace=True)
    data = data[["open", "high", "low", "close", "volume"]].copy()
    data.columns = ["Open", "High", "Low", "Close", "Volume"]

    data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Complete"] = [True for _ in range(len(data) - 1)] + [False]

    data.to_csv("Test_Data.csv", index=True)

    print(f"✅ Total of {len(data)} candles collected.\n")
    return data
