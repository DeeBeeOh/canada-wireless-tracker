import json
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

def generate_real_data():
    pytrends = TrendReq(hl='en-CA', tz=360)
    # High-intent Canadian cellphone activation queries
    keywords = ['activate SIM', 'eSIM Canada', 'port number', 'BYOD plan']
    
    # Pulls rolling 5 full years of real weekly Canadian data
    pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo='CA')
    df = pytrends.interest_over_time()
    
    if 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
        
    # Weighted real activation index (0-100)
    df['Actual'] = (
        df['activate SIM'] * 0.40 +
        df['eSIM Canada'] * 0.30 +
        df['port number'] * 0.20 +
        df['BYOD plan'] * 0.10
    )
    
    df.reset_index(inplace=True)
    df.rename(columns={'date': 'Date'}, inplace=True)
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Compute true 5-year median baseline for each calendar week
    norms = df.groupby('Week_of_Year')['Actual'].median().reset_index()
    norms.rename(columns={'Actual': 'Baseline'}, inplace=True)
    norms['Baseline'] = norms['Baseline'].rolling(window=3, center=True, min_periods=1).mean()
    
    df = pd.merge(df, norms, on='Week_of_Year', how='left')
    df.sort_values('Date', inplace=True)
    
    records = []
    for _, row in df.iterrows():
        m, d = row['Date'].month, row['Date'].day
        ctx = "Standard Market Period"
        if m == 11 and d >= 18:
            ctx = "Black Friday / Cyber Monday Surge"
        elif m == 12 and d >= 18:
            ctx = "Boxing Week Retail Peak"
        elif (m == 8 and d >= 15) or (m == 9 and d <= 18):
            ctx = "Back-to-School & Student Arrival Peak"
        elif m == 9 and d > 18:
            ctx = "Fall Flagship Hardware (iPhone) Launch"
        elif m == 1 and d <= 18:
            ctx = "New Year Immigration / Winter Term Inflow"
        elif m == 2 and 10 <= d <= 25:
            ctx = "Spring Flagship Hardware (Galaxy) Release"
            
        actual_val = round(float(row['Actual']), 1)
        base_val = round(float(row['Baseline']), 1)
        pct_diff = round(((actual_val - base_val) / base_val) * 100, 1) if base_val > 0 else 0.0
        
        records.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "actual": actual_val,
            "baseline": base_val,
            "pct_diff": pct_diff,
            "context": ctx
        })
        
    with open('data.json', 'w') as f:
        json.dump(records, f)
    print(f"Saved {len(records)} real weekly records to data.json")

if __name__ == "__main__":
    generate_real_data()
