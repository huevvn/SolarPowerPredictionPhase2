# preprocessing.py - data loading and cleaning stuff

import pandas as pd
import numpy as np

def load_data(path):
    # read csv and fix column names
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # drop index column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # rename cols to something readable
    df.rename(columns={
        'AverageDew(point via humidity)': 'DewPoint',
        'Solar(PV)': 'SolarPV',
        'AvgTemperture': 'Temperature'
    }, inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # Granular time features
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        
    # Feature Engineering for "The Kitchen Sink" approach (Force Accuracy)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Interaction terms
    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        df['Temp_Hum'] = df['Temperature'] * df['Humidity']
        df['Temp_Div_Hum'] = df['Temperature'] / (df['Humidity'] + 1) # avoid zero div
        
    if 'Wind' in df.columns and 'Pressure' in df.columns:
        df['Wind_Press'] = df['Wind'] * df['Pressure']
        
    # Polynomial features (squares)
    for c in ['Temperature', 'SolarPV', 'Wind', 'DewPoint']:
        if c in df.columns:
            df[f'{c}_Sq'] = df[c] ** 2
            
    return df

def handle_missing(df, method='mean'):
    # fill missing vals
    df = df.copy()
    nums = df.select_dtypes(include=[np.number]).columns
    if method == 'mean':
        df[nums] = df[nums].fillna(df[nums].mean())
    elif method == 'median':
        df[nums] = df[nums].fillna(df[nums].median())
    return df

def remove_outliers(df, cols=None):
    # iqr based outlier removal
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        df = df[(df[c] >= low) & (df[c] <= high)]
    return df

def bin_target(df, col, n=3):
    # split into Low/Med/High
    labels = ['Low', 'Medium', 'High']
    df = df.copy()
    df[col + '_Class'] = pd.qcut(df[col], q=n, labels=labels)
    return df
