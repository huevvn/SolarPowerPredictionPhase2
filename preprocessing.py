import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df.rename(columns={
        'AverageDew(point via humidity)': 'DewPoint',
        'Solar(PV)': 'SolarPV',
        'AvgTemperture': 'Temperature'
    }, inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        df['Temp_Hum'] = df['Temperature'] * df['Humidity']
        df['Temp_Div_Hum'] = df['Temperature'] / (df['Humidity'] + 1)

    if 'Wind' in df.columns and 'Pressure' in df.columns:
        df['Wind_Press'] = df['Wind'] * df['Pressure']

    for c in ['Temperature', 'SolarPV', 'Wind', 'DewPoint']:
        if c in df.columns:
            df[f'{c}_Sq'] = df[c] ** 2

    return df

def handle_missing(df, method='mean'):
    df = df.copy()
    nums = df.select_dtypes(include=[np.number]).columns
    if method == 'mean':
        df[nums] = df[nums].fillna(df[nums].mean())
    elif method == 'median':
        df[nums] = df[nums].fillna(df[nums].median())
    return df

def remove_outliers(df, cols=None):
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
    labels = ['Low', 'Medium', 'High']
    df = df.copy()
    df[col + '_Class'] = pd.qcut(df[col], q=n, labels=labels)
    return df
