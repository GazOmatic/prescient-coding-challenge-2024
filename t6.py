import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier as xgb

from sklearn.model_selection import StratifiedKFold


def log(message):
    print(f'{datetime.datetime.now()} - {message}')
print('---> Python Script Start', t0 := datetime.datetime.now())

# Parameters
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30)
start_test = datetime.date(2024, 1, 1)
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = True
retrain_interval = 300  # Retrain every n steps, set to None to disable retraining

print('---> Initial data setup')

# Sector data
df_sectors = pd.read_csv('data/data0.csv')

def feature_engineering(df):
    # Add Sharpe ratio as a feature
    df['sharpe_ratio'] = df['return30'] / df['price'].rolling(window=30).std()
    df = df.dropna()  # Drop rows with NaN values after feature engineering
    return df

# Price and financial data
df_data = pd.read_csv('data/data1.csv')
df_data['date'] = pd.to_datetime(df_data['date']).apply(lambda d: d.date())
df_data = feature_engineering(df_data)
# Use all available features dynamically
df_x = df_data.drop(columns=['label']).copy()
df_y = df_data[['date', 'security', 'label']].copy()

list_vars1 = [col for col in df_x.columns if col not in ['date', 'security']]

# Get test dates
df_signals = pd.DataFrame(data={'date': df_x.loc[(df_x['date'] >= start_test) & (df_x['date'] <= end_test), 'date'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='date', inplace=True)

# Scale features
dict_scaler = {}
for col in list_vars1:
    dict_scaler[col] = MinMaxScaler(feature_range=(-1, 1))

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models
def get_models():
    return [
        
        CatBoostClassifier(iterations=50, random_seed=23, verbose=0),
        # xgb(n_estimators=100,  learning_rate=0.1, random_state=42, verbosity=0),
    ]

models = get_models()

# Iterate over test dates
for i in range(len(df_signals)):
    current_date = df_signals.loc[i, 'date']
    if verbose:
        print('---> Doing', current_date)

    # Define training set for the current date (only the past 2 months)
    start_lookback = current_date - datetime.timedelta(days=30)
    df_trainx = df_x[(df_x['date'] < current_date) & (df_x['date'] >= start_lookback)].copy()
    df_trainy = df_y[(df_y['date'] < current_date) & (df_y['date'] >= start_lookback)].copy()

    # Scale features for training set
    for col in list_vars1:
        df_trainx[col] = dict_scaler[col].fit_transform(df_trainx[col].values.reshape(-1, 1))[:, 0]

    # Fit models
    for model in models:
        model.fit(df_trainx[list_vars1], df_trainy['label'].values)

    # Define test set for the current date
    df_testx = df_x[df_x['date'] == current_date].copy()
    df_testy = df_y[df_y['date'] == current_date].copy()

    # Scale features for test set
    for col in list_vars1:
        df_testx[col] = dict_scaler[col].transform(df_testx[col].values.reshape(-1, 1))[:, 0]

    # Predict and calculate accuracy
    signals = []
    predictions = []
    for model in models:
        signal = model.predict_proba(df_testx[list_vars1])[:, 1] if hasattr(model, 'predict_proba') else model.predict(df_testx[list_vars1])
        signals.append(signal)
        predictions.append(model.predict(df_testx[list_vars1]))
    df_testy['signal'] = np.mean(signals, axis=0)
    df_testy['pred'] = np.mean(predictions, axis=0) > 0.5
    df_testy['count'] = 1

    acc_current = (df_testy['label'] == df_testy['pred']).sum() / len(df_testy)
    df_signals.loc[i, 'acc_current'] = acc_current

    # Store buy signals for securities
    df_signals.loc[i, df_testy['security'].values] = df_testy['signal'].values

# Create buy matrix for payoff plot
df_signals['10th'] = df_signals[df_sectors['security'].values].apply(lambda x: sorted(x)[len(df_sectors) - n_buys - 1], axis=1)

df_index = pd.DataFrame(np.array(df_signals[df_sectors['security'].values]) > np.array(df_signals['10th']).reshape((len(df_signals), 1)))

# Set 1 for top 10 strongest signals
df_buys = pd.DataFrame()
df_buys[df_sectors['security'].values] = np.zeros((len(df_signals), len(df_sectors)))
df_buys[df_index.values] = 1
df_buys.insert(0, 'date', df_signals['date'].copy())

# Check some signal plots
fig_aapl = px.line(df_signals, x='date', y='AAPL')
fig_aapl.show()

fig_pixel = px.imshow(np.array(df_buys[df_sectors['security'].values]))
fig_pixel.show()

# Create return matrix
df_returns = pd.read_csv('data/returns.csv')
df_returns['date'] = pd.to_datetime(df_returns['date']).apply(lambda d: d.date())
df_returns = df_returns[df_returns['date'] >= start_test]
df_returns = df_returns.pivot(index='date', columns='security', values='return1')

def plot_payoff(df_buys):
    df = df_buys.copy()

    # Matrix of buys
    df_payoff = df[['date']].copy()
    del df['date']
    arr_buys = np.array(df)
    arr_buys = arr_buys * (1 / n_buys)  # Equally weighted

    # Return matrix
    arr_ret = np.array(df_returns)
    arr_ret = arr_ret + 1

    df_payoff['payoff'] = (arr_buys * arr_ret @ np.ones(len(df_sectors)).reshape((len(df_sectors), 1)))[:, 0]
    df_payoff['tri'] = df_payoff['payoff'].cumprod()

    fig_payoff = px.line(df_payoff, x='date', y='tri')
    fig_payoff.show()

    print(f"---> Payoff for these buys between period {df_payoff['date'].min()} and {df_payoff['date'].max()} is {(df_payoff['tri'].values[-1] - 1) * 100 :.2f}%")

    return df_payoff

df_payoff = plot_payoff(df_buys)

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)