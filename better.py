import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
# from amlp import AttentionalMLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import shap

from amlp import AttentionalMLPClassifier


# ============================
# Logging Utility
# ============================
def log(message):
    print(f"{datetime.datetime.now()} - {message}")


# ============================
# Parameters
# ============================
start_train = datetime.date(2017, 1, 1)
end_train = datetime.date(2023, 11, 30)
start_test = datetime.date(2024, 1, 1)
end_test = datetime.date(2024, 6, 30)

n_buys = 10
verbose = True
retrain_interval = 300  # Retrain every n steps, set to None to disable retraining
scaling_enabled = False  # Toggle for feature scaling

# ============================
# Initialize
# ============================
log("---> Python Script Start")
t0 = datetime.datetime.now()

# ============================
# Data Loading
# ============================
log("---> Initial data setup")

# Sector data
df_sectors = pd.read_csv("data/data0.csv")

# Price and financial data
df_data = pd.read_csv("data/data1.csv")
df_data["date"] = pd.to_datetime(df_data["date"]).dt.date


# ============================
# Feature Engineering
# ============================
def feature_engineering(df, df_sectors):
    # Add Sharpe ratio as a feature
    df["sharpe_ratio"] = df["return30"] / df["price"].rolling(window=30).std()
    # Add date-based features
    df["day"] = df["date"].apply(lambda x: x.day)
    # # Tomorrow day
    # df["day_tomorrow"] = df["day"].shift(-1)

    # df["weekday"] = df["date"].apply(lambda x: x.weekday())

    # # Day of weak tomorrow 
    # df["weekday_tomorrow"] = df["weekday"].shift(-1)

    # Dropping
    # df = df.drop(columns=["day", "weekday"])
    # Day of the week
    # Moving averages
    # df["ma7"] = df["price"].rolling(window=7).mean()
    # df["ma30"] = df["price"].rolling(window=30).mean()
    
    # Adding sectors and label encoding
    df = pd.merge(df, df_sectors, on="security", how="left")
    df["sector"] = df["sector"].astype("category").cat.codes

    # Getting 30 day return of each sector for last 30 days
    # for sec in df["sector"].unique():
    #     df[f"return30_{sec}"] = (
    #         df.loc[df["sector"] == sec, "return30"].rolling(window=5).mean()
    #     )
    df = df.fillna(0)
    df = df.dropna()  # Drop rows with NaN values after feature engineering
    return df


df_data = feature_engineering(df_data, df_sectors)

# Define features and target
df_x = df_data.drop(columns=["label"]).copy()
df_y = df_data[["date", "security", "label"]].copy()

list_vars1 = [col for col in df_x.columns if col not in ["date", "security"]]

# ============================
# Get Test Dates
# ============================
df_signals = pd.DataFrame(
    data={
        "date": df_x.loc[
            (df_x["date"] >= start_test) & (df_x["date"] <= end_test), "date"
        ].unique()
    }
)
df_signals.sort_values(by="date", inplace=True)
df_signals.reset_index(drop=True, inplace=True)

# ============================
# Feature Scaling
# ============================
log("---> Scaling features")

# If scaling enabled scale the data else create dict
dict_scaler = {}
if scaling_enabled:
    for col in list_vars1:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_x[col].values.reshape(-1, 1))
        dict_scaler[col] = scaler
    log("---> Feature scaling is enabled")
else:
    log("---> Feature scaling is disabled")
    for col in list_vars1:
        dict_scaler[col] = df_x[col].values

# ============================
# Model Setup
# ============================
log("---> Setting up models")

m1 = CatBoostClassifier(n_estimators=100, random_seed=42, verbose=0)
# m2 = RandomForestClassifier(n_estimators=10, random_state=23)
m2 = AttentionalMLPClassifier(
    input_dim=len(list_vars1),
    hidden_dim=16,
    num_classes=2,
    lr=0.01,
    batch_size=256,
    epochs=20,
    verbose=True,
    device=None,
    random_state=None,
)


models = [
    ('catboost', m1),
    # ('randomforest', m2)
    # ("amlp", m2)
]

# model = VotingClassifier(estimators=models, voting="soft")
model = m1  # Using CatBoost directly

# ============================
# Feature Importance Storage
# ============================
feature_importances = pd.DataFrame(columns=["feature", "importance"])

# ============================
# Prediction and Signal Generation
# ============================
log("---> Starting prediction loop")

for i in range(len(df_signals)):
    current_date = df_signals.loc[i, "date"]
    if verbose:
        log(f"---> Processing date: {current_date}")

    # Define training window (past 90 days as per updated script)
    start_lookback = current_date - datetime.timedelta(days=300)
    df_trainx = df_x[
        (df_x["date"] < current_date) & (df_x["date"] >= start_lookback)
    ].copy()
    df_trainy = df_y[
        (df_y["date"] < current_date) & (df_y["date"] >= start_lookback)
    ].copy()

    # Scale training features if scaling is enabled
    if scaling_enabled:
        for col in list_vars1:
            df_trainx[col] = dict_scaler[col].transform(
                df_trainx[col].values.reshape(-1, 1)
            )[:, 0]

    # Fit model at specified intervals
    if retrain_interval and (i % retrain_interval == 0 or i == 0):

        # Example 2: Time-based weights (optional)
        df_trainy["time_weight"] = df_trainy["date"].apply(
            lambda x: np.exp(-0.05 * (current_date - x).days)
        )
        sample_weights = df_trainy["time_weight"]

        if i == 0:

            model.fit(
                df_trainx[list_vars1],
                df_trainy["label"].values,
                # sample_weight=sample_weights,
            )
        else:
            model.fit(
                df_trainx[list_vars1],
                df_trainy["label"].values,
                sample_weight=sample_weights,
                init_model=model,
            )
        # Extract and store feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_[0]
        else:
            # Using permutation importance as a fallback
            perm_importance = permutation_importance(
                model,
                df_trainx[list_vars1],
                df_trainy["label"].values,
                n_repeats=10,
                random_state=42,
            )
            importances = perm_importance.importances_mean

        temp_importances = pd.DataFrame(
            {"feature": list_vars1, "importance": importances}
        ).sort_values(by="importance", ascending=False)

        feature_importances = pd.concat(
            [feature_importances, temp_importances], ignore_index=True
        )

    # Define test set for the current date
    df_testx = df_x[df_x["date"] == current_date].copy()
    df_testy = df_y[df_y["date"] == current_date].copy()

    if df_testx.empty:
        if verbose:
            log(f"---> No data for date: {current_date}")
        continue

    # Scale test features if scaling is enabled
    if scaling_enabled:
        for col in list_vars1:
            # To handle potential unseen values in test set, use transform with existing scaler
            df_testx[col] = dict_scaler[col].transform(
                df_testx[col].values.reshape(-1, 1)
            )[:, 0]

    # Predict and calculate accuracy
    if hasattr(model, "predict_proba"):
        signal = model.predict_proba(df_testx[list_vars1])[:, 1]
    else:
        # If the model does not have predict_proba, use decision function or predictions
        signal = model.predict(df_testx[list_vars1])

    prediction = (signal > 0.5).astype(int)
    df_testy["signal"] = signal
    df_testy["pred"] = prediction
    df_testy["count"] = 1

    acc_current = (
        (df_testy["label"] == df_testy["pred"]).sum() / len(df_testy)
        if len(df_testy) > 0
        else 0
    )
    df_signals.loc[i, "acc_current"] = acc_current

    # Store buy signals for securities
    for sec, sig in zip(df_testy["security"], df_testy["signal"]):
        df_signals.at[i, sec] = sig

    if verbose and (i + 1) % 50 == 0:
        log(f"---> Processed {i + 1} out of {len(df_signals)} dates")

# ============================
# Feature Importance Visualization
# ============================
log("---> Visualizing feature importances")

if not feature_importances.empty:
    # Aggregate feature importances
    avg_importances = feature_importances.groupby("feature").mean().reset_index()
    avg_importances.sort_values(by="importance", ascending=False, inplace=True)

    # Plot feature importances
    fig_importance = px.bar(
        avg_importances,
        x="importance",
        y="feature",
        orientation="h",
        title="Average Feature Importances",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_importance.show()
else:
    log("---> No feature importances to display")

# ============================
# Stock Selection
# ============================
log("---> Selecting stocks to buy")

# Ensure that df_sectors['security'] matches df_signals columns
securities = df_sectors["security"].unique()
existing_securities = [sec for sec in securities if sec in df_signals.columns]

if not existing_securities:
    log("---> No matching securities found in signals")
    existing_securities = []

# Create buy matrix for payoff plot
if existing_securities:
    df_signals["threshold"] = df_signals[existing_securities].apply(
        lambda x: np.percentile(x, 100 - (n_buys / len(x)) * 100) if len(x) > 0 else 0,
        axis=1,
    )

    df_index = df_signals[existing_securities].gt(df_signals["threshold"], axis=0)

    # Set 1 for top N strongest signals split among sectors
    
    df_buys = pd.DataFrame(0, index=df_signals.index, columns=existing_securities)
    df_buys[df_index] = 1
    df_buys.insert(0, "date", df_signals["date"].copy())

    # df_buys = pd.DataFrame(0, index=df_signals.index, columns=existing_securities)
    # df_buys[df_index] = 1
    # df_buys.insert(0, "date", df_signals["date"].copy())
else:
    df_buys = pd.DataFrame({"date": df_signals["date"]})


# compute SHAP values
explainer = shap.Explainer(model, df_x)
explanation = explainer(df_x[:1000])

shap.plots.scatter(explanation[:, "day"])

# ============================
# Payoff Calculation
# ============================
log("---> Calculating payoff")

# Create return matrix
df_returns = pd.read_csv("data/returns.csv")
df_returns["date"] = pd.to_datetime(df_returns["date"]).dt.date
df_returns = df_returns[df_returns["date"] >= start_test]
df_returns = df_returns.pivot(index="date", columns="security", values="return1")


def plot_payoff(df_buys, df_returns, n_buys):
    df = df_buys.copy()

    if "threshold" in df.columns:
        df = df.drop(columns=["threshold"])

    # Matrix of buys
    df_payoff = df[["date"]].copy()
    df_buys_matrix = (
        df.drop(columns=["date"]).values if "date" in df.columns else df.values
    )
    arr_buys = df_buys_matrix * (1 / n_buys)  # Equally weighted

    # Align returns with buys
    aligned_returns = df_returns.reindex(df["date"]).fillna(0).values + 1

    # Calculate daily payoff
    daily_payoff = (arr_buys * aligned_returns).sum(axis=1)
    df_payoff["payoff"] = daily_payoff
    df_payoff["tri"] = df_payoff["payoff"].cumprod()

    # Plot cumulative returns
    fig_payoff = px.line(df_payoff, x="date", y="tri", title="Cumulative Payoff")
    fig_payoff.show()

    if not df_payoff.empty:
        final_return = (df_payoff["tri"].iloc[-1] - 1) * 100
        log(
            f"---> Payoff for these buys between {df_payoff['date'].min()} and {df_payoff['date'].max()} is {final_return:.2f}%"
        )
    else:
        log("---> No payoffs to calculate")

    return df_payoff


df_payoff = plot_payoff(df_buys, df_returns, n_buys)

log("---> Python Script End")
t1 = datetime.datetime.now()
log(f"---> Total time taken: {t1 - t0}")
