import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# === Markov Chain Model ===
class MarkovChainModel:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.bins = None
        self.transition_matrix = None
        self.state_means = None

    def discretize(self, series):
        self.bins = pd.qcut(series, self.n_states, labels=False, duplicates='drop')
        return self.bins

    def fit(self, y_train):
        states = self.discretize(y_train)
        matrix = np.zeros((self.n_states, self.n_states))
        for i, j in zip(states[:-1], states[1:]):
            matrix[i, j] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = matrix / np.where(row_sums == 0, 1, row_sums)
        df = pd.DataFrame({"state": states, "value": y_train})
        self.state_means = df.groupby("state")["value"].mean().to_dict()

    def predict(self, y_last_state, steps):
        preds = []
        current_state = y_last_state
        for _ in range(steps):
            next_probs = self.transition_matrix[current_state]
            next_state = np.argmax(next_probs)
            preds.append(self.state_means.get(next_state, 0))
            current_state = next_state
        return np.array(preds)

# === Load and preprocess merged data ===
def load_data():
    df = pd.read_csv("data/merged_sorted.csv", parse_dates=['datetime'])
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["pm2.5_lag1"] = df["PM2.5"].shift(1)
    df["pm2.5_lag2"] = df["PM2.5"].shift(2)
    df["pm2.5_lag3"] = df["PM2.5"].shift(3)
    df["pm2.5_roll3"] = df["PM2.5"].rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Final dataset shape: {df.shape}")
    return df

# === Evaluation Utility ===
def evaluate(y_true, y_pred, model_name, y_train_true=None, y_train_pred=None):
    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_mse = mean_squared_error(y_true, y_pred)

    result = {
        "model": model_name,
        "Test MAE": round(test_mae, 2),
        "Test MSE": round(test_mse, 2),
        "Test RMSE": round(test_rmse, 2)
    }

    if y_train_true is not None and y_train_pred is not None:
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        result.update({
            "Train MAE": round(train_mae, 2),
            "Train MSE": round(train_mse, 2),
            "Train RMSE": round(train_rmse, 2)
        })

    print(f"✅ {model_name}:")
    print(f"   Train RMSE = {result.get('Train RMSE', 'N/A')}, Test RMSE = {result['Test RMSE']}")
    return result

# === Shared Model Training Function ===
def run_model(df, model_class, model_name, scale=False, model_store=None, **kwargs):
    features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10",
                "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR",
                "pm2.5_lag1", "pm2.5_lag2", "pm2.5_lag3", "pm2.5_roll3", "hour", "dayofweek"]
    target = "PM2.5"
    X = df[features].copy()
    y = df[target]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, "model/results/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = model_class(**kwargs)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    if model_store is not None:
        model_store[model_name] = model

    return evaluate(y_test, y_pred_test, model_name, y_train_true=y_train, y_train_pred=y_pred_train)

# === ARIMA Model with Auto Tuning ===
def run_arima(df):
    try:
        series = df["PM2.5"]
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        p = d = q = range(0, 3)
        pdq_combinations = list(itertools.product(p, d, q))
        best_score = float("inf")
        best_order = (1, 1, 1)

        for order in pdq_combinations:
            try:
                model = ARIMA(train, order=order).fit()
                if model.aic < best_score:
                    best_score = model.aic
                    best_order = order
            except:
                continue

        model = ARIMA(train, order=best_order).fit()
        forecast_test = model.forecast(steps=len(test))
        y_pred_train = model.fittedvalues
        y_true_train = train[1:] if best_order[1] > 0 else train

        return evaluate(test, forecast_test, "ARIMA", y_train_true=y_true_train, y_train_pred=y_pred_train)
    except Exception as e:
        print(f"❌ ARIMA error: {str(e)}")
        return {"model": "ARIMA", "Test MAE": float('inf'), "Test MSE": float('inf'), "Test RMSE": float('inf')}

# === Markov Chain Model ===
def run_markov(df):
    try:
        y = df["PM2.5"].values
        train, test = y[:-10], y[-10:]
        markov = MarkovChainModel(n_states=5)
        markov.fit(train)

        last_state = pd.qcut(train, markov.n_states, labels=False, duplicates='drop')[-1]
        y_pred_test = markov.predict(last_state, steps=10)

        # Training prediction using transition matrix
        train_preds = []
        states = pd.qcut(train, markov.n_states, labels=False, duplicates='drop')
        for i in range(len(states) - 1):
            next_state = np.argmax(markov.transition_matrix[states[i]])
            train_preds.append(markov.state_means.get(next_state, 0))

        y_train_true = train[1:]
        y_train_pred = np.array(train_preds)

        return evaluate(test, y_pred_test, "Markov Chain", y_train_true=y_train_true, y_train_pred=y_train_pred)
    except Exception as e:
        print(f"❌ Markov error: {str(e)}")
        return {"model": "Markov Chain", "Test MAE": float('inf'), "Test MSE": float('inf'), "Test RMSE": float('inf')}

# === Run All Models ===
def run_all_models():
    print("\n🚀 Starting PM2.5 prediction using merged_sorted.csv...")
    df = load_data()
    if df.shape[0] < 20:
        print("❌ Not enough usable rows after preprocessing. Please collect more data.")
        return

    model_objects = {}

    results = [
        run_model(df, LinearRegression, "Linear Regression", scale=True, model_store=model_objects),
        run_model(df, RandomForestRegressor, "Random Forest", n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42, model_store=model_objects),
        run_model(df, XGBRegressor, "XGBoost", n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8, random_state=42, model_store=model_objects),
        run_model(df, LGBMRegressor, "LightGBM", n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=8, random_state=42, model_store=model_objects),
        run_model(df, SVR, "SVR", scale=True, C=50, epsilon=0.05, kernel='rbf', model_store=model_objects),
        run_model(df, KNeighborsRegressor, "KNN", scale=True, n_neighbors=5, weights='distance', model_store=model_objects),
        run_arima(df),
        run_markov(df)
    ]

    report = pd.DataFrame(results).sort_values('Test MAE').reset_index(drop=True)
    print("\n📊 Final Results (sorted by Test MAE):")
    print(report.to_string(index=False))

    os.makedirs("model/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"model/results/training_report_{timestamp}.csv"
    report.to_csv(report_path, index=False)
    print(f"\n📄 Report saved to: {report_path}")

    excluded_models = {"Linear Regression"}
    filtered_report = report[~report["model"].isin(excluded_models)]

    if filtered_report.empty:
        print("⚠️ No valid models found after excluding certain models.")
        return

    best_model_name = filtered_report.iloc[0]['model']
    best_model_obj = model_objects.get(best_model_name)
    if best_model_obj:
        best_model_path = f"model/final_model_{timestamp}.pkl"
        joblib.dump(best_model_obj, best_model_path)
        print(f"💾 Final best model ({best_model_name}) saved to: {best_model_path}")
    else:
        print(f"⚠️ Best model object for {best_model_name} not found.")


if __name__ == "__main__":
    run_all_models()
