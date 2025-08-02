import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess(input_path="data/taipei_long0801.csv", output_path="data/processed.csv", scaler_path="model/scaler.pkl"):
    df = pd.read_csv(input_path)

    # Keep only relevant columns
    df = df[["sitename", "monitordate", "itemengname", "concentration"]]

    # Pivot: wide format
    df = df.pivot_table(
        index=["sitename", "monitordate"],
        columns="itemengname",
        values="concentration"
    ).reset_index()

    df.columns.name = None

    # Convert date/time
    df["datetime"] = pd.to_datetime(df["monitordate"], errors="coerce")
    df = df.sort_values("datetime")

    # Drop rows missing PM2.5
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["PM2.5"])

    # Lag & time features
    df["pm2.5_lag1"] = df["PM2.5"].shift(1)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    # Fill missing values:
    # 1. Forward fill
    df = df.fillna(method="ffill")

    # 2. Fill any remaining NaNs with global median
    df = df.fillna(df.median(numeric_only=True))

    # === 🧪 Standardize features ===
    exclude_cols = ['sitename', 'monitordate', 'datetime', 'PM2.5']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"📦 Scaler saved to {scaler_path}")

    # Save final processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    preprocess()
