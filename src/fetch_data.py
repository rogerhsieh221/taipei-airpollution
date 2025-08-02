# src/fetch_data.py

import os
import requests
import pandas as pd

API_KEY = "1da6204b-e100-4b0a-b565-d13dbb6b1468"
BASE_URL = "https://data.moenv.gov.tw/api/v2/aqx_p_136"

def fetch_taipei_air_quality_long(output_path="data/taipei_long.csv"):
    os.makedirs("data", exist_ok=True)

    params = {
        "api_key": API_KEY,
        "format": "JSON",
        "limit": 4000,
    }

    print("📡 Fetching real-time Taipei air quality data...")
    response = requests.get(BASE_URL, params=params)

    print(f"🛰️ Status Code: {response.status_code}")
    print(f"🧾 Raw response preview:\n{response.text[:300]}")

    if response.status_code != 200:
        print(f"❌ HTTP Error: {response.status_code}")
        return

    try:
        data = response.json()["records"]
        df = pd.DataFrame(data)

        # # ✅ Filter to 台北市
        # df = df[df["county"] == "臺北市"]

        if df.empty:
            print("⚠️ No Taipei City records found.")
            return

        # ✅ Cast numeric values
        df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")

        # ✅ Select and reorder columns
        df = df[["sitename", "monitordate", "itemengname", "concentration"]]

        # ✅ Save as long-format CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Long-format data saved to {output_path} ({len(df)} rows)")

    except Exception as e:
        print("❌ JSON parse error:", e)

if __name__ == "__main__":
    fetch_taipei_air_quality_long()
