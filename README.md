# Taipei PM2.5 & PM10 Prediction Project

本專案旨在預測台北市的 PM2.5 與 PM10 空氣品質指標。  
透過環境部 API 取得空品監測數據，經過資料清理、特徵工程與多種模型訓練後，選出最佳模型以供後續部署與視覺化應用。

---

## 📁 專案架構與流程

```
.
├── src/                 # 原始碼
│   ├── fetch_data.py      # 從 API 下載並整理資料
│   ├── preprocessed.py    # 資料前處理與特徵工程
│   └── train_model.py     # 訓練 PM2.5 模型
│   └── train_pm10_model.py# 訓練 PM10 模型
├── data/                # 原始與處理後資料
├── model/               # Scaler、模型與訓練報告
```

---

## 📡 資料來源

- **API Endpoint**:  
  `https://data.moenv.gov.tw/api/v2/aqx_p_136`

- **主要欄位**:
  | 欄位名稱       | 說明             |
  | -------------- | ---------------- |
  | sitename       | 測站名稱         |
  | monitordate    | 監測時間（字串） |
  | itemengname    | 監測項目名稱     |
  | concentration  | 濃度（字串需轉數值） |

> 將原始 JSON 轉為長格式 CSV：`data/taipei_long.csv`

---

## 🧹 資料前處理（`preprocessed.py`）

1. **格式轉換：長 → 寬格式**
```python
df = df.pivot_table(
    index=["sitename", "monitordate"],
    columns="itemengname",
    values="concentration"
).reset_index()
```

2. **時間處理與排序**
```python
df["datetime"] = pd.to_datetime(df["monitordate"])
df = df.sort_values("datetime")
```

3. **缺失值處理**
- 移除 PM2.5 或 PM10 缺失值
- 其他欄位：先 forward fill → 再以全域中位數補齊
```python
df = df.dropna(subset=["PM2.5", "PM10"])
df = df.fillna(method="ffill")
df = df.fillna(df.median(numeric_only=True))
```

4. **特徵工程（Feature Engineering）**

- 為了提升模型對時間序列中 PM2.5 和 PM10 變化趨勢的理解與預測能力，我們設計了以下幾類特徵：

---

#### 📌 滯後特徵簡介（What are Lag Features?）

- 在時間序列建模中，**滯後特徵（Lag Features）** 是指使用「前一個或多個時間點」的觀測值作為目前時間點的輸入特徵。這種方式能幫助模型捕捉時間上的延續性與趨勢，例如「昨天的空氣品質會影響今天的空氣品質」。

舉例來說：
- `lag1` 代表的是 t-1 時刻的觀測值，用於預測 t 時刻。
- `lag3` 則是 t-3 的值，用於了解三小時前的狀況對現在的影響。
- `roll3` 是前 3 小時的平均值，有助於平滑突變值，反映短期趨勢。

- 滯後特徵是許多時間序列預測模型（如 ARIMA、XGBoost、LSTM）中的基礎特徵類型。

---

#### 時間特徵（Temporal Features）
- `hour`：從時間戳中提取的「小時」資訊，幫助模型捕捉日內變化週期（例如通勤時段污染上升）。
- `dayofweek`：提取星期幾（0 = 星期一，6 = 星期日），捕捉一週中的變化規律，例如週末與平日的空氣品質差異。

```python
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
```

5. **標準化**
```python
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "model/scaler.pkl")
```

---

## 🎯 PM2.5 預測說明

### 使用特徵（共 23 項）
```python
[
  "AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10",
  "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR",
  "pm2.5_lag1", "pm2.5_lag2", "pm2.5_lag3", "pm2.5_roll3", "hour", "dayofweek"
]
```

- 目標變數：`PM2.5`

---

## 🎯 PM10 預測說明

本模組旨在預測 PM10 空氣污染指標，並使用與 PM2.5 模型類似的資料處理與特徵工程流程。

### 📥 使用特徵（共 23 項）

```python
[
  "AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM2.5",
  "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR",
  "pm10_lag1", "pm10_lag2", "pm10_lag3", "pm10_roll3", "hour", "dayofweek"
]
```

- 目標變數：`PM10`

---

## 🤖 支援模型類型（PM2.5 / PM10 通用）

| 模型             | 類型         | 說明                                 |
|------------------|--------------|--------------------------------------|
| Linear Regression| 基準模型     | 僅 PM2.5 使用                         |
| Random Forest    | 集成模型     | 高效處理非線性特徵                   |
| XGBoost          | 強化集成     | 精準快速，適合時間序列變數           |
| LightGBM         | 強化集成     | 訓練快速，支援大量特徵               |
| SVR              | 支援向量機   | 擅長處理非線性與小樣本               |
| KNN              | 距離型模型   | 以鄰近歷史數據推測                   |
| ARIMA            | 時間序列模型 | 使用 AIC 選擇最佳 (p,d,q) 組合       |
| Markov Chain     | 機率模型     | 將污染值離散為狀態並預測轉移機率     |

---

## 🧭 `sitename` 的處理策略

- 所有模型皆 **不拆站點**，使用整個台北地區混合訓練資料。

| 方法         | 優點                         | 缺點                         |
|--------------|------------------------------|------------------------------|
| 不拆站點     | 訓練資料更多，模型更穩定     | 無法考慮個別站點特性         |
| 拆開訓練     | 精準反映地區差異             | 資料稀疏、容易 overfit       |

---

## 📊 模型評估指標

- MAE（Mean Absolute Error）  
- MSE（Mean Squared Error）  
- RMSE（Root Mean Squared Error）

每次訓練會儲存：
1. 模型比較報告（CSV）
2. 最佳模型（排除 inf 結果）儲存為 `.pkl`

---

## 模型結果
| Model           | PM10 Test MAE | PM10 Test RMSE | PM10 Train MAE | PM10 Train RMSE | PM2.5 Test MAE | PM2.5 Test RMSE | PM2.5 Train MAE | PM2.5 Train RMSE |
|-----------------|----------------|----------------|----------------|-----------------|----------------|----------------|------------------|------------------|
| LightGBM        | 0.64           | 0.81           | 0.39           | 0.50            | 2.42           | 3.10           | 1.43             | 1.81             |
| SVR             | 0.66           | 0.87           | 0.05           | 0.05            | 2.13           | 2.75           | 0.05             | 0.05             |
| XGBoost         | 0.72           | 0.88           | 0.00           | 0.00            | 2.03           | 2.65           | 0.00             | 0.01             |
| KNN             | 0.74           | 0.95           | 0.00           | 0.00            | 2.53           | 3.09           | 0.00             | 0.00             |
| Random Forest   | 0.76           | 0.94           | 0.24           | 0.33            | 2.10           | 2.89           | 0.88             | 1.14             |
| ARIMA           | 0.83           | 0.99           | 0.80           | 1.03            | inf            | inf            | inf              | inf              |
| Markov Chain    | 1.04           | 1.27           | 1.14           | 1.43            | 4.53           | 5.35           | 3.57             | 4.56             |
| Linear Regression | —            | —              | —              | —               | 0.00           | 0.00           | 0.00             | 0.00             |


## 🧪 執行流程範例

```bash
# PM2.5 預測流程
python src/fetch_data.py
python src/preprocessed.py
python src/train_model_pm25.py

# PM10 預測流程
python src/train_model_pm10.py
```

### 🔍 產出結果

| 輸出檔案                                      | 說明                             |
|-----------------------------------------------|----------------------------------|
| `data/processedXXXX.csv`                      | 清理與特徵工程後的資料           |
| `model/scaler.pkl` / `scaler_pm10.pkl`        | 特徵標準化器                     |
| `model/final_model_*.pkl`                     | 儲存最佳模型                     |
| `model/results/training_report_*.csv`         | 模型評估報告（按 MAE 排序）     |

---

## 📌 備註

- 若預處理後資料少於 20 筆，將跳出提醒並中止訓練
- 預設排除線性模型儲存（僅作為 baseline 比較）
- 所有報告含訓練與測試集 MAE、MSE、RMSE 指標

---
