# 🚦 Urban Traffic Congestion Forecasting Using Deep Learning

## Overview

Traffic congestion is a major challenge in urban areas, leading to delays, increased fuel consumption, and reduced mobility. This project develops a **traffic congestion forecasting system** using **time series analysis** and **deep learning** to predict future traffic volume and congestion severity from historical traffic data.

The project explores traffic patterns across **4 junctions** using **48,120 hourly records**, engineers temporal and lag-based features, and compares **ARIMA**, **LSTM**, and **Bidirectional LSTM (BiLSTM)** models for traffic forecasting. Predicted traffic volumes are then mapped into congestion levels to make the results easier to interpret and apply in traffic management contexts.

---

## Project Objective

The goal of this project is to forecast urban traffic congestion in advance using historical traffic data, enabling more proactive traffic monitoring and decision-making.

---

## Dataset

* **Dataset size:** 48,120 hourly traffic records
* **Coverage:** 4 traffic junctions
* **City evaluated:** **Nairobi, Kenya** *

The dataset captures traffic volume observations over time, making it suitable for **time series forecasting** and congestion analysis.

---

## What the Notebook Covers

The notebook follows a complete traffic forecasting workflow:

### 1. Data Exploration

* Load and inspect traffic data
* Analyze traffic trends across junctions
* Visualize patterns using plots and heatmaps

### 2. Feature Engineering

A total of **16 features** were used, including:

* **Temporal features:** hour, day of week, month
* **Cyclical features:** sine/cosine encoding for time variables
* **Lag features:** previous traffic observations
* **Rolling statistics:** rolling mean and rolling standard deviation
* Additional traffic indicators such as peak-hour and weekend flags

### 3. Congestion Classification

Traffic volume predictions were translated into four congestion levels:

* **Free Flow**
* **Moderate**
* **Heavy**
* **Severe**

### 4. Forecasting Models

The project compares three forecasting approaches:

* **ARIMA** – baseline statistical model
* **LSTM** – deep learning model for sequential traffic forecasting
* **BiLSTM** – bidirectional sequence model for improved pattern learning

### 5. Model Training

The final deep learning setup includes:

* **Architecture:** 3-layer LSTM (**128 → 64 → 32**) + Dense layer
* **Training setup:** Huber loss, Adam optimizer, early stopping
* Hyperparameter tuning to improve forecasting performance

---

## Model Performance

For the evaluated test setup (**Junction 1**), the comparison showed:

| Model        |       MAE |      RMSE |         R² |  MAPE (%) |
| ------------ | --------: | --------: | ---------: | --------: |
| ARIMA(5,1,2) |     34.80 |     40.09 |    -3.0115 |         – |
| LSTM         | **18.38** | **23.54** | **0.3638** | **23.46** |
| BiLSTM       |     68.42 |     72.78 |    -7.5966 |     99.68 |

### Best Model

✅ **LSTM** achieved the best performance on the test set, with an **RMSE improvement of 41.3% over ARIMA**.

---

## Key Outcomes

* Traffic patterns vary significantly across junctions and time periods.
* Feature engineering improved the model’s ability to capture traffic behavior.
* Deep learning models performed better than the ARIMA baseline for forecasting.
* In the evaluated setup, **LSTM outperformed both ARIMA and BiLSTM**.
* Forecasted traffic values can be converted into congestion categories for easier interpretation and decision-making.

---

## Tech Stack

* **Python**
* **Pandas** & **NumPy**
* **Matplotlib** & **Seaborn**
* **Scikit-learn**
* **Statsmodels**
* **TensorFlow / Keras**

---

## Project Structure

```bash
urban-traffic-congestion-forecasting/
│
├── urban_traffic_prediction.ipynb
├── traffic.csv
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/urban-traffic-congestion-forecasting.git
cd urban-traffic-congestion-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook urban_traffic_prediction.ipynb
```

---

## Workflow Summary

```text
Traffic Data
   ↓
Data Exploration
   ↓
Feature Engineering
   ↓
Congestion Classification
   ↓
ARIMA Baseline
   ↓
LSTM / BiLSTM Training
   ↓
Model Evaluation
   ↓
Traffic Forecasting
   ↓
Congestion Prediction
```

---

## Limitations

* The project mainly relies on **historical traffic volume data**
* External factors such as **weather, accidents, road closures, and public events** were not included
* Real-time traffic data was not integrated in the current version

---

## Future Work

Potential extensions include:

* Integrating **real-time traffic data**
* Adding **weather and event-based features**
* Deploying the forecasting system as a live dashboard
* Exploring additional models such as **GRU** or **Transformers**

---

## Author

**Annex Ogeya**
Capstone Project — Urban Traffic Congestion Forecasting Using Deep Learning
