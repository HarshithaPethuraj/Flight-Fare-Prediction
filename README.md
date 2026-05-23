# ✈️ Flight Fare Prediction — ML Web App

> Predict Indian domestic flight fares in real-time using a tuned XGBoost model, deployed as a fully interactive Streamlit application.

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Streamlit-667eea?style=for-the-badge)](https://flight-fare-prediction-g8qstmvtbwrbxnketl34jo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

---

##  Live Demo

** [Try the app here](https://flight-fare-prediction-g8qstmvtbwrbxnketl34jo.streamlit.app/)**

Open the app, fill in your route details, and get an instant fare prediction — no setup required.

---

## 📌 Project Overview

This end-to-end ML project predicts flight ticket prices for Indian domestic routes. It covers the full pipeline: data cleaning → feature engineering → model training → evaluation → deployment as a public web application.

**Dataset:** 10,683 flight bookings (March–June 2019), sourced from Kaggle's Indian domestic flight dataset.

**Target variable:** Flight fare in Indian Rupees (₹)

---

##  Features of the App

| Feature | Description |
|---|---|
|  Fare Prediction | Instant price estimate for any airline, route, and date |
|  Interactive Route Map | 3D arc map showing source → destination with distance & speed metrics |
|  30-Day Price Trend | Bar chart showing how fares change day by day over the next month |
|  Airline Comparison | Side-by-side fare comparison across all 12 airlines for the same route |
|  Smart Travel Tips | Personalized tips based on your chosen airline, timing, and price |
|  Dark Mode | Full dark/light mode toggle |
|  Mobile Responsive | Works on phones and tablets |

---

## 🧠 ML Pipeline

```
Raw CSV  →  Data Cleaning  →  Feature Engineering  →  Model Training  →  Evaluation  →  Deployment
```

### Feature Engineering
- **Time buckets** — departure/arrival hour mapped to Early Morning / Morning / Afternoon / Evening / Night
- **Journey duration** — computed in minutes from departure and arrival times (handles overnight flights)
- **Date features** — day of week, day of month, month extracted from journey date
- **Route distance** — Haversine formula used to calculate great-circle distance between city coordinates
- **Stop count** — encoded as integer (0 = non-stop, 1–4 = stops)

### Model
- **Algorithm:** XGBoost Regressor
- **Wrapper:** scikit-learn `Pipeline` (preprocessor + model)
- **Preprocessing:** `ColumnTransformer` with `OneHotEncoder` for categoricals and `StandardScaler` for numerics
- **Tuning:** GridSearchCV / hyperparameter optimization
- **Accuracy:** ~89% R² on the test set

---

##  Model Performance

| Metric | Score |
|---|---|
| R² (Test Set) | ~0.89 |
| Algorithm | XGBoost Regressor |
| Training samples | ~8,500 |
| Test samples | ~2,100 |
| Features used | 11 engineered features |

---

##  Supported Routes

**Sources:** Bangalore · Chennai · Delhi · Kolkata · Mumbai

**Destinations:** Bangalore · Cochin · Delhi · Hyderabad · Kolkata · New Delhi

**Airlines (12):** Air Asia · Air India · GoAir · IndiGo · Jet Airways · Jet Airways Business · Multiple Carriers · Multiple Carriers Premium Economy · SpiceJet · Trujet · Vistara · Vistara Premium Economy

---

##  Project Structure

```
Flight-Fare-Prediction/
│
├── Flight_Fare_Prediction.ipynb   # Full EDA, feature engineering, training notebook
├── app.py                         # Streamlit web application
├── flight_fare_pipeline.pkl       # Trained & serialized ML pipeline
├── requirements.txt               # Python dependencies
└── README.md
```

---

##  Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/HarshithaPethuraj/Flight-Fare-Prediction.git
cd Flight-Fare-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | XGBoost, scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, PyDeck |
| Web App | Streamlit |
| Serialization | Joblib |
| Deployment | Streamlit Cloud |

---

##  Disclaimer

This model was trained on Indian domestic flight data from **March–June 2019**. Predictions are estimates for learning purposes and are not suitable for actual fare comparison post-2019 or for international routes.

---

## Author

**Harshitha Pethuraj**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/harshitha-pethuraj-13738129b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/HarshithaPethuraj)

---

*Built with Streamlit · XGBoost · scikit-learn · Plotly*
