# -*- coding: utf-8 -*-
"""
Flight Fare Predictor — Streamlit App
Author: Harshitha Pethuraj
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pydeck as pdk
from datetime import date, time, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flight Fare Predictor | ML App",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# DARK MODE STATE
# ============================================================

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


def get_theme_css(dark: bool):

    if dark:
        return """
        <style>

        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            color: #f1f5f9;
        }

        h2, h3, h4, h5, p, label, div {
            color: #f1f5f9 !important;
        }

        .stMarkdown {
            color: #f1f5f9;
        }

        [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
        }

        .tip-card {
            background: #1e293b !important;
            color: #f1f5f9 !important;
            border-left: 5px solid #f59e0b;
        }

        </style>
        """

    return """
    <style>

    .stApp {
        background: linear-gradient(180deg, #e0f2fe 0%, #ffffff 50%);
    }

    .tip-card {
        background: white;
        color: #1e293b;
        border-left: 5px solid #f59e0b;
    }

    </style>
    """


# ============================================================
# APPLY THEME
# ============================================================

st.markdown(
    get_theme_css(st.session_state.dark_mode),
    unsafe_allow_html=True
)

# ============================================================
# MAIN CSS
# ============================================================

st.markdown("""
<style>

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.hero h1 {
    color: white !important;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}

.hero p {
    color: rgba(255,255,255,0.9) !important;
    margin-top: 0.5rem;
    font-size: 1rem;
}

.prediction-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 1.5rem 0;
}

.prediction-card h2 {
    font-size: 3rem !important;
    font-weight: 800;
    margin: 0;
    color: white !important;
}

.best-day-card {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 1.5rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 1rem 0;
}

.tip-card {
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    font-size: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #312e81 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.footer {
    text-align: center;
    padding: 2rem;
    font-size: 0.9rem;
    margin-top: 3rem;
}

@media (max-width: 768px) {

    .hero {
        padding: 1.2rem !important;
    }

    .hero h1 {
        font-size: 1.6rem !important;
    }

    .hero p {
        font-size: 0.9rem !important;
    }

    .prediction-card {
        padding: 1.5rem !important;
    }

    .prediction-card h2 {
        font-size: 2.2rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load('flight_fare_pipeline.pkl')


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Failed to load model: {e}")

# ============================================================
# CONSTANTS
# ============================================================

AIRLINES = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo',
    'Jet Airways', 'Jet Airways Business',
    'Multiple carriers', 'Multiple carriers Premium economy',
    'SpiceJet', 'Trujet', 'Vistara',
    'Vistara Premium economy'
]

SOURCES = [
    'Banglore', 'Chennai', 'Delhi',
    'Kolkata', 'Mumbai'
]

DESTINATIONS = [
    'Banglore', 'Cochin', 'Delhi',
    'Hyderabad', 'Kolkata', 'New Delhi'
]

STOP_OPTIONS = {
    'Non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}

CITY_COORDS = {
    'Banglore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Delhi': (28.7041, 77.1025),
    'Kolkata': (22.5726, 88.3639),
    'Mumbai': (19.0760, 72.8777),
    'Cochin': (9.9312, 76.2673),
    'Hyderabad': (17.3850, 78.4867),
    'New Delhi': (28.6139, 77.2090),
}

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_duration(dep_hour, dep_minute,
                       arr_hour, arr_minute,
                       next_day=False):

    dep_total = dep_hour * 60 + dep_minute
    arr_total = arr_hour * 60 + arr_minute

    if next_day or arr_total < dep_total:
        arr_total += 24 * 60

    return arr_total - dep_total


def build_input(
    airline,
    source,
    destination,
    stops,
    journey_dt,
    dep_t,
    arr_t,
    next_day
):

    duration = calculate_duration(
        dep_t.hour,
        dep_t.minute,
        arr_t.hour,
        arr_t.minute,
        next_day
    )

    return pd.DataFrame([{
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Total_Stops': stops,
        'Journey_Day': journey_dt.day,
        'Journey_Month': journey_dt.month,
        'Dep_Hour': dep_t.hour,
        'Dep_Minute': dep_t.minute,
        'Arrival_Hour': arr_t.hour,
        'Arrival_Minute': arr_t.minute,
        'Duration_Mins': duration
    }])


def predict_price(row_df):

    log_price = model.predict(row_df)[0]

    return float(np.expm1(log_price))


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <h1>✈️ Flight Fare Predictor</h1>
    <p>
        AI-powered flight price prediction system
        built using Machine Learning & XGBoost
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    dark_mode_toggle = st.toggle(
        "🌗 Dark Mode",
        value=st.session_state.dark_mode
    )

    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()

    st.markdown("---")

    st.markdown("## 📊 Model Metrics")

    st.markdown("""
- R² Score: 0.89
- RMSE: ₹1,800
- MAE: ₹1,113
- Algorithm: XGBoost
- Dataset: 10,683 flights
""")

    st.markdown("---")

    st.markdown("### 🔗 Links")

    st.markdown(
        "[GitHub Repo](https://github.com/HarshithaPethuraj/Flight-Fare-Prediction)"
    )

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/harshitha-pethuraj-13738129b/)"
    )

# ============================================================
# INPUT FORM
# ============================================================

st.markdown("### 🛫 Plan Your Journey")

airline = st.selectbox(
    "Choose Airline",
    AIRLINES,
    index=4
)

source = st.selectbox(
    "From",
    SOURCES,
    index=2
)

destination = st.selectbox(
    "To",
    DESTINATIONS,
    index=1
)

stops_label = st.selectbox(
    "Number of Stops",
    list(STOP_OPTIONS.keys()),
    index=1
)

journey_date = st.date_input(
    "Journey Date",
    value=date(2019, 5, 1)
)

dep_time = st.time_input(
    "Departure Time",
    value=time(9, 30)
)

arr_time = st.time_input(
    "Arrival Time",
    value=time(12, 30)
)

next_day = st.checkbox(
    "✈️ Arrives the next day"
)

if source == destination:
    st.warning(
        "⚠️ Source and destination should be different."
    )

predict_clicked = st.button(
    "🔮 Predict My Flight Fare",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if source == destination:

        st.error(
            "❌ Source and destination cannot be the same."
        )

    elif not model_loaded:

        st.error("❌ Model not loaded.")

    else:

        try:

            journey_dt = pd.to_datetime(journey_date)

            stops_val = STOP_OPTIONS[stops_label]

            input_data = build_input(
                airline,
                source,
                destination,
                stops_val,
                journey_dt,
                dep_time,
                arr_time,
                next_day
            )

            predicted_price = predict_price(input_data)

            st.markdown(f"""
            <div class="prediction-card">
                <p>🎯 Estimated Flight Fare</p>
                <h2>₹{predicted_price:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.success(
                f"Estimated Fare: ₹ {predicted_price:,.0f}"
            )

            # ====================================================
            # FLIGHT SUMMARY
            # ====================================================

            st.markdown("### ✈️ Flight Summary")

            s1, s2 = st.columns(2)

            with s1:
                st.metric(
                    "Route",
                    f"{source} → {destination}"
                )

            with s2:
                st.metric(
                    "Stops",
                    stops_label
                )

            # ====================================================
            # MAP
            # ====================================================

            st.markdown("### 🗺️ Route Map")

            src_lat, src_lon = CITY_COORDS[source]
            dst_lat, dst_lon = CITY_COORDS[destination]

            points_df = pd.DataFrame([
                {
                    'lat': src_lat,
                    'lon': src_lon
                },
                {
                    'lat': dst_lat,
                    'lon': dst_lon
                }
            ])

            st.pydeck_chart(
                pdk.Deck(
                    map_style='light',
                    initial_view_state=pdk.ViewState(
                        latitude=(src_lat + dst_lat) / 2,
                        longitude=(src_lon + dst_lon) / 2,
                        zoom=4
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=points_df,
                            get_position='[lon, lat]',
                            get_radius=50000,
                        )
                    ]
                )
            )

        except Exception as e:

            st.error(f"❌ Prediction failed: {e}")

# ============================================================
# INFO SECTION
# ============================================================

st.markdown("---")

st.markdown("### 🤖 About The Model")

st.markdown("""
This application predicts Indian domestic flight fares
using a tuned XGBoost Regression model trained on
historical airline booking data.
""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit, XGBoost & scikit-learn
</div>
""", unsafe_allow_html=True)
