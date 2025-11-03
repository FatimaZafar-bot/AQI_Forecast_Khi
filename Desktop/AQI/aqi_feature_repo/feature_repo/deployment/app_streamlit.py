import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None

try:
    from lime import lime_tabular
except Exception:
    lime_tabular = None

try:
    import streamlit_shap
except Exception:
    streamlit_shap = None

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="aqi_refresh")

API_URL = "http://127.0.0.1:8000/api/v1/forecast/karachi"
FORECAST_FILE = "../model_train/forecast_next3days_all_models.csv"


@st.cache_data(ttl=600)
def fetch_forecast_data():
    """Fetches the AQI forecast data from FastAPI backend."""
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data['forecasts'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Karachi')

        for col in ['RandomForest', 'GradientBoosting', 'LightGBM', 'Ensemble_Prediction']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

        # sort by timestamp to ensure sequence
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['Date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df['Time'] = df['timestamp'].dt.strftime('%H:%M')

        # last_update comes from the API response (model run time)
        last_update = pd.to_datetime(data.get('last_update_time')).tz_convert('Asia/Karachi')

        return df, last_update
    except Exception as e:
        st.error(f"❌ Error fetching data from API: {e}")
        return None, None

AQI_CATEGORIES = {
    (0, 50): ("Good", "green", "#00e400"),
    (51, 100): ("Moderate", "yellow", "#ffff00"),
    (101, 150): ("Unhealthy for Sensitive Groups", "orange", "#ff7e00"),
    (151, 200): ("Unhealthy", "red", "#ff0000"),
    (201, 300): ("Very Unhealthy", "purple", "#8f3f97"),
    (301, 500): ("Hazardous", "maroon", "#7e0023"),
}

def get_aqi_category(aqi):
    for (low, high), (category, _, hex_color) in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return category, hex_color
    return "Extreme Hazard", "#4b0082"


# streamlit layout 
st.set_page_config(page_title="Karachi AQI Forecast Dashboard", layout="wide")


st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to bottom, #87CEEB 0%, #f0f8ff 100%);
        background-attachment: fixed;
        background-size: cover;
    }

    /* Add animated clouds (soft & subtle) */
    @keyframes moveclouds {
        0% {background-position: 0 0;}
        100% {background-position: 1000px 0;}
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: url('https://i.imgur.com/dT4gA2f.png'); /* transparent cloud pattern */
        background-repeat: repeat-x;
        background-size: 600px 300px;
        opacity: 0.25;
        animation: moveclouds 60s linear infinite;
        z-index: -1;
    }

    /* Optional: subtle sun glow in the corner */
    .stApp::after {
        content: "";
        position: fixed;
        top: 60px; right: 60px;
        width: 120px; height: 120px;
        background: radial-gradient(circle, rgba(255, 255, 224, 0.8), rgba(255, 255, 0, 0.2));
        border-radius: 50%;
        box-shadow: 0 0 60px 30px rgba(255, 255, 0, 0.2);
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  

st.sidebar.header("Appearance & Options")
st.session_state.dark_mode = st.sidebar.checkbox("🌗 Dark Mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    PRIMARY = "#00BFFF"
    BG_PRIMARY = "#0f2027"  
    BG_SECOND = "#203a43"
    BG_THIRD = "#2c5364"
    TEXT = "#FAFAFA"
    CARD_BG = "rgba(255,255,255,0.03)"
else:
    PRIMARY = "#0b66b2"
    BG_PRIMARY = "#f7fbff"
    BG_SECOND = "#eef6ff"
    BG_THIRD = "#e6f0fb"
    TEXT = "#0A0A0A"
    CARD_BG = "rgba(0,0,0,0.03)"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {BG_PRIMARY}, {BG_SECOND}, {BG_THIRD});
        color: {TEXT};
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }}

    /* Header */
    .main-title {{
        font-size: 2.6rem;
        color: {PRIMARY};
        text-align: center;
        margin-bottom: 6px;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(0,0,0,0.25);
    }}
    .subtitle {{
        text-align: center;
        font-size: 1rem;
        color: {'#cfd8dc' if st.session_state.dark_mode else '#42566b'};
        margin-bottom: 18px;
    }}

    /* Metric style */
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        color: {PRIMARY};
        font-weight: 700;
    }}

    /* Card / DataFrame container */
    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 6px;
        background-color: {CARD_BG};
        backdrop-filter: blur(6px);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {'#071018' if st.session_state.dark_mode else '#ffffff'} !important;
        color: {TEXT} !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }}

    /* Tabs styling */
    div[data-baseweb="tab-list"] {{
        justify-content: center;
        border-bottom: 2px solid rgba(255,255,255,0.03);
    }}
    div[data-baseweb="tab"] {{
        font-weight: 600;
        color: rgba(255,255,255,0.7) !important;
    }}
    div[data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY} !important;
        border-bottom: 3px solid {PRIMARY};
    }}

    /* Headings */
    h2, h3 {{
        color: {PRIMARY} !important;
    }}

    /* Small responsive tweaks */
    @media (max-width: 600px) {{
        .main-title {{ font-size: 1.8rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(f"<div class='main-title'>🏙️ Karachi AQI Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time 72-Hour Air Quality Prediction & Insights</div>", unsafe_allow_html=True)


st.sidebar.markdown("### 🧩 Feature Store")
simulate_features = st.sidebar.checkbox("Use simulated feature snapshot (sidebar)", value=True)

if simulate_features:
    feature_dict = {
        "Temperature (°C)": 33.5,
        "Humidity (%)": 60,
        "Wind Speed (km/h)": 8.2,
        "PM2.5 (µg/m³)": 45,
        "PM10 (µg/m³)": 78,
        "NO₂ (ppb)": 27
    }
else:
    try:
        feat_resp = requests.get(API_URL.replace("/forecast/karachi", "/features/karachi"), timeout=4)
        feat_resp.raise_for_status()
        feature_dict = feat_resp.json()
    except Exception:
        feature_dict = {"info": "Feature endpoint not available, keeping simulation."}

with st.sidebar.expander("🧾 Current Feature Values (Feature Store)"):
    st.json(feature_dict)

df_forecast, last_update = fetch_forecast_data()


MODEL_REGISTRY_DIR = os.path.join(
    r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\model_train\model_registry"
)
MODEL_FILES = {
    'RandomForest': os.path.join(MODEL_REGISTRY_DIR, 'RandomForest_model.pkl'),
    'GradientBoosting': os.path.join(MODEL_REGISTRY_DIR, 'GradientBoosting_model.pkl'),
    'LightGBM': os.path.join(MODEL_REGISTRY_DIR, 'LightGBM_model.pkl'),
}
TRAIN_TEST_FILE = os.path.join(MODEL_REGISTRY_DIR, 'train_test.pkl')  
SCALER_FILE = os.path.join(MODEL_REGISTRY_DIR, 'scaler.pkl')


def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load {os.path.basename(path)}: {e}")
        return None

# Cache loads for speed; avoid unhashable args in cache keys by not passing complex structures
@st.cache_data
def load_model_cached(path: str):
    return safe_joblib_load(path)

@st.cache_data
def load_train_test_cached(path: str):
    try:
        obj = joblib.load(path)
        return obj
    except Exception as e:
        return None

# main logic
if df_forecast is not None:
    st.sidebar.header("📊 Model Display Options")
    model_options = ['RandomForest', 'GradientBoosting', 'LightGBM', 'Ensemble_Prediction']
    selected_models = st.sidebar.multiselect(
        "Select models to display on the 72-hour trend chart:",
        options=model_options,
        default=model_options
    )

    #metric/current status
    st.markdown("---")
    if 'Ensemble_Prediction' in df_forecast.columns and not df_forecast['Ensemble_Prediction'].isna().all():
        next_hour_forecast = df_forecast.iloc[0]['Ensemble_Prediction']
    else:
        fallback_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM'] if c in df_forecast.columns]
        if fallback_cols:
            next_hour_forecast = df_forecast.iloc[0][fallback_cols[0]]
        else:
            next_hour_forecast = float('nan')

    try:
        category, color_hex = get_aqi_category(float(next_hour_forecast))
    except Exception:
        category, color_hex = "Unknown", "#888888"

    st.subheader("Current Forecast Status (Next Hour)")
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        try:
            st.metric("Ensemble AQI Forecast (Next Hour)", f"{float(next_hour_forecast):.2f}")
        except Exception:
            st.metric("Ensemble AQI Forecast (Next Hour)", "N/A")
    with col2:
        st.markdown(
            f"""
            <div style="background-color:{color_hex}; padding:12px; border-radius:10px; 
            text-align:center; color:{'white' if category in ['Very Unhealthy','Hazardous','Extreme Hazard'] else 'black'};
            font-weight:bold;">{category}</div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.info(f"Last Model Update/Forecast Run: **{last_update.strftime('%Y-%m-%d %H:%M %Z')}**")
        if os.path.exists(FORECAST_FILE):
            file_update = datetime.fromtimestamp(os.path.getmtime(FORECAST_FILE))
            st.caption(f"🕒 Last File Update Detected: **{file_update.strftime('%Y-%m-%d %H:%M:%S')}**")


    try:
        gauge_val = float(next_hour_forecast)
    except Exception:
        gauge_val = 0.0

    gauge_col1, gauge_col2 = st.columns([1,2])
    with gauge_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            title={'text': "Ensemble AQI (Next Hour)", 'font': {'size': 18, 'color': TEXT}},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': color_hex},
                'steps': [
                    {'range': [0, 50], 'color': "#00e400"},
                    {'range': [51, 100], 'color': "#ffff00"},
                    {'range': [101, 150], 'color': "#ff7e00"},
                    {'range': [151, 200], 'color': "#ff0000"},
                    {'range': [201, 300], 'color': "#8f3f97"},
                    {'range': [301, 500], 'color': "#7e0023"}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    
    if 'Ensemble_Prediction' in df_forecast.columns:
        try:
            if float(gauge_val) > 200:
                st.error("Hazardous AQI forecasted soon — limit outdoor exposure!")
            elif float(gauge_val) > 150:
                st.warning("AQI Unhealthy forecasted for sensitive groups.")
            elif float(gauge_val) > 100:
                st.info("Moderate AQI — sensitive people should be cautious.")
            else:
                st.success("AQI looks good for the next hour.")
        except Exception:
            pass

    
    st.subheader("Forecasted AQI Trend (72 Hours)")

    models_present = [m for m in selected_models if m in df_forecast.columns]
    if not models_present:
        st.warning("No selected model columns available in the data to plot.")
    else:
        df_plot = df_forecast.melt(
            id_vars=['timestamp'],
            value_vars=models_present,
            var_name='Model',
            value_name='AQI'
        )

        fig = px.line(
            df_plot,
            x='timestamp',
            y='AQI',
            color='Model',
            title='72-Hour Recursive AQI Forecast by Model',
            height=550,
        )

        line_dash_map = {
            'Ensemble_Prediction': 'solid',
            'RandomForest': 'dot',
            'GradientBoosting': 'dot',
            'LightGBM': 'dot',
        }
        line_width_map = {
            'Ensemble_Prediction': 3,
            'RandomForest': 1,
            'GradientBoosting': 1,
            'LightGBM': 1,
        }

        for trace in fig.data:
            name = trace.name
            trace.line.width = line_width_map.get(name, 1)
            trace.line.dash = line_dash_map.get(name, 'solid')

        for (low, high), (cat, _, color) in AQI_CATEGORIES.items():
            fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.1, line_width=0, layer='below',
                          annotation_text=cat, annotation_position='top left')

        fig.update_layout(xaxis_title="Time (Asia/Karachi)", yaxis_title="AQI Value",
                          template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
                          font=dict(color=TEXT))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📅 72-Hour AQI Forecast Table")

    if 'Ensemble_Prediction' in df_forecast.columns:
        df_table = df_forecast[['Date', 'Time', 'Ensemble_Prediction']].copy()
        df_table.rename(columns={'Ensemble_Prediction': 'AQI Forecast'}, inplace=True)
    else:
        fallback_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM'] if c in df_forecast.columns]
        if fallback_cols:
            df_table = df_forecast[['Date','Time',fallback_cols[0]]].copy()
            df_table.rename(columns={fallback_cols[0]: 'AQI Forecast'}, inplace=True)
        else:
            df_table = pd.DataFrame(columns=['Date','Time','AQI Forecast'])

    df_table['AQI Forecast'] = pd.to_numeric(df_table['AQI Forecast'], errors='coerce').round(2)
    st.dataframe(df_table, use_container_width=True)

    csv_data = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast Data (CSV)",
        data=csv_data,
        file_name=f"karachi_aqi_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Display Model Results
    st.markdown("### 🧪 Model Performance Results (RMSE, MAE, R²)")
    MODEL_RESULTS_FILE = r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\model_train\model_registry\model_results.csv"
    if os.path.exists(MODEL_RESULTS_FILE):
        df_model_results = pd.read_csv(MODEL_RESULTS_FILE)
        df_model_results[['RMSE','MAE','R2']] = df_model_results[['RMSE','MAE','R2']].round(3)
        st.dataframe(df_model_results, use_container_width=True)
    else:
        st.warning(f"Model results file not found at: {MODEL_RESULTS_FILE}")
        
    st.markdown("---")
    st.subheader("Advanced Analysis and Forecast Data")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Raw Forecast Data", "Model Comparison", "Hazard Alerts",
        "EDA", "Explainability", "Alerts", "Model Performance"
    ])


    with tab1:
        st.write("**Full 72-Hour Forecast Data (raw)**")
        # Display raw forecast without Date/Time columns if you prefer (you previously wanted to hide Date & Time in raw)
        raw_display = df_forecast.drop(columns=['Date', 'Time'], errors='ignore')
        st.dataframe(raw_display.set_index('timestamp'), use_container_width=True)


    with tab2:
        st.write("**Model-wise 72-Hour AQI Comparison**")
        model_comp = df_forecast[['Date','Time','RandomForest','GradientBoosting','LightGBM','Ensemble_Prediction']].copy()
        for c in ['RandomForest','GradientBoosting','LightGBM','Ensemble_Prediction']:
            if c in model_comp.columns:
                model_comp[c] = model_comp[c].round(2)
        st.dataframe(model_comp, use_container_width=True)

    with tab3:
        df_hazard = df_forecast[df_forecast['Ensemble_Prediction'] > 200] if 'Ensemble_Prediction' in df_forecast.columns else pd.DataFrame()
        if not df_hazard.empty:
            st.warning("🚨 **Hazardous/Very Unhealthy AQI Alert!**")
            st.dataframe(df_hazard[['Date','Time','Ensemble_Prediction']].rename(columns={'Ensemble_Prediction':'Forecasted AQI'}), hide_index=True)
            st.write("Recommendation: Limit outdoor exposure during these hours.")
        else:
            st.success("No 'Very Unhealthy' or 'Hazardous' AQI levels forecasted in the next 72 hours.")

    with tab4:
        st.subheader("Exploratory Data Analysis & Trends")
        st.markdown("**1) AQI Distribution (all available model forecasts)**")
        aqi_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM','Ensemble_Prediction'] if c in df_forecast.columns]
        if aqi_cols:

            df_melt = df_forecast.melt(id_vars=['timestamp'], value_vars=aqi_cols, var_name='Model', value_name='AQI')
            fig_hist = px.histogram(df_melt, x='AQI', color='Model', barmode='overlay', nbins=40,
                                    title="AQI Distribution across Models (next 72h)")
            fig_hist.update_layout(bargap=0.1, template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No model columns present for distribution plot.")

        st.markdown("**2) Correlation between model forecasts**")
        if len(aqi_cols) >= 2:
            corr_df = df_forecast[aqi_cols].corr()
            fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", title="Model Forecast Correlation")
            fig_corr.update_layout(template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 model columns to compute correlation.")

        st.markdown("**3) Average AQI by Hour of Day (trend)**")
        if 'Time' in df_forecast.columns and aqi_cols:
            try:
                df_forecast['Hour'] = pd.to_datetime(df_forecast['Time'], format='%H:%M').dt.hour
                hourly_avg = df_forecast.groupby('Hour')[aqi_cols].mean().reset_index()
                fig_hour = px.line(hourly_avg, x='Hour', y=aqi_cols, title='Average AQI by Hour (next 72h)')
                fig_hour.update_layout(template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
                st.plotly_chart(fig_hour, use_container_width=True)
            except Exception:
                st.info("Could not compute hourly aggregates (check Time format).")
        else:
            st.info("Time column or model columns missing for hourly trend.")

        st.markdown("---")
        st.markdown("**EDA notes:**\n\n- These charts are generated from the forecasts fetched from your FastAPI endpoint. \n- You can later replace these with additional analyses (seasonality decomposition, rolling stats, distribution per pollutant).")


    with tab5:
        st.subheader("Model Explainability (SHAP / LIME)")

        if shap is None:
            st.warning("`shap` not installed. Run `pip install shap` to enable SHAP explainability.")
        if lime_tabular is None:
            st.warning("`lime` not installed. Run `pip install lime` to enable LIME explainability.")
        if streamlit_shap is None:
            st.info("`streamlit_shap` not installed. SHAP visualizations will use matplotlib fallback. (Optional: pip install streamlit-shap)")

        available_models = [m for m, p in MODEL_FILES.items() if os.path.exists(p)]
        if not available_models:
            st.info("No model files found in model registry. Place RandomForest_model.pkl, GradientBoosting_model.pkl, or LightGBM_model.pkl in the model_registry folder.")
        else:
            chosen_model = st.selectbox("Select model for SHAP/LIME explanations:", options=available_models)

            model_path = MODEL_FILES.get(chosen_model)
            model_obj = load_model_cached(model_path)

            
            train_test_obj = None
            if os.path.exists(TRAIN_TEST_FILE):
                train_test_obj = load_train_test_cached(TRAIN_TEST_FILE)

            X_background = None
            feature_names = None
            if isinstance(train_test_obj, dict):
                
                X_background = train_test_obj.get('X_train') or train_test_obj.get('X_background') or train_test_obj.get('X')
                if X_background is None:
                    # try tuple fallback
                    for k in ['X_train','X_test','X','train','test']:
                        if k in train_test_obj:
                            X_background = train_test_obj[k]
                            break
            elif isinstance(train_test_obj, (list, tuple)):
                # assume (X_train, X_test, y_train, y_test)
                if len(train_test_obj) >= 1:
                    X_background = train_test_obj[0]
            else:
                X_background = train_test_obj

            if X_background is None:
                results_pkl = os.path.join(MODEL_REGISTRY_DIR, 'results.pkl')
                if os.path.exists(results_pkl):
                    try:
                        r = joblib.load(results_pkl)
                        if isinstance(r, dict):
                            X_background = r.get('X_train') or r.get('X') or X_background
                    except Exception:
                        pass

            if isinstance(X_background, pd.DataFrame):
                feature_names = list(X_background.columns)
                X_background_np = X_background.values
            elif isinstance(X_background, (np.ndarray, list)):
                X_background_np = np.array(X_background)
                feature_names = [f"f{i}" for i in range(X_background_np.shape[1])] if X_background_np.ndim == 2 else None
            else:
                X_background_np = None

            max_index = 0
            if X_background_np is not None and X_background_np.shape[0] > 0:
                max_index = min(499, X_background_np.shape[0] - 1)  
                sample_index = st.number_input("Select sample index to explain (from background):", min_value=0, max_value=int(max_index), value=0, step=1)
            else:
                fallback_len = len(df_forecast)
                sample_index = st.number_input("Select forecast row index to explain (fallback):", min_value=0, max_value=max(0, fallback_len-1), value=0, step=1)

            st.markdown("**Run explainability:**")
            col_shap, col_lime = st.columns([1,1])
            with col_shap:
                run_shap = st.button("Run SHAP")
            with col_lime:
                run_lime = st.button("Run LIME")

            def model_predict_proba_wrapper(X):
                try:
                    if hasattr(model_obj, "predict_proba"):
                        return model_obj.predict_proba(X)
                    else:
                        preds = model_obj.predict(X)
                        return preds.reshape(-1, 1)
                except Exception as e:
                    st.warning(f"Model predict/proba error: {e}")
                    return np.zeros((X.shape[0], 1))

            def model_predict_wrapper(X):
                try:
                    return model_obj.predict(X)
                except Exception as e:
                    st.warning(f"Model predict error: {e}")
                    return np.zeros(X.shape[0])

            # run Shap
            if run_shap:
                if shap is None:
                    st.error("SHAP package not available. Install with `pip install shap`.")
                elif model_obj is None:
                    st.error("Selected model could not be loaded.")
                else:
                    st.info("Computing SHAP values (may take a moment)...")
                    try:
                        
                        if X_background_np is not None:
                            background = X_background_np[np.linspace(0, X_background_np.shape[0]-1, min(100, X_background_np.shape[0]), dtype=int)]
                        else:
                            if 'Ensemble_Prediction' in df_forecast.columns:
                                proxy_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM'] if c in df_forecast.columns]
                                if proxy_cols:
                                    proxy_df = df_forecast[proxy_cols].dropna().head(100)
                                    background = proxy_df.values
                                else:
                                    background = None
                            else:
                                background = None

                        
                        explainer = None
                        try:
                            explainer = shap.Explainer(model_obj, background) if background is not None else shap.Explainer(model_obj)
                        except Exception:
                            try:
                                explainer = shap.TreeExplainer(model_obj, data=background)
                            except Exception:
                                explainer = None

                        if explainer is None:
                            st.error("Could not create SHAP explainer for this model.")
                        else:
                            if X_background_np is not None:
                                X_sample = X_background_np[int(sample_index)].reshape(1, -1)
                                X_for_pred = X_sample
                            else:
                                proxy_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM'] if c in df_forecast.columns]
                                if proxy_cols:
                                    X_sample = df_forecast.loc[int(sample_index), proxy_cols].values.reshape(1, -1)
                                    X_for_pred = X_sample
                                else:
                                    st.error("No suitable sample available for SHAP explanation (no train/test and no proxy features).")
                                    X_for_pred = None

                            if X_for_pred is not None:
                                shap_values = explainer(X_for_pred)
                                st.markdown("**SHAP: Global feature importance (bar)**")
                                try:
                                    fig = shap.plots.bar(shap_values, show=False)
                                    if streamlit_shap is not None:
                                        streamlit_shap.shap_plot(shap_values)
                                    else:
                                        plt.tight_layout()
                                        st.pyplot(bbox_inches="tight")
                                except Exception:
                                    try:
                                        shap.summary_plot(shap_values.values if hasattr(shap_values, 'values') else shap_values, features=X_for_pred, feature_names=feature_names)
                                        st.pyplot(bbox_inches="tight")
                                    except Exception as e:
                                        st.warning(f"Could not render SHAP global plot: {e}")

                             
                                st.markdown("**SHAP: Local explanation for selected sample**")
                                try:
                                    if streamlit_shap is not None:
                                        streamlit_shap.shap_interactive_plot(shap_values)
                                    else:
                                        try:
                                            shap.plots.waterfall(shap_values[0])
                                            st.pyplot(bbox_inches="tight")
                                        except Exception:
                                            vals = shap_values.values if hasattr(shap_values, 'values') else np.array(shap_values)
                                            feat_vals = X_for_pred.flatten()
                                            df_local = pd.DataFrame({
                                                'feature': feature_names if feature_names is not None else [f'f{i}' for i in range(len(feat_vals))],
                                                'value': feat_vals,
                                                'shap_value': vals.flatten()[:len(feat_vals)]
                                            })
                                            st.table(df_local)
                                except Exception as e:
                                    st.warning(f"Could not render SHAP local plot: {e}")

                    except Exception as e:
                        st.error(f"SHAP processing failed: {e}")

            # run lime
            if run_lime:
                if lime_tabular is None:
                    st.error("LIME not installed. Run `pip install lime` to enable LIME explanations.")
                elif model_obj is None:
                    st.error("Selected model could not be loaded.")
                else:
                    st.info("Running LIME explanation (may take a moment)...")
                    try:
                        if X_background_np is not None:
                            X_train_for_lime = X_background_np
                            fnames = feature_names if feature_names is not None else [f"f{i}" for i in range(X_background_np.shape[1])]
                        else:
                            proxy_cols = [c for c in ['RandomForest','GradientBoosting','LightGBM'] if c in df_forecast.columns]
                            if proxy_cols:
                                proxy_df = df_forecast[proxy_cols].dropna()
                                X_train_for_lime = proxy_df.values
                                fnames = proxy_cols
                            else:
                                st.error("No suitable training/background data found for LIME. Place train_test.pkl with X_train in the model registry.")
                                X_train_for_lime = None
                                fnames = None

                        if X_train_for_lime is not None:
                            explainer_lime = lime_tabular.LimeTabularExplainer(
                                training_data=np.array(X_train_for_lime),
                                feature_names=fnames,
                                mode='regression' if not hasattr(model_obj, "predict_proba") else 'classification'
                            )
                            
                            instance = X_train_for_lime[int(sample_index)].reshape(-1)
                            if hasattr(model_obj, "predict_proba"):
                                predict_fn = lambda x: model_predict_proba_wrapper(x)
                            else:
                                predict_fn = lambda x: model_predict_wrapper(x).reshape(-1, 1)

                            exp = explainer_lime.explain_instance(instance, predict_fn, num_features=min(10, len(fnames)))

                            
                            html = exp.as_html()
                            components.html(html, height=600, scrolling=True)
                        else:
                            st.error("LIME explanation could not be constructed due to missing background data.")
                    except Exception as e:
                        st.error(f"LIME failed: {e}")

    with tab6:
        st.subheader("Alerts & Notifications")
       
        if 'Ensemble_Prediction' in df_forecast.columns:
            total_hazard = int((df_forecast['Ensemble_Prediction'] > 200).sum())
            total_unhealthy = int(((df_forecast['Ensemble_Prediction'] > 150) & (df_forecast['Ensemble_Prediction'] <= 200)).sum())
            st.markdown(f"**Hazardous (AQI>200) timestamps in next 72h:** {total_hazard}")
            st.markdown(f"**Unhealthy (151-200) timestamps in next 72h:** {total_unhealthy}")
            if total_hazard > 0:
                hazardous_df = df_forecast[df_forecast['Ensemble_Prediction'] > 200][['Date','Time','Ensemble_Prediction']].rename(columns={'Ensemble_Prediction':'Forecasted AQI'})
                csv_hazard = hazardous_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Hazardous Timestamps (CSV)", data=csv_hazard, file_name=f"hazardous_timestamps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            else:
                st.info("No hazardous timestamps to export.")
        else:
            st.info("Ensemble predictions not available for alert summary.")

    with tab7:
        st.markdown("### Model Performance Results (RMSE, MAE, R²)")
        MODEL_RESULTS_FILE = r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\model_train\model_registry\model_results.csv"
        if os.path.exists(MODEL_RESULTS_FILE):
            df_model_results = pd.read_csv(MODEL_RESULTS_FILE)
            
            df_model_results[['RMSE','MAE','R2']] = df_model_results[['RMSE','MAE','R2']].round(3)
            
            try:
                df_model_results['RMSE_rank'] = df_model_results['RMSE'].rank(method='min')
                df_model_results = df_model_results.sort_values('RMSE')
            except Exception:
                pass
            st.dataframe(df_model_results.reset_index(drop=True), use_container_width=True)
        else:
            st.warning(f"Model results file not found at: {MODEL_RESULTS_FILE}")


    if 'Ensemble_Prediction' in df_forecast.columns:
        with tab5:
            st.markdown("---")
            st.subheader("72-Hour Ensemble Forecast Heatmap")
            heat_df = df_forecast[['timestamp','Date','Time','Ensemble_Prediction']].copy()
            heat_df['Hour'] = pd.to_datetime(heat_df['Time'], format='%H:%M').dt.hour
            pivot = heat_df.pivot_table(index='Date', columns='Hour', values='Ensemble_Prediction', aggfunc='mean')
            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Hour (24h)", y="Date", color="AQI"),
                aspect="auto",
                title="Ensemble Prediction Heatmap (Date x Hour)"
            )
            fig_heat.update_layout(template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
            st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.error("⚠️ Unable to load forecast data. Please check that FastAPI is running and the CSV is available.")

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center; color:{"#B0BEC5" if st.session_state.dark_mode else "#333333"};'>
    Built with ❤️ using <b>Streamlit</b> & <b>FastAPI</b> | Hybrid City Theme | {datetime.now().year}
    </div>
    """,
    unsafe_allow_html=True
)
