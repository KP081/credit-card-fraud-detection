###########################################################
#   |   |   |   |   Import Libraries
###########################################################

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


###########################################################
#   |   |   |   |   Confifuration
###########################################################

DEFAULT_API_URL = "https://fraud-detection-api-7fap.onrender.com"

st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon = "🕵️",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


###########################################################
#   |   |   |   |   Custom CSS
###########################################################

st.markdown(
    """
    <style>
        /* Main Header */
        .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #a8b2c3;
        margin: 0.5rem 0 0 0;
    }
    /* Result cards */
    .fraud-card {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .safe-card {
        background: linear-gradient(135deg, #00cc8822, #00cc8844);
        border: 2px solid #00cc88;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    /* Metric styling */
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html = True
)


###########################################################
#   |   |   |   |   Helper Functions
###########################################################

# Check FastAPI server on or not
def check_api_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"Status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "can't reach API server"}
    except requests.exceptions.Timeout:
        return False, {"error": "Request timeout : Server can't respond"}
    except Exception as e:
        return False, {"error": str(e)}
    

# prediction for single transaction
def predict_transaction(api_url, transaction_data):
    try:
        response = requests.post(
            f"{api_url}/predict",
            json = transaction_data,
            timeout = 10
        ) 
        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 422:
            return False, {"error": "Validation Error : Input Format Wrong"}
        else:
            return False, {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}
    
    
# predict for batch transaction
def predict_batch(api_url, transactions_list):
    try:
        payload = {"transactions": transactions_list}
        response = requests.post(
            f"{api_url}/batch",
            json = payload,
            timeout = 30
        )
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"API Error: {response.status_code}"}
    except Exception as e: 
        return False, {"error": str(e)}
    
    
# return color based on risk-level
def get_risk_color(risk_level):
    colors = {
        "CRITICAL" : "#ff0000",
        "HIGH"     : "#ff6b35",
        "MEDIUM"   : "#ffd700",
        "LOW"      : "#90ee90",
        "SAFE"     : "#00cc88"
    }
    
    return colors.get(risk_level, "#808080")


# return emoji based on risk-level
def get_risk_emoji(risk_level):
    emojis = {
        "CRITICAL" : "🔴",
        "HIGH"     : "🟠",
        "MEDIUM"   : "🟡",
        "LOW"      : "🟢",
        "SAFE"     : "✅"
    }
    return emojis.get(risk_level, "⚪")


###########################################################
#   |   |   |   |   Sidebar
###########################################################

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fraud.png", width=80)
    st.title("🔍 Fraud Detector")
    st.markdown("---")
    
    # API Configuration
    st.subheader("API Settings")
    api_url = st.text_input(
        "FastAPI URL",
        value = DEFAULT_API_URL,
        placeholder = "https://your-ngrok-url.ngrok.io",
        help = "Paste Ngrok URL from NB07"
    )
    
    # API Status
    if st.button("Check Connection"):
        with st.spinner("Checking..."):
            is_healthy, health_data = check_api_health(api_url)
        
        if is_healthy:
            st.success("✅ API Connected!")
            st.json(health_data)
        else:
            st.error(f"❌ {health_data.get('error', 'Connection failed')}")
            
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.selectbox(
        "Select Page",
        [
            "Home",
            "Single Transaction",
            "Batch Analysis",
            "Dashboard"
        ]
    )
    st.markdown("---")
    
    st.caption("Built with using Streamlit + FastAPI")
    

###########################################################
#   |   |   |   |   Page : Home
###########################################################

if page == "Home":
    
    # Header
    st.markdown(
    """
    <div class="main-header">
        <h1>Credit Card Fraud Detection</h1>
        <p>Real-time fraud detection powered by XGBoost + FastAPI</p>
    </div>
    """,
    unsafe_allow_html = True
    )
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", "XGBoost", "Optuna Tuned")
    with col2:
        st.metric("Dataset", "284,807", "transactions")
    with col3:
        st.metric("Fraud Cases", "492", "0.17% rate")
    with col4:
        st.metric("Target", "Recall", "90+ fraud catch")

    st.markdown("---")
    
    # Project Info
    coll1 , coll2 = st.columns(2)
    
    with coll1:
        st.subheader("Tech Stack")
        st.markdown(
        """
        --------------------------------------
        | Layer     |     Technology         |
        |-----------|------------------------|
        | ML Model  | XGBoost (Optuna Tuned) |
        | Backend   | FastAPI + Pydantic     |
        | Frontend  | Streamlit              |
        | Serving   | Uvicorn + Ngrok        |
        | Storage   | Google Drive           |
        --------------------------------------
        """
        )
        
    with coll2:
        st.subheader("Features")
        st.markdown(
        """
        - ✅ **Real-time** single transaction check
        - ✅ **Batch processing** via CSV upload
        - ✅ **Risk levels**: SAFE -> CRITICAL
        - ✅ **Confidence scores** for each prediction
        - ✅ **Analytics dashboard** with charts
        - ✅ **REST API** for integration
        """
        )
        
        st.markdown("---")
        
        
###########################################################
#   |   |   |   |   Page : Single Transaction
###########################################################

elif page == "Single Transaction":
    st.title("Single Transaction Check")
    st.markdown("First fill Transaction details, than check is fraud or not!")
    st.markdown("---")
    
    # Input Form
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        # Basic Info
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value = 0.01,
                max_value = 50000.0,
                value = 149.62,
                step = 0.01,
                help = "Put Transaction Amount in USD"
            )
        
        with col2:
            time = st.number_input(
                "Time (seconds since first transaction)",
                min_value = 0.0,
                max_value = 172792.0,
                value = 0.0,
                step = 1799.0
            )
        
        with col3:
            time_hour = st.number_input(
                "Time (hours since first transaction)",
                min_value = 0.0,
                max_value = 48.0,
                value = 0.0,
                step = 0.5
            )
            
        # PCA Features
        st.markdown("---")
        st.subheader("PCA Features (V1 - V28)")
        st.caption("These features are automatically generated from the bank’s internal data.")
        
        # Use expender to keep it clean
        with st.expander("Enter PCA Features (Advanced)", expanded = False):
            # Default values - from actual fraud transaction
            default_vals = [
                -1.3598, -0.0728,  2.5363,  1.3782, -0.3383,
                 0.4624,  0.2396,  0.0987,  0.3638,  0.0908,
                -0.5516, -0.6178, -0.9914, -0.3112,  1.4682,
                -0.4704,  0.2080,  0.0258,  0.4040,  0.2514,
                -0.0183,  0.2778, -0.1105,  0.0669,  0.1285,
                -0.1891,  0.1336, -0.0211
            ]
            
            v_values = []
            
            # show 28 features in 4 columns
            for row in range(7):
                cols = st.columns(4)
                
                for col_idx in range(4):
                    feature_idx = row * 4 + col_idx
                    
                    with cols[col_idx]:
                        val = st.number_input(
                            f"V{feature_idx + 1}",
                            value = float(default_vals[feature_idx]),
                            format = "%.4f",
                            label_visibility = "visible"
                        )
                        v_values.append(val)
                        
            # Submit button
            submitted = st.form_submit_button(
                "Ckeck for fraud",
                use_container_width = True,
                type = "primary"
            )
            
    # Predict when form submit
    if submitted:
        
        # Prepare Transaction data
        transaction_data = {
            f"V{i+1}": v_values[i] for i in range(28)
        }
        
        transaction_data["Amount"] = amount
        transaction_data["Time"] = time
        transaction_data["Time_Hour"] = time_hour
        
        # API call with loading spinner
        with st.spinner("Analyzing transaction..."):
            success, result = predict_transaction(api_url, transaction_data)
            
        if success:
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Main Result Display
            is_fraud = result.get("is_fraud", False)
            risk_level = result.get("risk_level", "UNKNOWN")
            confidence = result.get("confidence", 0)
            emoji = get_risk_emoji(risk_level)
            
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-card">
                    <h2>🚨 FRAUD DETECTED! 🚨</h2>
                    <h3>{emoji} Risk Level: {risk_level}</h3>
                    <p>Confidence: {result.get('confidence_pct', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    <h2>✅ TRANSACTION SAFE</h2>
                    <h3>{emoji} Risk Level: {risk_level}</h3>
                    <p>Confidence: {result.get('confidence_pct', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fraud?", "YES" if is_fraud else "NO")
            with col2:
                st.metric("Confidence", result.get("confidence_pct"))
            with col3:
                st.metric("Risk Level", risk_level)
            with col4:
                st.metric("Response Time", result.get("processing_time"))
                
            # confidence gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                title = {"text": "Fraud Probability (%)"},
                delta = {"reference": 50},
                
                gauge = {
                    "axis": {"range": [0, 100]},
                    "steps" : [
                        {"range": [0, 30],  "color": "#00cc88"},  # Safe
                        {"range": [30, 50], "color": "#90ee90"},  # Low
                        {"range": [50, 70], "color": "#ffd700"},  # Medium
                        {"range": [70, 90], "color": "#ff6b35"},  # High
                        {"range": [90, 100],"color": "#ff0000"},  # Critical
                    ],
                    "threshold": {
                        "line" : {"color": "white", "width": 4},
                        "value": result.get("threshold_used", 0.5) * 100
                    }
                }
            ))
            fig.update_layout(height = 300)
            st.plotly_chart(fig, use_container_width = True)
            
            # Action Recommendation
            st.info(f"**Recommended Action:** {result.get('recommendation')}")
            
            # Row Response Expender
            with st.expander("Raw API Response (Developer View!)"):
                st.json(result)
                
            # Save to session state history
            if "history" not in st.session_state:
                st.session_state.history = []
                
            st.session_state.history.append({
                "time"      : datetime.now().strftime('%H:%M:%S'),
                "amount"    : f"${amount:.2f}",
                "is_fraud"  : is_fraud,
                "risk"      : risk_level,
                "confidence": f"{confidence*100:.1f}%"
            })
            
        else:
            st.error(f"❌ API Error: {result.get('error', 'Unknown error')}")
            st.info("Check the API URL in the sidebar and click on **“Check Connection.”**")
        
    # Recent history
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.subheader("Rcent Checks")
        
        history_df = pd.DataFrame(st.session_state.history[::-1])
        
        st.dataframe(history_df, use_container_width = True)
        
        
###########################################################
#   |   |   |   |   Page : Batch Analysis
###########################################################

elif page == "Batch Analysis":
    
    st.title("Batch Transaction Analysis")
    st.markdown("Upload a CSV file - analyze all transactions at once!")
    st.markdown("---")
    
    # Instructions
    with st.expander("CSV Format Instructions", expanded = True):
        st.markdown(
            """
            **Required columns:**
            ```
            Time, V1, V2, V3, ..., V28, Amount, Time_Hour
            ```
            **Rules:**
            - Amount: must greter than 0
            - Time: between 0.0 to 172792.0
            - Time_Hour: between 0.0 to 48.0
            - Max 100 transactions per batch
            - CSV format (comma separated)
            """
        )
        
        # Sample CSV download
        sample_data = {
            "Time": [0.0, 12685.0],
            **{f"V{i}": [0.0, 0.0] for i in range(1, 29)},
            "Amount": [149.62, 2.69],
            "Time_Hour": [0.0, 0.5]
        }
        
        sample_df = pd.DataFrame(sample_data)
        # csv_buffer = io.StringIO()
        
        csv_data = sample_df.to_csv(index = False)
        
        # csv_buffer.seek(0)
        
        st.download_button(
            label = "Download Sample CSV",
            data = csv_data,
            file_name = "sample_transactions.csv",
            mime = "text/csv"
        )
        
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type = ["csv"],
        help = "Max 100 transactions"
    )
    
    if uploaded_file is not None:
        
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} transactions")
            st.dataframe(df.head(5), use_container_width=True)
            
            # validate columns
            required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Time_Hour"]
            missing_cols  = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                if len(df) > 100:
                    st.warning("If there are more than 100 rows, only the first 100 will be processed.")
                    df = df.head(100)
                    
                if st.button("Start Batch Analysis", type="primary", use_container_width=True):
                    
                    transactions_list = df[required_cols].to_dict(orient="records")
                    
                    with st.spinner(f"{len(df)} Transactions are being analyzed..."):
                        success, result = predict_batch(api_url, transactions_list)
                        
                    if success:
                        # save Results in session_state
                        st.session_state.batch_results = result
                        st.session_state.batch_df = df
                    else:
                        st.error(f"❌ {result.get('error')}")

                # session_state for show results 
                if "batch_results" in st.session_state and st.session_state.batch_results:
                    result = st.session_state.batch_results
                    df = st.session_state.batch_df

                    st.markdown("---")
                    st.subheader("Batch Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", result.get("total_transactions", 0))
                    with col2:
                        st.metric("Frauds", result.get("fraud_count", 0), delta=None)
                    with col3:
                        st.metric("Fraud Rate", result.get("fraud_rate", "0%"))
                    with col4:
                        st.metric("Time", result.get("processing_time", "N/A"))

                    # Per Transaction Result
                    preds = result.get("predictions", [])

                    if preds:
                        result_df = pd.DataFrame(preds)
                        result_df["Amount"] = df["Amount"].values[:len(preds)]
                        result_df["Status"] = result_df["is_fraud"].map(
                            {True: "🔴 FRAUD", False: "🟢 SAFE"}
                        )

                        # Filter Option
                        filter_opt = st.radio(
                            "Filter: ",
                            ["All", "Frauds Only", "Safe Only"],
                            horizontal=True
                        )

                        if filter_opt == "Frauds Only":
                            display_df = result_df[result_df["is_fraud"] == True]
                        elif filter_opt == "Safe Only":
                            display_df = result_df[result_df["is_fraud"] == False]
                        else:
                            display_df = result_df

                        st.dataframe(display_df, use_container_width=True)

                        # Distribution chart
                        fig = px.histogram(
                            result_df,
                            x      = "confidence",
                            color  = "Status",
                            nbins  = 20,
                            title  = "Fraud Probability Distribution",
                            labels = {"confidence": "Fraud Probability"},
                            color_discrete_map = {
                                "🔴 FRAUD": "#ff4b4b",
                                "🟢 SAFE" : "#00cc88"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Download results
                        csv_output = result_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            data      = csv_output,
                            file_name = "fraud_detection_results.csv",
                            mime      = "text/csv"
                        )
                    else:
                        st.error(f"❌ {result.get('error')}")
                        
        except Exception as e:
            st.error(f"❌ File read error: {str(e)}")
            
            
###########################################################
#   |   |   |   |   Page : Dashboard
###########################################################

elif page == "Dashboard":
    
    st.title("Analytics Dashboard")
    st.markdown("Model performance and system stats")
    st.markdown("---")
    
    # Fetch model info from API
    try:
        resp = requests.get(f"{api_url}/model-info", timeout=5)
        
        if resp.status_code == 200:
            model_info = resp.json()
            api_online = True
        else:
            model_info = {}
            api_online = False
    except:
        model_info = {}
        api_online = False
        
    if not api_online:
        st.warning("API is offline - some sections will show placeholder data.")
    
    # Model Info Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        if model_info:
            st.json(model_info)
        else:
            st.info("Connect to the API to get model information.")
            
    with col2:
        st.subheader("Model Performance")
        
        perf_data = {
            "Metric"    : ["Recall", "Precision", "F1-Score", "ROC-AUC"],
            "Score"     : [0.87, 0.66, 0.75, 0.98],  # <- put NB05 values!
            "Target"    : [0.90, 0.50, 0.70, 0.95]
        }
        
        perf_df = pd.DataFrame(perf_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name = "Achived",
            x = perf_df["Metric"],
            y = perf_df["Score"],
            marker_color = "#00cc88"
        ))
        
        fig.add_trace(go.Bar(
            name = "Target",
            x = perf_df["Metric"],
            y = perf_df["Target"],
            marker_color = "#ff6b35"
        ))
        
        fig.update_layout(
            title = "Model Performance vs Target",
            yaxis = dict(range=[0, 1]),
            barmode = "group"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    # Dataset statistics
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Pie chart - Class distribution
        fig = px.pie(
            values = [284315, 492],
            names  = ["Normal", "Fraud"],
            title  = "Dataset Class Distribution",
            color_discrete_sequence = ["#00cc88", "#ff4b4b"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Session predictions history
        if "history" in st.session_state and st.session_state.history:
            hist = st.session_state.history
            fraud_count = sum(1 for h in hist if h["is_fraud"])
            normal_count = len(hist) - fraud_count
            
            fig = px.pie(
                values = [normal_count, fraud_count],
                names = ["Safe", "Fraud"],
                title = f"This Session ({len(hist)} checks)",
                color_discrete_sequence = ["#00cc88", "#ff4b4b"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions have been made in this session yet.")
            st.caption("Go to the Single Transaction page!")
            
    with col3:
        st.subheader("Risk Threshold Info")
        threshold = model_info.get("threshold", 0.3)
        st.metric("Production Threshold", f"{threshold:.3f}")
        st.markdown(
            """
            -----------------------------
            | Risk          |   Range   |
            |---------------|-----------|
            | ✅ SAFE       |  0 - 30%  |
            | 🟢 LOW        | 30 - 50%  |
            | 🟡 MEDIUM     | 50 - 70%  |
            | 🟠 HIGH       | 70 - 90%  |
            | 🔴 CRITICAL   | 90 - 100% |
            -----------------------------
            """
        )
        
    # API Status footer
    st.markdown("---")
    status_color = "🟢" if api_online else "🔴"
    st.caption(f"{status_color} API Status: {'Online' if api_online else 'Offline'} | "
               f"Last updated: {datetime.now().strftime('%H:%M:%S')}")