# 🔍 Credit Card Fraud Detection System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optuna_Tuned-0769AD?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> Real-time credit card fraud detection system powered by **XGBoost** with Optuna hyperparameter tuning, served via **FastAPI REST API** with an interactive **Streamlit** web interface - fully deployed on cloud.

---

## 🌐 Live Demo

| Service | URL | Status |
|---------|-----|--------|
| 🎨 **Web App** | [credit-card-fraud-detection.streamlit.app](https://credit-card-fraud-detection-wfoc7ccnzscsx2ayxmc9rj.streamlit.app/) | ![Live](https://img.shields.io/badge/status-live-brightgreen) |
| ⚡ **API Docs** | [fraud-detection-api.onrender.com/docs](https://fraud-detection-api-7fap.onrender.com/docs) | ![Live](https://img.shields.io/badge/status-live-brightgreen) |

> ⚠️ API is hosted on Render free tier — first request may take 30-50 seconds to wake up.

---

## 📊 Model Performance

| Metric | Score | Target |
|--------|-------|--------|
| 🎯 **Recall** | **0.8673** | <= 0.90 & > 0.85 ✅ |
| 🎯 **Precision** | **0.6641** | > 0.50 ✅ |
| 🎯 **F1-Score** | **0.7522** | > 0.70 ✅ |
| 🎯 **ROC-AUC** | **0.9851** | > 0.95 ✅ |

> Model optimized for **high recall** - catching frauds is the priority over false alarms. This reflects real-world business logic where missing a fraud is far costlier than a false alert.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| 🤖 **ML Model** | XGBoost | Gradient boosting classifier |
| 🔧 **Tuning** | Optuna | Hyperparameter optimization (100 trials) |
| ⚖️ **Imbalance** | SMOTE | Synthetic oversampling (0.17% fraud rate) |
| 🔄 **Preprocessing** | RobustScaler | Outlier-resistant feature scaling |
| ⚡ **Backend** | FastAPI + Pydantic | REST API with automatic validation |
| 🎨 **Frontend** | Streamlit + Plotly | Interactive web dashboard |
| 📦 **Model Storage** | Hugging Face Hub | Large file hosting (ML standard) |
| 🚀 **API Hosting** | Render.com | Cloud deployment (auto-deploy via GitHub) |
| 🌐 **App Hosting** | Streamlit Cloud | Frontend deployment |
| 🗃️ **Version Control** | Git + GitHub | CI/CD pipeline |

---

## ✨ Features

- ✅ **Real-time** :- single transaction fraud check with confidence score
- ✅ **Batch processing** :- upload CSV, analyze up to 100 transactions at once
- ✅ **Risk levels** :- SAFE -> LOW -> MEDIUM -> HIGH -> CRITICAL
- ✅ **Interactive dashboard** :- model performance charts, dataset overview
- ✅ **REST API** :- 5 endpoints with full Swagger documentation
- ✅ **Input validation** :- Pydantic schema validation on all API inputs
- ✅ **Auto-deploy** :- GitHub push -> Render auto-redeploys

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Server health + uptime |
| `GET` | `/model-info` | Model parameters + threshold |
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/batch` | Batch predictions (max 100) |
| `GET` | `/docs` | Interactive Swagger UI |

### Example API Request

```bash
curl -X POST "https://fraud-detection-api-7fap.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{ "Time": 0.01
       "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
       "V4": 1.3782,  "V5": -0.3383, "V6": 0.4624,
       "V7": 0.2396,  "V8": 0.0987,  "V9": 0.3638,
       "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
       "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
       "V16": -0.4704, "V17": 0.2080, "V18": 0.0258,
       "V19": 0.4040, "V20": 0.2514,  "V21": -0.0183,
       "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
       "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
       "V28": -0.0211, "Amount": 149.62, "Time": 0.0, "Time_Hour": 0.0
     }'
```

### Example Response

```json
{
  "is_fraud": true,
  "confidence": 0.9423,
  "confidence_pct": "94.23%",
  "risk_level": "CRITICAL",
  "recommendation": "Block immediately",
  "threshold_used": 0.312,
  "processing_time": "18.3ms"
}
```

---

## 🗂️ Project Structure

```
fraud_detection_api/
│
├── api/
│   └── main.py                  # FastAPI REST API (5 endpoints)
│
├── app/
│   └── app.py                   # Streamlit Web App (4 pages)
│
├── pipeline/                    # ML Pipeline (hosted on HuggingFace)
│   ├── model.joblib             # Trained XGBoost model
│   ├── scaler.joblib            # Fitted RobustScaler
│   └── config.json              # Threshold + feature config
│
├── .python-version              # Python 3.10 (Render)
├── render.yaml                  # Render deployment config
├── requirements_main.txt        # API dependencies
├── requirements.txt             # Streamlit app dependencies
└── README.md
```

---

## 📈 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1-V28 PCA + Amount + Time) |
| Time period | 2 days of European cardholder transactions |

---

## 🚀 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/KP081/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements_main.txt

# 4. Run FastAPI (Terminal 1)
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 5. Run Streamlit (Terminal 2)
streamlit run app/app.py --server.port 8501
# → http://localhost:8501
```

> **Note:** Model files are automatically downloaded from Hugging Face Hub on first startup.

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│       STREAMLIT WEB APP         │
│   (Streamlit Cloud — Free)      │
│  Home | Predict | Batch | Dash  │
└──────────────┬──────────────────┘
               │ HTTP (requests)
               ▼
┌─────────────────────────────────┐
│        FASTAPI REST API         │
│      (Render.com — Free)        │
│  /predict  /batch  /health      │
└──────────────┬──────────────────┘
               │ joblib.load()
               ▼
┌─────────────────────────────────┐
│       ML PIPELINE               │
│   (Hugging Face Hub — Free)     │
│  XGBoost + RobustScaler         │
│  + Optuna-tuned threshold       │
└─────────────────────────────────┘
```

---

## 👤 Author

**Kathan Patel**

[![GitHub](https://img.shields.io/badge/GitHub-KP081-181717?style=for-the-badge&logo=github)](https://github.com/KP081)

---

## 📄 License

This project is licensed under the MIT License - feel free to use it for learning and portfolio purposes.