import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import preprocess_input

# MLflow imports and utilities
try:
    import mlflow
    import mlflow.tracking
    from utils.mlflow_utils import (
        get_all_runs, get_model_metrics, get_all_model_comparison,
        read_mlflow_metric, read_mlflow_tag, get_run_metadata
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Fallback functions if MLflow not available
    def get_all_runs(): return []
    def get_model_metrics(*args): return {}
    def get_all_model_comparison(): return {"classification": [], "regression": []}
except Exception as e:
    MLFLOW_AVAILABLE = False
    # Fallback if mlflow_utils has issues
    def get_all_runs(): return []
    def get_model_metrics(*args): return {}
    def get_all_model_comparison(): return {"classification": [], "regression": []}

# ---------- App Config ----------
st.set_page_config(
    page_title="EMIPredict AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💳"
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        margin-top: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Result display */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Responsive Design - Mobile */
    @media screen and (max-width: 768px) {
        .main {
            padding: 1rem 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.2rem;
        }
        
        .info-box {
            padding: 1rem;
            font-size: 0.9rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
        
        .result-box {
            padding: 1.5rem;
            font-size: 1.2rem;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    
    /* Responsive Design - Tablet */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 1.5rem;
        }
    }
    
    /* Error message styling */
    .error-message {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #721c24;
    }
    
    /* Loading spinner */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #1f77b4, transparent);
        margin: 2rem 0;
    }
    
    /* Dashboard metric cards */
    .metric-card {
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .metric-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    .model-detail-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Sidebar Navigation Styling - Premium Modern Design */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }
    
    .sidebar-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        border: none;
        padding: 0;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    
    .sidebar-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.95;
        font-size: 0.9rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Navigation Button Styling - Clean & Modern */
    div[data-testid="stSidebar"] > div:first-of-type {
        padding-top: 0 !important;
    }
    
    div[data-testid="stSidebar"] button {
        width: 100% !important;
        padding: 1rem 1.25rem !important;
        margin: 0.4rem 0 !important;
        background: #ffffff !important;
        color: #2c3e50 !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        text-align: left !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06) !important;
        border-left: 4px solid transparent !important;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stSidebar"] button::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: scaleY(0);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stSidebar"] button:hover {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%) !important;
        color: #667eea !important;
        transform: translateX(10px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25) !important;
    }
    
    div[data-testid="stSidebar"] button:hover::before {
        transform: scaleY(1);
    }
    
    /* Active button styling */
    div[data-testid="stSidebar"] button[kind="primary"],
    div[data-testid="stSidebar"] button[aria-pressed="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
        border-left: 4px solid #ffffff !important;
        font-weight: 600 !important;
        transform: translateX(5px) !important;
    }
    
    div[data-testid="stSidebar"] button[kind="primary"]:hover,
    div[data-testid="stSidebar"] button[aria-pressed="true"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%) !important;
        transform: translateX(10px) scale(1.02) !important;
        box-shadow: 0 10px 28px rgba(102, 126, 234, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Models (cached) ----------
@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    try:
        clf_path = "models/xgboost_clf.pkl"
        reg_path = "models/xgboost_reg.pkl"
        
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Classifier model not found at {clf_path}")
        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"Regressor model not found at {reg_path}")
        
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        
        return clf, reg
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.info("Please ensure model files are in the 'models/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please check that model files are valid and compatible.")
        st.stop()

# Load models with error handling
try:
    clf_model, reg_model = load_models()
except Exception as e:
    st.error("Failed to initialize application. Please check the error above.")
    st.stop()

# ---------- Load MLflow Data (cached) ----------
@st.cache_data
def load_mlflow_data():
    """Load MLflow metrics and model data"""
    try:
        final_metrics = get_model_metrics("final")
        all_models = get_all_model_comparison()
        all_runs = get_all_runs()
        
        # Find XGBoost final models
        xgb_clf = next((r for r in all_runs if r["type"] == "classification" and 
                       ("XGBoost" in r["run_name"] or "Final" in r["run_name"])), None)
        xgb_reg = next((r for r in all_runs if r["type"] == "regression" and 
                       ("XGBoost" in r["run_name"] or "Final" in r["run_name"])), None)
        
        return {
            "final_classifier": xgb_clf["metrics"] if xgb_clf else final_metrics.get("classifier", {}),
            "final_regressor": xgb_reg["metrics"] if xgb_reg else final_metrics.get("regressor", {}),
            "all_classification": all_models.get("classification", []),
            "all_regression": all_models.get("regression", []),
            "all_runs": all_runs
        }
    except Exception as e:
        return {
            "final_classifier": {},
            "final_regressor": {},
            "all_classification": [],
            "all_regression": [],
            "all_runs": []
        }

mlflow_data = load_mlflow_data()

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("### <div style='text-align: center;'>Navigation</div>", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Navigation menu items with icons
nav_items = [
    ("🏠", "Home"),
    ("📈", "System Overview"),
    ("📊", "EMI Eligibility Checker"),
    ("💰", "Max EMI Predictor"),
    ("📉", "Data Exploration"),
    ("🔬", "MLflow Dashboard")
]

# Create styled navigation buttons
for icon, item_name in nav_items:
    is_active = st.session_state.current_page == item_name
    
    # Apply custom styling for active state
    if is_active:
        st.sidebar.markdown(f"""
        <style>
        div[data-testid="stSidebar"] button[key="nav_{item_name}"] {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
            border-left: 4px solid #ffffff !important;
            font-weight: 600 !important;
            transform: translateX(5px) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    if st.sidebar.button(f"{icon} {item_name}", key=f"nav_{item_name}", use_container_width=True):
        st.session_state.current_page = item_name

# Update page variable from session state
page = st.session_state.current_page

st.sidebar.markdown("""
    <div style='padding: 1rem; background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%); border-radius: 10px; margin-top: 2rem; border-left: 4px solid #1f77b4;'>
        <p style='font-size: 0.9rem; color: #2c3e50; margin: 0; line-height: 1.6;'>
            <strong>💡 Tip:</strong> Fill in all required fields for accurate predictions.
        </p>
    </div>
""", unsafe_allow_html=True)

# Developer Credit Panel

# =====================================================
# 🏠 HOME
# =====================================================
if page == "Home":
    st.title("💳 EMIPredict AI")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="info-box">
        <h2 style='color: white; margin-top: 0;'>Intelligent Financial Risk Assessment Platform</h2>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
        Make smarter lending decisions with AI-powered EMI predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>📊 Eligibility Check</h4>
            <p>Instantly determine if a customer is eligible for EMI with our advanced classification model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>💰 Max EMI Predictor</h4>
            <p>Calculate the maximum safe EMI amount a customer can afford based on their financial profile.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Supported EMI Scenarios")
    
    scenarios = [
        "🛒 E-commerce Shopping EMI",
        "🏠 Home Appliances EMI",
        "🚗 Vehicle EMI",
        "💼 Personal Loan EMI",
        "🎓 Education EMI"
    ]
    
    for scenario in scenarios:
        st.markdown(f"""
        <div class="metric-container">
            <p style='margin: 0.5rem 0; font-size: 1rem;'>{scenario}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # About Section
    st.markdown("### 📘 About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Machine Learning Models")
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>📊 XGBoost Classifier</h4>
            <p><strong>Purpose:</strong> EMI Eligibility Prediction</p>
            <p>Classifies customers into three categories:</p>
            <ul>
                <li>✅ Eligible</li>
                <li>⚠️ High Risk</li>
                <li>❌ Not Eligible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>💰 XGBoost Regressor</h4>
            <p><strong>Purpose:</strong> Maximum EMI Amount Prediction</p>
            <p>Calculates the maximum safe EMI amount a customer can afford based on their financial profile and risk factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>📈 Training Data</h4>
            <ul>
                <li><strong>Size:</strong> 404,800 financial profiles</li>
                <li><strong>Original Features:</strong> 17 input features</li>
                <li><strong>Total Features:</strong> 46 (after engineering & encoding)</li>
                <li><strong>Domain:</strong> Banking & FinTech</li>
                <li><strong>Quality:</strong> Real-world financial data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        <div class="feature-card">
            <ul style='list-style: none; padding: 0;'>
                <li>🐍 <strong>Python</strong> - Core programming language</li>
                <li>📚 <strong>Scikit-learn</strong> - Machine learning utilities</li>
                <li>🚀 <strong>XGBoost</strong> - Gradient boosting framework</li>
                <li>🌐 <strong>Streamlit</strong> - Web application framework</li>
                <li>📊 <strong>MLflow</strong> - Experiment tracking & model registry</li>
                <li>💾 <strong>Joblib</strong> - Model serialization</li>
                <li>🐼 <strong>Pandas</strong> - Data manipulation</li>
                <li>🔢 <strong>NumPy</strong> - Numerical computing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Developer Credit Panel at Bottom
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='
        padding: 2rem;
        background: linear-gradient(135deg, #0077b5 0%, #005885 100%);
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 119, 181, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        margin-top: 2rem;
    '>
        <div style='color: white; margin-bottom: 0.8rem;'>
            <p style='font-size: 0.85rem; margin: 0; opacity: 0.95; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;'>Developed By</p>
        </div>
        <div style='color: white; margin: 0.8rem 0;'>
            <h2 style='font-size: 1.5rem; margin: 0; font-weight: 600; letter-spacing: 0.5px;'>Sumathi S</h2>
        </div>
        <div style='margin-top: 1.5rem;'>
            <a href='https://www.linkedin.com/in/sumathisaravanan/' 
               target='_blank' 
               style='
                   display: inline-block;
                   padding: 0.8rem 2rem;
                   background: white;
                   color: #0077b5;
                   text-decoration: none;
                   border-radius: 8px;
                   font-size: 1rem;
                   font-weight: 600;
                   transition: all 0.3s ease;
                   box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
               '
               onmouseover="this.style.background='#f0f0f0'; this.style.transform='translateY(-3px)'; this.style.boxShadow='0 6px 16px rgba(0, 0, 0, 0.2)';"
               onmouseout="this.style.background='white'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.15)';">
                🔗 Connect on LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 📈 SYSTEM OVERVIEW
# =====================================================
elif page == "System Overview":
    st.title("📈 System Overview")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Load MLflow metrics
    clf_metrics = mlflow_data.get("final_classifier", {})
    reg_metrics = mlflow_data.get("final_regressor", {})
    
    if not clf_metrics and not reg_metrics:
        st.warning("⚠️ MLflow data not found. Please ensure mlruns folder is in the project directory.")
        st.stop()
    
    # Dashboard Metrics Cards - XGBoost Models
    col1, col2 = st.columns(2)
    
    with col1:
        acc = clf_metrics.get("accuracy", 0) * 100 if clf_metrics.get("accuracy", 0) < 1 else clf_metrics.get("accuracy", 0)
        prec = clf_metrics.get("precision", 0) * 100 if clf_metrics.get("precision", 0) < 1 else clf_metrics.get("precision", 0)
        rec = clf_metrics.get("recall", 0) * 100 if clf_metrics.get("recall", 0) < 1 else clf_metrics.get("recall", 0)
        f1 = clf_metrics.get("f1_score", 0) * 100 if clf_metrics.get("f1_score", 0) < 1 else clf_metrics.get("f1_score", 0)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);">
            <span class="metric-icon">📊</span>
            <div class="metric-title">XGBoost Classifier</div>
            <div class="metric-value" style="font-size: 2.2rem; margin: 1rem 0;">{acc:.1f}%</div>
            <div class="metric-subtitle" style="margin-bottom: 1rem;">Accuracy</div>
            <div style="border-top: 1px solid rgba(255,255,255,0.3); padding-top: 1rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>Precision:</span>
                    <strong>{prec:.1f}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>Recall:</span>
                    <strong>{rec:.1f}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>F1-Score:</span>
                    <strong>{f1:.1f}%</strong>
                </div>
            </div>
            <div class="metric-subtitle" style="margin-top: 1rem; color: rgba(255,255,255,0.9);">✅ Active Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        r2 = reg_metrics.get("r2", 0)
        rmse = reg_metrics.get("rmse", 0)
        mae = reg_metrics.get("mae", 0)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);">
            <span class="metric-icon">💰</span>
            <div class="metric-title">XGBoost Regressor</div>
            <div class="metric-value" style="font-size: 2.2rem; margin: 1rem 0;">{r2:.3f}</div>
            <div class="metric-subtitle" style="margin-bottom: 1rem;">R² Score</div>
            <div style="border-top: 1px solid rgba(255,255,255,0.3); padding-top: 1rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>RMSE:</span>
                    <strong>₹{int(rmse):,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>MAE:</span>
                    <strong>₹{int(mae):,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>R²:</span>
                    <strong>{r2:.3f}</strong>
                </div>
            </div>
            <div class="metric-subtitle" style="margin-top: 1rem; color: rgba(255,255,255,0.9);">✅ Active Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Model Details Section
    st.markdown("### 🤖 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate metrics for display
        clf_acc = clf_metrics.get("accuracy", 0) * 100 if clf_metrics.get("accuracy", 0) < 1 else clf_metrics.get("accuracy", 0)
        clf_prec = clf_metrics.get("precision", 0) * 100 if clf_metrics.get("precision", 0) < 1 else clf_metrics.get("precision", 0)
        clf_rec = clf_metrics.get("recall", 0) * 100 if clf_metrics.get("recall", 0) < 1 else clf_metrics.get("recall", 0)
        clf_f1 = clf_metrics.get("f1_score", 0) * 100 if clf_metrics.get("f1_score", 0) < 1 else clf_metrics.get("f1_score", 0)
        
        st.markdown(f"""
        <div class="model-detail-card" style="border: 2px solid #10B981;">
            <h3 style='color: #10B981; margin-top: 0; display: flex; align-items: center;'>
                <span style='font-size: 2rem; margin-right: 0.5rem;'>📊</span>
                XGBoost Classifier (Final)
            </h3>
            <p><strong>Model Type:</strong> Gradient Boosting Classifier</p>
            <p><strong>Purpose:</strong> EMI Eligibility Prediction</p>
            <p><strong>Output Classes:</strong> 3 (Eligible, High Risk, Not Eligible)</p>
            <p><strong>Training Data:</strong> 404,800 financial profiles</p>
            <p><strong>Features:</strong> 46 features (17 original + 11 engineered + 18 categorical)</p>
            <p><strong>Algorithm:</strong> XGBoost (Extreme Gradient Boosting)</p>
            <hr style='margin: 1rem 0; border: 1px solid #e0e0e0;'>
            <p style='margin: 0.5rem 0;'><strong>Performance Metrics:</strong></p>
            <ul style='margin-top: 0.5rem;'>
                <li><strong>Accuracy:</strong> {clf_acc:.1f}%</li>
                <li><strong>Precision:</strong> {clf_prec:.1f}%</li>
                <li><strong>Recall:</strong> {clf_rec:.1f}%</li>
                <li><strong>F1-Score:</strong> {clf_f1:.1f}%</li>
            </ul>
            <p style='margin-top: 1rem; margin-bottom: 0;'><strong>Status:</strong> <span style='color: #10B981; font-weight: bold;'>✅ Active & Ready</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate metrics for display
        reg_r2 = reg_metrics.get("r2", 0)
        reg_rmse = int(reg_metrics.get("rmse", 0))
        reg_mae = int(reg_metrics.get("mae", 0))
        
        st.markdown(f"""
        <div class="model-detail-card" style="border: 2px solid #10B981;">
            <h3 style='color: #10B981; margin-top: 0; display: flex; align-items: center;'>
                <span style='font-size: 2rem; margin-right: 0.5rem;'>💰</span>
                XGBoost Regressor (Final)
            </h3>
            <p><strong>Model Type:</strong> Gradient Boosting Regressor</p>
            <p><strong>Purpose:</strong> Maximum EMI Amount Prediction</p>
            <p><strong>Output:</strong> Continuous value (₹)</p>
            <p><strong>Training Data:</strong> 404,800 financial profiles</p>
            <p><strong>Features:</strong> 46 features (17 original + 11 engineered + 18 categorical)</p>
            <p><strong>Algorithm:</strong> XGBoost (Extreme Gradient Boosting)</p>
            <hr style='margin: 1rem 0; border: 1px solid #e0e0e0;'>
            <p style='margin: 0.5rem 0;'><strong>Performance Metrics:</strong></p>
            <ul style='margin-top: 0.5rem;'>
                <li><strong>R² Score:</strong> {reg_r2:.3f}</li>
                <li><strong>RMSE:</strong> ₹{reg_rmse:,}</li>
                <li><strong>MAE:</strong> ₹{reg_mae:,}</li>
            </ul>
            <p style='margin-top: 1rem; margin-bottom: 0;'><strong>Status:</strong> <span style='color: #10B981; font-weight: bold;'>✅ Active & Ready</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Metrics - Classification Models
    st.markdown("### 📊 Classification Models Performance")
    
    clf_models = mlflow_data["all_classification"]
    
    if len(clf_models) > 0:
        # Display all classification models
        cols = st.columns(min(len(clf_models), 3))
        
        for idx, model in enumerate(clf_models[:3]):
            with cols[idx % 3]:
                is_final = "XGBoost" in model["name"] or "Final" in model["name"]
                border_style = "border: 3px solid #10B981;" if is_final else ""
                title_color = "#10B981" if is_final else "#1f77b4"
                metrics = model.get("metrics", {})
                
                acc = metrics.get("accuracy", 0) * 100 if metrics.get("accuracy", 0) < 1 else metrics.get("accuracy", 0)
                prec = metrics.get("precision", 0) * 100 if metrics.get("precision", 0) < 1 else metrics.get("precision", 0)
                rec = metrics.get("recall", 0) * 100 if metrics.get("recall", 0) < 1 else metrics.get("recall", 0)
                f1 = metrics.get("f1_score", 0) * 100 if metrics.get("f1_score", 0) < 1 else metrics.get("f1_score", 0)
                
                model_icon = "🚀" if is_final else "📈" if "Logistic" in model["name"] else "🌲"
                
                st.markdown(f"""
                <div class="model-detail-card" style="{border_style}">
                    <h4 style='color: {title_color}; margin-top: 0;'>{model_icon} {model["name"]}</h4>
                    <ul style='line-height: 2;'>
                        <li><strong>Accuracy:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>{acc:.1f}%</span></li>
                        <li><strong>Precision:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>{prec:.1f}%</span></li>
                        <li><strong>Recall:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>{rec:.1f}%</span></li>
                        <li><strong>F1-Score:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>{f1:.1f}%</span></li>
                    </ul>
                    {"<p style='margin-top: 0.5rem; color: #10B981; font-weight: bold;'>✅ Currently Active</p>" if is_final else ""}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No classification models found in MLflow. Please ensure models are logged to MLflow.")
    
    # Regression Models Performance
    st.markdown("### 📈 Regression Models Performance")
    
    reg_models = mlflow_data["all_regression"]
    
    if len(reg_models) > 0:
        # Display all regression models
        cols = st.columns(min(len(reg_models), 3))
        
        for idx, model in enumerate(reg_models[:3]):
            with cols[idx % 3]:
                is_final = "XGBoost" in model["name"] or "Final" in model["name"]
                border_style = "border: 3px solid #10B981;" if is_final else ""
                title_color = "#10B981" if is_final else "#1f77b4"
                metrics = model.get("metrics", {})
                
                r2 = metrics.get("r2", 0)
                rmse = metrics.get("rmse", 0)
                mae = metrics.get("mae", 0)
                
                model_icon = "🚀" if is_final else "📊" if "Linear" in model["name"] else "🌲"
                
                st.markdown(f"""
                <div class="model-detail-card" style="{border_style}">
                    <h4 style='color: {title_color}; margin-top: 0;'>{model_icon} {model["name"]}</h4>
                    <ul style='line-height: 2;'>
                        <li><strong>R² Score:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>{r2:.3f}</span></li>
                        <li><strong>RMSE:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>₹{int(rmse):,}</span></li>
                        <li><strong>MAE:</strong> <span style='{"color: #10B981; font-weight: bold;" if is_final else ""}'>₹{int(mae):,}</span></li>
                    </ul>
                    {"<p style='margin-top: 0.5rem; color: #10B981; font-weight: bold;'>✅ Currently Active</p>" if is_final else ""}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No regression models found in MLflow. Please ensure models are logged to MLflow.")
    
    # Model Architecture
    st.markdown("### 🏗️ Model Architecture & Training")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-detail-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>Training Configuration</h4>
            <ul style='line-height: 2;'>
                <li><strong>Data Split:</strong> 80% Train, 20% Test</li>
                <li><strong>Random State:</strong> 42 (for reproducibility)</li>
                <li><strong>Stratification:</strong> Yes (for classification)</li>
                <li><strong>Dataset Size:</strong> 404,800 records</li>
                <li><strong>Experiment Tracking:</strong> MLflow</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Get XGBoost classifier parameters from MLflow
        xgb_clf_run = next((r for r in mlflow_data["all_runs"] 
                           if r["type"] == "classification" and 
                           ("XGBoost" in r["run_name"] or "Final" in r["run_name"])), None)
        clf_params = xgb_clf_run.get("params", {}) if xgb_clf_run else {}
        
        st.markdown(f"""
        <div class="model-detail-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>XGBoost Classifier Parameters</h4>
            <ul style='line-height: 2;'>
                <li><strong>n_estimators:</strong> {clf_params.get('n_estimators', 'N/A')}</li>
                <li><strong>max_depth:</strong> {clf_params.get('max_depth', 'N/A')}</li>
                <li><strong>learning_rate:</strong> {clf_params.get('learning_rate', 'N/A')}</li>
                <li><strong>subsample:</strong> {clf_params.get('subsample', 'N/A')}</li>
                <li><strong>colsample_bytree:</strong> {clf_params.get('colsample_bytree', 'N/A')}</li>
                <li><strong>objective:</strong> {clf_params.get('objective', 'N/A')}</li>
                <li><strong>eval_metric:</strong> {clf_params.get('eval_metric', 'N/A')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get XGBoost regressor parameters from MLflow
        xgb_reg_run = next((r for r in mlflow_data["all_runs"] 
                           if r["type"] == "regression" and 
                           ("XGBoost" in r["run_name"] or "Final" in r["run_name"])), None)
        reg_params = xgb_reg_run.get("params", {}) if xgb_reg_run else {}
        
        st.markdown(f"""
        <div class="model-detail-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>XGBoost Regressor Parameters</h4>
            <ul style='line-height: 2;'>
                <li><strong>n_estimators:</strong> {reg_params.get('n_estimators', 'N/A')}</li>
                <li><strong>max_depth:</strong> {reg_params.get('max_depth', 'N/A')}</li>
                <li><strong>learning_rate:</strong> {reg_params.get('learning_rate', 'N/A')}</li>
                <li><strong>subsample:</strong> {reg_params.get('subsample', 'N/A')}</li>
                <li><strong>colsample_bytree:</strong> {reg_params.get('colsample_bytree', 'N/A')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Get other model parameters from MLflow
        other_models_info = []
        for run in mlflow_data["all_runs"]:
            if "XGBoost" not in run["run_name"] and "Final" not in run["run_name"]:
                params = run.get("params", {})
                if params:
                    key_params = {k: v for k, v in params.items() 
                                if k in ["n_estimators", "max_depth", "max_iter", "C"]}
                    if key_params:
                        other_models_info.append({
                            "name": run["run_name"],
                            "params": key_params
                        })
        
        params_html = ""
        if other_models_info:
            for model_info in other_models_info[:3]:  # Show max 3
                params_str = ", ".join([f"{k}={v}" for k, v in model_info["params"].items()])
                params_html += f"<li><strong>{model_info['name']}:</strong> {params_str}</li>"
        else:
            params_html = "<li>No additional model parameters found in MLflow</li>"
        
        st.markdown(f"""
        <div class="model-detail-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>Other Models</h4>
            <ul style='line-height: 2;'>
                {params_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 📊 EMI ELIGIBILITY CHECKER
# =====================================================
elif page == "EMI Eligibility Checker":
    st.title("📊 EMI Eligibility Checker")
    st.markdown("### Enter customer information to check EMI eligibility")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.form("eligibility_form"):
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            age = st.number_input("👤 Age", 18, 65, 30, help="Customer's age (18-65 years)")
            gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
            marital_status = st.selectbox("💑 Marital Status", ["Single", "Married"])
            education = st.selectbox("🎓 Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            monthly_salary = st.number_input("💰 Monthly Salary (₹)", 10000, 300000, 50000, help="Monthly income in rupees")
            employment_type = st.selectbox("💼 Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("📅 Years of Employment", 0, 40, 5, help="Total years in current employment")
            company_type = st.selectbox("🏢 Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            house_type = st.selectbox("🏠 House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("🏘️ Monthly Rent (₹)", 0, 100000, 0, help="Monthly housing expenses")
            existing_loans = st.selectbox("📋 Existing Loans", ["Yes", "No"])
            current_emi_amount = st.number_input("💳 Current EMI Amount (₹)", 0, 100000, 0, help="Current monthly EMI payments")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Check Eligibility", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": 3,
            "dependents": 1,
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": 2000,
            "groceries_utilities": 5000,
            "other_monthly_expenses": 3000,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": 750,
            "bank_balance": 50000,
            "emergency_fund": 30000,
            "emi_scenario": "Personal Loan EMI",
            "requested_amount": 300000,
            "requested_tenure": 36,
        }

        X_input = preprocess_input(input_data)
        pred = clf_model.predict(X_input)[0]

        label_map = {0: "✅ Eligible", 1: "⚠️ High Risk", 2: "❌ Not Eligible"}
        color_map = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
        icon_map = {0: "✅", 1: "⚠️", 2: "❌"}
        
        result_text = label_map[pred]
        result_color = color_map[pred]
        
        st.markdown(f"""
        <div class="result-box" style="background: linear-gradient(135deg, {result_color} 0%, {result_color}dd 100%);">
            <h2 style='color: white; margin: 0;'>{result_text}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if pred == 0:
            st.info("🎉 This customer is eligible for EMI. Proceed with the loan application.")
        elif pred == 1:
            st.warning("⚠️ This customer is at high risk. Review the application carefully before approval.")
        else:
            st.error("❌ This customer is not eligible for EMI based on the current financial profile.")

# =====================================================
# 💰 MAX EMI PREDICTOR
# =====================================================
elif page == "Max EMI Predictor":
    st.title("💰 Maximum EMI Predictor")
    st.markdown("### Calculate the maximum safe EMI amount for a customer")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.form("max_emi_form"):
        st.markdown("#### 👤 Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            age = st.number_input("👤 Age", 18, 65, 30, key="emi_age", help="Customer's age (18-65 years)")
            gender = st.selectbox("⚧️ Gender", ["Male", "Female"], key="emi_gender")
            marital_status = st.selectbox("💑 Marital Status", ["Single", "Married"], key="emi_marital")
            education = st.selectbox("🎓 Education", ["High School", "Graduate", "Post Graduate", "Professional"], key="emi_education")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            monthly_salary = st.number_input("💰 Monthly Salary (₹)", 10000, 300000, 50000, key="emi_salary", help="Monthly income in rupees")
            employment_type = st.selectbox("💼 Employment Type", ["Private", "Government", "Self-employed"], key="emi_employment")
            years_of_employment = st.number_input("📅 Years of Employment", 0, 40, 5, key="emi_years", help="Total years in current employment")
            company_type = st.selectbox("🏢 Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"], key="emi_company")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            house_type = st.selectbox("🏠 House Type", ["Rented", "Own", "Family"], key="emi_house")
            monthly_rent = st.number_input("🏘️ Monthly Rent (₹)", 0, 100000, 0, key="emi_rent", help="Monthly housing expenses")
            existing_loans = st.selectbox("📋 Existing Loans", ["Yes", "No"], key="emi_existing")
            current_emi_amount = st.number_input("💳 Current EMI Amount (₹)", 0, 100000, 0, key="emi_current", help="Current monthly EMI payments")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 💵 Loan Details")
        
        col4, col5 = st.columns(2)
        with col4:
            requested_amount = st.number_input("💵 Requested Loan Amount (₹)", 10000, 2000000, 500000, key="emi_requested", help="Total loan amount requested")
        with col5:
            emi_scenario = st.selectbox("📋 EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"], key="emi_scenario")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Max EMI", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": 3,
            "dependents": 1,
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": 2000,
            "groceries_utilities": 5000,
            "other_monthly_expenses": 3000,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": 750,
            "bank_balance": 50000,
            "emergency_fund": 30000,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": 36,
        }

        X_input = preprocess_input(input_data)
        emi = reg_model.predict(X_input)[0]

        st.markdown(f"""
        <div class="result-box">
            <h2 style='color: white; margin: 0;'>Maximum Safe EMI</h2>
            <h1 style='color: white; margin: 1rem 0; font-size: 3rem;'>₹ {int(emi):,}</h1>
            <p style='color: white; margin: 0; opacity: 0.9;'>Based on customer's financial profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Requested Amount", f"₹ {requested_amount:,}")
        with col2:
            monthly_payment_ratio = (emi / monthly_salary * 100) if monthly_salary > 0 else 0
            st.metric("EMI to Salary Ratio", f"{monthly_payment_ratio:.1f}%")
        with col3:
            st.metric("Monthly Salary", f"₹ {monthly_salary:,}")

# =====================================================
# 📉 DATA EXPLORATION
# =====================================================
elif page == "Data Exploration":
    st.title("📉 Data Exploration & Visualization")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Try to load from data folder first
    data_path = "data/emi_prediction_dataset.csv"
    df = None
    
    if os.path.exists(data_path):
        try:
            @st.cache_data
            def load_data_from_path(path):
                return pd.read_csv(path, low_memory=False)
            
            df = load_data_from_path(data_path)
            st.success(f"✅ Dataset loaded from {data_path}! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.warning(f"Could not load from {data_path}: {str(e)}")
    
    # File upload section (as fallback)
    if df is None:
        st.markdown("### 📁 Load Dataset")
        uploaded_file = st.file_uploader("Upload CSV file for exploration", type=['csv'], help="Upload the EMI prediction dataset CSV file")
        
        if uploaded_file is not None:
            try:
                @st.cache_data
                def load_data(file):
                    return pd.read_csv(file, low_memory=False)
                
                df = load_data(uploaded_file)
                
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    if df is not None:
        # Dataset Overview
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
        
        # Data Preview - Full Width
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Visualization Section - Full Width
        st.markdown("### 📈 Interactive Visualizations")
        
        viz_tabs = st.tabs(["📊 Distribution Analysis", "📉 Correlation Analysis", "🎯 Target Analysis", "📋 Feature Analysis"])
        
        with viz_tabs[0]:
            st.markdown("#### Distribution of Key Features")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select feature to visualize", numeric_cols[:10], key="dist_col")
                
                fig = px.histogram(df, x=selected_col, nbins=50, 
                                 title=f"Distribution of {selected_col}",
                                 labels={selected_col: selected_col.replace('_', ' ').title()})
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            st.markdown("#### Feature Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                corr_cols = st.multiselect("Select features for correlation", numeric_cols[:15], 
                                          default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols,
                                          key="corr_cols")
                
                if len(corr_cols) > 1:
                    corr_matrix = df[corr_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                   title="Correlation Heatmap",
                                   color_continuous_scale="RdBu")
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            st.markdown("#### Target Variable Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'emi_eligibility' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    # Eligibility distribution
                    eligibility_counts = df['emi_eligibility'].value_counts()
                    fig = px.pie(values=eligibility_counts.values, names=eligibility_counts.index,
                               title="EMI Eligibility Distribution")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Eligibility by scenario
                    if 'emi_scenario' in df.columns:
                        scenario_elig = pd.crosstab(df['emi_scenario'], df['emi_eligibility'])
                        fig = px.bar(scenario_elig, barmode='group', 
                                   title="Eligibility Distribution by EMI Scenario")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:
            st.markdown("#### Feature Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                selected_feature = st.selectbox("Select feature", numeric_cols, key="stat_feature")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Descriptive Statistics**")
                    st.dataframe(df[selected_feature].describe(), use_container_width=True)
                with col2:
                    st.markdown("**Box Plot**")
                    fig = px.box(df, y=selected_feature, title=f"Box Plot: {selected_feature}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please upload a CSV file to begin data exploration")
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>📊 Available Visualizations</h4>
            <ul>
                <li><strong>Distribution Analysis:</strong> Histograms and distribution plots for numeric features</li>
                <li><strong>Correlation Analysis:</strong> Interactive correlation heatmaps</li>
                <li><strong>Target Analysis:</strong> Analysis of EMI eligibility distribution</li>
                <li><strong>Feature Analysis:</strong> Statistical summaries and box plots</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 🔬 MLFLOW DASHBOARD
# =====================================================
elif page == "MLflow Dashboard":
    st.title("🔬 MLflow Dashboard")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if not MLFLOW_AVAILABLE:
        st.error("⚠️ MLflow is not installed. Please install it using: `pip install mlflow`")
        st.markdown("""
        <div class="feature-card">
            <h4 style='color: #1f77b4; margin-top: 0;'>📦 Installation</h4>
            <p>To enable MLflow features, install MLflow:</p>
            <code>pip install mlflow</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load MLflow data
        all_runs = mlflow_data["all_runs"]
        
        if len(all_runs) > 0:
            st.success(f"✅ Loaded {len(all_runs)} experiment runs from MLflow")
            
            # Experiment Summary
            st.markdown("### 📊 Experiment Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            clf_count = len([r for r in all_runs if r["type"] == "classification"])
            reg_count = len([r for r in all_runs if r["type"] == "regression"])
            
            with col1:
                st.metric("Total Runs", len(all_runs))
            with col2:
                st.metric("Classification Models", clf_count)
            with col3:
                st.metric("Regression Models", reg_count)
            with col4:
                st.metric("Experiment", "EMIPredict_AI")
            
            # All Runs Table
            st.markdown("### 📋 All Experiment Runs")
            
            runs_data = []
            for run in all_runs:
                run_info = {
                    "Run Name": run["run_name"],
                    "Type": run["type"].title(),
                    **{k: f"{v:.4f}" if isinstance(v, float) else str(v) 
                       for k, v in run["metrics"].items()}
                }
                runs_data.append(run_info)
            
            if runs_data:
                runs_df = pd.DataFrame(runs_data)
                st.dataframe(runs_df, use_container_width=True, hide_index=True)
            
            # Metrics Visualization
            st.markdown("### 📈 Model Performance Comparison")
            
            # Classification Metrics
            clf_runs = [r for r in all_runs if r["type"] == "classification"]
            if len(clf_runs) > 0:
                st.markdown("#### Classification Models")
                
                clf_data = []
                for run in clf_runs:
                    metrics = run.get("metrics", {})
                    clf_data.append({
                        "Model": run["run_name"],
                        "Accuracy": metrics.get("accuracy", 0) * 100 if metrics.get("accuracy", 0) < 1 else metrics.get("accuracy", 0),
                        "Precision": metrics.get("precision", 0) * 100 if metrics.get("precision", 0) < 1 else metrics.get("precision", 0),
                        "Recall": metrics.get("recall", 0) * 100 if metrics.get("recall", 0) < 1 else metrics.get("recall", 0),
                        "F1-Score": metrics.get("f1_score", 0) * 100 if metrics.get("f1_score", 0) < 1 else metrics.get("f1_score", 0)
                    })
                
                if clf_data:
                    clf_df = pd.DataFrame(clf_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Accuracy', x=clf_df['Model'], y=clf_df['Accuracy']))
                    fig.add_trace(go.Bar(name='Precision', x=clf_df['Model'], y=clf_df['Precision']))
                    fig.add_trace(go.Bar(name='Recall', x=clf_df['Model'], y=clf_df['Recall']))
                    fig.add_trace(go.Bar(name='F1-Score', x=clf_df['Model'], y=clf_df['F1-Score']))
                    fig.update_layout(barmode='group', title="Classification Metrics Comparison", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Regression Metrics
            reg_runs = [r for r in all_runs if r["type"] == "regression"]
            if len(reg_runs) > 0:
                st.markdown("#### Regression Models")
                
                reg_data = []
                for run in reg_runs:
                    metrics = run.get("metrics", {})
                    reg_data.append({
                        "Model": run["run_name"],
                        "R² Score": metrics.get("r2", 0),
                        "RMSE": metrics.get("rmse", 0),
                        "MAE": metrics.get("mae", 0)
                    })
                
                if reg_data:
                    reg_df = pd.DataFrame(reg_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='R² Score', x=reg_df['Model'], y=reg_df['R² Score'], yaxis='y'))
                    fig.add_trace(go.Bar(name='RMSE', x=reg_df['Model'], y=reg_df['RMSE'], yaxis='y2'))
                    fig.add_trace(go.Bar(name='MAE', x=reg_df['Model'], y=reg_df['MAE'], yaxis='y2'))
                    fig.update_layout(
                        title="Regression Metrics Comparison",
                        yaxis=dict(title="R² Score", side="left"),
                        yaxis2=dict(title="RMSE / MAE (₹)", side="right", overlaying="y"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Run Information
            st.markdown("### 🔍 Detailed Run Information")
            
            selected_run_name = st.selectbox("Select a run to view details", 
                                            [r["run_name"] for r in all_runs],
                                            key="run_selector")
            
            selected_run = next((r for r in all_runs if r["run_name"] == selected_run_name), None)
            
            if selected_run:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Metrics")
                    metrics_df = pd.DataFrame(list(selected_run["metrics"].items()), 
                                             columns=["Metric", "Value"])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Parameters")
                    if selected_run.get("params"):
                        params_df = pd.DataFrame(list(selected_run["params"].items()), 
                                                columns=["Parameter", "Value"])
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No parameters logged for this run")
        else:
            st.warning("No MLflow runs found. Please ensure the mlruns folder is in the project directory.")
            st.markdown("""
            <div class="feature-card">
                <h4 style='color: #1f77b4; margin-top: 0;'>💡 MLflow Data Location</h4>
                <p>The app is looking for MLflow data in: <code>./mlruns</code></p>
                <p>Make sure your mlruns folder is in the project root directory.</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
