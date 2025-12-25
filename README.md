# 💳 EMIPredict AI - Intelligent Financial Risk Assessment Platform

An AI-powered Streamlit application for making data-driven lending decisions through EMI eligibility prediction and maximum EMI amount calculation.

## 🚀 Features

- **EMI Eligibility Checker**: Classifies customers into Eligible, High Risk, or Not Eligible categories
- **Max EMI Predictor**: Calculates the maximum safe EMI amount based on financial profiles
- **Data Exploration**: Interactive visualizations and analysis of financial data
- **MLflow Dashboard**: Track and compare all ML model experiments
- **System Overview**: Comprehensive model performance metrics and details

## 🛠️ Technology Stack

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **XGBoost** - Gradient boosting for classification and regression
- **Scikit-learn** - Machine learning utilities
- **MLflow** - Experiment tracking & model registry
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data manipulation and numerical computing

## 📋 Prerequisites

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd Guvi_ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   - `models/xgboost_clf.pkl` - Classification model
   - `models/xgboost_reg.pkl` - Regression model
   - `models/feature_columns.pkl` - Feature columns reference

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

**Note**: The Data Exploration page requires the dataset file (`data/emi_prediction_dataset.csv`) which is not included in the GitHub repository due to its large size (71.93 MB). If you want to use the Data Exploration feature, you'll need to:
- Download the dataset separately
- Place it in the `data/` directory
- The app will automatically load it when you navigate to the Data Exploration page

## 🌐 Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Ensure all files are committed**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   ```

2. **Push to GitHub**
   ```bash
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Sign up/Login**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy New App**
   - Click "New app"
   - Select your repository
   - Select branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Access Your App**
   - Your app will be available at: `https://your-app-name.streamlit.app`
   - Auto-redeploy is enabled by default (redeploys on every push)

### Step 3: Verify Deployment

- ✅ All pages load correctly
- ✅ Models load successfully
- ✅ Predictions work
- ✅ MLflow dashboard displays
- ⚠️ **Note**: Data Exploration page may not work on Streamlit Cloud due to the large CSV file size (71.93 MB). To use the Data Exploration feature, please clone the repository and run the app locally.

## 📁 Project Structure

```
Guvi_ML/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── models/               # Trained ML models
│   ├── xgboost_clf.pkl
│   ├── xgboost_reg.pkl
│   └── feature_columns.pkl
├── utils/                # Utility modules
│   ├── preprocessing.py
│   └── mlflow_utils.py
├── data/                 # Dataset (optional)
│   └── emi_prediction_dataset.csv
└── mlruns/              # MLflow experiment data
```

## 📱 Responsive Design

The application is fully responsive and optimized for:
- **Desktop** (> 1200px)
- **Tablet** (768px - 1200px)
- **Mobile** (< 768px)

## 🔒 Error Handling

The application includes comprehensive error handling for:
- Model loading failures
- Missing data files
- Invalid user inputs
- MLflow connection issues
- Network errors

## 📊 Model Information

- **Classification Model**: XGBoost Classifier
  - Purpose: EMI Eligibility Prediction
  - Output: 3 classes (Eligible, High Risk, Not Eligible)
  
- **Regression Model**: XGBoost Regressor
  - Purpose: Maximum EMI Amount Prediction
  - Output: Continuous value (₹)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Developer

**Sumathi S**
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/sumathisaravanan/)

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit**
