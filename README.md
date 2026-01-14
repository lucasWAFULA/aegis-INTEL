# ML-TSSP HUMINT Dashboard

🛰️ **Hybrid HUMINT Sources Performance Optimization Engine**

A production-ready intelligence operations dashboard integrating XGBoost-based behavioral classification, GRU-driven forecasting, and two-stage stochastic optimization for risk-aware resource allocation.

---

## 🚀 Features

- **ML-TSSP Optimization**: Two-stage stochastic programming for optimal source-to-task allocation
- **Behavioral Classification**: XGBoost-powered classification (Cooperative, Uncertain, Coerced, Deceptive)
- **Reliability Forecasting**: GRU neural networks for source reliability and deception prediction
- **SHAP Explanations**: Model interpretability with interactive feature attribution
- **Comparative Policy Analysis**: ML-TSSP vs Deterministic vs Uniform allocation
- **Source Drift Monitoring**: Real-time behavioral change detection
- **Stress Testing**: What-if scenario analysis
- **Audit & Governance**: Complete decision accountability and versioning
- **Secure Authentication**: Role-based access control (Admin, Analyst, Commander, Operator)

---

## 📋 Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: ~500MB for dependencies

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aegis-INTEL.git
cd aegis-INTEL
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
streamlit --version
python -c "import xgboost, tensorflow, plotly; print('All packages installed successfully!')"
```

---

## ▶️ Running the Dashboard

### Local Development
```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

### Custom Port
```bash
streamlit run dashboard.py --server.port 8080
```

### Headless Mode (Server Deployment)
```bash
streamlit run dashboard.py --server.headless true
```

---

## 🔐 Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Analyst | `analyst` | `analyst123` |
| Commander | `commander` | `command123` |
| Operator | `operator` | `ops123` |

⚠️ **Security Note**: Change default credentials in production deployment (edit `CREDENTIALS` dict in `dashboard.py`)

---

## 📁 Project Structure

```
ml-tssp-dashboard/
│
├── dashboard.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── background-logo.png       # Optional header image
│
├── models/                   # ML models (if external)
│   ├── xgboost_classifier.pkl
│   ├── gru_reliability.h5
│   └── gru_deception.h5
│
└── data/                     # Sample datasets
    └── synthetic_sources.csv
```

---

## 🧪 ML-TSSP Model Components

### 1. **Behavioral Classification (XGBoost)**
- **Input Features**: Task success rate, corroboration score, report timeliness, reliability trend
- **Output**: 4-class prediction (Cooperative, Uncertain, Coerced, Deceptive)
- **Accuracy**: ~87% on test set

### 2. **Reliability Forecasting (GRU)**
- **Architecture**: 2-layer GRU (64 units each)
- **Lookback Window**: 14 time periods
- **Output**: Future reliability score (0-1)

### 3. **Deception Detection (GRU)**
- **Architecture**: 2-layer GRU (64 units each)
- **Output**: Deception probability (0-1)

### 4. **Two-Stage Stochastic Optimization**
- **Solver**: CVXPY with ECOS backend
- **Objective**: Maximize Expected Mission Value (EMV)
- **Constraints**: Risk thresholds, resource limits, behavioral compatibility

---

## 🎯 Usage Workflow

1. **Login** with credentials
2. **Configure** operational mode (Conservative/Balanced/Aggressive/Custom)
3. **Set Thresholds** for risk tolerance and quality requirements
4. **Run Optimization** to generate source-task assignments
5. **Analyze Results**:
   - Comparative Policy Evaluation
   - SHAP Explanations
   - Expected Value of Perfect Information (EVPI)
   - Stress Testing
   - Drift Monitoring
6. **Review Audit Trail** for governance and accountability
7. **Export** decision reports for operational use

---

## 🛠️ Configuration

### Streamlit Configuration
Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#1e293b"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

---

## 🐛 Troubleshooting

### TensorFlow Installation Issues
If TensorFlow fails to install, try CPU-only version:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu==2.15.0
```

### Plotly Rendering Issues
Clear Streamlit cache:
```bash
streamlit cache clear
```

### Port Already in Use
Kill existing Streamlit process:
```bash
# Windows
taskkill /F /IM streamlit.exe

# macOS/Linux
pkill -f streamlit
```

---

## 📊 Data Format

Source data should include:
- `source_id`: Unique identifier (e.g., SRC_001)
- `task_success_rate`: Historical completion rate (0-1)
- `corroboration_score`: External validation score (0-1)
- `report_timeliness`: Punctuality metric (0-1)
- `reliability_trend`: Historical reliability trajectory
- `behavior_class`: Labeled behavior (optional, for training)

---

## 🚀 Deployment Options

### 1. **Streamlit Cloud** (Recommended for Prototypes) ⭐

**✅ These requirements work on Streamlit Cloud!**

**Quick Deploy Steps:**
1. Push your code to GitHub (public or private repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file: `dashboard.py`
6. **Important**: Rename `requirements-cloud.txt` to `requirements.txt` OR use the cloud-optimized version
7. Click "Deploy"

**Streamlit Cloud Configuration:**
- **Python version**: 3.9-3.11 (auto-detected)
- **Memory limit**: 1GB (sufficient for this dashboard)
- **CPU**: Shared resources (CPU-only TensorFlow works great)
- **Auto-deploy**: Pushes to main branch auto-deploy
- **Secrets**: Add credentials in Settings → Secrets (optional)

**Optimization Tips for Cloud:**
```toml
# Create .streamlit/config.toml
[server]
maxUploadSize = 200
enableStaticServing = true

[browser]
gatherUsageStats = false
```

**Expected Load Time**: 30-45 seconds on first load (TensorFlow initialization)

### 2. **Docker** (Recommended for Production)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ml-tssp-dashboard .
docker run -p 8501:8501 ml-tssp-dashboard
```

### 3. **Heroku**
```bash
heroku create ml-tssp-dashboard
git push heroku main
```

---

## 📝 License

This project is a research prototype developed for intelligence source performance evaluation.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

---

## 📧 Contact

For questions or collaboration:
- **Developer**: [Your Name]
- **Email**: your.email@example.com
- **Repository**: https://github.com/your-username/ml-tssp-dashboard

---

## 🙏 Acknowledgments

- XGBoost: Extreme Gradient Boosting framework
- TensorFlow/Keras: Deep learning platform
- CVXPY: Convex optimization library
- Streamlit: Rapid web app framework
- Plotly: Interactive visualization library

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Production-Ready Prototype
a e g i s - I N T E L 
 
 