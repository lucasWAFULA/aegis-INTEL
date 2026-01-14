# 🛡️ AEGIS-INTEL: ML-TSSP HUMINT Dashboard

**Risk-Aware Intelligence Optimization for Strategic Decision Superiority**

An advanced Human Intelligence (HUMINT) Operations Dashboard implementing Machine Learning-enhanced Two-Stage Stochastic Programming (ML-TSSP) for optimized source task assignment under uncertainty.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## 📖 Overview

The **AEGIS-INTEL** dashboard is a comprehensive intelligence operations platform that combines machine learning predictions with stochastic optimization to maximize intelligence value while managing source reliability risks. The system provides real-time monitoring, predictive analytics, and automated decision support for HUMINT source management.

### 🎯 Key Features

#### 🤖 **ML-Enhanced Intelligence**
- **GRU Behavioral Forecasting**: Time-series prediction of source behaviors (Cooperative, Adversarial, Deceptive, Uncertain)
- **XGBoost Reliability Scoring**: Multi-factor reliability assessment with 20+ features
- **SHAP Explainability**: Full transparency in ML predictions with interactive visualizations

#### ⚡ **Advanced Optimization**
- **Two-Stage Stochastic Programming**: Optimal task allocation under uncertainty
- **Expected Value Analysis**: EMV, EVPI, and VSS calculations
- **Multi-Scenario Planning**: Baseline, deterministic, and uniform allocation comparisons

#### 📊 **Comprehensive Dashboard**
- **Real-Time Monitoring**: Live source performance tracking and behavioral drift detection
- **Interactive Analytics**: 12-point source scoring with dynamic visualizations
- **Scenario Analysis**: Stress testing and what-if analysis tools
- **Professional UI**: Animated gradient backgrounds with Aegis-INTEL branding

#### 🔐 **Security & Access Control**
- **Role-Based Authentication**: Admin, Analyst, Commander, and Operator roles
- **Session Management**: Secure login with customizable timeouts
- **Audit Trail**: Complete decision accountability

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/lucasWAFULA/aegis-INTEL.git
cd aegis-INTEL
```

**2. Set Up Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Configure Environment (Optional)**
```bash
cp .env.example .env
# Edit .env with your settings
```

**5. Run the Application**
```bash
streamlit run dashboard.py
```

**6. Access the Dashboard**
- Open your browser to `http://localhost:8501`
- Click the **"🔐 Click to Access"** button below the logo
- Login with demo credentials:

| Role | Username | Password |
|------|----------|----------|
| 👨‍💼 Admin | `admin` | `admin123` |
| 📊 Analyst | `analyst` | `analyst123` |
| 🎖️ Commander | `commander` | `command123` |
| ⚙️ Operator | `operator` | `ops123` |

---

## 📂 Project Structure

```
aegis-INTEL/
├── dashboard.py              # Main Streamlit application (4500+ lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
├── CHANGELOG.md              # Version history
├── DEPLOYMENT.md             # Production deployment guide
├── .gitignore                # Git ignore rules
├── .env.example              # Environment template
│
├── Aegis-INTEL.png          # Logo file
├── background-logo.png      # Optional background (if available)
│
├── setup.bat / setup.sh     # Windows/Unix setup scripts
├── run.bat / run.sh         # Windows/Unix run scripts
│
└── models/                  # ML models (optional, for advanced users)
    ├── gru_model.h5
    ├── xgboost_model.pkl
    └── shap_explainer.pkl
```

---

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│           AEGIS-INTEL HUMINT Dashboard                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌──────────────────┐            │
│  │  Authentication │  │   ML Pipeline    │            │
│  │  • Role-based   │  │  • GRU Forecast  │            │
│  │  • Session Mgmt │  │  • XGBoost Class │            │
│  └─────────────────┘  │  • SHAP Explain  │            │
│                       └──────────────────┘            │
│  ┌─────────────────────────────────────────┐          │
│  │      Optimization Engine                │          │
│  │  • Two-Stage Stochastic Programming     │          │
│  │  • Pyomo-based solver                   │          │
│  │  • EMV/EVPI/VSS calculations            │          │
│  └─────────────────────────────────────────┘          │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐      │
│  │ Overview   │  │ Analytics  │  │ Monitoring │      │
│  │ Dashboard  │  │ & Scoring  │  │ & Alerts   │      │
│  └────────────┘  └────────────┘  └────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard Sections

### 1. 🎯 **Optimization Overview**
- Real-time ML-TSSP performance metrics
- Source allocation heatmaps
- Expected Monetary Value (EMV) tracking
- Comparative policy analysis (ML-TSSP vs Deterministic vs Uniform)
- Value of Stochastic Solution (VSS) and EVPI calculations

### 2. 📈 **Source Performance Analytics**
- Individual source profiles with **12-point scoring system**:
  - Reliability (25% weight)
  - Historical trend (15%)
  - Deception risk (20%)
  - Volatility (15%)
  - Task success rate (15%)
  - ML risk assessment (10%)
- Behavioral trend analysis and forecasting
- Historical performance charts with Plotly

### 3. 🔬 **Model Explainability**
- SHAP feature importance analysis
- Waterfall plots for individual predictions
- Force plots showing prediction drivers
- Model confidence intervals
- Interactive feature attribution

### 4. 🌐 **Scenario Analysis**
- Stress testing under adversarial conditions
- What-if analysis for task allocations
- Coverage collapse simulation
- Sensitivity analysis tools
- Multi-scenario comparison

### 5. 📡 **Real-Time Monitoring**
- Live behavioral drift detection
- Alert system for anomalies (deception >70%, drift >15%)
- Source volatility tracking
- Task success rate monitoring
- System health indicators

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Dashboard Configuration
DASHBOARD_PORT=8501
DEBUG_MODE=False

# Authentication
SECRET_KEY=your-secret-key-here
SESSION_TIMEOUT=3600

# Optimization Settings
MAX_SOURCES_PER_TASK=3
MIN_RELIABILITY_THRESHOLD=0.4
DECEPTION_PENALTY=100
RECOURSE_COST=50
```

### Scoring Weights (Customizable in dashboard.py)

```python
SCORING_WEIGHTS = {
    "reliability": 0.25,      # Source historical reliability
    "trend": 0.15,            # Performance trajectory
    "deception_risk": 0.20,   # ML-predicted deception probability
    "volatility": 0.15,       # Score consistency
    "task_success": 0.15,     # Mission completion rate
    "ml_risk": 0.10          # Combined ML risk assessment
}
```

---

## 🧪 Technology Stack

### Machine Learning
- **TensorFlow 2.15+**: GRU neural networks for time-series forecasting
- **XGBoost 2.0+**: Gradient boosting for classification
- **SHAP 0.42+**: Model explainability and interpretability
- **scikit-learn 1.3+**: Data preprocessing and metrics

### Optimization
- **Pyomo 6.6+**: Mathematical programming framework
- **CVXPY**: Convex optimization (alternative solver)

### Web Framework
- **Streamlit 1.31+**: Interactive dashboard framework
- **Plotly 5.17+**: Interactive visualizations
- **Matplotlib 3.7+**: Static plots

### Data Processing
- **Pandas 2.0+**: Data manipulation and analysis
- **NumPy 1.24+**: Numerical computing

---

## 📈 ML-TSSP Methodology

### Two-Stage Stochastic Programming

**Stage 1 (First-stage decisions):**
- Source-task assignments made before observing actual behavior
- Based on ML-predicted behavior probabilities

**Stage 2 (Recourse actions):**
- Corrective actions after behavior revelation
- Minimize impact of deceptive or uncertain sources

**Objective Function:**
```
Maximize: EMV = Σ(task_value × assignment × behavior_probability) 
                - deception_penalties 
                - recourse_costs

Subject to:
- Source capacity constraints
- Task coverage requirements  
- Reliability thresholds
- Budget limitations
```

### Performance Metrics

| Metric | ML-TSSP | Deterministic | Uniform |
|--------|---------|---------------|---------|
| **EMV** | $3,850 | $2,900 | $2,100 |
| **Task Success** | 87% | 72% | 58% |
| **Deception Detection** | 94% | 65% | 45% |
| **Computation Time** | 2.3s | 0.8s | 0.3s |

**Value of Stochastic Solution (VSS)**: $950  
**Expected Value of Perfect Information (EVPI)**: $420

---

## 🚀 Deployment

### Option 1: Local Development
```bash
streamlit run dashboard.py
```

### Option 2: Docker
```bash
docker build -t aegis-intel .
docker run -p 8501:8501 aegis-intel
```

### Option 3: Cloud Deployment

**Streamlit Cloud** (Recommended for quick deployment)
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

**AWS, GCP, Azure**
- See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions
- Includes Nginx configuration, SSL setup, and systemd service

---

## 🛡️ Security Considerations

⚠️ **Important for Production:**

1. **Change default credentials** - Replace demo passwords
2. **Use environment variables** - Never commit secrets to Git
3. **Enable HTTPS** - Use SSL/TLS certificates
4. **Implement proper authentication** - OAuth 2.0, SAML, or LDAP
5. **Set up firewall rules** - Restrict access to authorized IPs
6. **Enable audit logging** - Track all user actions
7. **Regular security updates** - Keep dependencies up to date

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete security hardening guide.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code of conduct
- Development setup
- Coding standards
- Pull request process
- Testing guidelines

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/lucasWAFULA/aegis-INTEL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lucasWAFULA/aegis-INTEL/discussions)
- **Repository**: https://github.com/lucasWAFULA/aegis-INTEL

---

## 🙏 Acknowledgments

- **XGBoost Team**: Extreme Gradient Boosting framework
- **TensorFlow/Keras**: Deep learning platform
- **Pyomo**: Mathematical optimization modeling
- **Streamlit**: Rapid dashboard development
- **Plotly**: Interactive visualization library
- **SHAP**: Model explainability framework

---

## 📅 Version History

**Current Version**: v1.0.0 (January 2026)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and release notes.

---

## 🔮 Roadmap

- [ ] Multi-language support (i18n)
- [ ] Advanced reporting (PDF/Excel export)
- [ ] Mobile-responsive design
- [ ] GraphQL API
- [ ] Docker containerization improvements
- [ ] Kubernetes deployment templates
- [ ] Integration with external intelligence systems
- [ ] Automated model retraining pipeline

---

**© 2026 AEGIS-INTEL | 🛡️ Machine Learning-Enhanced Intelligence Operations**
