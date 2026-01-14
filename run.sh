#!/bin/bash

# ML-TSSP HUMINT Dashboard - Run Script
# Quick launcher for the dashboard

echo "🛡️  Starting ML-TSSP HUMINT Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run ./setup.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dashboard.py exists
if [ ! -f "dashboard.py" ]; then
    echo "❌ dashboard.py not found!"
    echo "   Make sure you're in the correct directory"
    exit 1
fi

# Run the dashboard
echo "🚀 Launching dashboard on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""
streamlit run dashboard.py
