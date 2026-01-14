#!/bin/bash

# ML-TSSP HUMINT Dashboard - Quick Start Script
# This script sets up and runs the dashboard

echo "🛡️  ML-TSSP HUMINT Dashboard Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists, if not create from example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "Creating .env file from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your configuration"
    fi
fi

# Check for required assets
echo ""
echo "Checking for required assets..."
if [ ! -f "Aegis-INTEL.png" ]; then
    echo "⚠️  Warning: Aegis-INTEL.png logo not found"
    echo "   Please add your logo to the project root"
fi

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo ""
echo "To start the dashboard, run:"
echo "  source venv/bin/activate"
echo "  streamlit run dashboard.py"
echo ""
echo "Or simply run: ./run.sh"
echo "=================================="
