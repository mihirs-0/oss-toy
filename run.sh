#!/bin/bash
echo "🚚 Starting Supply Chain Intelligence Demo..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "requirements_installed.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch requirements_installed.txt
fi

# Launch the app
echo "Launching Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 