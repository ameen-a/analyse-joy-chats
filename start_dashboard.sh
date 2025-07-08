#!/bin/bash

# Script to start the Chat Session Explorer dashboard
# This ensures the virtual environment is properly activated

echo "🚀 Starting Chat Session Explorer dashboard..."
echo "📊 The dashboard will open in your web browser"
echo "🔗 If it doesn't open automatically, go to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Please install dependencies first:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if dashboard file exists
DASHBOARD_FILE="$SCRIPT_DIR/src/streamlit_dashboard.py"
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "❌ Dashboard file not found at $DASHBOARD_FILE"
    exit 1
fi

echo "🐍 Using virtual environment Python: $(which python)"
echo "🎨 Starting with dark theme..."
echo ""

# Run streamlit with proper configuration
streamlit run "$DASHBOARD_FILE" \
    --theme.base dark \
    --theme.primaryColor "#ff6b6b" \
    --theme.backgroundColor "#0f0f0f" \
    --theme.secondaryBackgroundColor "#1e1e1e" \
    --theme.textColor "#ffffff" \
    --server.headless false \
    --server.port 8501 \
    --browser.serverAddress localhost 