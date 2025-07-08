#!/usr/bin/env python3
"""
Simple script to run the Chat Session Explorer dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit dashboard."""
    # Change to the src directory where the dashboard is located
    dashboard_path = Path(__file__).parent / "src" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Starting Chat Session Explorer dashboard...")
    print("📊 The dashboard will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard\n")
    
    # Check if we're in a virtual environment
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    if venv_python.exists():
        python_executable = str(venv_python)
        print("🐍 Using virtual environment Python")
    else:
        python_executable = sys.executable
        print("🐍 Using system Python")
    
    try:
        # Run streamlit
        subprocess.run([
            python_executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--theme.base", "dark",
            "--theme.primaryColor", "#ff6b6b",
            "--theme.backgroundColor", "#0f0f0f",
            "--theme.secondaryBackgroundColor", "#1e1e1e",
            "--theme.textColor", "#ffffff"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 