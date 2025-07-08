#!/usr/bin/env python3
"""
Setup script for the Chat Session Explorer Dashboard.
Installs dependencies and launches the dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing required dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_data_files():
    """Check if data files exist."""
    data_dir = Path("data")
    session_files = list(data_dir.glob("session_detection_results_*.xlsx"))
    
    if not session_files:
        print("⚠️  Warning: No session detection result files found in data/ directory")
        print("   Make sure you have .xlsx files with session data")
        return False
    
    print(f"✅ Found {len(session_files)} session data files")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Chat Session Explorer Dashboard\n")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your environment.")
        sys.exit(1)
    
    # Check data files
    check_data_files()
    
    print("\n" + "="*60)
    print("🎉 Setup complete! Your dashboard is ready to launch.")
    print("="*60)
    print("\n🚀 To start the dashboard, run:")
    print("   python run_dashboard.py")
    print("\n📖 For more information, see DASHBOARD_README.md")
    
    # Ask if they want to launch now
    response = input("\n❓ Would you like to launch the dashboard now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n🚀 Launching dashboard...")
        try:
            import run_dashboard
            run_dashboard.main()
        except Exception as e:
            print(f"❌ Error launching dashboard: {e}")
            print("💡 Try running manually: python run_dashboard.py")

if __name__ == "__main__":
    main() 