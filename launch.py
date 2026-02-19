#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                🛡️ UNIFIED SECURITY AI PLATFORM LAUNCHER 🛡️
═══════════════════════════════════════════════════════════════════════════════

This script checks everything and launches the platform automatically.

Usage:
    python launch.py              # Launch with auto-check
    python launch.py --setup      # Setup only (install + train)
    python launch.py --skip-check # Skip checks and launch directly
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    banner = f"""{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     🛡️  UNIFIED SECURITY AI PLATFORM LAUNCHER 🛡️               ║
║                                                                   ║
║     10 ML Models • Malware + Fraud Detection • 100% Local        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def check_python_version():
    """Check if Python version is 3.8+"""
    print(f"\n{Colors.BLUE}[1/5] Checking Python version...{Colors.ENDC}")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"{Colors.RED}❌ Python 3.8+ required. Found: {version.major}.{version.minor}{Colors.ENDC}")
        return False
    print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro}{Colors.ENDC}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print(f"\n{Colors.BLUE}[2/5] Checking dependencies...{Colors.ENDC}")
    required_packages = [
        'streamlit',
        'sklearn',
        'numpy',
        'pandas',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"{Colors.GREEN}  ✓ {package}{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.RED}  ✗ {package}{Colors.ENDC}")
            missing.append(package)
    
    if missing:
        print(f"\n{Colors.YELLOW}⚠️  Missing packages detected.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Run: pip install -r requirements_unified.txt{Colors.ENDC}")
        return False
    
    print(f"{Colors.GREEN}✅ All core dependencies installed{Colors.ENDC}")
    return True

def check_models():
    """Check if AI models are trained"""
    print(f"\n{Colors.BLUE}[3/5] Checking AI models...{Colors.ENDC}")
    
    # Check malware models
    malware_path = Path("trained_models")
    malware_ready = malware_path.exists() and (malware_path / "stacking_ensemble.pkl").exists()
    
    if malware_ready:
        print(f"{Colors.GREEN}  ✓ Malware detection models{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}  ⚠ Malware models not found{Colors.ENDC}")
        print(f"{Colors.YELLOW}    Run: python train_from_database.py{Colors.ENDC}")
    
    # Check fraud models
    fraud_path = Path("models")
    fraud_ready = fraud_path.exists() and (fraud_path / "metadata.json").exists()
    
    if fraud_ready:
        print(f"{Colors.GREEN}  ✓ Fraud detection models{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}  ⚠ Fraud models not found{Colors.ENDC}")
        print(f"{Colors.YELLOW}    Run: python train_fraud_models.py{Colors.ENDC}")
    
    if not malware_ready or not fraud_ready:
        print(f"\n{Colors.YELLOW}╭{'─'*65}╮{Colors.ENDC}")
        print(f"{Colors.YELLOW}│  Would you like to train the models now? (5-10 minutes)        │{Colors.ENDC}")
        print(f"{Colors.YELLOW}│  [Y]es / [N]o / [S]kip and launch anyway                        │{Colors.ENDC}")
        print(f"{Colors.YELLOW}╰{'─'*65}╯{Colors.ENDC}")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'y':
            train_models(malware_ready, fraud_ready)
        elif choice == 's':
            print(f"{Colors.YELLOW}⚠️  Launching without complete models. Some features may not work.{Colors.ENDC}")
            return True
        else:
            return False
    
    print(f"{Colors.GREEN}✅ All models trained and ready{Colors.ENDC}")
    return True

def train_models(malware_ready, fraud_ready):
    """Train missing models"""
    print(f"\n{Colors.BLUE}[TRAINING] Starting model training...{Colors.ENDC}")
    
    if not malware_ready:
        print(f"\n{Colors.CYAN}Training malware detection models...{Colors.ENDC}")
        result = subprocess.run([sys.executable, "train_from_database.py"], 
                              capture_output=False)
        if result.returncode != 0:
            print(f"{Colors.RED}❌ Malware training failed{Colors.ENDC}")
            return False
    
    if not fraud_ready:
        print(f"\n{Colors.CYAN}Training fraud detection models...{Colors.ENDC}")
        result = subprocess.run([sys.executable, "train_fraud_models.py"], 
                              capture_output=False)
        if result.returncode != 0:
            print(f"{Colors.RED}❌ Fraud training failed{Colors.ENDC}")
            return False
    
    print(f"{Colors.GREEN}✅ Model training complete!{Colors.ENDC}")
    return True

def check_files():
    """Check if required files exist"""
    print(f"\n{Colors.BLUE}[4/5] Checking required files...{Colors.ENDC}")
    
    required_files = [
        ("unified_security_platform.py", "Main application"),
        ("features.py", "Feature engineering"),
        ("fraud_inference.py", "Fraud detection engine"),
    ]
    
    all_exist = True
    for filename, description in required_files:
        if Path(filename).exists():
            print(f"{Colors.GREEN}  ✓ {description}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}  ✗ {filename} - {description}{Colors.ENDC}")
            all_exist = False
    
    if not all_exist:
        print(f"{Colors.RED}❌ Missing required files{Colors.ENDC}")
        return False
    
    print(f"{Colors.GREEN}✅ All required files present{Colors.ENDC}")
    return True

def launch_app():
    """Launch the Streamlit application"""
    print(f"\n{Colors.BLUE}[5/5] Launching application...{Colors.ENDC}")
    print(f"\n{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    print(f"{Colors.GREEN}  🚀 Starting Unified Security AI Platform...{Colors.ENDC}")
    print(f"{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    print(f"\n{Colors.CYAN}  Opening browser at: http://localhost:8501{Colors.ENDC}")
    print(f"{Colors.CYAN}  Press Ctrl+C to stop{Colors.ENDC}\n")
    
    time.sleep(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "unified_security_platform.py",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Shutting down gracefully...{Colors.ENDC}")
        print(f"{Colors.GREEN}✅ Thanks for using Unified Security AI Platform!{Colors.ENDC}\n")

def setup_only():
    """Setup mode - install and train only"""
    print_banner()
    print(f"\n{Colors.BOLD}SETUP MODE{Colors.ENDC}\n")
    
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    print(f"\n{Colors.BLUE}Installing dependencies...{Colors.ENDC}")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", 
        "requirements_unified.txt"
    ])
    
    if result.returncode != 0:
        print(f"{Colors.RED}❌ Installation failed{Colors.ENDC}")
        sys.exit(1)
    
    # Train models
    print(f"\n{Colors.BLUE}Training models...{Colors.ENDC}")
    train_models(False, False)
    
    print(f"\n{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    print(f"{Colors.GREEN}  ✅ Setup complete!{Colors.ENDC}")
    print(f"{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    print(f"\n{Colors.CYAN}Run: python launch.py{Colors.ENDC}")

def main():
    """Main launcher function"""
    # Check for setup flag
    if "--setup" in sys.argv:
        setup_only()
        return
    
    # Check for skip flag
    skip_checks = "--skip-check" in sys.argv
    
    print_banner()
    
    if skip_checks:
        print(f"\n{Colors.YELLOW}⚠️  Skipping checks (--skip-check flag detected){Colors.ENDC}")
        launch_app()
        return
    
    # Run all checks
    print(f"\n{Colors.BOLD}Running system checks...{Colors.ENDC}")
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print(f"\n{Colors.RED}Please install dependencies first:{Colors.ENDC}")
        print(f"{Colors.YELLOW}  pip install -r requirements_unified.txt{Colors.ENDC}")
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    if not check_models():
        print(f"\n{Colors.YELLOW}Models not ready. Exiting.{Colors.ENDC}")
        sys.exit(1)
    
    # All checks passed
    print(f"\n{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    print(f"{Colors.GREEN}  ✅ All checks passed! System ready.{Colors.ENDC}")
    print(f"{Colors.GREEN}{'═'*67}{Colors.ENDC}")
    
    launch_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        sys.exit(1)
