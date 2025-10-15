#!/usr/bin/env python3
"""
Quick test script to verify SageMaker setup is working
Run this before processing your full dataset
"""

import sys
import os
import subprocess
import json

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print('='*60)

def check_mark(success):
    return "✅" if success else "❌"

def test_python():
    """Test Python version"""
    print_header("Testing Python")
    version = sys.version_info
    success = version.major == 3 and version.minor >= 8
    print(f"{check_mark(success)} Python {version.major}.{version.minor}.{version.micro}")
    if not success:
        print("   ⚠️  Python 3.8+ recommended")
    return success

def test_dependencies():
    """Test required Python packages"""
    print_header("Testing Python Dependencies")
    packages = ['pandas', 'pydantic', 'aiohttp', 'tqdm']
    all_good = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Run: pip install {package}")
            all_good = False
    
    return all_good

def test_ollama():
    """Test Ollama installation"""
    print_header("Testing Ollama")
    
    # Check if ollama command exists
    try:
        result = subprocess.run(['which', 'ollama'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
        else:
            print("❌ Ollama not found")
            print("   Run: curl -fsSL https://ollama.com/install.sh | sh")
            return False
    except:
        print("❌ Ollama not found")
        return False
    
    # Check if ollama is running
    try:
        result = subprocess.run(['pgrep', '-f', 'ollama serve'],
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(f"✅ Ollama service running (PID: {result.stdout.strip()})")
        else:
            print("❌ Ollama service not running")
            print("   Run: OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &")
            return False
    except:
        print("⚠️  Could not check if Ollama is running")
    
    # Check if model is available
    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True)
        if 'qwen2.5:7b-instruct-q4_0' in result.stdout:
            print("✅ Model downloaded: qwen2.5:7b-instruct-q4_0")
        else:
            print("❌ Model not found")
            print("   Run: ollama pull qwen2.5:7b-instruct-q4_0")
            return False
    except:
        print("⚠️  Could not check Ollama models")
    
    return True

def test_ollama_api():
    """Test Ollama API connectivity"""
    print_header("Testing Ollama API")
    
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama API responding")
            models = response.json().get('models', [])
            print(f"   Available models: {len(models)}")
            return True
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
            return False
    except ImportError:
        print("⚠️  requests not installed (optional)")
        return True
    except Exception as e:
        print(f"❌ Could not connect to Ollama API: {e}")
        print("   Make sure Ollama is running")
        return False

def test_gpu():
    """Test GPU availability"""
    print_header("Testing GPU")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                               '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print("✅ GPU detected:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
            return True
        else:
            print("ℹ️  No GPU detected - will use CPU")
            print("   For faster processing, use ml.g4dn.xlarge or ml.g5.xlarge")
            return True  # Not an error
    except FileNotFoundError:
        print("ℹ️  No GPU detected - will use CPU")
        print("   For faster processing, use ml.g4dn.xlarge or ml.g5.xlarge")
        return True  # Not an error

def test_sqlite():
    """Test SQLite"""
    print_header("Testing SQLite")
    
    import sqlite3
    try:
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('SELECT sqlite_version()')
        version = cursor.fetchone()[0]
        print(f"✅ SQLite {version}")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ SQLite error: {e}")
        return False

def test_disk_space():
    """Test available disk space"""
    print_header("Testing Disk Space")
    
    try:
        result = subprocess.run(['df', '-h', '/tmp'],
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            header = lines[0]
            data = lines[1].split()
            available = data[3]
            print(f"✅ /tmp available space: {available}")
            
            # Parse available space (rough check)
            if 'G' in available:
                gb = float(available.replace('G', ''))
                if gb < 5:
                    print("   ⚠️  Less than 5GB free - might run out of space")
        return True
    except:
        print("⚠️  Could not check disk space")
        return True

def test_csv_file():
    """Test if CSV file exists"""
    print_header("Testing CSV File")
    
    # Common locations
    locations = [
        '/home/sagemaker-user/contributor_data.csv',
        '/home/sagemaker-user/*.csv',
        '/tmp/*.csv'
    ]
    
    import glob
    found_files = []
    for pattern in locations:
        found_files.extend(glob.glob(pattern))
    
    if found_files:
        print("✅ CSV files found:")
        for f in found_files[:5]:  # Show max 5
            size = os.path.getsize(f) / (1024*1024)  # MB
            print(f"   {f} ({size:.1f} MB)")
        if len(found_files) > 5:
            print(f"   ... and {len(found_files) - 5} more")
        print("\n   Update CSV_FILE_PATH in sagemaker_script.py with one of these paths")
        return True
    else:
        print("❌ No CSV files found")
        print("   Upload your CSV to SageMaker Studio")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" SageMaker Setup Verification")
    print("="*60)
    print("\nThis will test if your SageMaker environment is ready")
    print("to run the Contributor Intelligence Platform.\n")
    
    results = {
        'Python': test_python(),
        'Dependencies': test_dependencies(),
        'Ollama': test_ollama(),
        'Ollama API': test_ollama_api(),
        'GPU': test_gpu(),
        'SQLite': test_sqlite(),
        'Disk Space': test_disk_space(),
        'CSV File': test_csv_file()
    }
    
    # Summary
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        print(f"{check_mark(result)} {test}")
    
    print(f"\n{'='*60}")
    print(f"Tests Passed: {passed}/{total}")
    print('='*60)
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run:")
        print("   python sagemaker_script.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before continuing.")
        print("   See SAGEMAKER_SETUP.md for help")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

