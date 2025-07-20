#!/usr/bin/env python3
"""
Quick test runner for Jarvis DeepSeek functionality
Focused on essential tests with fast execution
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test(test_name, test_file):
    """Run a single test and return success status"""
    print(f"🧪 {test_name}...")
    
    project_root = Path(__file__).parent
    venv_python = project_root / "venv" / "bin" / "python"
    test_path = project_root / "tests" / test_file
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            str(venv_python), str(test_path)
        ], capture_output=True, text=True, timeout=60)
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        # Count tests from output
        tests_run = 0
        for line in result.stdout.split('\n'):
            if 'Ran ' in line and 'test' in line:
                try:
                    tests_run = int(line.split()[1])
                    break
                except (ValueError, IndexError):
                    pass
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} ({tests_run} tests, {duration:.1f}s)")
        
        if not success and result.stderr:
            print(f"   Error: {result.stderr.split(chr(10))[0][:80]}...")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        return False


def main():
    """Run essential AI provider tests"""
    print("🚀 Quick AI Provider Test Suite")
    print("=" * 40)
    
    # Essential tests only
    tests = [
        ("DeepSeek Flags", "test_deepseek_flag.py"),
        ("DeepSeek Integration", "test_deepseek_integration.py"),
        ("DeepSeek E2E", "test_deepseek_e2e.py"),
        ("Anthropic Flags", "test_anthropic_flag.py"),
        ("Anthropic Integration", "test_anthropic_integration.py"),
        ("Anthropic E2E", "test_anthropic_e2e.py")
    ]
    
    start_time = time.time()
    passed = 0
    total = len(tests)
    
    for test_name, test_file in tests:
        if run_test(test_name, test_file):
            passed += 1
        print()
    
    # Summary
    duration = time.time() - start_time
    success_rate = (passed / total) * 100
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} passed ({success_rate:.0f}%)")
    print(f"⏱️  Duration: {duration:.1f}s")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(130)