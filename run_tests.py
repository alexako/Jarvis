#!/usr/bin/env python3
"""
Comprehensive test runner for Jarvis Assistant
Runs all test suites with detailed reporting and coverage
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TestRunner:
    """Test runner with comprehensive reporting and coverage"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.venv_python = self.project_root / "venv" / "bin" / "python"
        
        # Test suites to run
        self.test_suites = [
            {
                "name": "DeepSeek Flag Unit Tests",
                "file": "test_deepseek_flag.py",
                "description": "Command line flag parsing and validation for DeepSeek",
                "critical": True
            },
            {
                "name": "DeepSeek Integration Tests", 
                "file": "test_deepseek_integration.py",
                "description": "DeepSeek API integration and functionality",
                "critical": True
            },
            {
                "name": "DeepSeek End-to-End Tests",
                "file": "test_deepseek_e2e.py", 
                "description": "Complete system integration with DeepSeek",
                "critical": True
            },
            {
                "name": "Anthropic Flag Unit Tests",
                "file": "test_anthropic_flag.py",
                "description": "Command line flag parsing and validation for Anthropic",
                "critical": True
            },
            {
                "name": "Anthropic Integration Tests",
                "file": "test_anthropic_integration.py",
                "description": "Anthropic Claude API integration and functionality",
                "critical": True
            },
            {
                "name": "Anthropic End-to-End Tests",
                "file": "test_anthropic_e2e.py",
                "description": "Complete system integration with Anthropic Claude",
                "critical": True
            },
            {
                "name": "Speech-to-Text Tests",
                "file": "test_stt.py",
                "description": "STT engine functionality",
                "critical": False
            },
            {
                "name": "Text-to-Speech Tests",
                "file": "test_pyttsx_tts.py",
                "description": "TTS engine functionality", 
                "critical": False
            },
            {
                "name": "Whisper STT Tests",
                "file": "test_whisper_stt.py",
                "description": "Whisper-specific STT functionality",
                "critical": False
            },
            {
                "name": "Force Transcribe Tests",
                "file": "force_transcribe_test.py",
                "description": "Audio transcription functionality",
                "critical": False
            }
        ]
        
        # Test results tracking
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
    
    def check_environment(self) -> bool:
        """Check if the test environment is properly set up"""
        print("🔍 Checking test environment...")
        
        # Check if virtual environment exists
        if not self.venv_python.exists():
            print(f"❌ Virtual environment not found at {self.venv_python}")
            print("   Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
            return False
        
        # Check if tests directory exists
        if not self.tests_dir.exists():
            print(f"❌ Tests directory not found at {self.tests_dir}")
            return False
        
        # Check for AI provider API keys
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if deepseek_key:
            print(f"✅ DeepSeek API key found (starts with: {deepseek_key[:10]}...)")
        else:
            print("⚠️  DeepSeek API key not found - DeepSeek tests may be skipped")
            
        if anthropic_key:
            print(f"✅ Anthropic API key found (starts with: {anthropic_key[:10]}...)")
        else:
            print("⚠️  Anthropic API key not found - Anthropic tests may be skipped")
            
        if not deepseek_key and not anthropic_key:
            print("❌ No AI provider API keys found - critical tests will fail")
            return False
        
        # Check Python version
        try:
            result = subprocess.run([str(self.venv_python), "--version"], 
                                 capture_output=True, text=True)
            print(f"✅ Python version: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Failed to check Python version: {e}")
            return False
        
        print("✅ Environment check complete\n")
        return True
    
    def run_single_test(self, test_suite: Dict) -> Tuple[bool, Dict]:
        """Run a single test suite and return results"""
        test_file = self.tests_dir / test_suite["file"]
        
        if not test_file.exists():
            return False, {
                "status": "MISSING",
                "stdout": "",
                "stderr": f"Test file not found: {test_file}",
                "duration": 0,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
        
        print(f"🧪 Running {test_suite['name']}...")
        print(f"   📝 {test_suite['description']}")
        
        start_time = time.time()
        
        try:
            # Run the test with verbose output
            result = subprocess.run([
                str(self.venv_python), str(test_file)
            ], 
            capture_output=True, 
            text=True, 
            cwd=str(self.project_root),
            timeout=120  # 2 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse test results from output
            stdout = result.stdout
            stderr = result.stderr
            
            # Count tests from unittest output
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_errors = 0
            
            # Parse unittest summary line
            for line in (stdout + stderr).split('\n'):
                if 'Ran ' in line and 'test' in line:
                    # Example: "Ran 7 tests in 1.090s"
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            tests_run = int(parts[1])
                        except ValueError:
                            pass
                
                if 'FAILED' in line and '=' in line:
                    # Example: "FAILED (failures=1, errors=2)"
                    if 'failures=' in line:
                        try:
                            failures_part = line.split('failures=')[1].split(',')[0].split(')')[0]
                            tests_failed = int(failures_part)
                        except (ValueError, IndexError):
                            pass
                    if 'errors=' in line:
                        try:
                            errors_part = line.split('errors=')[1].split(',')[0].split(')')[0]
                            tests_errors = int(errors_part)
                        except (ValueError, IndexError):
                            pass
            
            # If no failures/errors mentioned, assume all passed
            if tests_run > 0 and tests_failed == 0 and tests_errors == 0:
                tests_passed = tests_run
            else:
                tests_passed = tests_run - tests_failed - tests_errors
            
            success = result.returncode == 0
            status = "PASSED" if success else "FAILED"
            
            print(f"   {'✅' if success else '❌'} {status} - {tests_run} tests, {tests_passed} passed, {tests_failed + tests_errors} failed")
            print(f"   ⏱️  Duration: {duration:.2f}s\n")
            
            return success, {
                "status": status,
                "stdout": stdout,
                "stderr": stderr, 
                "duration": duration,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_errors": tests_errors
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ TIMEOUT after {duration:.1f}s\n")
            return False, {
                "status": "TIMEOUT",
                "stdout": "",
                "stderr": "Test timed out after 120 seconds",
                "duration": duration,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_errors": 0
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"   💥 ERROR: {e}\n")
            return False, {
                "status": "ERROR", 
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_errors": 0
            }
    
    def run_all_tests(self, critical_only: bool = False) -> bool:
        """Run all test suites"""
        self.start_time = time.time()
        
        print("🚀 Starting Jarvis Test Suite")
        print("=" * 60)
        
        if not self.check_environment():
            return False
        
        suites_to_run = [s for s in self.test_suites if not critical_only or s.get("critical", False)]
        
        print(f"📋 Running {len(suites_to_run)} test suite(s)...")
        if critical_only:
            print("   (Critical tests only)")
        print()
        
        overall_success = True
        
        for test_suite in suites_to_run:
            success, result = self.run_single_test(test_suite)
            self.results[test_suite["name"]] = result
            
            if not success:
                overall_success = False
                if test_suite.get("critical", False):
                    print(f"❌ CRITICAL TEST FAILED: {test_suite['name']}")
            
            # Update totals
            self.total_tests += result["tests_run"]
            self.passed_tests += result["tests_passed"] 
            self.failed_tests += result["tests_failed"]
            self.error_tests += result.get("tests_errors", 0)
        
        return overall_success
    
    def print_summary(self):
        """Print comprehensive test summary"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Overall stats
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"🧪 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"💥 Errors: {self.error_tests}")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print()
        
        # Per-suite breakdown
        print("📋 TEST SUITE BREAKDOWN:")
        print("-" * 40)
        
        for suite_name, result in self.results.items():
            status_icon = {
                "PASSED": "✅",
                "FAILED": "❌", 
                "ERROR": "💥",
                "TIMEOUT": "⏰",
                "MISSING": "❓"
            }.get(result["status"], "❓")
            
            print(f"{status_icon} {suite_name}")
            print(f"   Status: {result['status']}")
            print(f"   Tests: {result['tests_run']} run, {result['tests_passed']} passed")
            print(f"   Duration: {result['duration']:.2f}s")
            
            if result["status"] != "PASSED" and result["stderr"]:
                print(f"   Error: {result['stderr'][:100]}...")
            print()
        
        # Overall result
        overall_status = "PASSED" if self.failed_tests + self.error_tests == 0 and self.total_tests > 0 else "FAILED"
        print("=" * 60)
        print(f"🎯 OVERALL RESULT: {overall_status}")
        print("=" * 60)
    
    def save_results(self, output_file: Optional[str] = None):
        """Save test results to JSON file"""
        if not output_file:
            output_file = self.project_root / "test_results.json"
        
        results_data = {
            "timestamp": time.time(),
            "total_duration": time.time() - self.start_time if self.start_time else 0,
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "error_tests": self.error_tests,
                "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            "results": self.results
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"📄 Results saved to {output_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Jarvis test suite")
    parser.add_argument("--critical-only", action="store_true", 
                       help="Run only critical tests (DeepSeek functionality)")
    parser.add_argument("--save-results", type=str, metavar="FILE",
                       help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        success = runner.run_all_tests(critical_only=args.critical_only)
        
        if not args.quiet:
            runner.print_summary()
        
        if args.save_results:
            runner.save_results(args.save_results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()