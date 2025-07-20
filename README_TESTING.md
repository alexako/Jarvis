# Jarvis Testing Guide

This document describes how to run and understand the Jarvis test suite.

## Quick Start

```bash
# Run critical tests only (AI provider functionality)
python run_tests.py --critical-only

# Run all tests
python run_tests.py

# Run tests with results saved to file
python run_tests.py --save-results results.json

# Run quietly (minimal output)
python run_tests.py --critical-only --quiet
```

## Test Suites

### Critical Tests (Core AI Provider Functionality)
These tests verify the core AI provider integrations:

#### DeepSeek Tests
1. **DeepSeek Flag Unit Tests** (`test_deepseek_flag.py`)
   - Command line flag parsing (`--use-deepseek`)
   - Mutual exclusivity validation
   - Help text verification

2. **DeepSeek Integration Tests** (`test_deepseek_integration.py`)
   - DeepSeek brain initialization
   - API configuration and connectivity
   - Live API functionality
   - Error handling

3. **DeepSeek End-to-End Tests** (`test_deepseek_e2e.py`)
   - Complete system integration
   - Environment validation
   - Command line interface testing

#### Anthropic Tests
4. **Anthropic Flag Unit Tests** (`test_anthropic_flag.py`)
   - Command line flag parsing (`--use-anthropic`)
   - Default provider behavior
   - Mutual exclusivity validation

5. **Anthropic Integration Tests** (`test_anthropic_integration.py`)
   - Anthropic Claude brain initialization
   - API configuration and connectivity
   - Live API functionality with Jarvis personality
   - Provider priority management

6. **Anthropic End-to-End Tests** (`test_anthropic_e2e.py`)
   - Complete system integration
   - Performance testing
   - Default behavior validation

### Additional Tests (Optional)
These tests cover other system components:

4. **Speech-to-Text Tests** (`test_stt.py`)
5. **Text-to-Speech Tests** (`test_pyttsx_tts.py`)
6. **Whisper STT Tests** (`test_whisper_stt.py`)
7. **Force Transcribe Tests** (`force_transcribe_test.py`)

## Prerequisites

### Environment Setup
```bash
# Ensure virtual environment is active
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys
For AI provider tests to run successfully, you need:
```bash
# For DeepSeek tests
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For Anthropic tests  
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Note: You need at least one API key for the critical tests to pass. If both are available, the tests will verify provider switching functionality.

## Test Runner Features

- **Environment Validation**: Checks Python version, virtual environment, and API keys
- **Comprehensive Reporting**: Detailed pass/fail statistics and timing
- **Timeout Protection**: Tests timeout after 2 minutes to prevent hangs
- **JSON Export**: Save results for CI/CD integration
- **Critical Test Focus**: Run only essential tests for faster feedback

## Understanding Test Results

### Success Indicators
- ✅ **PASSED**: Test suite completed successfully
- 📈 **100.0% Success Rate**: All tests passed

### Failure Indicators
- ❌ **FAILED**: One or more tests failed
- 💥 **ERROR**: Test encountered an unexpected error
- ⏰ **TIMEOUT**: Test took longer than 2 minutes
- ❓ **MISSING**: Test file not found

### Sample Output
```
📊 TEST SUMMARY
============================================================
⏱️  Total Duration: 15.59s
🧪 Total Tests: 25
✅ Passed: 25
❌ Failed: 0
💥 Errors: 0
📈 Success Rate: 100.0%
```

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Fast critical tests for pull requests
python run_tests.py --critical-only --quiet --save-results ci_results.json

# Full test suite for main branch
python run_tests.py --save-results full_results.json
```

Exit codes:
- `0`: All tests passed
- `1`: One or more tests failed
- `130`: Interrupted by user (Ctrl+C)

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Found**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **DeepSeek API Key Missing**
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   ```

3. **Test Timeouts**
   - Check internet connectivity for API tests
   - Ensure system has sufficient resources

4. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration

### Debug Mode
For detailed debugging, run individual tests:
```bash
# Run specific test with verbose output
python tests/test_deepseek_flag.py -v

# Check test file directly
python -m unittest tests.test_deepseek_integration -v
```

## Test Development

### Adding New Tests
1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Add to `run_tests.py` test_suites list
4. Mark as critical if essential for core functionality

### Test Structure
```python
import unittest

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_feature_works(self):
        """Test that feature works correctly"""
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

## Performance Benchmarks

### Expected Test Durations
- **DeepSeek Flag Tests**: ~1-2 seconds
- **DeepSeek Integration Tests**: ~8-10 seconds (includes live API calls)
- **DeepSeek E2E Tests**: ~4-6 seconds
- **Anthropic Flag Tests**: ~1-2 seconds
- **Anthropic Integration Tests**: ~3-5 seconds (includes live API calls)
- **Anthropic E2E Tests**: ~4-6 seconds
- **Total Critical Tests**: ~25-35 seconds

### Resource Requirements
- **Memory**: ~100MB for test runner
- **Network**: Required for DeepSeek and Anthropic API tests
- **Storage**: ~1MB for test result files