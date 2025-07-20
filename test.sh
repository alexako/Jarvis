#!/bin/bash
# Simple test runner for Jarvis AI Provider functionality

set -e  # Exit on any error

echo "🚀 Running Jarvis AI Provider Tests"
echo "===================================="

# Activate virtual environment
source venv/bin/activate

# Run DeepSeek tests
echo "🧪 Running DeepSeek Flag Tests..."
python tests/test_deepseek_flag.py

echo ""
echo "🧪 Running DeepSeek Integration Tests..."
python tests/test_deepseek_integration.py

echo ""
echo "🧪 Running DeepSeek E2E Tests..."
python tests/test_deepseek_e2e.py

# Run Anthropic tests
echo ""
echo "🧪 Running Anthropic Flag Tests..."
python tests/test_anthropic_flag.py

echo ""
echo "🧪 Running Anthropic Integration Tests..."
python tests/test_anthropic_integration.py

echo ""
echo "🧪 Running Anthropic E2E Tests..."
python tests/test_anthropic_e2e.py

echo ""
echo "✅ All AI provider tests completed successfully!"
echo "🎉 Both DeepSeek and Anthropic integrations are working properly!"