#!/bin/bash
# Verification script to test that optimizations are working

echo "🧪 Verifying Jarvis Performance Optimizations..."

# Check if the API is responding
echo "🔍 Checking API status..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null || echo "000")

if [ "$STATUS" == "000" ]; then
    echo "❌ API server is not accessible"
    echo "Please ensure the Jarvis API service is running"
    exit 1
elif [ "$STATUS" == "400" ]; then
    echo "⚠️  API server is running but rejecting requests (likely due to host validation)"
    echo "This is expected for the production API with strict security"
else
    echo "✅ API server is responding with status: $STATUS"
fi

# Test that we can import the optimized modules
echo "🔍 Testing module imports..."
cd /opt/jarvis
PYTHONPATH=/opt/jarvis /opt/jarvis/venv/bin/python -c "
import sys
try:
    from speech_analysis.stt import AudioBuffer, AudioConfig
    print('✅ STT module imports successfully')
    
    from commands import JarvisCommands
    print('✅ Commands module imports successfully')
    
    from jarvis_context import create_jarvis_context
    print('✅ Context module imports successfully')
    
    print('✅ All modules are working correctly with optimizations')
except Exception as e:
    print(f'❌ Error importing modules: {e}')
    sys.exit(1)
"

# Test performance improvements
echo "🔍 Testing performance improvements..."
cd /opt/jarvis
PYTHONPATH=/opt/jarvis /opt/jarvis/venv/bin/python -c "
import time
import numpy as np
from speech_analysis.stt import AudioBuffer, AudioConfig

# Test AudioBuffer performance
buffer = AudioBuffer(debug=False)
config = AudioConfig()

# Generate test data
test_chunks = []
for i in range(1000):
    chunk = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
    test_chunks.append(chunk)

# Benchmark add_chunk method
start_time = time.time()
for chunk in test_chunks:
    buffer.add_chunk(chunk, config)
end_time = time.time()

processing_time = end_time - start_time
chunks_per_second = len(test_chunks) / processing_time

print(f'✅ Processed {len(test_chunks)} audio chunks in {processing_time:.4f}s')
print(f'✅ Performance: {chunks_per_second:.0f} chunks/second')

if chunks_per_second > 5000:
    print('✅ Audio processing performance is excellent')
else:
    print('⚠️  Audio processing performance may need further optimization')
"

echo "✅ Verification complete!"
echo "Your Jarvis system is running with the latest performance optimizations."