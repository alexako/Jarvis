# 🤖 Jarvis Voice Assistant

A comprehensive voice assistant system combining Speech-to-Text (STT), Text-to-Speech (TTS), and AI-powered responses, inspired by the Jarvis AI from Iron Man.

## 🧠 Architecture Overview

Jarvis consists of two main components:

1. **Jarvis Core** (this repository) - The central AI brain that processes requests and generates responses
2. **Argus** - The "eyes and ears" of the system, handling audio/video capture and basic interaction on edge devices like Raspberry Pi

## 🚀 Quick Start

### Basic Voice Assistant
```bash
# Start the complete voice assistant
python jarvis.py

# With AI features enabled (requires API keys)
python jarvis.py --enable-ai

# Use specific AI provider
python jarvis.py --enable-ai --use-deepseek
python jarvis.py --enable-ai --use-anthropic  # default

# Use local AI only (private, offline, no API keys required)
python jarvis.py --use-local

# Use enhanced neural TTS for natural voice
python jarvis.py --tts-engine piper

# Use different TTS engines
python jarvis.py --tts-engine pyttsx3
python jarvis.py --tts-engine system
```

### Demo Mode
```bash
python demo_jarvis.py
# Choose option 1 for TTS-only demo
# Choose option 2 for full voice interaction
```

### Testing
```bash
# Run all critical tests
python run_tests.py --critical-only

# Quick test suite
python quick_test.py

# Simple test runner
./test.sh
```

## 🎯 About Argus

**Argus** is the codename for the distributed "eyes and ears" component of the Jarvis system. It's designed to run on edge devices like Raspberry Pi equipped with microphones and cameras, serving as the sensory input layer for Jarvis.

### Role of Argus:
- **Audio Capture**: Listens for voice commands through USB microphones
- **Voice Activity Detection**: Detects when someone is speaking to avoid constant processing
- **Basic Command Processing**: Handles simple, pre-programmed responses locally
- **Video Capture**: Captures video from AI cameras for motion detection and security
- **Network Communication**: Sends complex requests to the main Jarvis server for processing
- **Response Playback**: Converts text responses to speech and plays them through speakers

### Key Features of Argus:
- Lightweight design optimized for Raspberry Pi
- Offline capability for basic functions
- Modular architecture for easy extension
- Text-to-Speech using pyttsx3 for local responses
- Motion detection for security applications
- Secure communication with Jarvis core

The Argus component can be found in its own repository: [github.com/alexako/Argus](https://github.com/alexako/Argus)

## ✨ Features

### 🎤 Speech-to-Text (STT)
- **Whisper STT**: High-accuracy speech recognition using OpenAI Whisper
- **Wake Word Detection**: Responds to "Jarvis" or "Hey Jarvis"
- **Voice Activity Detection**: Smart silence detection with adaptive thresholds
- **Background Noise Adaptation**: Automatically adjusts to ambient sound levels
- **Real-time Processing**: Continuous listening with callback-based processing

### 🔊 Text-to-Speech (TTS)
- **Piper Neural TTS**: High-quality neural TTS with natural British voice (default)
- **pyttsx3 Engine**: Reliable cross-platform TTS fallback
- **Coqui TTS**: Advanced neural TTS with voice cloning capabilities
- **System TTS**: Native OS text-to-speech (macOS `say` command)
- **British Voice**: Authentic British accent for Jarvis personality
- **Personality Enhancement**: Adds formal "sir" addressing and contextual responses

### 🤖 AI Provider Support
- **Triple AI Integration**: Support for DeepSeek, Anthropic Claude, and Local Llama
- **Provider Selection**: Choose primary AI provider via command line flags
- **Fallback System**: Automatic fallback between providers for reliability
- **Smart Prioritization**: Configure primary and secondary AI providers
- **Local AI**: Private, offline processing with Llama 3.2 (no API keys required)

#### AI Provider Flags
```bash
--enable-ai              # Enable cloud AI features (Claude/DeepSeek)
--use-anthropic          # Use Anthropic Claude as primary (default)
--use-deepseek          # Use DeepSeek as primary provider
--use-local             # Use local Llama 3.2 as primary (private, offline)
--disable-local-llm     # Disable local LLM support entirely
--tts-engine ENGINE     # TTS engine: piper (default), pyttsx3, coqui, system
```

### 🎯 Command System (54 Commands)

#### 🕒 Time & Date
- `"time"`, `"what time"`, `"current time"`, `"what's the time"`
- `"date"`, `"what date"`, `"today's date"`, `"what's the date"`, `"what day"`

#### 👋 Greetings
- `"hello"`, `"hi"`, `"good morning"`, `"good afternoon"`, `"good evening"`, `"hey"`

#### 🔧 System Status & Info
- `"how are you"`, `"status"`, `"system status"`, `"are you okay"`
- `"battery"` - Real battery percentage and charging status
- `"memory"` - System memory information
- `"disk space"` - Disk usage information

#### 🎭 Entertainment
- `"tell me a joke"`, `"joke"`, `"something funny"`
- Includes programming and tech humor

#### ℹ️ Information & Help
- `"who are you"`, `"what are you"`, `"introduce yourself"`
- `"help"`, `"what can you do"`, `"commands"`, `"capabilities"`
- `"weather"`, `"what's the weather"`, `"temperature"` (placeholder)

#### 🎛️ System Control
- `"stop listening"`, `"sleep"`, `"pause"` - Temporary deactivation
- `"shutdown"`, `"exit"`, `"quit"`, `"turn off"` - Complete shutdown

#### 👋 Farewells
- `"goodbye"`, `"bye"`, `"see you later"`, `"farewell"`, `"good night"`

#### 🧪 Testing
- `"test"`, `"test voice"`, `"test system"`

## 🧪 Testing Infrastructure

### Comprehensive Test Suite
- **62 Total Tests** across 6 test suites
- **DeepSeek Tests**: 25 tests (unit + integration + E2E)
- **Anthropic Tests**: 37 tests (unit + integration + E2E)
- **100% Pass Rate**: All tests validated and working

### Test Runners

#### Full-Featured Runner
```bash
# Run critical tests only
python run_tests.py --critical-only

# Run all tests with detailed reporting
python run_tests.py

# Save results to JSON
python run_tests.py --save-results results.json

# Quiet mode
python run_tests.py --critical-only --quiet
```

#### Quick Test Runner
```bash
# Fast essential tests (~30-40 seconds)
python quick_test.py
```

#### Simple Bash Runner
```bash
# Sequential test execution
./test.sh
```

### Test Coverage
- **Unit Tests**: Flag parsing, configuration, defaults
- **Integration Tests**: API connectivity, live functionality, error handling
- **End-to-End Tests**: Complete system integration, environment validation

## 📋 Installation & Requirements

### System Requirements
- **Python 3.8+**
- **Microphone access**
- **Audio output (speakers/headphones)**

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `openai-whisper` - STT engine
- `pyttsx3` - TTS engine
- `pyaudio` - Audio I/O
- `numpy` - Audio processing
- `anthropic` - Anthropic Claude API (optional)
- `openai` - DeepSeek API (optional)

### API Keys (Optional - for cloud AI features)
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### Local AI Setup (Optional - for offline AI features)
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.2 model
ollama pull llama3.2:latest

# Verify installation
ollama run llama3.2:latest "Hello"
```

### Enhanced TTS Setup (Optional - for natural voice)
```bash
# Install Piper TTS (if not already installed)
pip install piper-tts

# Download British voice model (auto-downloaded on first use)
mkdir -p ~/.local/share/piper/models
cd ~/.local/share/piper/models
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json

# Test Piper voice
echo "Good afternoon, sir." | piper -m ~/.local/share/piper/models/en_GB-alan-medium.onnx -f test.wav && afplay test.wav
```

## 🎯 Usage Examples

### Basic Voice Interaction
```
1. Start: python jarvis.py
2. Activate: Say "Jarvis"
3. Jarvis: "Yes, sir. How may I assist you?"
4. Command: "What time is it?"
5. Jarvis: "The current time is 5:30 PM, sir."
```

### AI-Powered Responses
```bash
# Start with AI enabled
python jarvis.py --enable-ai --use-deepseek

# Say "Jarvis" then ask complex questions
User: "Explain quantum computing"
Jarvis: "Quantum computing harnesses quantum mechanics..."
```

### System Information
```
User: "Battery status"
Jarvis: "Battery is at 44 percent and charging, sir."

User: "Tell me a joke"
Jarvis: "Why don't scientists trust atoms, sir? Because they make up everything."
```

## 🏗️ Architecture

### Core Components
- `src/core/jarvis_assistant.py` - Main voice assistant with AI integration
- `src/commands/commands.py` - Centralized command processing (54 commands)
- `src/ai/ai_brain.py` - AI provider management and brain classes
- `src/audio/` - STT and TTS engine implementations
- `src/context/` - Context and memory management
- `src/utils/` - Utility functions and configuration management

### AI Brain System
- **AnthropicBrain**: Claude integration with Jarvis personality
- **DeepSeekBrain**: DeepSeek integration with OpenAI-compatible API
- **AIBrainManager**: Provider prioritization and fallback management
- **Configurable**: Easy switching between providers and fallback settings

### Testing Structure
```
tests/
├── test_deepseek_flag.py      # DeepSeek CLI flag tests
├── test_deepseek_integration.py # DeepSeek API tests
├── test_deepseek_e2e.py       # DeepSeek end-to-end tests
├── test_anthropic_flag.py     # Anthropic CLI flag tests
├── test_anthropic_integration.py # Anthropic API tests
├── test_anthropic_e2e.py      # Anthropic end-to-end tests
└── ...                        # Additional component tests
```

## 🔧 Development

### Running Tests
```bash
# Critical AI provider tests
python run_tests.py --critical-only

# All tests with coverage
python run_tests.py

# Individual test suites
python tests/test_deepseek_integration.py
python tests/test_anthropic_flag.py
```

### Adding New Commands
1. Edit `commands.py`
2. Add command patterns to appropriate category
3. Implement response logic
4. Test with voice input

### Adding New AI Providers
1. Create new brain class in `ai_brain.py`
2. Implement `process_request()` and `is_healthy()` methods
3. Add provider to `BrainProvider` enum
4. Update configuration in `create_ai_config()`
5. Add corresponding tests

## 📊 Performance

### Audio Configuration
- **STT**: 16kHz sample rate, 1024 chunk size
- **TTS**: British English, 180 WPM, 90% volume
- **Latency**: <500ms response time for voice commands
- **Accuracy**: High-accuracy Whisper STT with noise adaptation

### Test Performance
- **Critical Tests**: ~25-35 seconds
- **Full Test Suite**: ~45-60 seconds
- **Memory Usage**: ~100MB during testing
- **API Response**: <30 seconds for AI queries

## 📋 TODO

### High Priority
- [ ] **Fix providers endpoint** - Currently only returning Anthropic and DeepSeek, missing local models
- [ ] **UI version compatibility warning** - Address version compatibility warnings in the UI interface
- [ ] **Dynamic local model detection** - Enhance providers endpoint to include available Ollama models (create script to check downloaded models)

### Future Enhancements
- [ ] Add real-time model switching capability
- [ ] Implement model performance monitoring
- [ ] Add support for additional local model providers
- [ ] Create automated model management system

## 🎉 Status: FULLY OPERATIONAL

The Jarvis Voice Assistant is a complete, production-ready system featuring:

✅ **Speech Recognition** - Whisper-powered STT with wake word detection  
✅ **Voice Responses** - Natural British-accented TTS  
✅ **54 Voice Commands** - Comprehensive command system  
✅ **Dual AI Integration** - DeepSeek and Anthropic Claude support  
✅ **62 Test Coverage** - Comprehensive testing infrastructure  
✅ **Professional Personality** - Formal "sir" addressing and contextual responses  
✅ **System Integration** - Real macOS system information and control  
✅ **Extensible Architecture** - Easy addition of new commands and AI providers  

Ready for use as a complete, AI-powered voice assistant system!

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Add tests**: Ensure new functionality has corresponding tests
4. **Run test suite**: `python run_tests.py` - ensure all tests pass
5. **Commit changes**: Follow existing commit message style
6. **Create pull request**: Include test results and feature description

### Testing Requirements
- All new features must include unit tests
- AI provider changes require integration tests
- CLI changes need end-to-end tests
- Maintain 100% test pass rate

## 📄 License

This project is open source and available under the MIT License.