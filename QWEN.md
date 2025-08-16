# Jarvis Voice Assistant - Project Context

## Project Overview

Jarvis is a comprehensive voice assistant system combining Speech-to-Text (STT), Text-to-Speech (TTS), and AI-powered responses, inspired by the Jarvis AI from Iron Man. It features a modular architecture with two main components:

1. **Jarvis Core** - The central AI brain that processes requests and generates responses
2. **Argus** - The "eyes and ears" component (separate repository) handling audio/video capture on edge devices

## Architecture

### Core Components

The project follows a modular architecture with these main directories:

- `src/core/` - Main voice assistant implementation
- `src/ai/` - AI brain providers (Anthropic Claude, DeepSeek, Local Llama)
- `src/audio/` - STT and TTS engine implementations
- `src/commands/` - Centralized command processing system
- `src/context/` - Context and memory management with multi-user support
- `src/utils/` - Utility functions and configuration management
- `tests/` - Comprehensive test suite with 62 tests across 6 test suites

### Key Features

1. **Speech-to-Text (STT)**
   - Whisper STT for high-accuracy speech recognition
   - Wake word detection ("Jarvis" or "Hey Jarvis")
   - Voice activity detection with adaptive thresholds
   - Real-time processing with callback-based processing

2. **Text-to-Speech (TTS)**
   - Piper Neural TTS with natural British voice (default)
   - pyttsx3 engine as fallback
   - Coqui TTS with voice cloning capabilities
   - System TTS (macOS `say` command)

3. **AI Provider Support**
   - Triple AI integration: DeepSeek, Anthropic Claude, and Local Llama
   - Configurable provider selection with command line flags
   - Automatic fallback between providers for reliability
   - Smart prioritization with health checks

4. **Multi-User Context System**
   - Persistent SQLite-based memory storage
   - User profiles with aliases and preferences
   - Conversation history tracking
   - Context-aware responses using previous interactions

5. **Command System (54 Commands)**
   - Time & Date queries
   - Greetings and Farewells
   - System Status & Info (battery, memory, disk space)
   - Entertainment (jokes)
   - Help and Identity queries
   - System Control (stop listening, shutdown)
   - Memory and Context management
   - Multi-user commands (switch user, list users)

## Development Workflow

### Running the Assistant

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
```

### Testing

```bash
# Run all critical tests
python run_tests.py --critical-only

# Run all tests with detailed reporting
python run_tests.py

# Save results to JSON
python run_tests.py --save-results results.json

# Quiet mode
python run_tests.py --critical-only --quiet
```

### AI Provider Flags

```bash
--enable-ai              # Enable cloud AI features (Claude/DeepSeek)
--use-anthropic          # Use Anthropic Claude as primary (default)
--use-deepseek          # Use DeepSeek as primary provider
--use-local             # Use local Llama 3.2 as primary (private, offline)
--disable-local-llm     # Disable local LLM support entirely
--tts-engine ENGINE     # TTS engine: piper (default), pyttsx3, coqui, system
```

## Code Structure

### Main Entry Points

- `jarvis.py` - Primary entry point for the voice assistant
- `src/core/jarvis_assistant.py` - EnhancedJarvisAssistant class with main logic
- `src/commands/commands.py` - JarvisCommands class with command processing
- `src/ai/ai_brain.py` - AIBrainManager with provider orchestration
- `src/context/jarvis_context.py` - JarvisContext with memory management
- `src/audio/speech_analysis/stt.py` - Speech-to-text engines
- `src/audio/speech_analysis/tts.py` - Text-to-speech engines

### AI Providers

1. **AnthropicBrain** - Claude integration with Jarvis personality
2. **DeepSeekBrain** - DeepSeek integration with OpenAI-compatible API
3. **LocalBrain** - Ollama-powered local LLM for private, offline processing

### Testing Infrastructure

62 total tests across 6 test suites:
- DeepSeek tests (25 tests): unit + integration + E2E
- Anthropic tests (37 tests): unit + integration + E2E
- Additional component tests for STT/TTS functionality

Test runners:
- `run_tests.py` - Full-featured test runner with reporting
- `quick_test.py` - Fast essential tests (~30-45 seconds)
- `test.sh` - Simple bash test runner

## Dependencies

Key packages:
- `openai-whisper` - STT engine
- `pyttsx3` - TTS engine
- `pyaudio` - Audio I/O
- `numpy` - Audio processing
- `anthropic` - Anthropic Claude API (optional)
- `openai` - DeepSeek API (optional)
- `ollama` - Local LLM support (optional)
- `piper-tts` - Neural TTS engine (optional)
- `psutil` - System monitoring

## API Keys (Optional)

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

## Local AI Setup (Optional)

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.2 model
ollama pull llama3.2:latest

# Verify installation
ollama run llama3.2:latest "Hello"
```

## Enhanced TTS Setup (Optional)

```bash
# Install Piper TTS
pip install piper-tts

# Download British voice model (auto-downloaded on first use)
mkdir -p ~/.local/share/piper/models
cd ~/.local/share/piper/models
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json

# Test Piper voice
echo "Good afternoon, sir." | piper -m ~/.local/share/piper/models/en_GB-alan-medium.onnx -f test.wav && afplay test.wav
```

## Development Guidelines

### Adding New Commands

1. Edit `src/commands/commands.py`
2. Add command patterns to appropriate category in `command_mappings`
3. Implement response logic in a handler method
4. Test with voice input

### Adding New AI Providers

1. Create new brain class in `src/ai/ai_brain.py` inheriting from `BaseBrain`
2. Implement `process_request()` and `is_healthy()` methods
3. Add provider to `BrainProvider` enum
4. Update configuration in `create_ai_config()`
5. Add corresponding tests

### Testing Requirements

- All new features must include unit tests
- AI provider changes require integration tests
- CLI changes need end-to-end tests
- Maintain 100% test pass rate

### Code Conventions

- Follow existing code style and patterns
- Add logging for important operations
- Handle exceptions gracefully with user-friendly messages
- Use type hints for function parameters and return values
- Document public methods with docstrings

## Performance Considerations

- STT: 16kHz sample rate, 1024 chunk size
- TTS: British English, 180 WPM, 90% volume
- Latency: <500ms response time for voice commands
- Critical Tests: ~25-35 seconds
- Full Test Suite: ~45-60 seconds
- API Response: <30 seconds for AI queries

## Project Status

✅ FULLY OPERATIONAL - The Jarvis Voice Assistant is a complete, production-ready system featuring:
- Speech Recognition with wake word detection
- Natural British-accented TTS
- 54 Voice Commands
- Dual AI Integration (DeepSeek and Anthropic Claude)
- 62 Test Coverage
- Professional Personality with "sir" addressing
- Real macOS system information and control
- Extensible Architecture