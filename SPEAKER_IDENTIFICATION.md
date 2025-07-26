# Speaker Identification System

Simple voice recognition training and testing for the Jarvis AI Assistant.

## Quick Start Guide

### Training New Speakers

**Option 1: Live Recording (Recommended)**
```bash
# Train with default settings (3 samples, 5 seconds each)
python simple_train.py --name "John" --live

# Custom settings (5 samples, 3 seconds each)
python simple_train.py --name "Sarah" --live --samples 5 --duration 3
```

**Option 2: From Audio Files**
```bash
# Train from WAV files
python simple_train.py --name "Mike" --files recording1.wav recording2.wav recording3.wav
```

### Testing Recognition

```bash
# Test identification (will identify you as one of the trained speakers)
python test_speaker_identification.py --test

# List all enrolled speakers
python test_speaker_identification.py --list
```

## Audio Setup

The system is configured to use your **Scarlett 2i2 4th Gen** USB audio interface (device index 3).

**Current Audio Settings:**
- Sample Rate: 16kHz
- Channels: 2 (stereo, automatically converted to mono)
- Input Device: Scarlett 2i2 4th Gen (device index 3)
- Confidence Threshold: 0.40

## Training Tips

### For Best Results:
1. **Speak clearly** - Use normal conversation volume and pace
2. **Consistent environment** - Train in the same room you'll use Jarvis
3. **Multiple samples** - 3-5 samples provide good accuracy
4. **Varied content** - Say different phrases for each sample
5. **Good audio quality** - Ensure your USB interface is working properly

### Sample Recording Tips:
- **Sample 1**: "Hello Jarvis, this is [Name]. How are you today?"
- **Sample 2**: "Hey Jarvis, can you help me with something?"  
- **Sample 3**: "Jarvis, what's the weather like today?"

## Management Commands

```bash
# Training and Management
python simple_train.py --name "PersonName" --live    # Train new person
python simple_train.py --list                        # List all speakers
python simple_train.py --delete "PersonName"         # Delete a speaker

# Testing and Verification
python test_speaker_identification.py --test         # Test identification
python test_speaker_identification.py --list         # List enrolled users
python test_speaker_identification.py --stats "name" # Get speaker statistics
```

## Example Workflow

```bash
# 1. Train yourself
python simple_train.py --name "Alex" --live --samples 3

# 2. Train family members
python simple_train.py --name "Sarah" --live --samples 3
python simple_train.py --name "Kids" --live --samples 5

# 3. Test the system
python test_speaker_identification.py --test

# 4. Use with Jarvis
python jarvis_assistant.py
# Now Jarvis will recognize who is speaking!
```

## How It Works

The system uses SpeechBrain's ECAPA-TDNN neural network model to create unique voice profiles (embeddings) for each user. When someone speaks:

1. **Audio Capture**: Records audio directly from your USB interface
2. **Feature Extraction**: Creates a voice embedding using deep learning
3. **Comparison**: Compares against enrolled user profiles using cosine similarity
4. **Identification**: Returns the best match with confidence score
5. **Context Switch**: Jarvis automatically switches to that user's context

## Troubleshooting

### Audio Issues
If you get recording errors:
1. **Check USB interface**: Ensure Scarlett 2i2 is connected and recognized
2. **Test other apps**: Verify it works in GarageBand or Voice Memo first
3. **Check device index**: 
   ```bash
   python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]} - {p.get_device_info_by_index(i)[\"maxInputChannels\"]} inputs') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]; p.terminate()"
   ```

### Low Recognition Confidence
If recognition confidence is low (< 0.5):
1. **Add more training samples**: Run training again to add more samples
2. **Check audio quality**: Ensure consistent recording conditions between training and testing
3. **Adjust threshold**: Edit `config.json` and lower `confidence_threshold` from 0.40 to 0.30
4. **Retrain completely**: Delete the speaker and train from scratch

### Recognition Not Working
1. **Check enrolled speakers**: `python simple_train.py --list`
2. **Verify database**: Look for `speaker_profiles.db` file
3. **Test basic functionality**: `python test_speaker_identification.py --test`
4. **Check audio device**: Ensure device index 3 is your Scarlett interface

## Configuration

Main settings are in `config.json`:

```json
{
  "audio": {
    "input_device_index": 3,
    "sample_rate": 16000,
    "channels": 2
  },
  "speaker_identification": {
    "enabled": true,
    "confidence_threshold": 0.40,
    "enrollment_threshold": 3,
    "backend_preference": "speechbrain"
  }
}
```

### Key Settings:
- **`input_device_index`**: Audio interface device (3 = Scarlett 2i2)
- **`confidence_threshold`**: Minimum confidence for identification (0.40 = 40%)
- **`enrollment_threshold`**: Minimum samples required for enrollment (3)

## Performance Benchmarks

Based on testing with your setup:

- **Confidence scores 0.7+**: Excellent recognition ✅
- **Confidence scores 0.5-0.7**: Good recognition ✅  
- **Confidence scores 0.3-0.5**: Fair recognition ⚠️ (consider retraining)
- **Confidence scores < 0.3**: Poor recognition ❌ (retrain required)

**Your recent test results:**
- Test 1: Alex (0.87 confidence) ✅
- Test 2: Alex (0.89 confidence) ✅  
- Test 3: Alex (0.77 confidence) ✅

## Files Overview

- **`simple_train.py`** - Main training script (live recording and file-based)
- **`test_speaker_identification.py`** - Testing and enrollment verification
- **`config.json`** - Audio configuration and thresholds
- **`speaker_profiles.db`** - SQLite database storing speaker profiles
- **`speech_analysis/speaker_identification.py`** - Core identification system

## Technical Details

### Backend: SpeechBrain ECAPA-TDNN
- **Model**: Pre-trained on VoxCeleb dataset
- **Accuracy**: 98%+ on clean audio
- **Embedding Size**: 192 dimensions
- **Processing Time**: ~100-200ms per identification

### Audio Processing
- **Format**: 16-bit PCM
- **Sample Rate**: 16kHz
- **Channels**: Stereo → Mono conversion
- **Minimum Speech**: 2-3 seconds for reliable identification

### Database
- **Type**: SQLite (`speaker_profiles.db`)
- **Storage**: Voice embeddings + metadata
- **Privacy**: All processing local, no external data transmission

## Integration with Jarvis

Once speakers are trained, the main Jarvis system automatically:
- ✅ **Identifies speakers** in real-time during conversation
- ✅ **Personalizes responses** based on identified user
- ✅ **Switches context** to user-specific preferences
- ✅ **Logs interactions** per user for history tracking

## Advanced Usage

### Batch Training from Directory
Organize files like this:
```
voice_data/
├── Alex/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── sample3.wav
└── Sarah/
    ├── recording1.wav
    └── recording2.wav
```

Then run:
```bash
python simple_train.py --batch-dir voice_data/
```

### Programmatic Usage
```python
from speech_analysis.speaker_identification import SpeakerIdentificationSystem

# Initialize
speaker_system = SpeakerIdentificationSystem()

# Enroll user
success = speaker_system.enroll_user("john", "John", audio_data, 16000)

# Identify speaker
result = speaker_system.identify_speaker(audio_data, 16000)
if result.user_id:
    print(f"Identified: {result.user_id} (confidence: {result.confidence:.2f})")
```

## Getting Help

If you encounter issues:

1. **Check this documentation** for common solutions
2. **Test basic functionality** with the test scripts
3. **Verify audio setup** with other applications first
4. **Check logs** in the console output for specific error messages

The system is designed to be simple and reliable - most issues are related to audio device configuration or insufficient training data.