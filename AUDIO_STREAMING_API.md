# Jarvis Audio Streaming API

This document describes the new audio streaming functionality added to the Jarvis REST API.

## Overview

The audio streaming feature allows real-time streaming of Jarvis's synthesized speech responses, providing a more interactive and responsive user experience compared to traditional file-based audio delivery.

## New Endpoints

### 1. Enhanced Chat Endpoint with Streaming Support

**Endpoint:** `POST /chat`

**New Parameters:**
- `stream_audio` (boolean): When `true` and `use_tts` is also `true`, provides a streaming audio URL instead of a static file URL.

**Enhanced Response:**
```json
{
  "response": "Jarvis's text response",
  "provider_used": "AI provider name",
  "processing_time": 0.123,
  "audio_url": "/audio/response_file.wav",     // Traditional file URL (when stream_audio=false)
  "stream_url": "/audio/stream/request_id",    // Streaming URL (when stream_audio=true)
  "request_id": "unique_request_id",
  "current_user": "user_name"
}
```

### 2. Audio Stream Response Endpoint

**Endpoint:** `GET /audio/stream/{request_id}`

**Description:** Streams the audio response for a specific chat request in real-time chunks.

**Features:**
- Chunked streaming (4KB chunks by default)
- WAV format audio
- Proper HTTP headers for streaming
- Automatic cleanup after streaming completes

**Example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello Jarvis", "use_tts": true, "stream_audio": true}'

# Response includes stream_url: "/audio/stream/abc123..."

curl http://localhost:8000/audio/stream/abc123... --output response.wav
```

### 3. Direct Audio Streaming Endpoint

**Endpoint:** `POST /audio/stream`

**Description:** Create a direct audio stream for any text without going through the chat system.

**Request Body:**
```json
{
  "text": "Text to convert to speech",
  "chunk_size": 4096,           // Optional: Chunk size in bytes (default: 4096)
  "format": "wav",              // Optional: Audio format (default: "wav")
  "quality": "medium"           // Optional: Audio quality (default: "medium")
}
```

**Response:** Streaming audio data with proper headers

**Example:**
```bash
curl -X POST http://localhost:8000/audio/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Direct streaming test", "chunk_size": 8192}' \
  --output direct_stream.wav
```

## Implementation Details

### Audio Format
- **Format:** WAV (RIFF)
- **Sample Rate:** 22,050 Hz
- **Channels:** Mono (1 channel)
- **Bit Depth:** 16-bit PCM
- **Chunk Size:** Configurable (default 4KB)

### Streaming Behavior
- Audio is generated completely first, then streamed in chunks
- Small delays between chunks simulate real-time streaming
- Streams include proper HTTP headers for browser compatibility
- CORS headers included for web application support

### Request Lifecycle
1. Chat request with `stream_audio: true` creates a streaming endpoint
2. Request metadata is stored temporarily in `active_requests`
3. Client accesses the stream URL to receive audio chunks
4. Request is cleaned up automatically after streaming completes

## Error Handling

### Common Error Responses

**404 - Stream Not Found:**
```json
{
  "error": "Audio stream not found or expired",
  "code": "HTTP_404"
}
```

**503 - TTS Service Unavailable:**
```json
{
  "error": "TTS service not available",
  "code": "HTTP_503"
}
```

## Usage Examples

### JavaScript Web Client
```javascript
// Request streaming audio
const response = await fetch('/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello Jarvis',
    use_tts: true,
    stream_audio: true
  })
});

const data = await response.json();

// Stream the audio
if (data.stream_url) {
  const audio = new Audio();
  audio.src = data.stream_url;
  audio.play();
}
```

### Python Client
```python
import requests

# Request streaming audio
response = requests.post('http://localhost:8000/chat', json={
    'text': 'Hello Jarvis',
    'use_tts': True,
    'stream_audio': True
})

data = response.json()

# Download the stream
if data['stream_url']:
    stream_response = requests.get(f"http://localhost:8000{data['stream_url']}")
    with open('jarvis_response.wav', 'wb') as f:
        f.write(stream_response.content)
```

## Testing

A comprehensive test suite is provided in `test_audio_streaming.py` which validates:

1. **Health Check:** Verifies API server and TTS engine are operational
2. **Regular Chat:** Tests traditional audio file generation
3. **Streaming Chat:** Tests streaming audio URL generation
4. **Audio Stream:** Tests actual streaming download and audio verification
5. **Direct Stream:** Tests direct audio streaming endpoint

Run tests with:
```bash
python test_audio_streaming.py
```

## Performance Considerations

- Audio generation happens synchronously before streaming begins
- Memory usage scales with audio length (entire audio held in memory during streaming)
- Concurrent streams are supported but limited by system resources
- Request cleanup prevents memory leaks from abandoned streams

## Future Enhancements

Potential improvements for future versions:

1. **True Real-time Synthesis:** Generate audio chunks as they're requested rather than pre-generating
2. **Compression Support:** Add MP3 and other compressed format support
3. **Quality Options:** Implement actual quality levels (low/medium/high)
4. **Streaming Protocol Upgrades:** Support for WebSockets or Server-Sent Events
5. **Caching:** Cache frequently requested audio for better performance

## Compatibility

- **HTTP Version:** HTTP/1.1 and HTTP/2
- **Browsers:** All modern browsers supporting streaming responses
- **Mobile:** Compatible with mobile applications and web views
- **Command Line:** Full curl and wget support