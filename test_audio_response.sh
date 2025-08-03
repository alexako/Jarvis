#!/bin/bash
# Quick test script for Jarvis audio responses
# Usage: ./test_audio_response.sh "Your message here"

set -e

# Configuration
API_URL="http://localhost:8000"
OUTPUT_DIR="./audio_responses"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_api() {
    log_info "Checking if Jarvis API is running..."
    if ! curl -s "$API_URL/health" > /dev/null; then
        log_error "Jarvis API is not running at $API_URL"
        log_info "Start it with: python jarvis_api.py --host 0.0.0.0 --port 8000"
        exit 1
    fi
    log_success "API is running"
}

get_audio_response() {
    local message="$1"
    local output_file="$2"
    
    log_info "Sending message: '$message'"
    
    # Send request to Jarvis
    local response=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$message\", \"use_tts\": true}")
    
    # Check if request was successful
    if [[ -z "$response" ]]; then
        log_error "No response from API"
        return 1
    fi
    
    # Extract audio URL and text response
    local audio_url=$(echo "$response" | jq -r '.audio_url // empty')
    local text_response=$(echo "$response" | jq -r '.response // empty')
    local provider=$(echo "$response" | jq -r '.provider_used // empty')
    local processing_time=$(echo "$response" | jq -r '.processing_time // empty')
    
    if [[ -z "$audio_url" || "$audio_url" == "null" ]]; then
        log_error "No audio URL in response"
        echo "Response: $response"
        return 1
    fi
    
    log_success "Text response: $text_response"
    log_info "Provider: $provider, Processing time: ${processing_time}s"
    log_info "Audio URL: $audio_url"
    
    # Download audio file
    log_info "Downloading audio..."
    if curl -s "$API_URL$audio_url" -o "$output_file"; then
        log_success "Audio saved to: $output_file"
        return 0
    else
        log_error "Failed to download audio"
        return 1
    fi
}

play_audio() {
    local audio_file="$1"
    
    if command -v afplay >/dev/null 2>&1; then
        # macOS
        log_info "Playing audio with afplay..."
        afplay "$audio_file"
    elif command -v aplay >/dev/null 2>&1; then
        # Linux
        log_info "Playing audio with aplay..."
        aplay "$audio_file"
    elif command -v paplay >/dev/null 2>&1; then
        # PulseAudio
        log_info "Playing audio with paplay..."
        paplay "$audio_file"
    else
        log_warning "No audio player found. Audio saved to: $audio_file"
        log_info "You can play it manually with your preferred audio player"
    fi
}

test_streaming() {
    local message="$1"
    
    log_info "Testing streaming audio for: '$message'"
    
    local response=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$message\", \"stream_audio\": true}")
    
    local stream_url=$(echo "$response" | jq -r '.stream_url // empty')
    local text_response=$(echo "$response" | jq -r '.response // empty')
    
    if [[ -n "$stream_url" && "$stream_url" != "null" ]]; then
        log_success "Stream response: $text_response"
        log_info "Stream URL: $stream_url"
        log_info "You can stream with: curl '$API_URL$stream_url'"
    else
        log_error "No stream URL in response"
    fi
}

main() {
    echo "🎵 Jarvis Audio Response Tester"
    echo "=============================="
    
    # Get message from command line or prompt
    local message="$1"
    if [[ -z "$message" ]]; then
        echo -n "Enter message for Jarvis: "
        read -r message
    fi
    
    if [[ -z "$message" ]]; then
        log_error "No message provided"
        echo "Usage: $0 \"Your message here\""
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if API is running
    check_api
    
    # Generate output filename
    local safe_message=$(echo "$message" | tr ' ' '_' | tr -cd '[:alnum:]_' | cut -c1-20)
    local output_file="$OUTPUT_DIR/${TIMESTAMP}_${safe_message}.wav"
    
    echo
    log_info "=== Testing Standard Audio Response ==="
    if get_audio_response "$message" "$output_file"; then
        echo
        log_info "=== Playing Audio ==="
        play_audio "$output_file"
        
        echo
        log_info "=== File Info ==="
        if command -v file >/dev/null 2>&1; then
            file "$output_file"
        fi
        if command -v du >/dev/null 2>&1; then
            echo "File size: $(du -h "$output_file" | cut -f1)"
        fi
    fi
    
    echo
    log_info "=== Testing Streaming Audio ==="
    test_streaming "$message"
    
    echo
    log_success "Test completed!"
    echo "Audio files saved in: $OUTPUT_DIR"
}

# Check dependencies
if ! command -v curl >/dev/null 2>&1; then
    log_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    log_error "jq is required but not installed"
    log_info "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    exit 1
fi

# Run main function
main "$@"