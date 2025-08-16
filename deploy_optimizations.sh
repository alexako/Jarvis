#!/bin/bash
# Deployment script to update Jarvis with latest optimizations

echo "🚀 Updating Jarvis with latest performance optimizations..."

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root or with sudo" 
   exit 1
fi

# Define source and destination directories
SOURCE_DIR="/Users/alex/Code/jarvis"
DEST_DIR="/opt/jarvis"

echo "📁 Source directory: $SOURCE_DIR"
echo "📂 Destination directory: $DEST_DIR"

# Backup current files
echo "📦 Creating backup of current files..."
cp "$DEST_DIR/speech_analysis/stt.py" "$DEST_DIR/speech_analysis/stt.py.backup.$(date +%s)" 2>/dev/null || echo "No existing STT file to backup"
cp "$DEST_DIR/commands.py" "$DEST_DIR/commands.py.backup.$(date +%s)" 2>/dev/null || echo "No existing commands file to backup"
cp "$DEST_DIR/jarvis_context.py" "$DEST_DIR/jarvis_context.py.backup.$(date +%s)" 2>/dev/null || echo "No existing context file to backup"
cp "$DEST_DIR/jarvis_assistant.py" "$DEST_DIR/jarvis_assistant.py.backup.$(date +%s)" 2>/dev/null || echo "No existing assistant file to backup"
cp "$DEST_DIR/speech_analysis/tts.py" "$DEST_DIR/speech_analysis/tts.py.backup.$(date +%s)" 2>/dev/null || echo "No existing TTS file to backup"
cp "$DEST_DIR/config_manager.py" "$DEST_DIR/config_manager.py.backup.$(date +%s)" 2>/dev/null || echo "No existing config manager file to backup"

# Copy optimized files
echo "🔄 Copying optimized files..."
cp "$SOURCE_DIR/src/audio/speech_analysis/stt.py" "$DEST_DIR/speech_analysis/stt.py"
cp "$SOURCE_DIR/src/commands/commands.py" "$DEST_DIR/commands.py"
cp "$SOURCE_DIR/src/context/jarvis_context.py" "$DEST_DIR/jarvis_context.py"
cp "$SOURCE_DIR/src/core/jarvis_assistant.py" "$DEST_DIR/jarvis_assistant.py"
cp "$SOURCE_DIR/src/audio/speech_analysis/tts.py" "$DEST_DIR/speech_analysis/tts.py"
cp "$SOURCE_DIR/src/utils/config_manager.py" "$DEST_DIR/config_manager.py"

# Restart the service
echo "🔄 Restarting Jarvis API service..."
launchctl unload /Library/LaunchDaemons/com.jarvis.api.plist 2>/dev/null
sleep 2
launchctl load /Library/LaunchDaemons/com.jarvis.api.plist

echo "✅ Jarvis has been updated with the latest optimizations!"
echo "📊 Performance improvements include:"
echo "  • Lazy module imports for faster startup"
echo "  • Optimized audio processing algorithms"
echo "  • Improved command processing efficiency"
echo "  • Better memory and resource management"
echo "  • Enhanced TTS processing"
echo "  • Optimized health checks"
echo ""
echo "The API service has been restarted with these improvements."