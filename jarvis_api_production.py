#!/usr/bin/env python3
"""
Jarvis REST API Server - Production Configuration
Production-ready deployment with enterprise security features
"""

__version__ = "1.5.0-production"

import asyncio
import logging
import time
import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from io import BytesIO

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Production security imports
from production_security import (
    ProductionSecurity, get_current_user, require_permission, 
    SecurityMiddleware, limiter, validate_request_size, check_ip_access,
    log_security_event, cleanup_expired_sessions
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Jarvis components
from ai_brain import AIBrainManager, BrainProvider
from speech_analysis.tts import JarvisTTS
from commands import JarvisCommands, create_ai_config
from jarvis_context import create_jarvis_context

# Configure production logging
handlers = [logging.StreamHandler()]
try:
    # Try to add file handler if directory exists and is writable
    import os
    os.makedirs('/var/log/jarvis', exist_ok=True)
    handlers.append(logging.FileHandler('/var/log/jarvis/api.log'))
except (OSError, PermissionError):
    # Fall back to just console logging if file logging isn't available
    print("Warning: Cannot write to /var/log/jarvis/api.log, using console logging only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Global Jarvis instance
jarvis_brain: Optional[AIBrainManager] = None
jarvis_tts: Optional[JarvisTTS] = None
jarvis_commands: Optional[JarvisCommands] = None
jarvis_context = None
security_config = ProductionSecurity()

# API Models (same as before but with enhanced validation)
class TextRequest(BaseModel):
    """Request model for text interaction with enhanced validation"""
    text: str = Field(..., description="Text input to send to Jarvis", min_length=1, max_length=2000)
    use_tts: bool = Field(default=False, description="Whether to generate audio response")
    stream_audio: bool = Field(default=False, description="Whether to stream audio response in real-time")
    ai_provider: Optional[str] = Field(default=None, description="Preferred AI provider (local, anthropic, deepseek)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the request")
    user: Optional[str] = Field(default=None, description="User identifier for multi-user context (optional)")

    class Config:
        # Prevent additional fields to avoid injection attacks
        extra = "forbid"

class TextResponse(BaseModel):
    """Response model for text interaction"""
    response: str = Field(..., description="Jarvis's text response")
    provider_used: str = Field(..., description="AI provider that handled the request")
    processing_time: float = Field(..., description="Time taken to process request in seconds")
    audio_url: Optional[str] = Field(default=None, description="URL to generated audio file if TTS was requested")
    stream_url: Optional[str] = Field(default=None, description="URL to audio stream if streaming was requested")
    request_id: str = Field(..., description="Unique identifier for this request")
    current_user: Optional[str] = Field(default=None, description="Current active user in context system")

class StreamingAudioRequest(BaseModel):
    """Request model for streaming audio with validation"""
    text: str = Field(..., description="Text to convert to streaming audio", min_length=1, max_length=5000)
    chunk_size: int = Field(default=4096, description="Audio chunk size for streaming (bytes)", ge=1024, le=16384)
    format: str = Field(default="wav", description="Audio format", pattern="^(wav|mp3)$")
    quality: str = Field(default="medium", description="Audio quality", pattern="^(low|medium|high)$")

    class Config:
        extra = "forbid"

class StatusResponse(BaseModel):
    """System status response"""
    version: str = Field(..., description="Jarvis API version")
    status: str = Field(..., description="System status (healthy, degraded, unhealthy)")
    uptime: float = Field(..., description="Server uptime in seconds")
    ai_providers: Dict[str, bool] = Field(..., description="Status of each AI provider")
    tts_engine: str = Field(..., description="Current TTS engine")
    local_mode: bool = Field(..., description="Whether running in local-only mode")

class HealthResponse(BaseModel):
    """Health check response"""
    healthy: bool = Field(..., description="Overall health status")
    components: Dict[str, bool] = Field(..., description="Health status of individual components")
    timestamp: datetime = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    request_id: Optional[str] = Field(default=None, description="Request ID if applicable")

# Global state
server_start_time = time.time()
active_requests: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with production setup"""
    # Startup
    logger.info("🚀 Starting Jarvis API Server (Production Mode)...")
    await initialize_jarvis()
    
    # Start background tasks
    asyncio.create_task(periodic_cleanup())
    
    logger.info("✅ Jarvis API Server ready for production!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Jarvis API Server...")
    await cleanup_jarvis()
    logger.info("✅ Jarvis API Server stopped")

async def periodic_cleanup():
    """Periodic cleanup tasks"""
    while True:
        try:
            cleanup_expired_sessions()
            # Clean up old active requests (older than 1 hour)
            now = time.time()
            expired_requests = [
                req_id for req_id, req_data in active_requests.items()
                if now - req_data.get('start_time', now) > 3600
            ]
            for req_id in expired_requests:
                active_requests.pop(req_id, None)
            
            if expired_requests:
                logger.info(f"Cleaned up {len(expired_requests)} expired requests")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
        
        await asyncio.sleep(300)  # Run every 5 minutes

# Initialize FastAPI app with production settings
app = FastAPI(
    title="Jarvis Voice Assistant API",
    description="Production-ready RESTful API for Jarvis AI voice assistant",
    version=__version__,
    docs_url="/docs" if os.getenv('JARVIS_ENABLE_DOCS', 'false').lower() == 'true' else None,
    redoc_url="/redoc" if os.getenv('JARVIS_ENABLE_DOCS', 'false').lower() == 'true' else None,
    lifespan=lifespan
)

# Add production security middleware
app.add_middleware(SecurityMiddleware)

# Add trusted host middleware
trusted_hosts = os.getenv('JARVIS_TRUSTED_HOSTS', '').split(',')
if trusted_hosts and trusted_hosts[0]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Add CORS middleware with production settings
allowed_origins = security_config.allowed_origins
if not allowed_origins:
    # Development fallback
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict methods
    allow_headers=["Authorization", "Content-Type"],  # Restrict headers
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

async def initialize_jarvis():
    """Initialize Jarvis components for production"""
    global jarvis_brain, jarvis_tts, jarvis_commands, jarvis_context
    
    try:
        # Initialize AI brain with production settings
        ai_config = create_ai_config(
            anthropic_enabled=True,
            deepseek_enabled=True,
            local_enabled=True,
            prefer_local=False  # Prefer cloud for production reliability
        )
        
        jarvis_brain = AIBrainManager(ai_config)
        
        # Initialize TTS with production settings
        jarvis_tts = JarvisTTS(tts_engine="piper")
        
        # Initialize context/memory system
        db_path = os.getenv('JARVIS_DB_PATH', '/var/lib/jarvis/jarvis_memory.db')
        jarvis_context = create_jarvis_context(
            db_path=db_path,
            max_session_history=50,  # Increased for production
            default_user="user"
        )
        
        # Initialize commands system
        jarvis_commands = JarvisCommands(jarvis_tts, None, ai_config, jarvis_context)
        
        logger.info("🧠 AI Brain initialized for production")
        logger.info("🎙️ TTS engine initialized")
        logger.info("💾 Context system initialized")
        logger.info("⚙️ Command system initialized")
        
    except Exception as e:
        logger.error("❌ Failed to initialize Jarvis: %s", e)
        raise

async def cleanup_jarvis():
    """Cleanup Jarvis components"""
    global jarvis_brain, jarvis_tts, jarvis_commands, jarvis_context
    
    jarvis_brain = None
    jarvis_tts = None
    jarvis_commands = None
    jarvis_context = None

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

def process_jarvis_command(text: str) -> str:
    """Process command through Jarvis commands system (synchronous wrapper)"""
    if not jarvis_commands:
        return "Jarvis command system not available"
    
    try:
        class ResponseCapture:
            def __init__(self):
                self.responses = []
            
            def speak_direct(self, text):
                self.responses.append(text)
            
            def speak_with_feedback_control(self, text):
                self.responses.append(text)
        
        # Temporarily replace TTS to capture responses
        original_tts = jarvis_commands.tts
        response_capture = ResponseCapture()
        jarvis_commands.tts = response_capture
        
        # Process the command
        jarvis_commands.process_command(text)
        
        # Restore original TTS
        jarvis_commands.tts = original_tts
        
        # Return captured response or default
        if response_capture.responses:
            return response_capture.responses[-1]
        else:
            return "Command processed successfully"
            
    except Exception as e:
        logger.error(f"Error in process_jarvis_command: {e}")
        return f"Error processing command: {str(e)}"

# Production API Endpoints

@app.get("/", response_model=Dict[str, str])
@limiter.limit("10/minute")
async def root(request: Request):
    """Root endpoint with rate limiting"""
    return {
        "service": "Jarvis Voice Assistant API",
        "version": __version__,
        "status": "operational",
        "environment": "production"
    }

@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    components = {}
    
    # Check AI brain health
    if jarvis_brain:
        components["ai_brain"] = jarvis_brain.is_available()
        for provider, brain in jarvis_brain.brains.items():
            components[f"ai_provider_{provider.value}"] = brain.is_healthy()
    else:
        components["ai_brain"] = False
    
    # Check TTS health
    components["tts"] = jarvis_tts is not None
    
    # Check commands system
    components["commands"] = jarvis_commands is not None
    
    overall_health = all(components.values())
    
    return HealthResponse(
        healthy=overall_health,
        components=components,
        timestamp=datetime.now()
    )

@app.get("/status", response_model=StatusResponse)
@limiter.limit("20/minute")
async def get_status(
    request: Request,
    current_user: dict = Depends(require_permission("read"))
):
    """Get system status (requires authentication)"""
    uptime = time.time() - server_start_time
    
    # Get AI provider status
    ai_providers = {}
    if jarvis_brain:
        for provider, brain in jarvis_brain.brains.items():
            ai_providers[provider.value] = brain.is_healthy()
    
    # Determine overall status
    if not jarvis_brain or not jarvis_tts:
        status_value = "unhealthy"
    elif any(ai_providers.values()):
        status_value = "healthy"
    else:
        status_value = "degraded"
    
    return StatusResponse(
        version=__version__,
        status=status_value,
        uptime=uptime,
        ai_providers=ai_providers,
        tts_engine=jarvis_tts.tts_engine.__class__.__name__ if jarvis_tts else "none",
        local_mode=jarvis_brain.primary_brain.provider_name == "Local Gemma" if jarvis_brain else False
    )

@app.post("/chat", response_model=TextResponse)
@limiter.limit("60/minute")
async def chat_with_jarvis(
    request: Request,
    chat_request: TextRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_permission("chat")),
    _: bool = Depends(validate_request_size(1)),  # 1MB max
    __: bool = Depends(check_ip_access)
):
    """Chat with Jarvis via text with enhanced security"""
    request_id = generate_request_id()
    start_time = time.time()
    
    # Log the request
    log_security_event('chat_request', {
        'user': current_user['user']['name'],
        'text_length': len(chat_request.text),
        'use_tts': chat_request.use_tts,
        'stream_audio': chat_request.stream_audio
    }, request)
    
    try:
        if not jarvis_commands or not jarvis_context:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Jarvis command system or context not available"
            )
        
        # Handle user switching if specified
        if chat_request.user:
            user_switched = jarvis_context.switch_user(chat_request.user)
            if not user_switched:
                logger.warning(f"Failed to switch to user: {chat_request.user}")
        
        # Store request for tracking
        active_requests[request_id] = {
            "start_time": start_time,
            "text": chat_request.text,
            "user": current_user['user']['name'],
            "jarvis_user": jarvis_context.current_user_id
        }
        
        # Process through Jarvis commands system
        response = await asyncio.get_event_loop().run_in_executor(
            None, process_jarvis_command, chat_request.text
        )
        
        processing_time = time.time() - start_time
        
        # Determine provider used
        provider_used = "jarvis_commands"
        if jarvis_commands.ai_enabled and jarvis_commands.ai_brain:
            provider_used = jarvis_commands.ai_brain.primary_brain.provider_name if jarvis_commands.ai_brain.primary_brain else "unknown"
        
        # Generate audio if requested
        audio_url = None
        stream_url = None
        if chat_request.use_tts and jarvis_tts:
            if chat_request.stream_audio and "stream" in current_user.get('permissions', []):
                stream_url = f"/audio/stream/{request_id}"
                active_requests[request_id]["response_text"] = response
            else:
                audio_filename = f"response_{request_id}.wav"
                background_tasks.add_task(generate_audio_response, response, audio_filename)
                audio_url = f"/audio/{audio_filename}"
        
        # Get current user info
        current_jarvis_user = jarvis_context.get_current_user()
        
        # Don't clean up request tracking here if streaming is requested
        if not (chat_request.use_tts and chat_request.stream_audio):
            active_requests.pop(request_id, None)
        
        return TextResponse(
            response=response,
            provider_used=provider_used,
            processing_time=processing_time,
            audio_url=audio_url,
            stream_url=stream_url,
            request_id=request_id,
            current_user=current_jarvis_user['display_name']
        )
        
    except HTTPException:
        active_requests.pop(request_id, None)
        raise
    except Exception as e:
        active_requests.pop(request_id, None)
        logger.error(f"Error processing chat request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/audio/stream/{request_id}")
@limiter.limit("30/minute")
async def stream_audio_response(
    request: Request,
    request_id: str,
    current_user: dict = Depends(require_permission("stream"))
):
    """Stream audio response in real-time chunks"""
    # Check if request exists and has response text
    if request_id not in active_requests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio stream not found or expired"
        )
    
    request_info = active_requests[request_id]
    response_text = request_info.get("response_text")
    
    if not response_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No response text available for streaming"
        )
    
    if not jarvis_tts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service not available"
        )
    
    # Log streaming access
    log_security_event('audio_stream_access', {
        'request_id': request_id,
        'user': current_user['user']['name']
    }, request)
    
    async def generate_audio_stream():
        """Generate audio stream chunks"""
        try:
            # Generate complete audio first
            audio_data = jarvis_tts.tts_engine.synthesize(response_text)
            
            if not audio_data:
                yield b"Audio generation failed"
                return
            
            # Stream audio in chunks
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in audio streaming {request_id}: {e}")
            yield b"Streaming error occurred"
        finally:
            # Clean up request after streaming
            active_requests.pop(request_id, None)
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/audio/stream")
@limiter.limit("30/minute")
async def create_audio_stream(
    request: Request,
    stream_request: StreamingAudioRequest,
    current_user: dict = Depends(require_permission("stream")),
    _: bool = Depends(validate_request_size(1))
):
    """Create a new audio stream for text"""
    if not jarvis_tts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service not available"
        )
    
    request_id = generate_request_id()
    
    # Log streaming request
    log_security_event('direct_audio_stream', {
        'user': current_user['user']['name'],
        'text_length': len(stream_request.text),
        'chunk_size': stream_request.chunk_size
    }, request)
    
    async def generate_streaming_audio():
        """Generate streaming audio chunks for text"""
        try:
            # Generate audio data
            audio_data = jarvis_tts.tts_engine.synthesize(stream_request.text)
            
            if not audio_data:
                yield b"Audio generation failed"
                return
            
            # Stream in specified chunks
            for i in range(0, len(audio_data), stream_request.chunk_size):
                chunk = audio_data[i:i + stream_request.chunk_size]
                yield chunk
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in direct audio streaming {request_id}: {e}")
            yield b"Streaming error occurred"
    
    return StreamingResponse(
        generate_streaming_audio(),
        media_type="audio/wav" if stream_request.format == "wav" else "audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Request-ID": request_id,
        }
    )

async def generate_audio_response(text: str, filename: str):
    """Background task to generate audio response"""
    try:
        import tempfile
        import os
        
        if jarvis_tts:
            # Generate audio
            audio_data = jarvis_tts.tts_engine.synthesize(text)
            
            if audio_data:
                # Save to secure temporary file
                audio_dir = os.getenv('JARVIS_AUDIO_DIR', tempfile.gettempdir())
                os.makedirs(audio_dir, exist_ok=True)
                audio_path = os.path.join(audio_dir, filename)
                
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                logger.info(f"Generated audio file: {filename}")
            else:
                logger.warning(f"Failed to generate audio for: {filename}")
    except Exception as e:
        logger.error(f"Error generating audio {filename}: {e}")

# Enhanced error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Jarvis REST API Server - Production')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Jarvis API Server (Production) on {args.host}:{args.port}")
    print(f"👥 Workers: {args.workers}")
    print(f"🔒 Security: Enhanced")
    
    uvicorn.run(
        "jarvis_api_production:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )