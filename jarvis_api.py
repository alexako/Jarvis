#!/usr/bin/env python3
"""
Jarvis REST API Server
Provides RESTful access to Jarvis voice assistant capabilities

Features:
- Text-based interaction with Jarvis AI
- Multiple AI provider support (local, cloud)
- Audio synthesis endpoints  
- Status and health monitoring
- Extensible for future video stream integration
"""

__version__ = "1.5.0"

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Jarvis components
from ai_brain import AIBrainManager, BrainProvider
from speech_analysis.tts import JarvisTTS
from commands import JarvisCommands, create_ai_config
from jarvis_context import create_jarvis_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Jarvis instance
jarvis_brain: Optional[AIBrainManager] = None
jarvis_tts: Optional[JarvisTTS] = None
jarvis_commands: Optional[JarvisCommands] = None
jarvis_context = None

# API Models
class TextRequest(BaseModel):
    """Request model for text interaction"""
    text: str = Field(..., description="Text input to send to Jarvis", min_length=1, max_length=1000)
    use_tts: bool = Field(default=False, description="Whether to generate audio response")
    ai_provider: Optional[str] = Field(default=None, description="Preferred AI provider (local, anthropic, deepseek)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the request")
    user: Optional[str] = Field(default=None, description="User identifier for multi-user context (optional)")

class TextResponse(BaseModel):
    """Response model for text interaction"""
    response: str = Field(..., description="Jarvis's text response")
    provider_used: str = Field(..., description="AI provider that handled the request")
    processing_time: float = Field(..., description="Time taken to process request in seconds")
    audio_url: Optional[str] = Field(default=None, description="URL to generated audio file if TTS was requested")
    request_id: str = Field(..., description="Unique identifier for this request")
    current_user: Optional[str] = Field(default=None, description="Current active user in context system")

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

class VideoUploadRequest(BaseModel):
    """Request model for video analysis"""
    description: Optional[str] = Field(default=None, description="Description of what to analyze in the video")
    analyze_motion: bool = Field(default=True, description="Whether to analyze motion in the video")
    extract_frames: bool = Field(default=False, description="Whether to extract key frames")
    ai_provider: Optional[str] = Field(default=None, description="Preferred AI provider for analysis")

class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis"""
    analysis: str = Field(..., description="AI analysis of the video content")
    motion_detected: bool = Field(..., description="Whether motion was detected")
    frame_count: int = Field(..., description="Total number of frames processed")
    duration: float = Field(..., description="Video duration in seconds")
    key_frames: Optional[List[str]] = Field(default=None, description="URLs to extracted key frames")
    processing_time: float = Field(..., description="Time taken to process video")
    request_id: str = Field(..., description="Unique identifier for this request")

# Global state
server_start_time = time.time()
active_requests: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting Jarvis API Server...")
    await initialize_jarvis()
    logger.info("✅ Jarvis API Server ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Jarvis API Server...")
    await cleanup_jarvis()
    logger.info("✅ Jarvis API Server stopped")

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis Voice Assistant API",
    description="RESTful API for interacting with Jarvis AI voice assistant",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token-based authentication (extend for production use)"""
    if credentials is None:
        return None  # Allow unauthenticated access for now
    
    # TODO: Implement proper JWT validation
    if credentials.credentials == "jarvis-api-key":
        return {"user": "api_user"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def initialize_jarvis():
    """Initialize Jarvis components"""
    global jarvis_brain, jarvis_tts, jarvis_commands, jarvis_context
    
    try:
        # Initialize AI brain with all providers
        ai_config = create_ai_config(
            anthropic_enabled=True,
            deepseek_enabled=True,
            local_enabled=True,
            prefer_local=True  # Prefer local for API use
        )
        
        jarvis_brain = AIBrainManager(ai_config)
        
        # Initialize TTS with Piper neural voice
        jarvis_tts = JarvisTTS(tts_engine="piper")
        
        # Initialize context/memory system - SHARED with voice assistant
        jarvis_context = create_jarvis_context(
            db_path="jarvis_memory.db",  # Same database as voice assistant
            max_session_history=20,
            default_user="user"  # Same default as voice assistant
        )
        
        # Initialize commands system with context support
        jarvis_commands = JarvisCommands(jarvis_tts, None, ai_config, jarvis_context)
        
        logger.info("🧠 AI Brain initialized with providers: %s", list(jarvis_brain.brains.keys()))
        logger.info("🎙️ TTS engine initialized")
        logger.info("💾 Context system initialized for API")  
        logger.info("⚙️ Command system initialized")
        
    except Exception as e:
        logger.error("❌ Failed to initialize Jarvis: %s", e)
        raise

async def cleanup_jarvis():
    """Cleanup Jarvis components"""
    global jarvis_brain, jarvis_tts, jarvis_commands, jarvis_context
    
    # Clean up resources
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
    
    # This is a simplified version - in a real implementation you'd want to capture
    # the actual response from the commands system
    try:
        # Use a mock TTS that captures responses instead of speaking
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
            return response_capture.responses[-1]  # Return last response
        else:
            return "Command processed successfully"
            
    except Exception as e:
        logger.error(f"Error in process_jarvis_command: {e}")
        return f"Error processing command: {str(e)}"

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Jarvis Voice Assistant API",
        "version": __version__,
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
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
async def get_status():
    """Get system status"""
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
        local_mode=jarvis_brain.primary_brain.provider_name == "Local Llama" if jarvis_brain else False
    )

@app.post("/chat", response_model=TextResponse)
async def chat_with_jarvis(
    request: TextRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Chat with Jarvis via text with multi-user context support"""
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        if not jarvis_commands or not jarvis_context:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Jarvis command system or context not available"
            )
        
        # Handle user switching if specified
        if request.user:
            # Switch to specified user
            user_switched = jarvis_context.switch_user(request.user)
            if not user_switched:
                logger.warning(f"Failed to switch to user: {request.user}")
        
        # Store request for tracking
        active_requests[request_id] = {
            "start_time": start_time,
            "text": request.text,
            "user": current_user.get("user") if current_user else "anonymous",
            "jarvis_user": jarvis_context.current_user_id
        }
        
        # Process through Jarvis commands system (includes AI fallback and context)
        response = await asyncio.get_event_loop().run_in_executor(
            None, process_jarvis_command, request.text
        )
        
        processing_time = time.time() - start_time
        
        # Determine which provider was used (simplified for commands system)
        provider_used = "jarvis_commands"
        if jarvis_commands.ai_enabled and jarvis_commands.ai_brain:
            provider_used = jarvis_commands.ai_brain.primary_brain.provider_name if jarvis_commands.ai_brain.primary_brain else "unknown"
        
        # Generate audio if requested
        audio_url = None
        if request.use_tts and jarvis_tts:
            audio_filename = f"response_{request_id}.wav"
            background_tasks.add_task(generate_audio_response, response, audio_filename)
            audio_url = f"/audio/{audio_filename}"
        
        # Get current user info
        current_jarvis_user = jarvis_context.get_current_user()
        
        # Clean up request tracking
        active_requests.pop(request_id, None)
        
        return TextResponse(
            response=response,
            provider_used=provider_used,
            processing_time=processing_time,
            audio_url=audio_url,
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
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Retrieve generated audio file"""
    import os
    import tempfile
    
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found"
        )
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename
    )

@app.get("/providers", response_model=Dict[str, Dict[str, Any]])
async def get_ai_providers():
    """Get information about available AI providers"""
    if not jarvis_brain:
        return {}
    
    providers = {}
    for provider, brain in jarvis_brain.brains.items():
        providers[provider.value] = {
            "name": brain.provider_name,
            "healthy": brain.is_healthy(),
            "is_primary": brain == jarvis_brain.primary_brain,
            "is_fallback": brain == jarvis_brain.fallback_brain
        }
    
    return providers

@app.get("/users", response_model=Dict[str, Any])
async def get_users():
    """Get information about users in the system"""
    if not jarvis_context:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Context system not available"
        )
    
    users = jarvis_context.list_users()
    current_user = jarvis_context.get_current_user()
    
    return {
        "current_user": current_user,
        "users": users,
        "total_users": len(users)
    }

@app.post("/users/switch")
async def switch_user(user_identifier: str):
    """Switch to a different user"""
    if not jarvis_context:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Context system not available"
        )
    
    success = jarvis_context.switch_user(user_identifier)
    if success:
        current_user = jarvis_context.get_current_user()
        return {
            "success": True,
            "message": f"Switched to user {current_user['display_name']}",
            "current_user": current_user
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to switch to user: {user_identifier}"
        )

@app.get("/users/current", response_model=Dict[str, Any])
async def get_current_user_info():
    """Get current user information and aliases"""
    if not jarvis_context:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Context system not available"
        )
    
    current_user = jarvis_context.get_current_user()
    aliases = jarvis_context.get_user_aliases()
    recent_context = jarvis_context.get_recent_context(5)
    
    return {
        "user": current_user,
        "aliases": aliases,
        "recent_exchanges": len(recent_context),
        "preferences": jarvis_context.session_preferences
    }

@app.post("/video/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyze"),
    description: Optional[str] = None,
    analyze_motion: bool = True,
    extract_frames: bool = False,
    ai_provider: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Analyze uploaded video for security monitoring and content analysis"""
    request_id = generate_request_id()
    start_time = time.time()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video files are supported"
        )
    
    # Validate file size (limit to 100MB for now)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Video file too large (max 100MB)"
        )
    
    try:
        # Store request for tracking
        active_requests[request_id] = {
            "start_time": start_time,
            "filename": file.filename,
            "user": current_user.get("user") if current_user else "anonymous",
            "type": "video_analysis"
        }
        
        # Process video in background task
        background_tasks.add_task(
            process_video_analysis, 
            file, 
            request_id, 
            description, 
            analyze_motion, 
            extract_frames,
            ai_provider
        )
        
        processing_time = time.time() - start_time
        
        # Return immediate response with placeholder data
        # Real processing happens in background
        return VideoAnalysisResponse(
            analysis="Video analysis in progress. Check back for results.",
            motion_detected=False,  # Will be updated by background task
            frame_count=0,
            duration=0.0,
            key_frames=None,
            processing_time=processing_time,
            request_id=request_id
        )
        
    except Exception as e:
        active_requests.pop(request_id, None)
        logger.error(f"Error processing video upload {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/video/result/{request_id}", response_model=VideoAnalysisResponse)
async def get_video_analysis_result(request_id: str):
    """Get results of video analysis by request ID"""
    # In a real implementation, this would check a database or cache
    # For now, return a placeholder response
    return VideoAnalysisResponse(
        analysis="Video analysis feature is prepared but not yet fully implemented. This endpoint is ready for future enhancement.",
        motion_detected=False,
        frame_count=0,
        duration=0.0,
        key_frames=None,
        processing_time=0.0,
        request_id=request_id
    )

@app.get("/video/frame/{filename}")
async def get_video_frame(filename: str):
    """Retrieve extracted video frame"""
    import os
    import tempfile
    
    frame_path = os.path.join(tempfile.gettempdir(), "frames", filename)
    
    if not os.path.exists(frame_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame not found"
        )
    
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        filename=filename
    )

async def process_video_analysis(
    file: UploadFile, 
    request_id: str, 
    description: Optional[str],
    analyze_motion: bool,
    extract_frames: bool,
    ai_provider: Optional[str]
):
    """Background task to process video analysis"""
    try:
        import tempfile
        import os
        
        logger.info(f"Starting video analysis for request {request_id}")
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"video_{request_id}_{file.filename}")
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Placeholder for actual video processing
        # In a real implementation, this would:
        # 1. Extract frames using OpenCV or similar
        # 2. Analyze motion using computer vision techniques  
        # 3. Send frames to AI provider for content analysis
        # 4. Store results in database or cache
        
        analysis_text = f"Video analysis prepared for {file.filename}. "
        if description:
            analysis_text += f"Analysis request: {description}. "
        
        analysis_text += "This feature framework is complete and ready for computer vision implementation."
        
        logger.info(f"Video analysis completed for request {request_id}")
        
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in video analysis background task {request_id}: {e}")
    finally:
        # Clean up request tracking
        active_requests.pop(request_id, None)

async def generate_audio_response(text: str, filename: str):
    """Background task to generate audio response"""
    try:
        import tempfile
        import os
        
        if jarvis_tts:
            # Generate audio
            audio_data = jarvis_tts.tts_engine.synthesize(text)
            
            if audio_data:
                # Save to temporary file
                audio_path = os.path.join(tempfile.gettempdir(), filename)
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                logger.info(f"Generated audio file: {filename}")
            else:
                logger.warning(f"Failed to generate audio for: {filename}")
    except Exception as e:
        logger.error(f"Error generating audio {filename}: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
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
    
    parser = argparse.ArgumentParser(description='Jarvis REST API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Jarvis API Server on {args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {args.reload}")
    
    uvicorn.run(
        "jarvis_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )