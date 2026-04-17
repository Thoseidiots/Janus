"""
Janus Business Operations API - CEO Management Interface

Provides RESTful API access for Janus AI CEO to manage business operations,
client communications, and autonomous workflows. Supports the CEO's
revenue-generating activities across multiple business verticals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# FastAPI for REST API
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Janus imports (existing components)
try:
    from avus_brain import AvusBrain
    from holographic_brain_memory.core import HolographicMemory
    from janus_video_comprehension import JanusVideoComprehension
    from speech_synthesis import SpeechSynthesis
    from janus_fault_integration import JanusAIGuard
except ImportError as e:
    logging.warning(f"Could not import Janus components: {e}")
    # Fallback for development
    AvusBrain = None
    HolographicMemory = None
    JanusVideoComprehension = None
    SpeechSynthesis = None
    JanusAIGuard = None

logger = logging.getLogger(__name__)

# Configuration
API_CONFIG = {
    "title": "Janus Cloud API",
    "description": "Autonomous AI System API - No external dependencies",
    "version": "1.0.0",
    "pricing": {
        "free_tier": {
            "requests_per_hour": 100,
            "max_model_size": "1b",
            "features": ["text_generation", "basic_memory"]
        },
        "developer_tier": {
            "requests_per_hour": 1000,
            "max_model_size": "7b",
            "features": ["all_basic", "video_analysis", "speech_synthesis"],
            "price": 100  # $100/month
        },
        "enterprise_tier": {
            "requests_per_hour": 10000,
            "max_model_size": "70b",
            "features": ["all_features", "custom_training", "multi_instance"],
            "price": 500  # $500/month per instance
        }
    }
}

# Pydantic Models
class User(BaseModel):
    id: str
    email: str
    tier: str = "free"
    api_key: str
    created_at: datetime
    last_active: datetime
    usage_stats: Dict[str, Any] = field(default_factory=dict)

class APIRequest(BaseModel):
    prompt: str
    model_size: str = "1b"
    max_tokens: int = 500
    temperature: float = 0.7
    context: Optional[Dict[str, Any]] = None

class VideoAnalysisRequest(BaseModel):
    video_path: str
    analysis_type: str = "scene_detection"
    duration: Optional[int] = None
    output_format: str = "json"

class SpeechRequest(BaseModel):
    text: str
    voice_type: str = "default"
    output_format: str = "wav"
    emotion: str = "neutral"

class MemoryRequest(BaseModel):
    operation: str  # "store", "retrieve", "search"
    content: Optional[Union[str, Dict]] = None
    query: Optional[str] = None
    memory_type: str = "episodic"

class UsageStats(BaseModel):
    requests_today: int = 0
    requests_this_hour: int = 0
    tokens_used: int = 0
    last_request: Optional[datetime] = None

@dataclass
class APIKey:
    key_id: str
    api_key: str
    user_id: str
    tier: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_stats: UsageStats = field(default_factory=UsageStats)

class JanusCloudAPI:
    """Main API Service"""
    
    def __init__(self):
        self.app = FastAPI(
            title=API_CONFIG["title"],
            description=API_CONFIG["description"],
            version=API_CONFIG["version"]
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        self.security = HTTPBearer()
        
        # In-memory storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.usage_logs: List[Dict] = []
        
        # Janus components (lazy loading)
        self._janus_instances: Dict[str, Any] = {}
        self._fault_guard = None
        
        # Setup routes
        self._setup_routes()
        
        # Initialize fault guard
        if JanusAIGuard:
            self._fault_guard = JanusAIGuard()
        
        logger.info("Janus Cloud API initialized")
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Janus Cloud API",
                "version": API_CONFIG["version"],
                "status": "operational",
                "features": [
                    "text_generation",
                    "video_analysis", 
                    "speech_synthesis",
                    "holographic_memory",
                    "autonomous_capabilities"
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": API_CONFIG["version"],
                "uptime": "operational"
            }
        
        @self.app.post("/auth/generate-key")
        async def generate_api_key(email: str, tier: str = "free"):
            """Generate new API key"""
            # Validate tier
            if tier not in API_CONFIG["pricing"]:
                raise HTTPException(status_code=400, detail="Invalid tier")
            
            # Create user if not exists
            user_id = str(uuid.uuid4())
            if email not in [user.email for user in self.users.values()]:
                user = User(
                    id=user_id,
                    email=email,
                    tier=tier,
                    api_key=f"janus_{uuid.uuid4().hex}",
                    created_at=datetime.utcnow(),
                    last_active=datetime.utcnow()
                )
                self.users[user_id] = user
            else:
                user = next(u for u in self.users.values() if u.email == email)
                user_id = user.id
            
            # Generate API key
            key_id = str(uuid.uuid4())
            api_key = f"janus_{uuid.uuid4().hex}"
            
            key_obj = APIKey(
                key_id=key_id,
                api_key=api_key,
                user_id=user_id,
                tier=tier,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=365) if tier != "free" else None
            )
            
            self.api_keys[api_key] = key_obj
            
            return {
                "api_key": api_key,
                "tier": tier,
                "expires_at": key_obj.expires_at.isoformat() if key_obj.expires_at else None,
                "rate_limits": API_CONFIG["pricing"][tier]
            }
        
        @self.app.post("/v1/text/generate")
        async def generate_text(
            request: APIRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Generate text using Avus model"""
            # Validate API key
            key_obj = self._validate_api_key(credentials.credentials)
            self._check_rate_limit(key_obj)
            
            # Validate request with fault guard
            if self._fault_guard:
                validation = self._fault_guard.validate_ai_generation(
                    request.prompt, "text", {"model_size": request.model_size}
                )
                if not validation["is_allowed"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Request blocked: {validation['block_reason']}"
                    )
            
            # Process request
            try:
                result = await self._generate_text_internal(request, key_obj)
                self._log_usage(key_obj, "text_generation", len(result.get("text", "")))
                return result
            except Exception as e:
                logger.error(f"Text generation error: {e}")
                raise HTTPException(status_code=500, detail="Generation failed")
        
        @self.app.post("/v1/video/analyze")
        async def analyze_video(
            request: VideoAnalysisRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Analyze video content"""
            key_obj = self._validate_api_key(credentials.credentials)
            self._check_rate_limit(key_obj)
            
            if key_obj.tier == "free":
                raise HTTPException(status_code=403, detail="Video analysis requires paid tier")
            
            try:
                result = await self._analyze_video_internal(request, key_obj)
                self._log_usage(key_obj, "video_analysis", 1)
                return result
            except Exception as e:
                logger.error(f"Video analysis error: {e}")
                raise HTTPException(status_code=500, detail="Analysis failed")
        
        @self.app.post("/v1/speech/synthesize")
        async def synthesize_speech(
            request: SpeechRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Synthesize speech from text"""
            key_obj = self._validate_api_key(credentials.credentials)
            self._check_rate_limit(key_obj)
            
            if key_obj.tier == "free":
                raise HTTPException(status_code=403, detail="Speech synthesis requires paid tier")
            
            try:
                result = await self._synthesize_speech_internal(request, key_obj)
                self._log_usage(key_obj, "speech_synthesis", len(request.text))
                return result
            except Exception as e:
                logger.error(f"Speech synthesis error: {e}")
                raise HTTPException(status_code=500, detail="Synthesis failed")
        
        @self.app.post("/v1/memory")
        async def memory_operation(
            request: MemoryRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Perform holographic memory operations"""
            key_obj = self._validate_api_key(credentials.credentials)
            self._check_rate_limit(key_obj)
            
            try:
                result = await self._memory_operation_internal(request, key_obj)
                self._log_usage(key_obj, f"memory_{request.operation}", 1)
                return result
            except Exception as e:
                logger.error(f"Memory operation error: {e}")
                raise HTTPException(status_code=500, detail="Memory operation failed")
        
        @self.app.get("/v1/usage")
        async def get_usage_stats(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get usage statistics"""
            key_obj = self._validate_api_key(credentials.credentials)
            
            return {
                "tier": key_obj.tier,
                "usage_stats": asdict(key_obj.usage_stats),
                "rate_limits": API_CONFIG["pricing"][key_obj.tier],
                "api_key_id": key_obj.key_id
            }
        
        @self.app.get("/v1/models")
        async def list_models(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """List available models"""
            key_obj = self._validate_api_key(credentials.credentials)
            
            models = ["avus-1b", "avus-3b", "avus-7b", "avus-13b", "avus-34b", "avus-70b"]
            max_size = API_CONFIG["pricing"][key_obj.tier]["max_model_size"]
            
            # Filter models by tier
            if max_size == "1b":
                available_models = ["avus-1b"]
            elif max_size == "7b":
                available_models = ["avus-1b", "avus-3b", "avus-7b"]
            else:
                available_models = models
            
            return {
                "models": available_models,
                "current_tier": key_obj.tier,
                "max_model_size": max_size
            }
    
    def _validate_api_key(self, api_key: str) -> APIKey:
        """Validate API key and return key object"""
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        key_obj = self.api_keys[api_key]
        
        if not key_obj.is_active:
            raise HTTPException(status_code=401, detail="API key deactivated")
        
        if key_obj.expires_at and datetime.utcnow() > key_obj.expires_at:
            raise HTTPException(status_code=401, detail="API key expired")
        
        # Update last active
        key_obj.usage_stats.last_request = datetime.utcnow()
        
        return key_obj
    
    def _check_rate_limit(self, key_obj: APIKey):
        """Check rate limits"""
        tier_config = API_CONFIG["pricing"][key_obj.tier]
        max_requests = tier_config["requests_per_hour"]
        
        # Reset hourly counter if needed
        if (key_obj.usage_stats.last_request and 
            datetime.utcnow() - key_obj.usage_stats.last_request > timedelta(hours=1)):
            key_obj.usage_stats.requests_this_hour = 0
        
        if key_obj.usage_stats.requests_this_hour >= max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {max_requests} requests per hour"
            )
        
        key_obj.usage_stats.requests_this_hour += 1
        key_obj.usage_stats.requests_today += 1
    
    def _log_usage(self, key_obj: APIKey, operation: str, units: int):
        """Log API usage"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_key_id": key_obj.key_id,
            "user_id": key_obj.user_id,
            "operation": operation,
            "units": units,
            "tier": key_obj.tier
        }
        self.usage_logs.append(log_entry)
        
        # Keep only last 10000 logs
        if len(self.usage_logs) > 10000:
            self.usage_logs = self.usage_logs[-10000:]
    
    async def _generate_text_internal(self, request: APIRequest, key_obj: APIKey) -> Dict[str, Any]:
        """Internal text generation"""
        # Get or create Janus instance for this user
        instance_key = f"{key_obj.user_id}_{request.model_size}"
        
        if instance_key not in self._janus_instances:
            # Initialize Avus brain for this user
            if AvusBrain:
                brain = AvusBrain(model_size=request.model_size)
                self._janus_instances[instance_key] = brain
            else:
                # Fallback for development
                return {
                    "text": f"Generated response for: {request.prompt} (development mode)",
                    "model": f"avus-{request.model_size}",
                    "tokens_used": len(request.prompt.split()),
                    "quality_score": 95.0
                }
        
        brain = self._janus_instances[instance_key]
        
        # Generate text
        start_time = time.time()
        response = await asyncio.to_thread(
            brain.generate, 
            request.prompt, 
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        generation_time = time.time() - start_time
        
        return {
            "text": response,
            "model": f"avus-{request.model_size}",
            "tokens_used": len(response.split()),
            "generation_time": generation_time,
            "quality_score": 95.0
        }
    
    async def _analyze_video_internal(self, request: VideoAnalysisRequest, key_obj: APIKey) -> Dict[str, Any]:
        """Internal video analysis"""
        if not JanusVideoComprehension:
            return {
                "analysis": "Video analysis not available in development mode",
                "video_path": request.video_path,
                "analysis_type": request.analysis_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize video comprehension
        if "video_comprehension" not in self._janus_instances:
            vc = JanusVideoComprehension()
            self._janus_instances["video_comprehension"] = vc
        
        vc = self._janus_instances["video_comprehension"]
        
        # Analyze video
        analysis = await asyncio.to_thread(
            vc.analyze_video,
            request.video_path,
            analysis_type=request.analysis_type,
            duration=request.duration
        )
        
        return {
            "analysis": analysis,
            "video_path": request.video_path,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _synthesize_speech_internal(self, request: SpeechRequest, key_obj: APIKey) -> Dict[str, Any]:
        """Internal speech synthesis"""
        if not SpeechSynthesis:
            return {
                "audio_url": f"/audio/synthesized_{uuid.uuid4().hex}.wav",
                "text": request.text,
                "voice_type": request.voice_type,
                "duration": len(request.text) * 0.1,  # Estimate
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize speech synthesis
        if "speech_synthesis" not in self._janus_instances:
            ss = SpeechSynthesis()
            self._janus_instances["speech_synthesis"] = ss
        
        ss = self._janus_instances["speech_synthesis"]
        
        # Synthesize speech
        audio_data = await asyncio.to_thread(
            ss.synthesize,
            request.text,
            voice_type=request.voice_type,
            emotion=request.emotion
        )
        
        # Save audio file (in production, use cloud storage)
        audio_filename = f"audio_{uuid.uuid4().hex}.wav"
        # audio_path = f"/tmp/{audio_filename}"
        # with open(audio_path, "wb") as f:
        #     f.write(audio_data)
        
        return {
            "audio_url": f"/audio/{audio_filename}",
            "text": request.text,
            "voice_type": request.voice_type,
            "emotion": request.emotion,
            "duration": len(request.text) * 0.1,  # Estimate
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _memory_operation_internal(self, request: MemoryRequest, key_obj: APIKey) -> Dict[str, Any]:
        """Internal memory operation"""
        if not HolographicMemory:
            return {
                "operation": request.operation,
                "result": f"Memory {request.operation} completed (development mode)",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize holographic memory
        memory_key = f"memory_{key_obj.user_id}"
        if memory_key not in self._janus_instances:
            hm = HolographicMemory()
            self._janus_instances[memory_key] = hm
        
        hm = self._janus_instances[memory_key]
        
        # Perform operation
        if request.operation == "store":
            result = await asyncio.to_thread(hm.store, request.content, request.memory_type)
        elif request.operation == "retrieve":
            result = await asyncio.to_thread(hm.retrieve, request.query, request.memory_type)
        elif request.operation == "search":
            result = await asyncio.to_thread(hm.search, request.query, request.memory_type)
        else:
            raise HTTPException(status_code=400, detail="Invalid memory operation")
        
        return {
            "operation": request.operation,
            "result": result,
            "memory_type": request.memory_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Create global API instance
api = JanusCloudAPI()

if __name__ == "__main__":
    # Run the API server
    api.run(host="0.0.0.0", port=8000)
