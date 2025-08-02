#!/usr/bin/env python3
"""
Production Security Configuration for Jarvis API
Implements enterprise-grade security for external API exposure
"""

import os
import secrets
import hashlib
import jwt
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

logger = logging.getLogger(__name__)

class ProductionSecurity:
    """Production security configuration and utilities"""
    
    def __init__(self):
        # Load configuration from environment variables
        self.jwt_secret = os.getenv('JARVIS_JWT_SECRET', self._generate_jwt_secret())
        self.api_keys = self._load_api_keys()
        self.allowed_origins = self._load_allowed_origins()
        self.admin_key = os.getenv('JARVIS_ADMIN_KEY', self._generate_admin_key())
        
    def _generate_jwt_secret(self) -> str:
        """Generate a secure JWT secret if not provided"""
        secret = secrets.token_urlsafe(64)
        logger.warning("Generated new JWT secret - save this to JARVIS_JWT_SECRET environment variable")
        logger.warning(f"JARVIS_JWT_SECRET={secret}")
        return secret
    
    def _generate_admin_key(self) -> str:
        """Generate a secure admin key if not provided"""
        key = secrets.token_urlsafe(32)
        logger.warning("Generated new admin key - save this to JARVIS_ADMIN_KEY environment variable")
        logger.warning(f"JARVIS_ADMIN_KEY={key}")
        return key
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or configuration"""
        api_keys = {}
        
        # Load from environment variables
        for i in range(1, 11):  # Support up to 10 API keys
            key = os.getenv(f'JARVIS_API_KEY_{i}')
            name = os.getenv(f'JARVIS_API_KEY_{i}_NAME', f'client_{i}')
            permissions = os.getenv(f'JARVIS_API_KEY_{i}_PERMISSIONS', 'read,chat').split(',')
            
            if key:
                api_keys[key] = {
                    'name': name,
                    'permissions': permissions,
                    'created': datetime.now(),
                    'last_used': None
                }
        
        # Generate a default key if none exist
        if not api_keys:
            default_key = secrets.token_urlsafe(32)
            api_keys[default_key] = {
                'name': 'default_client',
                'permissions': ['read', 'chat', 'stream'],
                'created': datetime.now(),
                'last_used': None
            }
            logger.warning(f"Generated default API key: {default_key}")
            logger.warning("Set JARVIS_API_KEY_1 environment variable for production")
        
        return api_keys
    
    def _load_allowed_origins(self) -> List[str]:
        """Load allowed CORS origins from environment"""
        origins_str = os.getenv('JARVIS_ALLOWED_ORIGINS', '')
        if origins_str:
            return [origin.strip() for origin in origins_str.split(',')]
        return []  # Empty list = no CORS restrictions in dev
    
    def create_jwt_token(self, user_data: Dict[str, Any], expires_hours: int = 24) -> str:
        """Create a JWT token for authenticated users"""
        payload = {
            'user': user_data,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow(),
            'iss': 'jarvis-api'
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload.get('user')
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return associated permissions"""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            key_info['last_used'] = datetime.now()
            return key_info
        return None
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, pwd_hash = password_hash.split('$')
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() == pwd_hash
        except ValueError:
            return False

# Global security instance
security_config = ProductionSecurity()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Enhanced authentication
security_bearer = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    """Enhanced authentication with JWT and API key support"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Try JWT token first
    user_data = security_config.verify_jwt_token(token)
    if user_data:
        return {
            'type': 'jwt',
            'user': user_data,
            'permissions': user_data.get('permissions', ['read', 'chat'])
        }
    
    # Try API key
    key_info = security_config.verify_api_key(token)
    if key_info:
        return {
            'type': 'api_key',
            'user': {'name': key_info['name']},
            'permissions': key_info['permissions']
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_permission(permission: str):
    """Decorator to require specific permissions"""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        if permission not in current_user.get('permissions', []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker

def log_security_event(event_type: str, details: Dict[str, Any], request: Request):
    """Log security events for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'client_ip': get_remote_address(request),
        'user_agent': request.headers.get('user-agent', ''),
        'details': details
    }
    
    # In production, send to SIEM/logging system
    logger.warning(f"SECURITY_EVENT: {log_entry}")

class SecurityMiddleware:
    """Custom security middleware for additional protection"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Basic security headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Security headers
                    security_headers = {
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                        b"content-security-policy": b"default-src 'self'",
                        b"referrer-policy": b"strict-origin-when-cross-origin"
                    }
                    
                    for key, value in security_headers.items():
                        headers[key] = value
                    
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Request validation
def validate_request_size(max_size_mb: int = 10):
    """Validate request size to prevent DoS attacks"""
    async def validator(request: Request):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large (max {max_size_mb}MB)"
            )
        return True
    return validator

# IP allowlist/blocklist
class IPFilter:
    """IP filtering for additional access control"""
    
    def __init__(self):
        self.allowed_ips = set(os.getenv('JARVIS_ALLOWED_IPS', '').split(',')) if os.getenv('JARVIS_ALLOWED_IPS') else set()
        self.blocked_ips = set(os.getenv('JARVIS_BLOCKED_IPS', '').split(',')) if os.getenv('JARVIS_BLOCKED_IPS') else set()
    
    def is_allowed(self, ip: str) -> bool:
        """Check if IP is allowed to access the API"""
        if self.blocked_ips and ip in self.blocked_ips:
            return False
        if self.allowed_ips and ip not in self.allowed_ips:
            return False
        return True

ip_filter = IPFilter()

async def check_ip_access(request: Request):
    """Check IP access permissions"""
    client_ip = get_remote_address(request)
    if not ip_filter.is_allowed(client_ip):
        log_security_event('ip_blocked', {'ip': client_ip}, request)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied from this IP address"
        )
    return True

# Session management
sessions = {}

def create_session(user_data: Dict[str, Any]) -> str:
    """Create a new user session"""
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        'user': user_data,
        'created': datetime.now(),
        'last_activity': datetime.now()
    }
    return session_id

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data"""
    if session_id in sessions:
        session = sessions[session_id]
        session['last_activity'] = datetime.now()
        return session
    return None

def cleanup_expired_sessions():
    """Clean up expired sessions (call periodically)"""
    now = datetime.now()
    expired = []
    for session_id, session in sessions.items():
        if (now - session['last_activity']).total_seconds() > 3600:  # 1 hour timeout
            expired.append(session_id)
    
    for session_id in expired:
        del sessions[session_id]
    
    logger.info(f"Cleaned up {len(expired)} expired sessions")