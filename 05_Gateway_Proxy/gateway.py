"""
API Gateway Service
Handles rate limiting, caching, and request proxying to backend services
"""

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
import httpx
import sqlite3
import time
import hashlib
import json
import os
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
from typing import Optional, Dict, List
from datetime import datetime, timezone
from collections import defaultdict
import logging
from functools import lru_cache
import contextvars

app = FastAPI(title="API Gateway", version="1.0.0")

# Environment variable to control error-only logging
ERROR_ONLY_LOGGING = os.getenv("ERROR_ONLY_LOGGING", "false").lower() == "true"

# Context variable to store response status code
response_status: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar('response_status', default=None)

# Custom logging filter for error-only mode
class ErrorOnlyFilter(logging.Filter):
    """Filter to only show 400 and 500 errors when ERROR_ONLY_LOGGING is enabled"""
    
    def filter(self, record):
        if not ERROR_ONLY_LOGGING:
            return True  # Show all logs when disabled
        
        # Always show ERROR level and above
        if record.levelno >= logging.ERROR:
            return True
        
        # Check if this is an HTTP status code log
        status = response_status.get(None)
        if status is not None:
            # Only show 400 and 500 errors
            if status >= 400:
                return True
            return False
        
        # For non-HTTP logs, only show ERROR and above
        return record.levelno >= logging.ERROR

# Configure logging with filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add filter to root logger
if ERROR_ONLY_LOGGING:
    for handler in logging.root.handlers:
        handler.addFilter(ErrorOnlyFilter())
    logger.info("ERROR_ONLY_LOGGING enabled - showing only 400 and 500 errors")

# Middleware to capture response status codes
class StatusCodeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        status = response.status_code
        response_status.set(status)
        
        # Log 400 and 500 errors even in error-only mode
        if ERROR_ONLY_LOGGING and status >= 400:
            logger.error(f"{request.method} {request.url.path} - Status: {status}")
        
        return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add status code middleware (after CORS)
app.add_middleware(StatusCodeMiddleware)

# Configuration
USERS_DB = os.path.join(os.path.dirname(__file__), "..", "04_User_Service", "users.db")
DATA_INGESTION_SERVICE = os.getenv("DATA_INGESTION_SERVICE", "http://127.0.0.1:8000")
MODEL_SERVICE = os.getenv("MODEL_SERVICE", "http://127.0.0.1:8001")
USER_SERVICE = os.getenv("USER_SERVICE", "http://127.0.0.1:8002")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8003"))

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT = 100  # requests per window
PREMIUM_RATE_LIMIT = 500  # requests per window for premium users

# Server-side cache configuration
CACHE_TTL = 300  # 5 minutes default
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# In-memory cache
cache_store: Dict[str, Dict] = {}
rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# Cache statistics
cache_stats = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "expirations": 0
}


def cleanup_expired_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = []
    
    for key, cached in cache_store.items():
        if current_time - cached["timestamp"] > cached["ttl"]:
            expired_keys.append(key)
    
    for key in expired_keys:
        del cache_store[key]
        cache_stats["expirations"] += 1
    
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    # Clean up user rate limit cache
    expired_user_keys = []
    for user_id, cached in user_rate_limit_cache.items():
        if current_time - cached["timestamp"] > USER_RATE_LIMIT_CACHE_TTL:
            expired_user_keys.append(user_id)
    
    for user_id in expired_user_keys:
        del user_rate_limit_cache[user_id]
    
    return len(expired_keys) + len(expired_user_keys)


def get_user_id_from_request(request: Request) -> Optional[int]:
    """Extract user ID from request headers or query params"""
    # Try to get user_id from header
    user_id_header = request.headers.get("X-User-ID")
    if user_id_header:
        try:
            return int(user_id_header)
        except ValueError:
            pass
    
    # Try to get from query params
    user_id_param = request.query_params.get("user_id")
    if user_id_param:
        try:
            return int(user_id_param)
        except ValueError:
            pass
    
    # Try to get from Authorization header (if using token-based auth)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # In a real implementation, you'd decode the token to get user_id
        # For now, we'll use IP-based rate limiting as fallback
        pass
    
    return None


# Cache for user rate limits (TTL: 60 seconds)
user_rate_limit_cache: Dict[int, Dict] = {}
USER_RATE_LIMIT_CACHE_TTL = 60

def get_user_rate_limit(user_id: Optional[int]) -> int:
    """Get rate limit for a user from database (with caching)"""
    if user_id is None:
        return DEFAULT_RATE_LIMIT
    
    # Check cache first
    current_time = time.time()
    if user_id in user_rate_limit_cache:
        cached = user_rate_limit_cache[user_id]
        if current_time - cached["timestamp"] < USER_RATE_LIMIT_CACHE_TTL:
            logger.debug(f"✅ User rate limit cache HIT for user_id={user_id}")
            return cached["rate_limit"]
        else:
            # Cache expired
            del user_rate_limit_cache[user_id]
    
    try:
        conn = sqlite3.connect(USERS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT block_status, block_type 
            FROM users 
            WHERE id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            block_status = row["block_status"] or "active"
            # Check if user is blocked
            if block_status != "active":
                rate_limit = 0  # Blocked users have 0 rate limit
            else:
                # Premium users get higher rate limit (you can add a premium field to users table)
                # For now, all active users get default rate limit
                rate_limit = DEFAULT_RATE_LIMIT
            
            # Cache the result
            user_rate_limit_cache[user_id] = {
                "rate_limit": rate_limit,
                "timestamp": current_time
            }
            logger.debug(f"💾 User rate limit cached for user_id={user_id}: {rate_limit}")
            return rate_limit
        
        # User not found, use default and cache it
        user_rate_limit_cache[user_id] = {
            "rate_limit": DEFAULT_RATE_LIMIT,
            "timestamp": current_time
        }
        return DEFAULT_RATE_LIMIT
    except Exception as e:
        logger.error(f"Error getting user rate limit: {e}")
        return DEFAULT_RATE_LIMIT


def check_rate_limit(identifier: str, rate_limit: int) -> bool:
    """Check if request is within rate limit"""
    current_time = time.time()
    
    # Clean old entries outside the window
    rate_limit_store[identifier] = [
        timestamp 
        for timestamp in rate_limit_store[identifier] 
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(rate_limit_store[identifier]) >= rate_limit:
        return False
    
    # Add current request
    rate_limit_store[identifier].append(current_time)
    return True


def get_cache_key(request: Request) -> str:
    """Generate cache key from request"""
    path = request.url.path
    query = str(sorted(request.query_params.items()))
    method = request.method
    return hashlib.md5(f"{method}:{path}:{query}".encode()).hexdigest()


def get_cached_response(cache_key: str, path: str = "") -> Optional[Dict]:
    """Get cached response if available and not expired"""
    if not CACHE_ENABLED:
        cache_stats["misses"] += 1
        return None
    
    if cache_key not in cache_store:
        cache_stats["misses"] += 1
        return None
    
    cached = cache_store[cache_key]
    age = time.time() - cached["timestamp"]
    if age > cached["ttl"]:
        del cache_store[cache_key]
        cache_stats["expirations"] += 1
        cache_stats["misses"] += 1
        logger.debug(f"Cache expired for {path} (age: {age:.1f}s, ttl: {cached['ttl']}s)")
        return None
    
    cache_stats["hits"] += 1
    hit_rate = (cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) * 100) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
    logger.info(f"✅ CACHE HIT for {path} (age: {age:.1f}s, ttl: {cached['ttl']}s, key: {cache_key[:8]}..., hit_rate: {hit_rate:.1f}%)")
    return cached["response"]


def set_cached_response(cache_key: str, response: Dict, ttl: int = CACHE_TTL, path: str = ""):
    """Cache a response"""
    if not CACHE_ENABLED:
        return
    
    cache_store[cache_key] = {
        "response": response,
        "timestamp": time.time(),
        "ttl": ttl
    }
    cache_stats["stores"] += 1
    logger.info(f"💾 CACHE STORE for {path} (ttl: {ttl}s, key: {cache_key[:8]}..., total_entries: {len(cache_store)})")


def get_target_service(path: str) -> str:
    """Determine which backend service to route to based on path"""
    # Remove leading slash for comparison
    path_clean = path.lstrip("/")
    
    # Data Ingestion Service routes
    if (path_clean.startswith("api/data") or path_clean.startswith("upload") or 
        path_clean.startswith("view") or path_clean.startswith("training") or 
        path_clean.startswith("testing") or path_clean.startswith("validate") or 
        path_clean.startswith("insert") or path_clean.startswith("stats") or 
        path_clean.startswith("type-stats") or 
        (path_clean.startswith("health") and "data" in path_clean.lower())):
        return DATA_INGESTION_SERVICE
    
    # Model Service routes
    elif (path_clean.startswith("api/model") or path_clean.startswith("train") or 
          path_clean.startswith("test") or path_clean.startswith("predict") or 
          path_clean.startswith("models") or path_clean.startswith("model-types") or 
          path_clean.startswith("model/status") or path_clean.startswith("model/metrics")):
        return MODEL_SERVICE
    
    # User Service routes
    elif (path_clean.startswith("api/user") or path_clean.startswith("users") or 
          path_clean.startswith("history") or path_clean.startswith("network-logs") or 
          path_clean.startswith("ws/") or path_clean.startswith("set-model") or 
          path_clean.startswith("get-model") or path_clean.startswith("publish")):
        return USER_SERVICE
    
    # Default to data ingestion service
    return DATA_INGESTION_SERVICE


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limiting"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/gateway/health"]:
            return await call_next(request)
        
        user_id = get_user_id_from_request(request)
        
        # Use user_id if available, otherwise use IP address
        if user_id:
            identifier = f"user_{user_id}"
        else:
            identifier = f"ip_{request.client.host}"
        
        rate_limit = get_user_rate_limit(user_id)
        
        # Check if user is blocked
        if rate_limit == 0:
            return Response(
                content=json.dumps({"detail": "User is blocked"}),
                status_code=403,
                media_type="application/json"
            )
        
        # Check rate limit
        if not check_rate_limit(identifier, rate_limit):
            return Response(
                content=json.dumps({
                    "detail": f"Rate limit exceeded. Maximum {rate_limit} requests per {RATE_LIMIT_WINDOW} seconds."
                }),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Window": str(RATE_LIMIT_WINDOW),
                    "Retry-After": str(RATE_LIMIT_WINDOW)
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = rate_limit - len(rate_limit_store[identifier])
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + RATE_LIMIT_WINDOW))
        
        return response


app.add_middleware(RateLimitMiddleware)


@app.get("/health")
@app.get("/gateway/health")
async def health_check():
    """Gateway health check"""
    total_requests = cache_stats["hits"] + cache_stats["misses"]
    hit_rate = (cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "status": "healthy",
        "service": "API Gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_enabled": CACHE_ENABLED,
        "cache_stats": {
            "total_entries": len(cache_store),
            "user_rate_limit_entries": len(user_rate_limit_cache),
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "stores": cache_stats["stores"],
            "expirations": cache_stats["expirations"],
            "hit_rate_percent": round(hit_rate, 2)
        }
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_request(request: Request, path: str):
    """Proxy requests to appropriate backend service"""
    
    # Periodically clean up expired cache entries (every 100 requests, roughly)
    if len(cache_store) > 0 and (cache_stats["hits"] + cache_stats["misses"]) % 100 == 0:
        cleanup_expired_cache()
    
    # Skip caching for certain endpoints (analytics, write operations, etc.)
    skip_cache_paths = [
        "/train", "/test", "/predict", "/upload", "/validate", "/insert", "/publish",
        "/history", "/network-logs", "/get-model", "/set-model"
    ]
    # Note: /models (list) and /model-types are cacheable, but /model/status and /model/metrics are not
    should_cache = not any(path.startswith(skip) for skip in skip_cache_paths)
    
    # Allow caching for health endpoints (with short TTL)
    if "/health" in path:
        should_cache = True
    
    # Check cache for GET requests
    cache_key = None
    if request.method == "GET" and should_cache:
        cache_key = get_cache_key(request)
        cached_response = get_cached_response(cache_key, path)
        if cached_response:
            response = Response(
                content=json.dumps(cached_response["content"]),
                status_code=cached_response["status_code"],
                media_type="application/json",
                headers={
                    **cached_response.get("headers", {}),
                    "X-Cache": "HIT",
                    "X-Cache-Key": cache_key[:16],
                    "Cache-Control": f"public, max-age={CACHE_TTL}",
                    "ETag": f'"{cache_key}"'
                }
            )
            return response
    
    # Determine target service
    target_service = get_target_service(path)
    # Ensure path starts with / for proper URL construction
    path_with_slash = path if path.startswith("/") else f"/{path}"
    target_url = f"{target_service}{path_with_slash}"
    
    # Handle query parameters
    if request.query_params:
        from urllib.parse import urlencode
        query_string = urlencode(list(request.query_params.items()))
        separator = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{separator}{query_string}"
    
    if request.method == "GET" and should_cache:
        logger.info(f"🔄 Proxying {request.method} {path} to {target_url} (cache: MISS)")
    else:
        logger.info(f"🔄 Proxying {request.method} {path} to {target_url} (cache: BYPASS)")
    
    # Prepare headers (exclude host and connection)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("connection", None)
    headers.pop("content-length", None)
    
    # Forward request to backend service
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            # Get request body if present
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
            
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=True
            )
            
            # Read response content
            content = await response.aread()
            
            # Parse JSON if applicable
            response_data = None
            if response.headers.get("content-type", "").startswith("application/json"):
                try:
                    response_data = json.loads(content.decode())
                except:
                    response_data = {"content": content.decode()}
            
            # Cache successful GET responses with appropriate TTL based on endpoint
            if request.method == "GET" and should_cache and response.status_code == 200 and response_data:
                # Determine TTL based on endpoint type
                cache_ttl = CACHE_TTL
                if "/health" in path:
                    cache_ttl = 30  # Health checks: 30 seconds
                elif "/model-types" in path:
                    cache_ttl = 600  # Model types: 10 minutes (rarely changes)
                elif "/models" in path and "/model/" not in path:
                    cache_ttl = 180  # Model list: 3 minutes
                elif "/stats" in path or "/type-stats" in path:
                    cache_ttl = 120  # Statistics: 2 minutes
                elif "/tables" in path or "/fields" in path:
                    cache_ttl = 300  # Table/field metadata: 5 minutes
                
                set_cached_response(cache_key, {
                    "content": response_data,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }, ttl=cache_ttl, path=path)
            
            # Prepare response headers
            response_headers = dict(response.headers)
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            
            # Add caching headers for GET requests
            if request.method == "GET" and should_cache:
                if response.status_code == 200:
                    # Determine TTL for Cache-Control header
                    cache_ttl = CACHE_TTL
                    if "/health" in path:
                        cache_ttl = 30
                    elif "/model-types" in path:
                        cache_ttl = 600
                    elif "/models" in path and "/model/" not in path:
                        cache_ttl = 180
                    elif "/stats" in path or "/type-stats" in path:
                        cache_ttl = 120
                    elif "/tables" in path or "/fields" in path:
                        cache_ttl = 300
                    
                    response_headers["Cache-Control"] = f"public, max-age={cache_ttl}"
                    response_headers["ETag"] = f'"{cache_key}"'
                    response_headers["X-Cache"] = "MISS"
                    response_headers["X-Cache-Key"] = cache_key[:16] if cache_key else ""
                else:
                    response_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    response_headers["X-Cache"] = "BYPASS"
            
            # Create response
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type", "application/json")
            )
    
    except httpx.TimeoutException:
        logger.error(f"Timeout proxying request to {target_url}")
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError:
        logger.error(f"Connection error proxying to {target_url}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except Exception as e:
        logger.error(f"Error proxying request: {e}")
        raise HTTPException(status_code=502, detail=f"Gateway error: {str(e)}")


async def connect_to_data_generation_websocket():
    """Connect to the data generation WebSocket endpoint in User Service"""
    ws_url = f"{USER_SERVICE.replace('http://', 'ws://')}/ws/generate-data"
    logger.info(f"Connecting to data generation WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("Connected to data generation WebSocket")
            while True:
                try:
                    # Keep connection alive - we don't need to process messages
                    message = await websocket.recv()
                    logger.debug(f"Received message from data generation WebSocket: {message[:100]}")
                except ConnectionClosed:
                    logger.warning("Data generation WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error in data generation WebSocket: {e}")
                    break
    except Exception as e:
        logger.error(f"Failed to connect to data generation WebSocket: {e}")
    
    # Reconnect after delay
    await asyncio.sleep(5)
    asyncio.create_task(connect_to_data_generation_websocket())


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    # Connect to data generation WebSocket
    asyncio.create_task(connect_to_data_generation_websocket())
    logger.info("Started data generation WebSocket connection task")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API Gateway on port {GATEWAY_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=GATEWAY_PORT)
