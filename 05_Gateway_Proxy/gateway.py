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
from typing import Optional, Dict, List, Tuple
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


# Input validation middleware
class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize input parameters"""
    
    def validate_integer_param(self, value: str, param_name: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
        """Validate integer parameter"""
        try:
            int_value = int(value)
            if min_val is not None and int_value < min_val:
                return None, f"{param_name} must be >= {min_val}"
            if max_val is not None and int_value > max_val:
                return None, f"{param_name} must be <= {max_val}"
            return int_value, None
        except ValueError:
            return None, f"{param_name} must be a valid integer"
    
    def validate_string_param(self, value: str, param_name: str, min_len: int = 1, max_len: Optional[int] = None, allow_empty: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Validate string parameter"""
        if not value and not allow_empty:
            return None, f"{param_name} cannot be empty"
        if len(value) < min_len:
            return None, f"{param_name} must be at least {min_len} characters"
        if max_len and len(value) > max_len:
            return None, f"{param_name} must be <= {max_len} characters"
        return value.strip(), None
    
    def validate_boolean_param(self, value: str, param_name: str) -> Tuple[Optional[bool], Optional[str]]:
        """Validate boolean parameter"""
        value_lower = value.lower()
        if value_lower in ["true", "1", "yes"]:
            return True, None
        elif value_lower in ["false", "0", "no", ""]:
            return False, None
        return None, f"{param_name} must be 'true' or 'false'"
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for health checks
        if request.url.path in ["/health", "/gateway/health"]:
            return await call_next(request)
        
        # Validate query parameters
        validation_errors = []
        path = request.url.path
        
        # Validate limit parameter (if present)
        if "limit" in request.query_params:
            limit, error = self.validate_integer_param(request.query_params["limit"], "limit", min_val=1, max_val=10000)
            if error:
                validation_errors.append(error)
        
        # Validate offset parameter (if present)
        if "offset" in request.query_params:
            offset, error = self.validate_integer_param(request.query_params["offset"], "offset", min_val=0)
            if error:
                validation_errors.append(error)
        
        # Validate user_id parameter (if present)
        if "user_id" in request.query_params:
            user_id, error = self.validate_integer_param(request.query_params["user_id"], "user_id", min_val=1)
            if error:
                validation_errors.append(error)
        
        # Validate network_id parameter (if present)
        if "network_id" in request.query_params:
            network_id, error = self.validate_string_param(request.query_params["network_id"], "network_id", max_len=255)
            if error:
                validation_errors.append(error)
        
        # Validate boolean parameters
        for bool_param in ["has_prediction", "is_active"]:
            if bool_param in request.query_params:
                _, error = self.validate_boolean_param(request.query_params[bool_param], bool_param)
                if error:
                    validation_errors.append(error)
        
        # Validate dataset_name/database_name (if present)
        for name_param in ["dataset_name", "database_name"]:
            if name_param in request.query_params:
                name, error = self.validate_string_param(request.query_params[name_param], name_param, max_len=255)
                if error:
                    validation_errors.append(error)
        
        # Validate X-User-ID header (if present)
        user_id_header = request.headers.get("X-User-ID")
        if user_id_header:
            user_id, error = self.validate_integer_param(user_id_header, "X-User-ID header", min_val=1)
            if error:
                validation_errors.append(error)
        
        # Validate request body for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # Check content-type
            if "application/json" in content_type:
                try:
                    # Only read body if it's manageable size for validation
                    # For very large JSON, we should stream or skip basic validation
                    content_length = request.headers.get("content-length")
                    if content_length and int(content_length) > 10 * 1024 * 1024:
                        logger.warning(f"JSON body too large ({content_length} bytes), bypassing gateway validation")
                    else:
                        body = await request.body()
                        # Validate JSON structure
                        try:
                            json_data = json.loads(body.decode('utf-8'))
                            # Store validated body for later use
                            request.state.validated_body = body
                            request.state.validated_json = json_data
                        except json.JSONDecodeError as e:
                            validation_errors.append(f"Invalid JSON: {str(e)}")
                except Exception as e:
                    validation_errors.append(f"Error reading request body: {str(e)}")
            elif "multipart/form-data" in content_type or "text/csv" in content_type:
                # For file uploads, we'll let the backend handle validation via streaming
                content_length = request.headers.get("content-length")
                if content_length:
                    try:
                        size = int(content_length)
                        MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # Increased to 500MB
                        if size > MAX_UPLOAD_SIZE:
                            validation_errors.append(f"Upload too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB")
                    except ValueError:
                        pass
        
        # Validate path parameters (basic sanitization)
        # Check for path traversal attempts
        if ".." in path or "//" in path:
            validation_errors.append("Invalid path: path traversal detected")
        
        # Check for null bytes in path
        if "\x00" in path:
            validation_errors.append("Invalid path: null byte detected")
        
        # If validation errors exist, return 400
        if validation_errors:
            logger.warning(f"Validation errors for {request.method} {path} from {request.client.host if request.client else 'unknown'}: {validation_errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation failed",
                    "errors": validation_errors,
                    "path": path
                }
            )
        
        response = await call_next(request)
        return response


# Add input validation middleware
app.add_middleware(InputValidationMiddleware)

# Configuration
USERS_DB = os.path.join(os.path.dirname(__file__), "..", "04_User_Service", "users.db")

# Upstream service configuration
# You can provide a single URL or a comma-separated list of URLs for basic L4-style load balancing.
# Example:
#   DATA_INGESTION_SERVICE="http://data-ingestion-1:8000,http://data-ingestion-2:8000"
DATA_INGESTION_SERVICE = os.getenv("DATA_INGESTION_SERVICE", "http://127.0.0.1:8000")
MODEL_SERVICE = os.getenv("MODEL_SERVICE", "http://127.0.0.1:8001")
USER_SERVICE = os.getenv("USER_SERVICE", "http://127.0.0.1:8002")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8003"))

# Parsed upstream lists for simple round-robin load balancing
def _parse_upstream_list(value: str) -> List[str]:
    """Parse a comma-separated list of upstream URLs into a clean list."""
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or [value.strip()]

DATA_INGESTION_UPSTREAMS: List[str] = _parse_upstream_list(DATA_INGESTION_SERVICE)
MODEL_SERVICE_UPSTREAMS: List[str] = _parse_upstream_list(MODEL_SERVICE)
USER_SERVICE_UPSTREAMS: List[str] = _parse_upstream_list(USER_SERVICE)

# Round-robin indices (simple in-memory counters; per-process, not shared)
_data_ingestion_rr_index = 0
_model_service_rr_index = 0
_user_service_rr_index = 0


def _get_next_upstream(upstreams: List[str], index_attr_name: str, service_label: str) -> str:
    """
    Very simple round-robin upstream selector.
    This is per-process and not strictly thread-safe, but good enough for this gateway.
    """
    global _data_ingestion_rr_index, _model_service_rr_index, _user_service_rr_index

    if not upstreams:
        raise RuntimeError(f"No upstreams configured for {service_label}")

    # Select and advance index
    if index_attr_name == "data_ingestion":
        idx = _data_ingestion_rr_index % len(upstreams)
        _data_ingestion_rr_index += 1
    elif index_attr_name == "model":
        idx = _model_service_rr_index % len(upstreams)
        _model_service_rr_index += 1
    elif index_attr_name == "user":
        idx = _user_service_rr_index % len(upstreams)
        _user_service_rr_index += 1
    else:
        # Fallback: first upstream
        idx = 0

    upstream = upstreams[idx]
    return upstream

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
    """Generate cache key from request including dataset/model headers"""
    path = request.url.path
    query = str(sorted(request.query_params.items()))
    method = request.method
    ds = request.headers.get("dataset-name", "")
    mn = request.headers.get("model-name", "")
    return hashlib.md5(f"{method}:{path}:{query}:{ds}:{mn}".encode()).hexdigest()


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
    """
    Determine which backend service to route to based on path
    and select an upstream instance using simple round-robin.
    """
    # Remove leading slash for comparison
    path_clean = path.lstrip("/")
    
    # Data Ingestion Service routes
    if (path_clean.startswith("api/data") or path_clean.startswith("upload") or 
        path_clean.startswith("view") or path_clean.startswith("training") or 
        path_clean.startswith("testing") or path_clean.startswith("random-test") or 
        path_clean.startswith("validate") or 
        path_clean.startswith("insert") or path_clean.startswith("stats") or 
        path_clean.startswith("type-stats") or 
        (path_clean.startswith("health") and "data" in path_clean.lower())):
        return _get_next_upstream(DATA_INGESTION_UPSTREAMS, "data_ingestion", "Data Ingestion Service")
    
    # Model Service routes
    elif (path_clean.startswith("api/model") or path_clean.startswith("train") or 
          path_clean.startswith("test") or path_clean.startswith("predict") or 
          path_clean.startswith("models") or path_clean.startswith("model-types") or 
          path_clean.startswith("model/status") or path_clean.startswith("model/metrics")):
        return _get_next_upstream(MODEL_SERVICE_UPSTREAMS, "model", "Model Service")
    
    # User Service routes
    elif (
        path_clean.startswith("api/user")
        or path_clean.startswith("users")
        or path_clean.startswith("history")
        or path_clean.startswith("network-logs")
        or path_clean.startswith("ws/")
        or path_clean.startswith("set-model")
        or path_clean.startswith("get-model")
        or path_clean.startswith("publish")
        or path_clean.startswith("recompute-predictions")
        or path_clean.startswith("dashboard-kpis")
    ):
        return _get_next_upstream(USER_SERVICE_UPSTREAMS, "user", "User Service")
    
    # Default to data ingestion service
    return _get_next_upstream(DATA_INGESTION_UPSTREAMS, "data_ingestion", "Data Ingestion Service")


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
    # Normalize path to ensure it starts with / and remove query params for matching
    normalized_path = path if path.startswith("/") else f"/{path}"
    # Extract just the path part (without query string) for cache skip checking
    path_without_query = normalized_path.split("?")[0] if "?" in normalized_path else normalized_path
    
    skip_cache_paths = [
        "/train", "/test", "/predict", "/upload", "/validate", "/insert", "/publish",
        "/history", "/network-logs", "/get-model", "/set-model", "/models", "/model-types",
        "/model/status", "/model/metrics", "/recompute-predictions", "/dashboard-kpis",
        "/tables", "/fields", "/new", "/clear", "/view", "/stats", "/type-stats"
    ]
    # Check if path (without query) starts with any skip path
    should_cache = not any(path_without_query.startswith(skip) for skip in skip_cache_paths)
    
    # Allow caching for health endpoints (with short TTL)
    if "/health" in path_without_query:
        should_cache = True
    
    # IMPORTANT: Never cache analytics endpoints, even if there's an old cache entry
    # If this is an analytics endpoint, clear any existing cache entry for it
    if any(path_without_query.startswith(skip) for skip in skip_cache_paths):
        # Clear any existing cache entries for this path to prevent stale data
        if request.method == "GET":
            temp_cache_key = get_cache_key(request)
            if temp_cache_key in cache_store:
                del cache_store[temp_cache_key]
                logger.debug(f"Cleared stale cache entry for analytics endpoint: {path_without_query}")
    
    # Check cache for GET requests ONLY if should_cache is True
    # Also verify the path is not in skip_cache_paths as a safety check
    cache_key = None
    if request.method == "GET" and should_cache and not any(path_without_query.startswith(skip) for skip in skip_cache_paths):
        cache_key = get_cache_key(request)
        cached_response = get_cached_response(cache_key, path)
        if cached_response:
            # Remove Content-Length from cached headers to avoid mismatches
            cached_headers = {}
            for key, value in cached_response.get("headers", {}).items():
                if key.lower() not in ["content-length", "transfer-encoding", "content-encoding"]:
                    cached_headers[key] = value
            
            cached_content = json.dumps(cached_response["content"])
            response = Response(
                content=cached_content,
                status_code=cached_response["status_code"],
                media_type="application/json",
                headers={
                    **cached_headers,
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
    # headers.pop("content-length", None)  # KEEP content-length for large uploads
    
    # Forward request to backend service
    try:
        logger.debug(f"Making {request.method} request to {target_url} with headers: {list(headers.keys())}")
        async with httpx.AsyncClient(timeout=None) as client:
            # Get request body if present (use validated body if available)
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                # Use validated body if available (from validation middleware)
                if hasattr(request.state, "validated_body"):
                    body = request.state.validated_body
                else:
                    body = await request.body()
            
            try:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                    follow_redirects=True
                )
                logger.debug(f"Received response from {target_url}: {response.status_code}")
            except Exception as req_err:
                logger.error(f"Error making request to {target_url}: {type(req_err).__name__}: {req_err}", exc_info=True)
                raise
            
            # Get response content as bytes
            # Use response.content which is a property that returns bytes synchronously
            # This ensures we get the exact bytes without any encoding issues
            try:
                content_bytes = response.content
                # Ensure we have bytes
                if not isinstance(content_bytes, bytes):
                    if isinstance(content_bytes, str):
                        content_bytes = content_bytes.encode('utf-8')
                    else:
                        content_bytes = bytes(content_bytes)
                logger.debug(f"Retrieved {len(content_bytes)} bytes from response")
            except Exception as content_err:
                logger.error(f"Error reading response content: {type(content_err).__name__}: {content_err}", exc_info=True)
                raise
            
            # Parse JSON if applicable (for caching purposes only)
            response_data = None
            if response.headers.get("content-type", "").startswith("application/json"):
                try:
                    response_data = json.loads(content_bytes.decode('utf-8'))
                except Exception as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    response_data = None
            
            # Cache successful GET responses with appropriate TTL based on endpoint
            # Double-check we're not caching analytics endpoints (defense in depth)
            if request.method == "GET" and should_cache and response.status_code == 200 and response_data and not any(path_without_query.startswith(skip) for skip in skip_cache_paths):
                # Determine TTL based on endpoint type
                cache_ttl = CACHE_TTL
                if "/health" in path_without_query:
                    cache_ttl = 30  # Health checks: 30 seconds
                elif "/model-types" in path_without_query:
                    cache_ttl = 600  # Model types: 10 minutes (rarely changes)
                elif "/stats" in path_without_query or "/type-stats" in path_without_query:
                    cache_ttl = 120  # Statistics: 2 minutes
                elif "/tables" in path_without_query or "/fields" in path_without_query:
                    cache_ttl = 300  # Table/field metadata: 5 minutes
                
                set_cached_response(cache_key, {
                    "content": response_data,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }, ttl=cache_ttl, path=path)
            
            # Prepare response headers - create a new dict and explicitly exclude problematic headers
            response_headers = {}
            for key, value in response.headers.items():
                # Skip headers that FastAPI should handle automatically
                if key.lower() not in ["content-length", "transfer-encoding", "connection", "keep-alive"]:
                    response_headers[key] = value
            
            # Always add no-cache headers for analytics endpoints
            analytics_paths = ["/history", "/network-logs", "/get-model", "/set-model", "/models", "/recompute-predictions", "/dashboard-kpis"]
            is_analytics_endpoint = any(path_without_query.startswith(analytics_path) for analytics_path in analytics_paths)
            
            # Add caching headers for GET requests (skip for analytics endpoints)
            if request.method == "GET" and should_cache and not is_analytics_endpoint:
                if response.status_code == 200:
                    # Determine TTL for Cache-Control header
                    cache_ttl = CACHE_TTL
                    if "/health" in path_without_query:
                        cache_ttl = 30
                    elif "/model-types" in path_without_query:
                        cache_ttl = 600
                    elif "/stats" in path_without_query or "/type-stats" in path_without_query:
                        cache_ttl = 120
                    elif "/tables" in path_without_query or "/fields" in path_without_query:
                        cache_ttl = 300
                    
                    response_headers["Cache-Control"] = f"public, max-age={cache_ttl}"
                    response_headers["ETag"] = f'"{cache_key}"'
                    response_headers["X-Cache"] = "MISS"
                    response_headers["X-Cache-Key"] = cache_key[:16] if cache_key else ""
                else:
                    response_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    response_headers["Pragma"] = "no-cache"
                    response_headers["Expires"] = "0"
                    response_headers["X-Cache"] = "BYPASS"
            
            # Force no-cache for analytics endpoints
            if is_analytics_endpoint:
                response_headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response_headers["Pragma"] = "no-cache"
                response_headers["Expires"] = "0"
                response_headers["X-Cache"] = "BYPASS"
            
            # Create response with original content bytes
            # FastAPI will automatically calculate the correct content-length based on content_bytes
            # Ensure content is bytes (not string) to avoid encoding issues
            if not isinstance(content_bytes, bytes):
                if isinstance(content_bytes, str):
                    content_bytes = content_bytes.encode('utf-8')
                else:
                    content_bytes = bytes(content_bytes)
            
            # Get content type from original response
            content_type = response.headers.get("content-type", "application/json")
            
            # Clean content-type header (remove charset for JSON, FastAPI will handle it)
            if content_type.startswith("application/json"):
                if "charset" in content_type:
                    content_type = "application/json"
            
            logger.debug(f"Returning response with status {response.status_code}, {len(content_bytes)} bytes, content-type: {content_type}")
            
            # Create Response - FastAPI will automatically set Content-Length based on content_bytes size
            # Do NOT include Content-Length in headers - let FastAPI calculate it
            # Double-check headers don't contain Content-Length or Transfer-Encoding
            final_headers = {}
            for key, value in response_headers.items():
                key_lower = key.lower()
                # Explicitly exclude any headers that might interfere with Content-Length calculation
                if key_lower not in ["content-length", "transfer-encoding", "content-encoding", "content-range"]:
                    final_headers[key] = value
            
            # Use Response with explicit content_length=None to let FastAPI calculate it
            # This ensures the Content-Length matches the actual content size
            return Response(
                content=content_bytes,
                status_code=response.status_code,
                headers=final_headers,
                media_type=content_type
            )
    
    except httpx.TimeoutException:
        logger.error(f"Timeout proxying {request.method} {path} to {target_url}")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "Gateway timeout",
                "message": f"Backend service did not respond in time",
                "service": target_service,
                "path": path
            }
        )
    except httpx.ConnectError as e:
        # Determine service name for better error message
        service_name = "Unknown"
        # Map based on which upstream list contains the base URL (best-effort)
        if any(str(target_service).startswith(u) for u in DATA_INGESTION_UPSTREAMS):
            service_name = "Data Ingestion Service"
        elif any(str(target_service).startswith(u) for u in MODEL_SERVICE_UPSTREAMS):
            service_name = "Model Service"
        elif any(str(target_service).startswith(u) for u in USER_SERVICE_UPSTREAMS):
            service_name = "User Service"
        
        logger.error(f"Connection error proxying {request.method} {path} to {target_url} - {service_name} appears to be down: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Backend service unavailable",
                "message": f"{service_name} is not responding. Please ensure the service is running.",
                "service": target_service,
                "service_name": service_name,
                "path": path,
                "target_url": target_url
            }
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} from {target_url}: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "error": "Backend service error",
                "message": f"Backend service returned error status {e.response.status_code}",
                "service": target_service,
                "path": path
            }
        )
    except Exception as e:
        logger.error(f"Error proxying {request.method} {path} to {target_url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Gateway error",
                "message": f"Unexpected error while proxying request: {str(e)}",
                "service": target_service,
                "path": path
            }
        )


async def connect_to_data_generation_websocket():
    """Connect to the data generation WebSocket endpoint in User Service"""
    # Use the first configured User Service upstream for WebSocket connection
    user_ws_base = USER_SERVICE_UPSTREAMS[0] if USER_SERVICE_UPSTREAMS else USER_SERVICE
    ws_url = f"{user_ws_base.replace('http://', 'ws://')}/ws/generate-data"
    reconnect_delay = 5  # Start with 5 seconds
    max_delay = 60  # Maximum delay of 60 seconds
    consecutive_failures = 0
    
    while True:
        try:
            # Check if User Service is up before attempting WebSocket connection
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    health_check = await client.get(f"{user_ws_base}/health")
                    if health_check.status_code != 200:
                        raise Exception("Health check failed")
            except Exception:
                # Service is down, wait before retrying
                if consecutive_failures == 0:
                    logger.warning(f"User Service appears to be down. Will retry WebSocket connection in {reconnect_delay}s")
                consecutive_failures += 1
                await asyncio.sleep(reconnect_delay)
                # Exponential backoff with max delay
                reconnect_delay = min(reconnect_delay * 1.5, max_delay)
                continue
            
            # Service is up, attempt WebSocket connection
            logger.info(f"Connecting to data generation WebSocket: {ws_url}")
            async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10) as websocket:
                logger.info("Connected to data generation WebSocket")
                consecutive_failures = 0
                reconnect_delay = 5  # Reset delay on successful connection
                
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
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {ws_url}")
            await asyncio.sleep(30)  # Wait longer for configuration errors
            continue
        except (ConnectionRefusedError, OSError) as e:
            # Connection refused - service is likely down
            consecutive_failures += 1
            if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                # Only log every 10th failure to reduce log noise
                logger.warning(f"User Service WebSocket connection refused (attempt {consecutive_failures}). Service may be down. Retrying in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_delay)
            continue
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures == 1 or consecutive_failures % 5 == 0:
                logger.error(f"Failed to connect to data generation WebSocket (attempt {consecutive_failures}): {e}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_delay)
            continue


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
