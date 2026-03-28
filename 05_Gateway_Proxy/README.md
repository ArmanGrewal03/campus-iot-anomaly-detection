# API Gateway Service

The API Gateway provides rate limiting, caching, and request proxying for all backend services.

## Features

### Rate Limiting
- **User-based rate limiting**: Checks user status from the user database
- **IP-based fallback**: Uses IP address if user ID is not provided
- **Configurable limits**: 
  - Default: 100 requests per 60 seconds
  - Blocked users: 0 requests (automatically blocked)
  - Premium users: 500 requests per 60 seconds (configurable)
- **Rate limit headers**: Returns `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers

### Caching
- **Server-side caching**: In-memory cache for GET requests (5 minutes TTL by default)
- **Client-side caching**: Adds `Cache-Control` and `ETag` headers
- **Cache headers**: 
  - `X-Cache: HIT` for cached responses
  - `X-Cache: MISS` for fresh responses
- **Smart caching**: Automatically skips caching for write operations (POST, PUT, DELETE, PATCH)

### Request Proxying & Load Balancing
Routes requests to appropriate backend services and performs simple round-robin load balancing:
- **Data Ingestion Service** (port 8000): `/upload`, `/view`, `/training`, `/testing`, `/validate`, `/insert`, `/stats`, `/type-stats`
- **Model Service** (port 8001): `/train`, `/test`, `/predict`, `/models`, `/model-types`, `/model/status`, `/model/metrics`
- **User Service** (port 8002): `/users`, `/history`, `/network-logs`, `/set-model`, `/get-model`, `/publish`

Each service supports **multiple upstream instances** via comma-separated environment variables:

- `DATA_INGESTION_SERVICE` (default: `http://127.0.0.1:8000`)
- `MODEL_SERVICE` (default: `http://127.0.0.1:8001`)
- `USER_SERVICE` (default: `http://127.0.0.1:8002`)

Example (docker-compose override or k8s):

```yaml
environment:
  - DATA_INGESTION_SERVICE=http://data-ingestion-1:8000,http://data-ingestion-2:8000
  - MODEL_SERVICE=http://model-service-1:8001,http://model-service-2:8001
  - USER_SERVICE=http://user-service-1:8002,http://user-service-2:8002
```

The gateway will round-robin across the configured upstreams for each request. This is a **per-process, in-memory** L4-style load balancer (no shared state across multiple gateway instances).

## Configuration

Environment variables:
- `DATA_INGESTION_SERVICE`: Data Ingestion Service URL (default: `http://127.0.0.1:8000`)
- `MODEL_SERVICE`: Model Service URL (default: `http://127.0.0.1:8001`)
- `USER_SERVICE`: User Service URL (default: `http://127.0.0.1:8002`)
- `GATEWAY_PORT`: Gateway port (default: `8003`)
- `CACHE_ENABLED`: Enable/disable caching (default: `true`)

## Usage

### Starting the Gateway

```powershell
# From the project root
cd 05_Gateway_Proxy
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python gateway.py
```

Or use the provided script:
```powershell
.\scripts\run-05-gateway.ps1
```

### Making Requests

All requests should go through the gateway at `http://127.0.0.1:8003`:

```bash
# With user ID header
curl -H "X-User-ID: 1" http://127.0.0.1:8003/view

# With query parameter
curl "http://127.0.0.1:8003/view?user_id=1"

# Rate limit headers in response
curl -I http://127.0.0.1:8003/view
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1234567890
```

### Rate Limit Responses

When rate limit is exceeded:
```json
{
  "detail": "Rate limit exceeded. Maximum 100 requests per 60 seconds."
}
```
Status code: `429 Too Many Requests`
Headers: `Retry-After: 60`

### Blocked Users

Blocked users receive:
```json
{
  "detail": "User is blocked"
}
```
Status code: `403 Forbidden`

## Architecture

```
Client Request
    ↓
API Gateway (Port 8003)
    ├─ Rate Limiting Middleware
    ├─ Cache Check (GET requests)
    ├─ Request Routing
    └─ Backend Service
        ├─ Data Ingestion (8000)
        ├─ Model Service (8001)
        └─ User Service (8002)
```

## Notes

- WebSocket connections should connect directly to the User Service (port 8002)
- Health checks (`/health`, `/gateway/health`) bypass rate limiting
- Cache is automatically invalidated after TTL expires
- Rate limiting uses a sliding window algorithm
