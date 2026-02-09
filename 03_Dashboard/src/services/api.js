import cache from "./cache";

const API_BASE = process.env.VUE_APP_API_BASE || "";

async function request(path, useCache = true) {
  // Check cache first
  if (useCache) {
    const cached = cache.get(path);
    if (cached !== null) {
      return cached;
    }
  }
  
  const url = `${API_BASE}${path}`;
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.detail || `Request failed: ${response.status}`);
  }
  
  // Cache successful responses
  if (useCache) {
    cache.set(path, data);
  }
  
  return data;
}

export async function getHealth(useCache = true) {
  return request("/api/health", useCache);
}

export async function getStats(useCache = true) {
  return request("/api/stats", useCache);
}

export async function getTypeStats(useCache = true) {
  return request("/api/type-stats", useCache);
}

export async function getView(limit = 100, offset = 0, useCache = true) {
  // Don't cache view requests with different params
  const cacheKey = `/api/view?limit=${limit}&offset=${offset}`;
  return request(cacheKey, useCache);
}

export function clearCache() {
  cache.clear();
}
