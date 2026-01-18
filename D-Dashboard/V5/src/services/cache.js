// Simple in-memory cache - no expiration
// Cache persists for instant display, but fresh data is always fetched in background
const cache = {
  data: new Map(),
  
  set(key, value) {
    this.data.set(key, {
      value,
      timestamp: Date.now(),
    });
  },
  
  get(key) {
    const entry = this.data.get(key);
    if (!entry) return null;
    // Always return cached value (no expiration check)
    // Fresh data is fetched in background on every page visit
    return entry.value;
  },
  
  clear() {
    this.data.clear();
  },
};

export default cache;
