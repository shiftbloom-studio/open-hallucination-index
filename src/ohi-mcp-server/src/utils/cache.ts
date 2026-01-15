/**
 * Response Cache
 * ==============
 * 
 * LRU cache with TTL for API responses.
 */

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

export class ResponseCache {
  private cache: Map<string, CacheEntry<unknown>> = new Map();
  private maxSize = 1000;
  private hits = 0;
  private misses = 0;

  constructor(
    private ttlSeconds: number = 300,
    private enabled: boolean = true
  ) {}

  private generateKey(source: string, operation: string, params: Record<string, unknown>): string {
    const paramStr = JSON.stringify(params, Object.keys(params).sort());
    return `${source}:${operation}:${paramStr}`;
  }

  get<T>(source: string, operation: string, params: Record<string, unknown>): T | undefined {
    if (!this.enabled) return undefined;

    const key = this.generateKey(source, operation, params);
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return undefined;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      this.misses++;
      return undefined;
    }

    this.hits++;
    return entry.value as T;
  }

  set<T>(source: string, operation: string, params: Record<string, unknown>, value: T): void {
    if (!this.enabled) return;

    const key = this.generateKey(source, operation, params);

    // LRU eviction
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      value,
      expiresAt: Date.now() + this.ttlSeconds * 1000,
    });
  }

  invalidate(source: string): void {
    for (const key of this.cache.keys()) {
      if (key.startsWith(`${source}:`)) {
        this.cache.delete(key);
      }
    }
  }

  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  getStats(): { size: number; hits: number; misses: number; hitRate: string } {
    const total = this.hits + this.misses;
    const hitRate = total > 0 ? ((this.hits / total) * 100).toFixed(1) + "%" : "N/A";
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate,
    };
  }
}
