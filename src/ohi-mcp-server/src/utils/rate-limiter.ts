/**
 * Rate Limiter
 * ============
 * 
 * Per-source rate limiting with sliding window.
 */

interface RateLimitConfig {
  requestsPerSecond: number;
  burstSize: number;
}

interface TokenBucket {
  tokens: number;
  lastRefill: number;
  config: RateLimitConfig;
}

// Default rate limits per source (requests per second)
const DEFAULT_LIMITS: Record<string, RateLimitConfig> = {
  wikidata: { requestsPerSecond: 5, burstSize: 10 },
  mediawiki: { requestsPerSecond: 10, burstSize: 20 },
  wikimedia_rest: { requestsPerSecond: 10, burstSize: 20 },
  dbpedia: { requestsPerSecond: 2, burstSize: 5 },
  openalex: { requestsPerSecond: 10, burstSize: 20 },
  crossref: { requestsPerSecond: 5, burstSize: 10 },
  europepmc: { requestsPerSecond: 5, burstSize: 10 },
  ncbi: { requestsPerSecond: 3, burstSize: 5 }, // 3/s without API key
  clinicaltrials: { requestsPerSecond: 5, burstSize: 10 },
  opencitations: { requestsPerSecond: 5, burstSize: 10 },
  gdelt: { requestsPerSecond: 2, burstSize: 5 },
  worldbank: { requestsPerSecond: 5, burstSize: 10 },
  osv: { requestsPerSecond: 5, burstSize: 10 },
  context7: { requestsPerSecond: 0.5, burstSize: 1 },
};

export class RateLimiter {
  private buckets: Map<string, TokenBucket> = new Map();
  private requestCounts: Map<string, number> = new Map();

  constructor(private customLimits: Record<string, RateLimitConfig> = {}) {}

  private getBucket(source: string): TokenBucket {
    let bucket = this.buckets.get(source);
    if (!bucket) {
      const config = this.customLimits[source] || DEFAULT_LIMITS[source] || {
        requestsPerSecond: 5,
        burstSize: 10,
      };
      bucket = {
        tokens: config.burstSize,
        lastRefill: Date.now(),
        config,
      };
      this.buckets.set(source, bucket);
    }
    return bucket;
  }

  private refillTokens(bucket: TokenBucket): void {
    const now = Date.now();
    const elapsed = (now - bucket.lastRefill) / 1000;
    const tokensToAdd = elapsed * bucket.config.requestsPerSecond;
    bucket.tokens = Math.min(bucket.config.burstSize, bucket.tokens + tokensToAdd);
    bucket.lastRefill = now;
  }

  /**
   * Check if a request can proceed. Returns delay in ms if rate limited.
   */
  async acquire(source: string): Promise<void> {
    const bucket = this.getBucket(source);
    this.refillTokens(bucket);

    if (bucket.tokens >= 1) {
      bucket.tokens -= 1;
      this.incrementCount(source);
      return;
    }

    // Calculate wait time
    const waitTime = (1 - bucket.tokens) / bucket.config.requestsPerSecond * 1000;
    await this.sleep(waitTime);
    
    // Refill and try again
    this.refillTokens(bucket);
    bucket.tokens -= 1;
    this.incrementCount(source);
  }

  private incrementCount(source: string): void {
    const current = this.requestCounts.get(source) || 0;
    this.requestCounts.set(source, current + 1);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getStats(): Record<string, { requests: number; tokensRemaining: number }> {
    const stats: Record<string, { requests: number; tokensRemaining: number }> = {};
    for (const [source, bucket] of this.buckets) {
      stats[source] = {
        requests: this.requestCounts.get(source) || 0,
        tokensRemaining: Math.floor(bucket.tokens),
      };
    }
    return stats;
  }
}
