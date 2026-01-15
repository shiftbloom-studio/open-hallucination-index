/**
 * Source Registry
 * ================
 * 
 * Central registry for all knowledge sources.
 * Manages initialization, health checks, and source discovery.
 */

import { BaseSource, SearchResult } from "./base.js";
import { WikidataSource } from "./wikidata.js";
import { MediaWikiSource } from "./mediawiki.js";
import { WikimediaRESTSource } from "./wikimedia-rest.js";
import { DBpediaSource } from "./dbpedia.js";
import { OpenAlexSource } from "./openalex.js";
import { CrossrefSource } from "./crossref.js";
import { EuropePMCSource } from "./europepmc.js";
import { NCBISource } from "./ncbi.js";
import { ClinicalTrialsSource } from "./clinicaltrials.js";
import { OpenCitationsSource } from "./opencitations.js";
import { GDELTSource } from "./gdelt.js";
import { WorldBankSource } from "./worldbank.js";
import { OSVSource } from "./osv.js";

export type SourceCategory =
  | "wikipedia"
  | "academic"
  | "medical"
  | "news"
  | "economic"
  | "security"
  | "all";

interface SourceHealth {
  name: string;
  healthy: boolean;
  lastCheck: Date;
  latency?: number;
}

class SourceRegistry {
  private sources: Map<string, BaseSource> = new Map();
  private healthStatus: Map<string, SourceHealth> = new Map();
  private categories: Map<SourceCategory, string[]> = new Map();
  private initialized = false;
  private maxParallel = 1;
  private perSourceDelayMs = 0;

  /**
   * Initialize all sources and perform health checks.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    console.log("[Registry] Initializing knowledge sources...");

    // Get polite pool email from environment for Crossref and OpenAlex
    const politeEmail = process.env.POLITE_POOL_EMAIL || process.env.CONTACT_EMAIL;
    
    if (politeEmail) {
      console.log(`[Registry] Using polite pool with email: ${politeEmail}`);
    } else {
      console.log("[Registry] Warning: No POLITE_POOL_EMAIL set - using anonymous access (lower rate limits)");
    }

    // Read throttling settings (defaults keep MCP gentle for external APIs)
    this.maxParallel = Math.max(1, parseInt(process.env.MCP_MAX_PARALLEL_SOURCES || "1", 10));
    this.perSourceDelayMs = Math.max(0, parseInt(process.env.MCP_SOURCE_DELAY_MS || "0", 10));

    if (this.maxParallel > 1) {
      console.log(`[Registry] MCP parallelism set to ${this.maxParallel}`);
    } else {
      console.log("[Registry] MCP parallelism set to 1 (serialized)");
    }
    if (this.perSourceDelayMs > 0) {
      console.log(`[Registry] MCP per-source delay ${this.perSourceDelayMs}ms`);
    }

    // Register all sources
    this.register(new WikidataSource());
    this.register(new MediaWikiSource());
    this.register(new WikimediaRESTSource());
    this.register(new DBpediaSource());
    this.register(new OpenAlexSource(politeEmail));  // Polite pool
    this.register(new CrossrefSource(politeEmail));  // Polite pool
    this.register(new EuropePMCSource());
    this.register(new NCBISource());
    this.register(new ClinicalTrialsSource());
    this.register(new OpenCitationsSource());
    this.register(new GDELTSource());
    this.register(new WorldBankSource());
    this.register(new OSVSource());

    // Define categories
    this.categories.set("wikipedia", ["wikidata", "mediawiki", "wikimedia-rest", "dbpedia"]);
    this.categories.set("academic", ["openalex", "crossref", "europepmc", "opencitations"]);
    this.categories.set("medical", ["europepmc", "ncbi", "clinicaltrials"]);
    this.categories.set("news", ["gdelt"]);
    this.categories.set("economic", ["worldbank"]);
    this.categories.set("security", ["osv"]);
    this.categories.set("all", Array.from(this.sources.keys()));

    // Run initial health checks in parallel
    await this.checkAllHealth();

    this.initialized = true;
    console.log(`[Registry] Initialized ${this.sources.size} sources`);
  }

  /**
   * Register a new source.
   */
  private register(source: BaseSource): void {
    this.sources.set(source.name, source);
    console.log(`[Registry] Registered source: ${source.name}`);
  }

  /**
   * Get a source by name.
   */
  get(name: string): BaseSource | undefined {
    return this.sources.get(name);
  }

  /**
   * Get all sources.
   */
  getAll(): BaseSource[] {
    return Array.from(this.sources.values());
  }

  /**
   * Get sources by category.
   */
  getByCategory(category: SourceCategory): BaseSource[] {
    const names = this.categories.get(category) || [];
    return names
      .map((name) => this.sources.get(name))
      .filter((s): s is BaseSource => s !== undefined);
  }

  /**
   * Get all healthy sources.
   */
  getHealthy(): BaseSource[] {
    return Array.from(this.sources.values()).filter((source) => {
      const status = this.healthStatus.get(source.name);
      return status?.healthy ?? true;
    });
  }

  /**
   * Check health of all sources.
   */
  async checkAllHealth(): Promise<Map<string, SourceHealth>> {
    const checks = Array.from(this.sources.entries()).map(async ([name, source]) => {
      const start = Date.now();
      try {
        const healthy = await Promise.race([
          source.healthCheck(),
          new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 5000)),
        ]);
        const latency = Date.now() - start;
        const status: SourceHealth = {
          name,
          healthy,
          lastCheck: new Date(),
          latency,
        };
        this.healthStatus.set(name, status);
        return status;
      } catch {
        const status: SourceHealth = {
          name,
          healthy: false,
          lastCheck: new Date(),
          latency: Date.now() - start,
        };
        this.healthStatus.set(name, status);
        return status;
      }
    });

    await Promise.all(checks);
    return this.healthStatus;
  }

  /**
   * Get health status for all sources.
   */
  getHealthStatus(): SourceHealth[] {
    return Array.from(this.healthStatus.values());
  }

  /**
   * Search across multiple sources in parallel.
   */
  async searchParallel(
    query: string,
    sources: BaseSource[],
    limit = 5
  ): Promise<SearchResult[]> {
    const results: SearchResult[] = [];

    // Process sources in small batches to honor external rate limits
    for (let i = 0; i < sources.length; i += this.maxParallel) {
      const batch = sources.slice(i, i + this.maxParallel);

      const batchResults = await Promise.all(
        batch.map(async (source) => {
          try {
            const res = await source.search(query, limit);
            return res;
          } catch (error) {
            console.error(`[Registry] Search failed for ${source.name}:`, error);
            return [];
          }
        })
      );

      for (const res of batchResults) {
        results.push(...res);
      }

      if (this.perSourceDelayMs > 0 && i + this.maxParallel < sources.length) {
        await new Promise((resolve) => setTimeout(resolve, this.perSourceDelayMs));
      }
    }

    return results;
  }

  /**
   * Get registry statistics.
   */
  getStats(): Record<string, unknown> {
    const healthyCount = Array.from(this.healthStatus.values()).filter(
      (s) => s.healthy
    ).length;

    return {
      total_sources: this.sources.size,
      healthy_sources: healthyCount,
      unhealthy_sources: this.sources.size - healthyCount,
      sources: Array.from(this.sources.keys()),
      categories: Object.fromEntries(this.categories),
      health: this.getHealthStatus(),
    };
  }
}

// Singleton instance
export const sourceRegistry = new SourceRegistry();
