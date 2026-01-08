/**
 * Knowledge Source Base
 * =====================
 * 
 * Abstract interface for knowledge sources.
 */

export interface SearchResult {
  source: string;
  title: string;
  content: string;
  url?: string;
  metadata?: Record<string, unknown>;
  score?: number;
}

export interface KnowledgeSource {
  name: string;
  description: string;
  
  /**
   * Check if source is available
   */
  healthCheck(): Promise<boolean>;
  
  /**
   * General search
   */
  search(query: string, limit?: number): Promise<SearchResult[]>;
}

export abstract class BaseSource implements KnowledgeSource {
  abstract name: string;
  abstract description: string;
  
  protected baseUrl: string;
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }
  
  abstract healthCheck(): Promise<boolean>;
  abstract search(query: string, limit?: number): Promise<SearchResult[]>;
  
  protected sanitizeQuery(query: string): string {
    return query.replace(/[^\w\s\-\.]/g, " ").trim().slice(0, 200);
  }
}
