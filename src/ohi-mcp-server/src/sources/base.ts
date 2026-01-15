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

  protected compactQuery(query: string, maxWords = 6): string {
    const sanitized = this.sanitizeQuery(query);
    if (!sanitized) return "";

    const stopwords = new Set([
      "a", "an", "the", "and", "or", "but", "if", "then", "else",
      "is", "are", "was", "were", "be", "been", "being",
      "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
      "about", "into", "over", "after", "before", "between", "during",
      "than", "that", "this", "these", "those", "it", "its",
      "has", "have", "had", "do", "does", "did",
      "not", "no", "yes", "true", "false",
      "study", "studies", "recent", "paper", "publication", "published",
    ]);

    const originalWords = sanitized.split(/\s+/).filter(Boolean);
    const keywords: string[] = [];

    for (const word of originalWords) {
      const lowered = word.toLowerCase();
      if (!stopwords.has(lowered)) {
        keywords.push(word);
      }
    }

    const selected = (keywords.length ? keywords : originalWords).slice(0, maxWords);
    return selected.join(" ");
  }

  protected primaryKeyword(query: string): string | null {
    const compacted = this.compactQuery(query, 8);
    if (!compacted) return null;
    const words = compacted.split(/\s+/).filter(Boolean);
    if (!words.length) return null;
    return words.reduce((longest, word) => (word.length > longest.length ? word : longest), words[0]);
  }

  protected firstKeyword(query: string): string | null {
    const compacted = this.compactQuery(query, 8);
    if (!compacted) return null;
    const words = compacted.split(/\s+/).filter(Boolean);
    return words[0] || null;
  }
}
