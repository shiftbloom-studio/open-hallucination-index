/**
 * Crossref Source
 * ===============
 * 
 * DOI metadata and scholarly publications.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface CrossrefWork {
  DOI?: string;
  title?: string[];
  author?: Array<{ given?: string; family?: string }>;
  abstract?: string;
  published?: { "date-parts"?: number[][] };
  "container-title"?: string[];
  publisher?: string;
  type?: string;
  "is-referenced-by-count"?: number;
}

interface CrossrefResponse {
  message?: {
    items?: CrossrefWork[];
  };
}

export class CrossrefSource extends BaseSource {
  name = "crossref";
  description = "Crossref DOI metadata";
  private email?: string;

  constructor(email?: string) {
    super("https://api.crossref.org");
    this.email = email;
  }

  /**
   * Remove HTML-like tags from an abstract string in a robust way by
   * repeatedly stripping `<...>` patterns until the string stabilizes.
   */
  private sanitizeAbstract(raw: string): string {
    let previous: string;
    let current = raw;
    do {
      previous = current;
      // Recreate the global regex each iteration to avoid stateful `lastIndex`.
      const tagRegex = /<[^>]+>/g;
      current = current.replace(tagRegex, "");
    } while (current !== previous);
    return current;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/works`, {
        params: { rows: 1 },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const doiMatch = query.match(/10\.\d{4,9}\/[^\s]+/i);
    if (doiMatch?.[0]) {
      const byDoi = await this.getByDOI(doiMatch[0]);
      if (byDoi) return [byDoi];
    }

    const compacted = this.compactQuery(query) || query;
    // Crossref polite pool: add mailto parameter for higher rate limits (50 req/sec vs 1 req/sec)
    // https://github.com/CrossRef/rest-api-doc#good-manners--more-reliable-service
    const params: Record<string, string | number> = {
      query: compacted,
      rows: limit,
      sort: "relevance",
    };
    
    if (this.email) {
      params.mailto = this.email;
    }

    const response = await httpClient.get<CrossrefResponse>(`${this.baseUrl}/works`, {
      params,
    });

    return (response.data.message?.items || []).map((work) => {
      const title = work.title?.[0] || "Untitled";
      const authors = (work.author || [])
        .slice(0, 3)
        .map((a) => `${a.given || ""} ${a.family || ""}`.trim())
        .join(", ");

      const abstract = work.abstract
        ? this.sanitizeAbstract(work.abstract).slice(0, 800)
        : "";

      const pubDate = this.formatDate(work.published?.["date-parts"]?.[0]);

      return {
        source: this.name,
        title,
        content: `Authors: ${authors || "Unknown"}\n${pubDate ? `Published: ${pubDate}\n` : ""}${work["container-title"]?.[0] ? `Journal: ${work["container-title"][0]}\n` : ""}\n${abstract}`,
        url: work.DOI ? `https://doi.org/${work.DOI}` : undefined,
        metadata: {
          doi: work.DOI,
          type: work.type,
          publisher: work.publisher,
          citation_count: work["is-referenced-by-count"],
        },
        score: work["is-referenced-by-count"],
      };
    });
  }

  async getByDOI(doi: string): Promise<SearchResult | null> {
    try {
      // Add polite pool mailto parameter if email is configured
      const params: Record<string, string> = {};
      if (this.email) {
        params.mailto = this.email;
      }
      
      const response = await httpClient.get<{ message?: CrossrefWork }>(
        `${this.baseUrl}/works/${encodeURIComponent(doi)}`,
        { params }
      );
      
      const work = response.data.message;
      if (!work) return null;

      const title = work.title?.[0] || doi;
      const authors = (work.author || [])
        .map((a) => `${a.given || ""} ${a.family || ""}`.trim())
        .join(", ");

      return {
        source: this.name,
        title,
        content: `DOI: ${doi}\nAuthors: ${authors}\nPublisher: ${work.publisher || "Unknown"}`,
        url: `https://doi.org/${doi}`,
        metadata: {
          doi,
          type: work.type,
          citation_count: work["is-referenced-by-count"],
        },
      };
    } catch {
      return null;
    }
  }

  private formatDate(parts?: number[]): string {
    if (!parts || parts.length === 0) return "";
    if (parts.length >= 3) return `${parts[0]}-${String(parts[1]).padStart(2, "0")}-${String(parts[2]).padStart(2, "0")}`;
    if (parts.length >= 2) return `${parts[0]}-${String(parts[1]).padStart(2, "0")}`;
    return String(parts[0]);
  }
}
