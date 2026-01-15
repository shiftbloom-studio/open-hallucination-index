/**
 * GDELT Source
 * ============
 * 
 * Global news and events.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface GDELTArticle {
  url?: string;
  title?: string;
  seendate?: string;
  domain?: string;
  language?: string;
  sourcecountry?: string;
}

interface GDELTResponse {
  articles?: GDELTArticle[];
}

export class GDELTSource extends BaseSource {
  name = "gdelt";
  description = "GDELT global news and events";

  constructor() {
    super("https://api.gdeltproject.org/api/v2");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/doc/doc`, {
        params: { query: "test", mode: "artlist", maxrecords: 1, format: "json" },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 10): Promise<SearchResult[]> {
    return this.searchArticles(query, limit);
  }

  async searchArticles(query: string, limit = 10): Promise<SearchResult[]> {
    try {
      const response = await httpClient.get<GDELTResponse>(`${this.baseUrl}/doc/doc`, {
        params: {
          query: this.compactQuery(query) || this.sanitizeQuery(query),
          mode: "artlist",
          maxrecords: limit,
          format: "json",
          sort: "relevance",
          timespan: "1y",
        },
      });

      return (response.data.articles || []).map((article) => ({
        source: this.name,
        title: article.title || "Untitled Article",
        content: `Source: ${article.domain || "Unknown"}\nCountry: ${article.sourcecountry || "Unknown"}\nDate: ${this.formatDate(article.seendate)}`,
        url: article.url,
        metadata: {
          domain: article.domain,
          language: article.language,
          country: article.sourcecountry,
          seendate: article.seendate,
        },
      }));
    } catch {
      return [];
    }
  }

  async getTimelineVolume(query: string): Promise<SearchResult[]> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/doc/doc`, {
        params: {
          query: this.sanitizeQuery(query),
          mode: "timelinevol",
          format: "json",
        },
      });

      return [
        {
          source: this.name,
          title: `News volume for "${query}"`,
          content: JSON.stringify(response.data, null, 2),
          metadata: { type: "timeline_volume" },
        },
      ];
    } catch {
      return [];
    }
  }

  private formatDate(gdeltDate?: string): string {
    if (!gdeltDate || gdeltDate.length < 8) return "Unknown";
    // GDELT date format: YYYYMMDDHHMMSS
    const year = gdeltDate.slice(0, 4);
    const month = gdeltDate.slice(4, 6);
    const day = gdeltDate.slice(6, 8);
    return `${year}-${month}-${day}`;
  }
}
