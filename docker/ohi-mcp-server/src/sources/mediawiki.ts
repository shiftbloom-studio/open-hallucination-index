/**
 * MediaWiki Source
 * ================
 * 
 * Wikipedia via MediaWiki Action API.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface MediaWikiSearchResponse {
  query?: {
    search?: Array<{
      pageid: number;
      title: string;
      snippet?: string;
    }>;
    pages?: Record<string, {
      pageid: number;
      title: string;
      extract?: string;
    }>;
  };
}

export class MediaWikiSource extends BaseSource {
  name = "mediawiki";
  description = "Wikipedia articles via MediaWiki Action API";

  constructor(baseUrl = "https://en.wikipedia.org/w/api.php") {
    super(baseUrl);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(this.baseUrl, {
        params: { action: "query", meta: "siteinfo", format: "json" },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const response = await httpClient.get<MediaWikiSearchResponse>(this.baseUrl, {
      params: {
        action: "query",
        list: "search",
        srsearch: query,
        srlimit: limit,
        srprop: "snippet|titlesnippet",
        format: "json",
      },
    });

    const results = response.data.query?.search || [];
    return results.map((item) => ({
      source: this.name,
      title: item.title,
      content: this.stripHtml(item.snippet || ""),
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(item.title.replace(/ /g, "_"))}`,
      metadata: { pageid: item.pageid },
    }));
  }

  async getExtract(title: string, sentences = 5): Promise<string> {
    const response = await httpClient.get<MediaWikiSearchResponse>(this.baseUrl, {
      params: {
        action: "query",
        titles: title,
        prop: "extracts",
        exsentences: sentences,
        exlimit: 1,
        explaintext: true,
        format: "json",
      },
    });

    const pages = response.data.query?.pages || {};
    for (const page of Object.values(pages)) {
      if (page.extract) return page.extract;
    }
    return "";
  }

  async getSummary(title: string): Promise<SearchResult | null> {
    const extract = await this.getExtract(title, 5);
    if (!extract) return null;

    return {
      source: this.name,
      title,
      content: extract,
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`,
    };
  }

  private stripHtml(html: string): string {
    return html.replace(/<[^>]*>/g, "").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
  }
}
