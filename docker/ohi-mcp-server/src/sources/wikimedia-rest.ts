/**
 * Wikimedia REST Source
 * =====================
 * 
 * Wikipedia content via REST API.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface WikimediaSummary {
  title?: string;
  displaytitle?: string;
  extract?: string;
  description?: string;
  pageid?: number;
  type?: string;
  content_urls?: {
    desktop?: { page?: string };
  };
}

export class WikimediaRESTSource extends BaseSource {
  name = "wikimedia_rest";
  description = "Wikipedia content via Wikimedia REST API";

  constructor(baseUrl = "https://en.wikipedia.org/api/rest_v1") {
    super(baseUrl);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/page/random/summary`);
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, _limit = 5): Promise<SearchResult[]> {
    // REST API doesn't have search, get summary directly
    const summary = await this.getSummary(query);
    return summary ? [summary] : [];
  }

  async getSummary(title: string): Promise<SearchResult | null> {
    try {
      const encodedTitle = encodeURIComponent(title.replace(/ /g, "_"));
      const response = await httpClient.get<WikimediaSummary>(
        `${this.baseUrl}/page/summary/${encodedTitle}`
      );

      if (response.status !== 200 || !response.data.extract) {
        return null;
      }

      const data = response.data;
      return {
        source: this.name,
        title: data.title || title,
        content: data.extract || "",
        url: data.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodedTitle}`,
        metadata: {
          description: data.description,
          pageid: data.pageid,
          type: data.type,
        },
      };
    } catch {
      return null;
    }
  }

  async getOnThisDay(month: number, day: number): Promise<SearchResult[]> {
    try {
      const response = await httpClient.get<{ events?: Array<{ text: string; year: number; pages?: Array<{ title: string }> }> }>(
        `${this.baseUrl}/feed/onthisday/events/${month.toString().padStart(2, "0")}/${day.toString().padStart(2, "0")}`
      );

      return (response.data.events || []).slice(0, 10).map((event) => ({
        source: this.name,
        title: `${event.year}: Historical Event`,
        content: event.text,
        metadata: { year: event.year, relatedPages: event.pages?.map((p) => p.title) },
      }));
    } catch {
      return [];
    }
  }
}
