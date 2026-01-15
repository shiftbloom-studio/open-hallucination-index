/**
 * OpenAlex Source
 * ===============
 * 
 * Academic works, authors, institutions.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface OpenAlexWork {
  id?: string;
  title?: string;
  doi?: string;
  publication_date?: string;
  cited_by_count?: number;
  type?: string;
  abstract_inverted_index?: Record<string, number[]>;
  authorships?: Array<{ author?: { display_name?: string } }>;
  primary_location?: { source?: { display_name?: string } };
}

interface OpenAlexResponse {
  results?: OpenAlexWork[];
}

export class OpenAlexSource extends BaseSource {
  name = "openalex";
  description = "OpenAlex academic catalog";
  private email?: string;

  constructor(email?: string) {
    super("https://api.openalex.org");
    this.email = email;
  }

  private getParams(extra: Record<string, string | number> = {}): Record<string, string | number> {
    const params: Record<string, string | number> = { ...extra };
    if (this.email) params.mailto = this.email;
    return params;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/works`, {
        params: this.getParams({ per_page: 1 }),
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const response = await httpClient.get<OpenAlexResponse>(`${this.baseUrl}/works`, {
      params: this.getParams({
        search: query,
        per_page: limit,
        sort: "relevance_score:desc",
      }),
    });

    return (response.data.results || []).map((work) => {
      const authors = (work.authorships || [])
        .slice(0, 3)
        .map((a) => a.author?.display_name)
        .filter(Boolean)
        .join(", ");

      const abstract = work.abstract_inverted_index
        ? this.reconstructAbstract(work.abstract_inverted_index)
        : "";

      return {
        source: this.name,
        title: work.title || "Untitled",
        content: `Authors: ${authors || "Unknown"}\n\n${abstract.slice(0, 800)}`,
        url: work.doi || work.id,
        metadata: {
          doi: work.doi,
          openalex_id: work.id?.replace("https://openalex.org/", ""),
          publication_date: work.publication_date,
          cited_by_count: work.cited_by_count,
          type: work.type,
          venue: work.primary_location?.source?.display_name,
        },
        score: work.cited_by_count,
      };
    });
  }

  private reconstructAbstract(invertedIndex: Record<string, number[]>): string {
    const words: [number, string][] = [];
    for (const [word, positions] of Object.entries(invertedIndex)) {
      for (const pos of positions) {
        words.push([pos, word]);
      }
    }
    words.sort((a, b) => a[0] - b[0]);
    return words.map((w) => w[1]).join(" ");
  }
}
