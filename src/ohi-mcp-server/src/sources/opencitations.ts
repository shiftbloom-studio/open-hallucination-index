/**
 * OpenCitations Source
 * ====================
 * 
 * Citation data for DOIs.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface OpenCitationMeta {
  title?: string;
  author?: string;
  pub_date?: string;
  source_title?: string;
  oa_link?: string;
}

interface OpenCitationRef {
  citing?: string;
  cited?: string;
}

export class OpenCitationsSource extends BaseSource {
  name = "opencitations";
  description = "OpenCitations citation data";

  constructor() {
    super("https://opencitations.net/index/api/v1");
  }

  async healthCheck(): Promise<boolean> {
    try {
      // Simple check - the API returns 404 for unknown DOIs but that's OK
      const response = await httpClient.get(`${this.baseUrl}/metadata/doi:10.1038/nature12373`);
      return response.status === 200 || response.status === 404;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    // OpenCitations doesn't have text search, only DOI lookup
    const dois = this.extractDOIs(query);
    const results: SearchResult[] = [];

    for (const doi of dois.slice(0, limit)) {
      const meta = await this.getMetadata(doi);
      if (meta) results.push(meta);
    }

    return results;
  }

  async getMetadata(doi: string): Promise<SearchResult | null> {
    try {
      const response = await httpClient.get<OpenCitationMeta[]>(
        `${this.baseUrl}/metadata/doi:${encodeURIComponent(doi)}`
      );

      if (!response.data || response.data.length === 0) return null;

      const meta = response.data[0];
      const [citations, references] = await Promise.all([
        this.getCitationCount(doi),
        this.getReferenceCount(doi),
      ]);

      return {
        source: this.name,
        title: meta.title || doi,
        content: `Authors: ${meta.author || "Unknown"}\nPublished: ${meta.pub_date || "Unknown"}\nVenue: ${meta.source_title || "Unknown"}\n\nCited by: ${citations} works\nReferences: ${references} works`,
        url: `https://doi.org/${doi}`,
        metadata: {
          doi,
          citation_count: citations,
          reference_count: references,
          oa_link: meta.oa_link,
        },
        score: citations,
      };
    } catch {
      return null;
    }
  }

  async getCitations(doi: string): Promise<SearchResult[]> {
    try {
      const response = await httpClient.get<OpenCitationRef[]>(
        `${this.baseUrl}/citations/doi:${encodeURIComponent(doi)}`
      );

      return (response.data || []).slice(0, 20).map((ref, idx) => ({
        source: this.name,
        title: `Citation ${idx + 1}`,
        content: `Citing DOI: ${ref.citing}`,
        url: ref.citing ? `https://doi.org/${ref.citing.replace("doi:", "")}` : undefined,
        metadata: { ...ref } as Record<string, unknown>,
      }));
    } catch {
      return [];
    }
  }

  private async getCitationCount(doi: string): Promise<number> {
    try {
      const response = await httpClient.get<OpenCitationRef[]>(
        `${this.baseUrl}/citations/doi:${encodeURIComponent(doi)}`
      );
      return response.data?.length || 0;
    } catch {
      return 0;
    }
  }

  private async getReferenceCount(doi: string): Promise<number> {
    try {
      const response = await httpClient.get<OpenCitationRef[]>(
        `${this.baseUrl}/references/doi:${encodeURIComponent(doi)}`
      );
      return response.data?.length || 0;
    } catch {
      return 0;
    }
  }

  private extractDOIs(text: string): string[] {
    const pattern = /10\.\d{4,}\/[^\s\]\)\>]+/g;
    const matches = text.match(pattern) || [];
    return matches.map((m) => m.replace(/[.,;:]$/, ""));
  }
}
