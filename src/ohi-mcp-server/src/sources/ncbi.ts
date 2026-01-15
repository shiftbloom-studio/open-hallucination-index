/**
 * NCBI E-utilities Source
 * =======================
 * 
 * PubMed and other NCBI databases.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface NCBISearchResult {
  esearchresult?: {
    idlist?: string[];
  };
}

interface NCBISummaryResult {
  result?: Record<string, {
    uid?: string;
    title?: string;
    authors?: Array<{ name?: string }>;
    source?: string;
    pubdate?: string;
  }>;
}

export class NCBISource extends BaseSource {
  name = "ncbi";
  description = "PubMed/NCBI biomedical literature";
  private apiKey?: string;
  private email?: string;

  constructor(apiKey?: string, email?: string) {
    super("https://eutils.ncbi.nlm.nih.gov/entrez/eutils");
    this.apiKey = apiKey;
    this.email = email;
  }

  private getParams(extra: Record<string, string | number> = {}): Record<string, string | number> {
    const params: Record<string, string | number> = { ...extra, retmode: "json" };
    if (this.apiKey) params.api_key = this.apiKey;
    if (this.email) params.email = this.email;
    return params;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/einfo.fcgi`, {
        params: this.getParams({}),
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    // Step 1: Search for PMIDs
    const searchResponse = await httpClient.get<NCBISearchResult>(`${this.baseUrl}/esearch.fcgi`, {
      params: this.getParams({
        db: "pubmed",
        term: query,
        retmax: limit,
        sort: "relevance",
      }),
    });

    const pmids = searchResponse.data.esearchresult?.idlist || [];
    if (pmids.length === 0) return [];

    // Step 2: Fetch article summaries
    const summaryResponse = await httpClient.get<NCBISummaryResult>(`${this.baseUrl}/esummary.fcgi`, {
      params: this.getParams({
        db: "pubmed",
        id: pmids.join(","),
      }),
    });

    const result = summaryResponse.data.result || {};
    return pmids
      .filter((pmid) => pmid in result)
      .map((pmid) => {
        const article = result[pmid];
        const authors = (article.authors || [])
          .slice(0, 3)
          .map((a) => a.name)
          .join(", ");

        return {
          source: this.name,
          title: article.title || "Untitled",
          content: `Authors: ${authors || "Unknown"}\nJournal: ${article.source || "Unknown"}\nPublished: ${article.pubdate || "Unknown"}`,
          url: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
          metadata: {
            pmid,
            journal: article.source,
            pubdate: article.pubdate,
          },
        };
      });
  }
}
