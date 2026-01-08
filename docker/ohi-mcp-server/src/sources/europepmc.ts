/**
 * Europe PMC Source
 * =================
 * 
 * Life sciences literature.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface EuropePMCArticle {
  pmid?: string;
  pmcid?: string;
  doi?: string;
  title?: string;
  authorString?: string;
  abstractText?: string;
  pubYear?: string;
  journalTitle?: string;
  isOpenAccess?: string;
  citedByCount?: number;
}

interface EuropePMCResponse {
  resultList?: {
    result?: EuropePMCArticle[];
  };
}

export class EuropePMCSource extends BaseSource {
  name = "europepmc";
  description = "Europe PMC life sciences literature";

  constructor() {
    super("https://www.ebi.ac.uk/europepmc/webservices/rest");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/search`, {
        params: { query: "test", resultType: "lite", pageSize: 1, format: "json" },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const response = await httpClient.get<EuropePMCResponse>(`${this.baseUrl}/search`, {
      params: {
        query,
        resultType: "core",
        pageSize: limit,
        format: "json",
        sort: "RELEVANCE",
      },
    });

    return (response.data.resultList?.result || []).map((article) => {
      const content = [
        `Authors: ${article.authorString || "Unknown"}`,
        article.pubYear ? `Published: ${article.pubYear}` : "",
        article.journalTitle ? `Journal: ${article.journalTitle}` : "",
        "",
        article.abstractText?.slice(0, 800) || "",
      ]
        .filter(Boolean)
        .join("\n");

      let url = "";
      if (article.pmcid) url = `https://europepmc.org/article/PMC/${article.pmcid}`;
      else if (article.pmid) url = `https://europepmc.org/article/MED/${article.pmid}`;
      else if (article.doi) url = `https://doi.org/${article.doi}`;

      return {
        source: this.name,
        title: article.title || "Untitled",
        content,
        url,
        metadata: {
          pmid: article.pmid,
          pmcid: article.pmcid,
          doi: article.doi,
          isOpenAccess: article.isOpenAccess === "Y",
          citedByCount: article.citedByCount,
        },
        score: article.citedByCount,
      };
    });
  }
}
