/**
 * Wikidata Source
 * ===============
 * 
 * Search and SPARQL queries against Wikidata.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface WikidataSearchResult {
  search?: Array<{
    id: string;
    label?: string;
    description?: string;
    concepturi?: string;
  }>;
}

interface SparqlResults {
  results?: {
    bindings?: Array<Record<string, { value: string; type: string }>>;
  };
}

export class WikidataSource extends BaseSource {
  name = "wikidata";
  description = "Wikidata knowledge graph via SPARQL and entity search";

  constructor() {
    super("https://www.wikidata.org");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(
        `${this.baseUrl}/w/api.php`,
        { params: { action: "wbsearchentities", search: "test", language: "en", limit: 1, format: "json" } }
      );
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const response = await httpClient.get<WikidataSearchResult>(
      `${this.baseUrl}/w/api.php`,
      {
        params: {
          action: "wbsearchentities",
          search: this.sanitizeQuery(query),
          language: "en",
          limit,
          format: "json",
        },
      }
    );

    return (response.data.search || []).map((item) => ({
      source: this.name,
      title: item.label || item.id,
      content: item.description || "",
      url: item.concepturi || `https://www.wikidata.org/wiki/${item.id}`,
      metadata: { entityId: item.id },
    }));
  }

  async sparqlQuery(sparql: string): Promise<SearchResult[]> {
    const response = await httpClient.get<SparqlResults>(
      "https://query.wikidata.org/sparql",
      {
        params: { query: sparql, format: "json" },
        headers: { Accept: "application/sparql-results+json" },
      }
    );

    const bindings = response.data.results?.bindings || [];
    return bindings.slice(0, 20).map((binding, idx) => {
      const values = Object.entries(binding)
        .map(([key, val]) => `${key}: ${val.value}`)
        .join("\n");
      return {
        source: this.name,
        title: `SPARQL Result ${idx + 1}`,
        content: values,
        metadata: binding,
      };
    });
  }

  async getEntityProperties(entityId: string, limit = 10): Promise<SearchResult[]> {
    const sparql = `
      SELECT ?propLabel ?valueLabel WHERE {
        wd:${entityId} ?prop ?value .
        ?property wikibase:directClaim ?prop .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      }
      LIMIT ${limit}
    `;
    return this.sparqlQuery(sparql);
  }
}
