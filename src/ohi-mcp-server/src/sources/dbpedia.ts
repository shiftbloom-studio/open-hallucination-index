/**
 * DBpedia Source
 * ==============
 * 
 * DBpedia via SPARQL endpoint.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface SparqlResults {
  results?: {
    bindings?: Array<Record<string, { value: string }>>;
  };
}

export class DBpediaSource extends BaseSource {
  name = "dbpedia";
  description = "DBpedia structured data via SPARQL";

  constructor() {
    super("https://dbpedia.org");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/sparql`, {
        params: { query: "ASK { ?s ?p ?o } LIMIT 1", format: "json" },
        headers: { Accept: "application/sparql-results+json" },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const compacted = this.compactQuery(query) || query;
    const sanitized = this.sanitizeForSparql(compacted);
    const sparql = this.compactSparql(`
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      PREFIX dbo: <http://dbpedia.org/ontology/>
      
      SELECT DISTINCT ?resource ?label ?abstract WHERE {
        ?resource rdfs:label ?label .
        ?resource dbo:abstract ?abstract .
        FILTER(LANG(?label) = 'en')
        FILTER(LANG(?abstract) = 'en')
        FILTER(CONTAINS(LCASE(?label), LCASE("${sanitized}")))
      }
      LIMIT ${limit}
    `);

    let response = await httpClient.get<SparqlResults>(`${this.baseUrl}/sparql`, {
      params: { query: sparql, format: "json" },
      headers: { Accept: "application/sparql-results+json" },
    });

    let bindings = response.data.results?.bindings || [];

    if (bindings.length === 0) {
      const fallback = this.firstKeyword(query);
      if (fallback && fallback !== compacted) {
        const fallbackSparql = this.compactSparql(`
          PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
          PREFIX dbo: <http://dbpedia.org/ontology/>
          
          SELECT DISTINCT ?resource ?label ?abstract WHERE {
            ?resource rdfs:label ?label .
            ?resource dbo:abstract ?abstract .
            FILTER(LANG(?label) = 'en')
            FILTER(LANG(?abstract) = 'en')
            FILTER(CONTAINS(LCASE(?label), LCASE("${this.sanitizeForSparql(fallback)}")))
          }
          LIMIT ${limit}
        `);

        response = await httpClient.get<SparqlResults>(`${this.baseUrl}/sparql`, {
          params: { query: fallbackSparql, format: "json" },
          headers: { Accept: "application/sparql-results+json" },
        });

        bindings = response.data.results?.bindings || [];
      }
    }

    if (bindings.length === 0) {
      const lookupQuery = this.firstKeyword(query) || compacted;
      if (lookupQuery) {
        const lookup = await this.lookupFallback(lookupQuery, limit);
        if (lookup.length > 0) return lookup;
      }
    }

    return bindings.map((binding) => ({
      source: this.name,
      title: binding.label?.value || "",
      content: (binding.abstract?.value || "").slice(0, 1500),
      url: binding.resource?.value,
      metadata: { resource: binding.resource?.value },
    }));
  }

  async getResource(resourceUri: string): Promise<SearchResult | null> {
    const sparql = this.compactSparql(`
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      PREFIX dbo: <http://dbpedia.org/ontology/>
      
      SELECT ?label ?abstract WHERE {
        <${resourceUri}> rdfs:label ?label .
        <${resourceUri}> dbo:abstract ?abstract .
        FILTER(LANG(?label) = 'en')
        FILTER(LANG(?abstract) = 'en')
      }
      LIMIT 1
    `);

    const response = await httpClient.get<SparqlResults>(`${this.baseUrl}/sparql`, {
      params: { query: sparql, format: "json" },
      headers: { Accept: "application/sparql-results+json" },
    });

    const bindings = response.data.results?.bindings || [];
    if (bindings.length === 0) return null;

    const binding = bindings[0];
    return {
      source: this.name,
      title: binding.label?.value || resourceUri,
      content: binding.abstract?.value || "",
      url: resourceUri,
    };
  }

  private sanitizeForSparql(text: string): string {
    return this.sanitizeQuery(text)
      .replace(/[\\"]/g, "")
      .replace(/'/g, "\\'")
      .slice(0, 100);
  }

  private compactSparql(query: string): string {
    return query.replace(/\s+/g, " ").trim();
  }

  private async lookupFallback(query: string, limit: number): Promise<SearchResult[]> {
    try {
      const response = await httpClient.get<string>("https://lookup.dbpedia.org/api/search", {
        params: { query, maxResults: limit },
        headers: { Accept: "application/xml" },
      });

      const xml = String(response.data || "");
      const results: SearchResult[] = [];
      const matches = xml.match(/<Result>[\s\S]*?<\/Result>/g) || [];
      for (const block of matches) {
        const label = this.extractXmlTag(block, "Label");
        const uri = this.extractXmlTag(block, "URI");
        const desc = this.extractXmlTag(block, "Description");
        if (!label && !uri) continue;
        results.push({
          source: this.name,
          title: label || uri || "",
          content: desc?.slice(0, 1500) || "",
          url: uri,
          metadata: { resource: uri },
        });
      }
      return results;
    } catch {
      return [];
    }
  }

  private extractXmlTag(xml: string, tag: string): string | undefined {
    const match = xml.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`));
    if (!match?.[1]) return undefined;
    return match[1]
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .replace(/&apos;/g, "'")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .trim();
  }
}
