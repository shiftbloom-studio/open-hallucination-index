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
    super("http://dbpedia.org");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/sparql`, {
        params: { query: "ASK { ?s ?p ?o } LIMIT 1", format: "json" },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    const sanitized = this.sanitizeForSparql(query);
    const sparql = `
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
    `;

    const response = await httpClient.get<SparqlResults>(`${this.baseUrl}/sparql`, {
      params: { query: sparql, format: "application/sparql-results+json" },
    });

    return (response.data.results?.bindings || []).map((binding) => ({
      source: this.name,
      title: binding.label?.value || "",
      content: (binding.abstract?.value || "").slice(0, 1500),
      url: binding.resource?.value,
      metadata: { resource: binding.resource?.value },
    }));
  }

  async getResource(resourceUri: string): Promise<SearchResult | null> {
    const sparql = `
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      PREFIX dbo: <http://dbpedia.org/ontology/>
      
      SELECT ?label ?abstract WHERE {
        <${resourceUri}> rdfs:label ?label .
        <${resourceUri}> dbo:abstract ?abstract .
        FILTER(LANG(?label) = 'en')
        FILTER(LANG(?abstract) = 'en')
      }
      LIMIT 1
    `;

    const response = await httpClient.get<SparqlResults>(`${this.baseUrl}/sparql`, {
      params: { query: sparql, format: "application/sparql-results+json" },
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
    return text.replace(/[\\"]/g, "").replace(/'/g, "\\'").slice(0, 100);
  }
}
