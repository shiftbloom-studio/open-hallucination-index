/**
 * Tool Aggregator
 * ================
 * 
 * Routes MCP tool calls to the appropriate knowledge sources.
 * Handles parallel execution and result aggregation.
 */

import { sourceRegistry, SourceCategory } from "./sources/registry.js";
import { SearchResult } from "./sources/base.js";
import { WikidataSource } from "./sources/wikidata.js";
import { MediaWikiSource } from "./sources/mediawiki.js";
import { WikimediaRESTSource } from "./sources/wikimedia-rest.js";
import { DBpediaSource } from "./sources/dbpedia.js";
import { OpenAlexSource } from "./sources/openalex.js";
import { CrossrefSource } from "./sources/crossref.js";
import { EuropePMCSource } from "./sources/europepmc.js";
import { NCBISource } from "./sources/ncbi.js";
import { ClinicalTrialsSource } from "./sources/clinicaltrials.js";
import { OpenCitationsSource } from "./sources/opencitations.js";
import { GDELTSource } from "./sources/gdelt.js";
import { WorldBankSource } from "./sources/worldbank.js";
import { OSVSource } from "./sources/osv.js";

export interface ToolResult {
  success: boolean;
  results?: SearchResult[];
  error?: string;
  metadata?: Record<string, unknown>;
}

type ToolHandler = (args: Record<string, unknown>) => Promise<ToolResult>;

class ToolAggregator {
  private handlers: Map<string, ToolHandler> = new Map();

  constructor() {
    this.registerHandlers();
  }

  private registerHandlers(): void {
    // Universal search
    this.handlers.set("search_all", this.searchAll.bind(this));

    // Wikipedia/Knowledge Graph
    this.handlers.set("search_wikipedia", this.searchWikipedia.bind(this));
    this.handlers.set("get_wikipedia_summary", this.getWikipediaSummary.bind(this));
    this.handlers.set("search_wikidata", this.searchWikidata.bind(this));
    this.handlers.set("query_wikidata_sparql", this.queryWikidataSPARQL.bind(this));
    this.handlers.set("search_dbpedia", this.searchDBpedia.bind(this));

    // Academic
    this.handlers.set("search_academic", this.searchAcademic.bind(this));
    this.handlers.set("search_openalex", this.searchOpenAlex.bind(this));
    this.handlers.set("search_crossref", this.searchCrossref.bind(this));
    this.handlers.set("get_doi_metadata", this.getDOIMetadata.bind(this));
    this.handlers.set("search_pubmed", this.searchPubMed.bind(this));
    this.handlers.set("search_europepmc", this.searchEuropePMC.bind(this));

    // Medical
    this.handlers.set("search_clinical_trials", this.searchClinicalTrials.bind(this));

    // Citations
    this.handlers.set("get_citations", this.getCitations.bind(this));

    // News
    this.handlers.set("search_gdelt", this.searchGDELT.bind(this));

    // Economic
    this.handlers.set("get_world_bank_indicator", this.getWorldBankIndicator.bind(this));

    // Security
    this.handlers.set("search_vulnerabilities", this.searchVulnerabilities.bind(this));
    this.handlers.set("get_vulnerability", this.getVulnerability.bind(this));
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    const handler = this.handlers.get(name);
    if (!handler) {
      return { success: false, error: `Unknown tool: ${name}` };
    }

    try {
      return await handler(args);
    } catch (error) {
      console.error(`[Aggregator] Tool ${name} failed:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // ============ Universal Search ============

  private async searchAll(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 5;
    const category = (args.category as SourceCategory) || "all";

    const sources = sourceRegistry.getByCategory(category);
    const results = await sourceRegistry.searchParallel(query, sources, limit);

    return {
      success: true,
      results: this.deduplicateResults(results),
      metadata: { sources_queried: sources.map((s) => s.name) },
    };
  }

  // ============ Wikipedia/Knowledge Graph ============

  private async searchWikipedia(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 5;

    const mediawiki = sourceRegistry.get("mediawiki") as MediaWikiSource;
    const results = await mediawiki?.search(query, limit) || [];

    return { success: true, results };
  }

  private async getWikipediaSummary(args: Record<string, unknown>): Promise<ToolResult> {
    const title = args.title as string;

    const wikimediaRest = sourceRegistry.get("wikimedia-rest") as WikimediaRESTSource;
    const result = await wikimediaRest?.getSummary(title);

    return {
      success: !!result,
      results: result ? [result] : [],
      error: result ? undefined : "Article not found",
    };
  }

  private async searchWikidata(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 5;

    const wikidata = sourceRegistry.get("wikidata") as WikidataSource;
    const results = await wikidata?.search(query, limit) || [];

    return { success: true, results };
  }

  private async queryWikidataSPARQL(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.sparql as string;

    const wikidata = sourceRegistry.get("wikidata") as WikidataSource;
    const results = await wikidata?.sparqlQuery(query) || [];

    return { success: true, results };
  }

  private async searchDBpedia(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 5;

    const dbpedia = sourceRegistry.get("dbpedia") as DBpediaSource;
    const results = await dbpedia?.search(query, limit) || [];

    return { success: true, results };
  }

  // ============ Academic ============

  private async searchAcademic(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 5;

    const sources = sourceRegistry.getByCategory("academic");
    const results = await sourceRegistry.searchParallel(query, sources, limit);

    return {
      success: true,
      results: this.deduplicateResults(results),
      metadata: { sources_queried: sources.map((s) => s.name) },
    };
  }

  private async searchOpenAlex(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const openalex = sourceRegistry.get("openalex") as OpenAlexSource;
    const results = await openalex?.search(query, limit) || [];

    return { success: true, results };
  }

  private async searchCrossref(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const crossref = sourceRegistry.get("crossref") as CrossrefSource;
    const results = await crossref?.search(query, limit) || [];

    return { success: true, results };
  }

  private async getDOIMetadata(args: Record<string, unknown>): Promise<ToolResult> {
    const doi = args.doi as string;

    const crossref = sourceRegistry.get("crossref") as CrossrefSource;
    const result = await crossref?.getByDOI(doi);

    return {
      success: !!result,
      results: result ? [result] : [],
      error: result ? undefined : "DOI not found",
    };
  }

  private async searchPubMed(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const ncbi = sourceRegistry.get("ncbi") as NCBISource;
    const results = await ncbi?.search(query, limit) || [];

    return { success: true, results };
  }

  private async searchEuropePMC(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const europepmc = sourceRegistry.get("europepmc") as EuropePMCSource;
    const results = await europepmc?.search(query, limit) || [];

    return { success: true, results };
  }

  // ============ Medical ============

  private async searchClinicalTrials(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const clinicaltrials = sourceRegistry.get("clinicaltrials") as ClinicalTrialsSource;
    const results = await clinicaltrials?.search(query, limit) || [];

    return { success: true, results };
  }

  // ============ Citations ============

  private async getCitations(args: Record<string, unknown>): Promise<ToolResult> {
    const doi = args.doi as string;

    const opencitations = sourceRegistry.get("opencitations") as OpenCitationsSource;
    const results = await opencitations?.getCitations(doi) || [];

    return { success: true, results };
  }

  // ============ News ============

  private async searchGDELT(args: Record<string, unknown>): Promise<ToolResult> {
    const query = args.query as string;
    const limit = (args.limit as number) || 10;

    const gdelt = sourceRegistry.get("gdelt") as GDELTSource;
    const results = await gdelt?.searchArticles(query, limit) || [];

    return { success: true, results };
  }

  // ============ Economic ============

  private async getWorldBankIndicator(args: Record<string, unknown>): Promise<ToolResult> {
    const indicator = args.indicator as string;
    const country = (args.country as string) || "all";
    const year = args.year as string | undefined;

    const worldbank = sourceRegistry.get("worldbank") as WorldBankSource;
    const results = await worldbank?.getIndicator(indicator, country, year) || [];

    return { success: true, results };
  }

  // ============ Security ============

  private async searchVulnerabilities(args: Record<string, unknown>): Promise<ToolResult> {
    const packageName = args.package as string;
    const ecosystem = args.ecosystem as string | undefined;

    const osv = sourceRegistry.get("osv") as OSVSource;
    const results = await osv?.queryPackage(packageName, ecosystem) || [];

    return { success: true, results };
  }

  private async getVulnerability(args: Record<string, unknown>): Promise<ToolResult> {
    const id = args.id as string;

    const osv = sourceRegistry.get("osv") as OSVSource;
    const result = await osv?.getVulnerability(id);

    return {
      success: !!result,
      results: result ? [result] : [],
      error: result ? undefined : "Vulnerability not found",
    };
  }

  // ============ Helpers ============

  private deduplicateResults(results: SearchResult[]): SearchResult[] {
    const seen = new Set<string>();
    return results.filter((r) => {
      const key = r.url || `${r.source}:${r.title}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  getToolNames(): string[] {
    return Array.from(this.handlers.keys());
  }
}

export const toolAggregator = new ToolAggregator();
