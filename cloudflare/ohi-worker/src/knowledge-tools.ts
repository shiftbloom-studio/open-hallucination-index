import { z } from "zod";

export interface SearchResult {
  source: string;
  title: string;
  content: string;
  url?: string;
  metadata?: Record<string, unknown>;
  score?: number;
}

export type KnowledgeToolResult =
  | string
  | {
      success: boolean;
      results?: SearchResult[];
      error?: string;
      metadata?: Record<string, unknown>;
    };

export interface KnowledgeToolDefinition {
  name: string;
  description: string;
  schema: z.ZodRawShape;
}

const querySchema = {
  query: z.string().min(1).max(500),
  limit: z.number().int().min(1).max(50).optional(),
};

export const KNOWLEDGE_TOOLS: KnowledgeToolDefinition[] = [
  {
    name: "search_all",
    description: "Search across all migrated OHI knowledge sources.",
    schema: {
      query: z.string().min(1).max(500),
      limit: z.number().int().min(1).max(20).optional(),
      sources: z.array(z.string()).optional(),
      category: z.enum(["wikipedia", "academic", "medical", "news", "economic", "security", "all"]).optional(),
    },
  },
  { name: "search_wikipedia", description: "Search Wikipedia via MediaWiki.", schema: querySchema },
  { name: "get_wikipedia_summary", description: "Get a Wikipedia article summary.", schema: { title: z.string().min(1).max(200) } },
  { name: "get_summary", description: "Get a Wikipedia article summary as text.", schema: { title: z.string().min(1).max(200) } },
  { name: "search_wikidata", description: "Search Wikidata entities.", schema: querySchema },
  { name: "query_wikidata_sparql", description: "Run a bounded Wikidata SPARQL query.", schema: { sparql: z.string().min(1).max(5000) } },
  { name: "search_dbpedia", description: "Search DBpedia structured data.", schema: querySchema },
  { name: "resolve-library-id", description: "Resolve a library name with Context7.", schema: { query: z.string().min(1).max(500), libraryName: z.string().min(1).max(160) } },
  { name: "query-docs", description: "Fetch Context7 documentation context.", schema: { libraryId: z.string().min(1).max(200), query: z.string().min(1).max(500) } },
  { name: "search_academic", description: "Search academic literature across OpenAlex, Crossref, and Europe PMC.", schema: querySchema },
  { name: "search_openalex", description: "Search OpenAlex.", schema: querySchema },
  { name: "search_crossref", description: "Search Crossref.", schema: querySchema },
  { name: "get_doi_metadata", description: "Get Crossref metadata for a DOI.", schema: { doi: z.string().min(4).max(200) } },
  { name: "search_pubmed", description: "Search PubMed/NCBI.", schema: querySchema },
  { name: "search_europepmc", description: "Search Europe PMC.", schema: querySchema },
  { name: "search_clinical_trials", description: "Search ClinicalTrials.gov.", schema: querySchema },
  { name: "get_citations", description: "Get OpenCitations citation data for a DOI.", schema: { doi: z.string().min(4).max(200) } },
  {
    name: "search_gdelt",
    description: "Search GDELT global news.",
    schema: {
      query: z.string().min(1).max(500),
      mode: z.enum(["artlist", "timelinevol", "tonechart"]).optional(),
      limit: z.number().int().min(1).max(50).optional(),
    },
  },
  {
    name: "get_world_bank_indicator",
    description: "Fetch World Bank indicator data.",
    schema: {
      indicator: z.string().min(1).max(80),
      country: z.string().min(1).max(20).optional(),
      year: z.string().min(4).max(20).optional(),
    },
  },
  {
    name: "search_vulnerabilities",
    description: "Search OSV vulnerabilities.",
    schema: {
      query: z.string().min(1).max(200),
      ecosystem: z.string().min(1).max(80).optional(),
    },
  },
  { name: "get_vulnerability", description: "Get a vulnerability by OSV/CVE/GHSA id.", schema: { id: z.string().min(1).max(120) } },
  {
    name: "ohi_check_balance",
    description: "Legacy OHI API key balance check.",
    schema: { api_key: z.string().min(1).max(300), api_url: z.string().url().optional() },
  },
];

const CATEGORY_SOURCES: Record<string, string[]> = {
  wikipedia: ["search_wikipedia", "search_wikidata", "search_dbpedia"],
  academic: ["search_openalex", "search_crossref", "search_europepmc"],
  medical: ["search_europepmc", "search_pubmed", "search_clinical_trials"],
  news: ["search_gdelt"],
  economic: ["get_world_bank_indicator"],
  security: ["search_vulnerabilities"],
  all: [
    "search_wikipedia",
    "search_wikidata",
    "search_dbpedia",
    "search_openalex",
    "search_crossref",
    "search_europepmc",
    "search_pubmed",
    "search_clinical_trials",
    "search_gdelt",
    "search_vulnerabilities",
  ],
};

export async function callKnowledgeTool(name: string, args: Record<string, unknown>, env: Env): Promise<KnowledgeToolResult> {
  try {
    switch (name) {
      case "search_all":
        return searchAll(args, env);
      case "search_wikipedia":
        return success(await searchWikipedia(str(args.query), num(args.limit, 5)));
      case "get_wikipedia_summary":
        return success(await getWikipediaSummary(str(args.title)));
      case "get_summary": {
        const result = await getWikipediaSummary(str(args.title));
        return result[0]?.content ?? "";
      }
      case "search_wikidata":
        return success(await searchWikidata(str(args.query), num(args.limit, 5)));
      case "query_wikidata_sparql":
        return success(await queryWikidataSparql(str(args.sparql)));
      case "search_dbpedia":
        return success(await searchDbpedia(str(args.query), num(args.limit, 5)));
      case "resolve-library-id":
        return resolveLibraryId(str(args.query), str(args.libraryName), env);
      case "query-docs":
        return queryDocs(str(args.libraryId), str(args.query), env);
      case "search_academic":
        return searchAcademic(args, env);
      case "search_openalex":
        return success(await searchOpenAlex(str(args.query), num(args.limit, 5), politeEmail(env)));
      case "search_crossref":
        return success(await searchCrossref(str(args.query), num(args.limit, 5), politeEmail(env)));
      case "get_doi_metadata":
        return success(await getDoiMetadata(str(args.doi), politeEmail(env)));
      case "search_pubmed":
        return success(await searchPubMed(str(args.query), num(args.limit, 5)));
      case "search_europepmc":
        return success(await searchEuropePmc(str(args.query), num(args.limit, 5)));
      case "search_clinical_trials":
        return success(await searchClinicalTrials(str(args.query), num(args.limit, 5)));
      case "get_citations":
        return success(await getCitations(str(args.doi)));
      case "search_gdelt":
        return success(await searchGdelt(str(args.query), str(args.mode || "artlist"), num(args.limit, 10)));
      case "get_world_bank_indicator":
        return success(await getWorldBankIndicator(str(args.indicator), str(args.country || "all"), typeof args.year === "string" ? args.year : undefined));
      case "search_vulnerabilities":
        return success(await searchVulnerabilities(str(args.query), typeof args.ecosystem === "string" ? args.ecosystem : undefined));
      case "get_vulnerability":
        return success(await getVulnerability(str(args.id)));
      case "ohi_check_balance":
        return {
          success: false,
          error: "The token-balance endpoint belongs to the legacy API and is not exposed by the Cloudflare-hosted OHI v2 Worker.",
        };
      default:
        return { success: false, error: `Unknown tool: ${name}` };
    }
  } catch (error) {
    return { success: false, error: errorMessage(error) };
  }
}

async function searchAll(args: Record<string, unknown>, env: Env): Promise<KnowledgeToolResult> {
  const query = str(args.query);
  const limit = num(args.limit, 3);
  const requested = Array.isArray(args.sources)
    ? args.sources.map((source) => String(source))
    : CATEGORY_SOURCES[str(args.category || "all")] ?? CATEGORY_SOURCES.all;
  const calls = requested.filter((source) => source !== "search_all").map(async (source) => {
    const result = await callKnowledgeTool(source, { query, limit }, env);
    return typeof result === "object" && result.success ? result.results ?? [] : [];
  });
  const settled = await Promise.allSettled(calls);
  const results = settled.flatMap((item) => item.status === "fulfilled" ? item.value : []);
  return {
    success: true,
    results: dedupe(results).slice(0, Math.max(limit * requested.length, limit)),
    metadata: { sources_queried: requested },
  };
}

async function searchAcademic(args: Record<string, unknown>, env: Env): Promise<KnowledgeToolResult> {
  const query = str(args.query);
  const limit = num(args.limit, 3);
  const [openalex, crossref, europepmc] = await Promise.all([
    searchOpenAlex(query, limit, politeEmail(env)).catch(() => []),
    searchCrossref(query, limit, politeEmail(env)).catch(() => []),
    searchEuropePmc(query, limit).catch(() => []),
  ]);
  return { success: true, results: dedupe([...openalex, ...crossref, ...europepmc]) };
}

async function searchWikipedia(query: string, limit: number): Promise<SearchResult[]> {
  const url = new URL("https://en.wikipedia.org/w/api.php");
  url.searchParams.set("action", "query");
  url.searchParams.set("list", "search");
  url.searchParams.set("srsearch", query);
  url.searchParams.set("srlimit", String(limit));
  url.searchParams.set("srprop", "snippet|titlesnippet");
  url.searchParams.set("format", "json");
  url.searchParams.set("origin", "*");
  const data = await fetchJson<{ query?: { search?: Array<{ pageid: number; title: string; snippet?: string }> } }>(url);
  return (data.query?.search ?? []).map((item) => ({
    source: "mediawiki",
    title: item.title,
    content: stripHtml(item.snippet ?? ""),
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(item.title.replaceAll(" ", "_"))}`,
    metadata: { pageid: item.pageid },
  }));
}

async function getWikipediaSummary(title: string): Promise<SearchResult[]> {
  const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
  const data = await fetchJson<{
    title?: string;
    extract?: string;
    description?: string;
    content_urls?: { desktop?: { page?: string } };
    pageid?: number;
  }>(url);
  if (!data.extract) return [];
  return [{
    source: "wikimedia-rest",
    title: data.title ?? title,
    content: data.extract,
    url: data.content_urls?.desktop?.page,
    metadata: { description: data.description, pageid: data.pageid },
  }];
}

async function searchWikidata(query: string, limit: number): Promise<SearchResult[]> {
  const url = new URL("https://www.wikidata.org/w/api.php");
  url.searchParams.set("action", "wbsearchentities");
  url.searchParams.set("search", compactQuery(query));
  url.searchParams.set("language", "en");
  url.searchParams.set("format", "json");
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("origin", "*");
  const data = await fetchJson<{ search?: Array<{ id: string; label?: string; description?: string; concepturi?: string }> }>(url);
  return (data.search ?? []).map((item) => ({
    source: "wikidata",
    title: item.label ?? item.id,
    content: item.description ?? "",
    url: item.concepturi ?? `https://www.wikidata.org/wiki/${item.id}`,
    metadata: { entityId: item.id },
  }));
}

async function queryWikidataSparql(sparql: string): Promise<SearchResult[]> {
  const url = new URL("https://query.wikidata.org/sparql");
  url.searchParams.set("query", sparql);
  url.searchParams.set("format", "json");
  const data = await fetchJson<{ results?: { bindings?: Array<Record<string, { value: string; type?: string }>> } }>(url, {
    accept: "application/sparql-results+json",
  });
  return (data.results?.bindings ?? []).slice(0, 20).map((binding, index) => ({
    source: "wikidata",
    title: `SPARQL Result ${index + 1}`,
    content: Object.entries(binding).map(([key, value]) => `${key}: ${value.value}`).join("\n"),
    metadata: binding,
  }));
}

async function searchDbpedia(query: string, limit: number): Promise<SearchResult[]> {
  const sparql = `
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?resource ?label ?abstract WHERE {
      ?resource rdfs:label ?label .
      ?resource dbo:abstract ?abstract .
      FILTER(LANG(?label) = 'en')
      FILTER(LANG(?abstract) = 'en')
      FILTER(CONTAINS(LCASE(?label), LCASE("${escapeSparql(compactQuery(query))}")))
    }
    LIMIT ${limit}
  `.replace(/\s+/g, " ").trim();
  const body = new URLSearchParams({ query: sparql, format: "json" });
  const data = await fetchJson<{ results?: { bindings?: Array<Record<string, { value: string }>> } }>(
    "https://dbpedia.org/sparql",
    { method: "POST", body, accept: "application/sparql-results+json" },
  );
  return (data.results?.bindings ?? []).map((binding) => ({
    source: "dbpedia",
    title: binding.label?.value ?? "",
    content: (binding.abstract?.value ?? "").slice(0, 1500),
    url: binding.resource?.value,
    metadata: { resource: binding.resource?.value },
  }));
}

async function searchOpenAlex(query: string, limit: number, email?: string): Promise<SearchResult[]> {
  const url = new URL("https://api.openalex.org/works");
  url.searchParams.set("search", query);
  url.searchParams.set("per_page", String(limit));
  url.searchParams.set("sort", "relevance_score:desc");
  if (email) url.searchParams.set("mailto", email);
  const data = await fetchJson<{ results?: Array<Record<string, unknown>> }>(url);
  return (data.results ?? []).map((work) => ({
    source: "openalex",
    title: str(work.title || "Untitled"),
    content: [
      `Authors: ${authorsFromOpenAlex(work) || "Unknown"}`,
      abstractFromInvertedIndex(work.abstract_inverted_index as Record<string, number[]> | undefined).slice(0, 900),
    ].join("\n\n"),
    url: typeof work.doi === "string" ? work.doi : str(work.id),
    metadata: {
      openalex_id: typeof work.id === "string" ? work.id.replace("https://openalex.org/", "") : undefined,
      doi: work.doi,
      publication_date: work.publication_date,
      cited_by_count: work.cited_by_count,
      type: work.type,
    },
    score: Number(work.cited_by_count ?? 0),
  }));
}

async function searchCrossref(query: string, limit: number, email?: string): Promise<SearchResult[]> {
  const doi = query.match(/10\.\d{4,9}\/[^\s]+/i)?.[0];
  if (doi) return getDoiMetadata(doi, email);
  const url = new URL("https://api.crossref.org/works");
  url.searchParams.set("query", compactQuery(query));
  url.searchParams.set("rows", String(limit));
  url.searchParams.set("sort", "relevance");
  if (email) url.searchParams.set("mailto", email);
  const data = await fetchJson<{ message?: { items?: Array<Record<string, unknown>> } }>(url);
  return (data.message?.items ?? []).map(crossrefWorkToResult);
}

async function getDoiMetadata(doi: string, email?: string): Promise<SearchResult[]> {
  const url = new URL(`https://api.crossref.org/works/${encodeURIComponent(doi)}`);
  if (email) url.searchParams.set("mailto", email);
  const data = await fetchJson<{ message?: Record<string, unknown> }>(url);
  return data.message ? [crossrefWorkToResult(data.message)] : [];
}

async function searchEuropePmc(query: string, limit: number): Promise<SearchResult[]> {
  const url = new URL("https://www.ebi.ac.uk/europepmc/webservices/rest/search");
  url.searchParams.set("query", query);
  url.searchParams.set("format", "json");
  url.searchParams.set("pageSize", String(limit));
  const data = await fetchJson<{ resultList?: { result?: Array<Record<string, unknown>> } }>(url);
  return (data.resultList?.result ?? []).map((item) => ({
    source: "europepmc",
    title: str(item.title || "Untitled"),
    content: [
      item.authorString ? `Authors: ${item.authorString}` : "",
      item.journalTitle ? `Journal: ${item.journalTitle}` : "",
      str(item.abstractText || "").slice(0, 1000),
    ].filter(Boolean).join("\n"),
    url: item.doi ? `https://doi.org/${item.doi}` : item.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${item.pmid}/` : undefined,
    metadata: { pmid: item.pmid, doi: item.doi, publication_date: item.firstPublicationDate },
  }));
}

async function searchPubMed(query: string, limit: number): Promise<SearchResult[]> {
  const searchUrl = new URL("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi");
  searchUrl.searchParams.set("db", "pubmed");
  searchUrl.searchParams.set("term", query);
  searchUrl.searchParams.set("retmode", "json");
  searchUrl.searchParams.set("retmax", String(limit));
  const search = await fetchJson<{ esearchresult?: { idlist?: string[] } }>(searchUrl);
  const ids = search.esearchresult?.idlist ?? [];
  if (ids.length === 0) return [];
  const summaryUrl = new URL("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi");
  summaryUrl.searchParams.set("db", "pubmed");
  summaryUrl.searchParams.set("id", ids.join(","));
  summaryUrl.searchParams.set("retmode", "json");
  const summary = await fetchJson<{ result?: Record<string, Record<string, unknown> | string[]> }>(summaryUrl);
  return ids.map((id) => {
    const item = summary.result?.[id] as Record<string, unknown> | undefined;
    return {
      source: "ncbi",
      title: str(item?.title || `PubMed ${id}`),
      content: [item?.fulljournalname, item?.pubdate].filter(Boolean).join("\n"),
      url: `https://pubmed.ncbi.nlm.nih.gov/${id}/`,
      metadata: { pmid: id, publication_date: item?.pubdate },
    };
  });
}

async function searchClinicalTrials(query: string, limit: number): Promise<SearchResult[]> {
  const url = new URL("https://clinicaltrials.gov/api/v2/studies");
  url.searchParams.set("query.term", query);
  url.searchParams.set("pageSize", String(limit));
  url.searchParams.set("format", "json");
  const data = await fetchJson<{ studies?: Array<{ protocolSection?: Record<string, Record<string, unknown>> }> }>(url);
  return (data.studies ?? []).map((study) => {
    const identification = study.protocolSection?.identificationModule ?? {};
    const status = study.protocolSection?.statusModule ?? {};
    const conditions = study.protocolSection?.conditionsModule ?? {};
    const nctId = str(identification.nctId);
    return {
      source: "clinicaltrials",
      title: str(identification.briefTitle || nctId),
      content: [status.overallStatus, Array.isArray(conditions.conditions) ? conditions.conditions.join(", ") : ""].filter(Boolean).join("\n"),
      url: nctId ? `https://clinicaltrials.gov/study/${nctId}` : undefined,
      metadata: { nct_id: nctId, status: status.overallStatus },
    };
  });
}

async function getCitations(doi: string): Promise<SearchResult[]> {
  const data = await fetchJson<Array<Record<string, unknown>>>(`https://opencitations.net/index/api/v2/citations/${encodeURIComponent(doi)}`);
  return data.slice(0, 30).map((item, index) => ({
    source: "opencitations",
    title: `Citation ${index + 1}`,
    content: Object.entries(item).map(([key, value]) => `${key}: ${String(value)}`).join("\n"),
    url: typeof item.citing === "string" ? `https://doi.org/${item.citing}` : undefined,
    metadata: item,
  }));
}

async function searchGdelt(query: string, mode: string, limit: number): Promise<SearchResult[]> {
  const url = new URL("https://api.gdeltproject.org/api/v2/doc/doc");
  url.searchParams.set("query", query);
  url.searchParams.set("mode", mode);
  url.searchParams.set("format", "json");
  url.searchParams.set("maxrecords", String(limit));
  const data = await fetchJson<{ articles?: Array<Record<string, unknown>> }>(url);
  return (data.articles ?? []).map((article) => ({
    source: "gdelt",
    title: str(article.title || article.url || "GDELT article"),
    content: [article.seendate, article.domain, article.sourcecountry].filter(Boolean).join("\n"),
    url: typeof article.url === "string" ? article.url : undefined,
    metadata: article,
  }));
}

async function getWorldBankIndicator(indicator: string, country: string, year?: string): Promise<SearchResult[]> {
  const url = new URL(`https://api.worldbank.org/v2/country/${encodeURIComponent(country || "all")}/indicator/${encodeURIComponent(indicator)}`);
  url.searchParams.set("format", "json");
  url.searchParams.set("per_page", "50");
  if (year) url.searchParams.set("date", year);
  const data = await fetchJson<unknown[]>(url);
  const rows = Array.isArray(data[1]) ? data[1] as Array<Record<string, unknown>> : [];
  return rows.slice(0, 50).map((row) => ({
    source: "worldbank",
    title: `${indicator} ${str((row.country as Record<string, unknown> | undefined)?.value || country)} ${row.date ?? ""}`.trim(),
    content: `Value: ${row.value ?? "n/a"}`,
    url: "https://data.worldbank.org/",
    metadata: row,
    score: Number(row.value ?? 0),
  }));
}

async function searchVulnerabilities(query: string, ecosystem?: string): Promise<SearchResult[]> {
  const id = query.match(/(CVE-\d{4}-\d{4,7}|GHSA-[a-z0-9-]{10,}|PYSEC-\d{4}-\d{1,7})/i)?.[1];
  if (id) return getVulnerability(id.toUpperCase());
  const body: Record<string, unknown> = ecosystem
    ? { package: { name: query, ecosystem } }
    : { query };
  const data = await fetchJson<{ vulns?: Array<Record<string, unknown>> }>("https://api.osv.dev/v1/query", {
    method: "POST",
    json: body,
  });
  return (data.vulns ?? []).slice(0, 10).map(formatVulnerability);
}

async function getVulnerability(id: string): Promise<SearchResult[]> {
  const data = await fetchJson<Record<string, unknown>>(`https://api.osv.dev/v1/vulns/${encodeURIComponent(id)}`);
  return data.id ? [formatVulnerability(data)] : [];
}

async function resolveLibraryId(query: string, libraryName: string, env: Env): Promise<string> {
  const base = context7Base(env);
  const data = await fetchJson<string | Record<string, unknown>>(`${base}/context7/resolve-library-id`, {
    method: "POST",
    json: { query, libraryName },
    headers: context7Headers(env),
  }).catch(() => fetchJson<string | Record<string, unknown>>(`${base}/resolve-library-id?${new URLSearchParams({ query, libraryName })}`, {
    headers: context7Headers(env),
  }));
  if (typeof data === "string") return data.trim();
  const libraryId = str(data.libraryId || data.library_id);
  if (!libraryId) return `No libraries found matching "${libraryName}".`;
  return [`- Title: ${str(data.name || libraryName)}`, `- Context7-compatible library ID: ${libraryId}`].join("\n");
}

async function queryDocs(libraryId: string, query: string, env: Env): Promise<string> {
  const base = context7Base(env);
  const url = `${base}/api/v2/context?${new URLSearchParams({ libraryId, query, type: "txt" })}`;
  const data = await fetchJson<string | Record<string, unknown>>(url, { headers: context7Headers(env) });
  if (typeof data === "string") return data.trim();
  return JSON.stringify(data, null, 2);
}

function crossrefWorkToResult(work: Record<string, unknown>): SearchResult {
  const title = Array.isArray(work.title) ? str(work.title[0]) : str(work.title || "Untitled");
  const authors = Array.isArray(work.author)
    ? work.author.slice(0, 3).map((author) => {
        const item = author as Record<string, unknown>;
        return `${str(item.given)} ${str(item.family)}`.trim();
      }).filter(Boolean).join(", ")
    : "";
  const container = Array.isArray(work["container-title"]) ? str(work["container-title"][0]) : "";
  return {
    source: "crossref",
    title,
    content: [authors ? `Authors: ${authors}` : "", container ? `Journal: ${container}` : "", stripHtml(str(work.abstract || "")).slice(0, 900)].filter(Boolean).join("\n"),
    url: work.DOI ? `https://doi.org/${work.DOI}` : undefined,
    metadata: { doi: work.DOI, type: work.type, publisher: work.publisher, citation_count: work["is-referenced-by-count"] },
    score: Number(work["is-referenced-by-count"] ?? 0),
  };
}

function formatVulnerability(vuln: Record<string, unknown>): SearchResult {
  const id = str(vuln.id || "Unknown");
  const affected = Array.isArray(vuln.affected) ? vuln.affected[0] as Record<string, unknown> | undefined : undefined;
  const pkg = affected?.package as Record<string, unknown> | undefined;
  const references = Array.isArray(vuln.references) ? vuln.references as Array<Record<string, unknown>> : [];
  const reference = references.find((item) => item.type === "ADVISORY" || item.type === "WEB");
  return {
    source: "osv",
    title: `${id}: ${str(vuln.summary || "Security vulnerability").slice(0, 100)}`,
    content: [vuln.summary, pkg ? `Package: ${pkg.ecosystem}/${pkg.name}` : "", str(vuln.details || "").slice(0, 800)].filter(Boolean).join("\n"),
    url: typeof reference?.url === "string" ? reference.url : `https://osv.dev/vulnerability/${id}`,
    metadata: { id, package: pkg ? `${pkg.ecosystem}/${pkg.name}` : undefined, published: vuln.published },
  };
}

interface FetchJsonOptions {
  method?: "GET" | "POST";
  body?: BodyInit;
  json?: unknown;
  headers?: Record<string, string>;
  accept?: string;
}

async function fetchJson<T>(url: URL | string, options: FetchJsonOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    accept: options.accept ?? "application/json",
    "user-agent": "OHI-MCP-Cloudflare/1.0 (https://ohi.shiftbloom.studio)",
    ...(options.headers ?? {}),
  };
  let body = options.body;
  if (options.json !== undefined) {
    body = JSON.stringify(options.json);
    headers["content-type"] = "application/json";
  }
  const response = await fetch(url.toString(), {
    method: options.method ?? (body ? "POST" : "GET"),
    headers,
    body,
  });
  if (!response.ok && response.status !== 404) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  const text = await response.text();
  if (!text) return null as T;
  if ((response.headers.get("content-type") ?? "").includes("json")) return JSON.parse(text) as T;
  return text as T;
}

function success(results: SearchResult[]): KnowledgeToolResult {
  return { success: true, results };
}

function dedupe(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  const out: SearchResult[] = [];
  for (const result of results) {
    const key = result.url || `${result.source}:${result.title}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(result);
  }
  return out;
}

function politeEmail(env: Env): string | undefined {
  const extended = env as Env & { POLITE_POOL_EMAIL?: string; CONTACT_EMAIL?: string };
  return extended.POLITE_POOL_EMAIL ?? extended.CONTACT_EMAIL;
}

function context7Base(env: Env): string {
  return ((env as Env & { CONTEXT7_BASE_URL?: string }).CONTEXT7_BASE_URL ?? "https://context7.com").replace(/\/$/, "");
}

function context7Headers(env: Env): Record<string, string> {
  const key = (env as Env & { CONTEXT7_API_KEY?: string }).CONTEXT7_API_KEY;
  return key ? { authorization: `Bearer ${key}` } : {};
}

function compactQuery(query: string, maxWords = 6): string {
  const stop = new Set(["a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "study", "paper"]);
  const words = query.replace(/[^\w\s.-]/g, " ").split(/\s+/).filter(Boolean);
  const filtered = words.filter((word) => !stop.has(word.toLowerCase()));
  return (filtered.length ? filtered : words).slice(0, maxWords).join(" ");
}

function abstractFromInvertedIndex(index?: Record<string, number[]>): string {
  if (!index) return "";
  const words: Array<[number, string]> = [];
  for (const [word, positions] of Object.entries(index)) {
    for (const position of positions) words.push([position, word]);
  }
  return words.sort((a, b) => a[0] - b[0]).map(([, word]) => word).join(" ");
}

function authorsFromOpenAlex(work: Record<string, unknown>): string {
  const authorships = Array.isArray(work.authorships) ? work.authorships as Array<Record<string, unknown>> : [];
  return authorships.slice(0, 3).map((authorship) => {
    const author = authorship.author as Record<string, unknown> | undefined;
    return str(author?.display_name);
  }).filter(Boolean).join(", ");
}

function stripHtml(text: string): string {
  return text.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

function escapeSparql(text: string): string {
  return text.replace(/\\/g, "\\\\").replace(/"/g, '\\"').slice(0, 100);
}

function str(value: unknown): string {
  return String(value ?? "").trim();
}

function num(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(1, Math.min(50, Math.floor(number))) : fallback;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
