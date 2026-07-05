const EMBEDDING_MODEL = "@cf/baai/bge-m3";
const CORPUS_QUEUE_BATCH_SIZE = 25;
const MAX_INGESTION_LIMIT = 5_000;
const DEFAULT_RANDOM_SEED_LIMIT = 250;
const DEFAULT_CHUNK_CHARS = 1_400;
const DEFAULT_CHUNK_OVERLAP = 180;

type CorpusSource = "wikipedia" | "wikidata";
type CorpusMode = "seed" | "backfill";
type CorpusStrategy = "curated" | "random" | "ids";

export interface CorpusWorkflowParams {
  source?: CorpusSource;
  mode?: CorpusMode;
  strategy?: CorpusStrategy;
  limit?: number;
  seed_titles?: string[];
  wikidata_ids?: string[];
  query?: string;
}

export interface CorpusIngestQueueMessage {
  kind: "corpus-ingest";
  runId: string;
  source: CorpusSource;
  mode: CorpusMode;
  strategy: CorpusStrategy;
  batch: number;
  titles?: string[];
  qids?: string[];
  query?: string;
  randomLimit?: number;
  searchLimit?: number;
}

interface CorpusRunConfig {
  source: CorpusSource;
  mode: CorpusMode;
  strategy: CorpusStrategy;
  limit: number;
  seedTitles: string[];
  wikidataIds: string[];
  query?: string;
}

interface CorpusDocument {
  docId: string;
  source: CorpusSource;
  sourceId: string;
  title: string;
  url: string;
  lang: string;
  revision?: string;
  text: string;
  license: string;
  metadata: Record<string, unknown>;
}

interface CorpusChunk {
  chunkId: string;
  vectorId: string;
  docId: string;
  chunkIndex: number;
  text: string;
  metadata: Record<string, unknown>;
}

interface WikidataEntity {
  id: string;
  labels?: Record<string, { value?: string }>;
  descriptions?: Record<string, { value?: string }>;
  aliases?: Record<string, Array<{ value?: string }>>;
  claims?: Record<string, Array<{ mainsnak?: { datavalue?: { value?: unknown } } }>>;
}

const DEFAULT_WIKIPEDIA_SEED_TITLES = [
  "Artificial intelligence",
  "Machine learning",
  "Natural language processing",
  "Hallucination (artificial intelligence)",
  "Large language model",
  "Fact-checking",
  "Wikipedia",
  "Wikidata",
  "Scientific method",
  "Evidence-based medicine",
  "Clinical trial",
  "Climate change",
  "Vaccine",
  "Albert Einstein",
  "Marie Curie",
  "World Health Organization",
  "European Union",
  "United Nations",
  "Internet",
  "Cloudflare",
  "Open-source software",
  "Computer security",
  "Cryptography",
  "Quantum computing",
  "Blockchain",
  "COVID-19",
  "Earth",
  "Moon",
  "Eiffel Tower",
  "Great Wall of China",
  "United States",
  "Germany",
  "Japan",
  "Brazil",
  "Nigeria",
  "India",
  "Economics",
  "Gross domestic product",
  "Inflation",
  "Human rights",
  "Democracy",
  "Law",
  "Copyright",
  "Data protection",
  "General Data Protection Regulation",
  "Software engineering",
  "TypeScript",
  "Python (programming language)",
  "SQL",
  "Graph database",
  "Vector database",
];

const DEFAULT_WIKIPEDIA_SEARCH_QUERIES = [
  "artificial intelligence factual accuracy",
  "large language model hallucination",
  "machine learning evaluation",
  "natural language inference",
  "information retrieval",
  "knowledge graph",
  "semantic search",
  "vector database",
  "graph database",
  "probability calibration",
  "Bayesian inference",
  "statistical hypothesis testing",
  "scientific evidence",
  "peer review",
  "clinical trial phases",
  "evidence based medicine",
  "public health",
  "vaccine efficacy",
  "epidemiology",
  "World Health Organization",
  "climate change",
  "greenhouse gas",
  "renewable energy",
  "biodiversity",
  "earth science",
  "astronomy",
  "space exploration",
  "quantum mechanics",
  "general relativity",
  "particle physics",
  "chemistry",
  "molecular biology",
  "genetics",
  "neuroscience",
  "psychology",
  "economics",
  "gross domestic product",
  "inflation",
  "central bank",
  "international trade",
  "World Bank",
  "United Nations",
  "European Union",
  "human rights",
  "constitutional law",
  "copyright law",
  "data protection",
  "privacy law",
  "General Data Protection Regulation",
  "computer security",
  "cryptography",
  "software vulnerability",
  "open source software",
  "supply chain attack",
  "operating system",
  "database",
  "SQL",
  "Python programming language",
  "TypeScript",
  "JavaScript",
  "Cloudflare Workers",
  "serverless computing",
  "edge computing",
  "HTTP",
  "Domain Name System",
  "Internet protocol suite",
  "cybersecurity",
  "blockchain",
  "cryptocurrency",
  "financial technology",
  "democracy",
  "election",
  "international relations",
  "geopolitics",
  "history of science",
  "World War II",
  "ancient history",
  "philosophy of science",
  "ethics of artificial intelligence",
  "education",
  "university",
  "library science",
  "journalism",
  "misinformation",
  "fact checking",
  "Wikipedia reliability",
  "Wikidata",
  "DBpedia",
  "OpenAlex",
  "Crossref",
  "PubMed",
  "clinicaltrials.gov",
  "OpenCitations",
  "GDELT",
  "World Bank indicator",
  "OSV vulnerability",
  "Eiffel Tower",
  "Moon",
  "Earth",
  "United States",
  "Germany",
  "India",
  "Brazil",
  "Nigeria",
  "Japan",
];

const DEFAULT_WIKIDATA_IDS = [
  "Q11660",
  "Q2539",
  "Q309",
  "Q170658",
  "Q7748",
  "Q180160",
  "Q52",
  "Q2013",
  "Q12483",
  "Q12140",
  "Q30612",
  "Q7942",
  "Q35869",
  "Q937",
  "Q7186",
  "Q7817",
  "Q458",
  "Q1065",
  "Q75",
  "Q13335",
  "Q341",
  "Q131257",
  "Q8789",
  "Q2529",
  "Q80",
];

export function isCorpusIngestQueueMessage(value: unknown): value is CorpusIngestQueueMessage {
  return Boolean(value && typeof value === "object" && (value as { kind?: unknown }).kind === "corpus-ingest");
}

export async function startCorpusRun(env: Env, params: CorpusWorkflowParams): Promise<Record<string, unknown>> {
  const config = normalizeRunConfig(params);
  const runId = crypto.randomUUID();
  const now = secondsNow();
  const messages = buildQueueMessages(runId, config);

  await env.OHI_DB.prepare(
    `INSERT INTO corpus_ingestion_runs
      (run_id, source, mode, strategy, status, total_seen, total_indexed, total_chunks, total_errors, batches_total, batches_completed, config_json, created_at, updated_at)
     VALUES (?, ?, ?, ?, 'queued', 0, 0, 0, 0, ?, 0, ?, ?, ?)`,
  )
    .bind(runId, config.source, config.mode, config.strategy, messages.length, JSON.stringify(config), now, now)
    .run();

  for (const message of messages) {
    await env.CORPUS_QUEUE.send(message);
  }

  await env.OHI_DB.prepare("UPDATE corpus_ingestion_runs SET status = 'running', updated_at = ? WHERE run_id = ?")
    .bind(secondsNow(), runId)
    .run();

  return {
    run_id: runId,
    status: "running",
    source: config.source,
    mode: config.mode,
    strategy: config.strategy,
    limit: config.limit,
    batches_enqueued: messages.length,
  };
}

export async function getCorpusRun(env: Env, runId: string): Promise<Record<string, unknown> | null> {
  const row = await env.OHI_DB.prepare("SELECT * FROM corpus_ingestion_runs WHERE run_id = ?").bind(runId).first<Record<string, unknown>>();
  if (!row) return null;
  const vectorInfo = await env.OHI_VECTOR.describe().catch(() => null);
  return {
    ...row,
    config: parseJson(row.config_json),
    vectorize: vectorInfo,
  };
}

export async function getCorpusOverview(env: Env): Promise<Record<string, unknown>> {
  const totals = await env.OHI_DB.prepare(
    `SELECT
      (SELECT COUNT(*) FROM corpus_documents) AS documents,
      (SELECT COUNT(*) FROM corpus_chunks) AS chunks,
      (SELECT COUNT(*) FROM wikidata_entities) AS wikidata_entities,
      (SELECT COUNT(*) FROM corpus_graph_edges) AS graph_edges`,
  ).first<Record<string, number>>();
  const runs = await env.OHI_DB.prepare(
    "SELECT run_id, source, mode, strategy, status, total_seen, total_indexed, total_chunks, total_errors, batches_total, batches_completed, created_at, updated_at, completed_at FROM corpus_ingestion_runs ORDER BY created_at DESC LIMIT 10",
  ).all<Record<string, unknown>>();
  const vectorInfo = await env.OHI_VECTOR.describe().catch(() => null);
  return {
    totals: totals ?? {},
    recent_runs: runs.results ?? [],
    vectorize: vectorInfo,
    object_store: hasCorpusBucket(env) ? "r2" : "d1-vectorize",
  };
}

export async function processCorpusIngestMessage(env: Env, message: CorpusIngestQueueMessage): Promise<void> {
  const started = secondsNow();
  try {
    const shouldProcess = await markBatchStarted(env, message);
    if (!shouldProcess) return;

    const docs = message.source === "wikipedia"
      ? await fetchWikipediaDocuments(message)
      : await fetchWikidataDocuments(message);

    let indexed = 0;
    let chunks = 0;
    for (const doc of docs) {
      const docChunks = await indexDocument(env, message.runId, doc);
      indexed += 1;
      chunks += docChunks;
    }

    const seen = expectedSeenForMessage(message, docs.length);
    const completed = await markBatchComplete(env, message, { seen, indexed, chunks });
    if (!completed) return;

    await env.OHI_DB.prepare(
      `UPDATE corpus_ingestion_runs
       SET total_seen = total_seen + ?,
           total_indexed = total_indexed + ?,
           total_chunks = total_chunks + ?,
           batches_completed = CASE
             WHEN batches_total > 0 THEN MIN(batches_total, batches_completed + 1)
             ELSE batches_completed + 1
           END,
           status = CASE WHEN status = 'complete' THEN status ELSE 'running' END,
           updated_at = ?
       WHERE run_id = ?`,
    )
      .bind(seen, indexed, chunks, secondsNow(), message.runId)
      .run();

    await maybeCompleteRun(env, message.runId);
  } catch (error) {
    await markBatchFailed(env, message, error);
    await env.OHI_DB.prepare(
      `UPDATE corpus_ingestion_runs
       SET total_errors = total_errors + 1,
           status = 'running',
           error = ?,
           updated_at = ?
       WHERE run_id = ?`,
    )
      .bind(errorMessage(error).slice(0, 2000), secondsNow(), message.runId)
      .run();
    throw error;
  } finally {
    void started;
  }
}

async function markBatchStarted(env: Env, message: CorpusIngestQueueMessage): Promise<boolean> {
  const existing = await env.OHI_DB.prepare(
    "SELECT status FROM corpus_ingestion_batches WHERE run_id = ? AND batch = ?",
  )
    .bind(message.runId, message.batch)
    .first<{ status: string }>();
  if (existing?.status === "complete") return false;

  const now = secondsNow();
  await env.OHI_DB.prepare(
    `INSERT INTO corpus_ingestion_batches
      (run_id, batch, status, attempts, started_at, updated_at)
     VALUES (?, ?, 'processing', 1, ?, ?)
     ON CONFLICT(run_id, batch) DO UPDATE SET
       status = CASE WHEN status = 'complete' THEN status ELSE 'processing' END,
       attempts = attempts + 1,
       updated_at = excluded.updated_at`,
  )
    .bind(message.runId, message.batch, now, now)
    .run();

  const updated = await env.OHI_DB.prepare(
    "SELECT status FROM corpus_ingestion_batches WHERE run_id = ? AND batch = ?",
  )
    .bind(message.runId, message.batch)
    .first<{ status: string }>();
  return updated?.status !== "complete";
}

async function markBatchComplete(
  env: Env,
  message: CorpusIngestQueueMessage,
  result: { seen: number; indexed: number; chunks: number },
): Promise<boolean> {
  const now = secondsNow();
  const update = await env.OHI_DB.prepare(
    `UPDATE corpus_ingestion_batches
     SET status = 'complete',
         seen = ?,
         indexed = ?,
         chunks = ?,
         error = NULL,
         completed_at = ?,
         updated_at = ?
     WHERE run_id = ? AND batch = ? AND status != 'complete'`,
  )
    .bind(result.seen, result.indexed, result.chunks, now, now, message.runId, message.batch)
    .run();
  return Number(update.meta.changes ?? 0) > 0;
}

async function markBatchFailed(env: Env, message: CorpusIngestQueueMessage, error: unknown): Promise<void> {
  await env.OHI_DB.prepare(
    `UPDATE corpus_ingestion_batches
     SET status = CASE WHEN status = 'complete' THEN status ELSE 'error' END,
         error = CASE WHEN status = 'complete' THEN error ELSE ? END,
         updated_at = ?
     WHERE run_id = ? AND batch = ?`,
  )
    .bind(errorMessage(error).slice(0, 2000), secondsNow(), message.runId, message.batch)
    .run();
}

function normalizeRunConfig(params: CorpusWorkflowParams): CorpusRunConfig {
  const source: CorpusSource = params.source === "wikidata" ? "wikidata" : "wikipedia";
  const mode: CorpusMode = params.mode === "backfill" ? "backfill" : "seed";
  const explicitTitles = cleanList(params.seed_titles);
  const explicitQids = cleanList(params.wikidata_ids).map((qid) => qid.toUpperCase());
  const strategy: CorpusStrategy =
    params.strategy === "ids" || explicitTitles.length > 0 || explicitQids.length > 0
      ? "ids"
      : params.strategy === "random"
        ? "random"
        : "curated";
  const defaultLimit = source === "wikipedia" && strategy === "random" ? DEFAULT_RANDOM_SEED_LIMIT : 50;
  const limit = clamp(Math.floor(Number(params.limit ?? defaultLimit) || defaultLimit), 1, MAX_INGESTION_LIMIT);
  return {
    source,
    mode,
    strategy,
    limit,
    seedTitles: explicitTitles,
    wikidataIds: explicitQids,
    query: typeof params.query === "string" ? params.query.slice(0, 200) : undefined,
  };
}

function buildQueueMessages(runId: string, config: CorpusRunConfig): CorpusIngestQueueMessage[] {
  if (config.source === "wikidata") {
    const ids = (config.wikidataIds.length > 0 ? config.wikidataIds : DEFAULT_WIKIDATA_IDS).slice(0, config.limit);
    return chunkArray(ids, 10).map((qids, batch) => ({
      kind: "corpus-ingest",
      runId,
      source: "wikidata",
      mode: config.mode,
      strategy: config.wikidataIds.length > 0 ? "ids" : "curated",
      batch,
      qids,
      query: config.query,
    }));
  }

  if (config.strategy === "random") {
    const batches = Math.ceil(config.limit / CORPUS_QUEUE_BATCH_SIZE);
    return Array.from({ length: batches }, (_, batch): CorpusIngestQueueMessage => ({
      kind: "corpus-ingest",
      runId,
      source: "wikipedia",
      mode: config.mode,
      strategy: "random",
      batch,
      randomLimit: Math.min(CORPUS_QUEUE_BATCH_SIZE, config.limit - batch * CORPUS_QUEUE_BATCH_SIZE),
      query: config.query,
    }));
  }

  if (config.seedTitles.length > 0) {
    return chunkArray(config.seedTitles.slice(0, config.limit), 8).map((batchTitles, batch) => ({
      kind: "corpus-ingest",
      runId,
      source: "wikipedia",
      mode: config.mode,
      strategy: "ids",
      batch,
      titles: batchTitles,
      query: config.query,
    }));
  }

  const titleMessages = chunkArray(DEFAULT_WIKIPEDIA_SEED_TITLES.slice(0, Math.min(config.limit, DEFAULT_WIKIPEDIA_SEED_TITLES.length)), 8)
    .map((batchTitles, batch): CorpusIngestQueueMessage => ({
      kind: "corpus-ingest",
      runId,
      source: "wikipedia",
      mode: config.mode,
      strategy: "curated",
      batch,
      titles: batchTitles,
      query: config.query,
    }));

  const remaining = Math.max(0, config.limit - DEFAULT_WIKIPEDIA_SEED_TITLES.length);
  const queryMessages = DEFAULT_WIKIPEDIA_SEARCH_QUERIES.slice(0, Math.ceil(remaining / 10))
    .map((query, index): CorpusIngestQueueMessage => ({
      kind: "corpus-ingest",
      runId,
      source: "wikipedia",
      mode: config.mode,
      strategy: "curated",
      batch: titleMessages.length + index,
      query,
      searchLimit: Math.min(10, remaining - index * 10),
    }))
    .filter((message) => (message.searchLimit ?? 0) > 0);

  return [...titleMessages, ...queryMessages];
}

function expectedSeenForMessage(message: CorpusIngestQueueMessage, fetched: number): number {
  if (message.source === "wikipedia" && message.strategy === "random") {
    return Math.max(1, message.randomLimit ?? fetched);
  }
  if (message.titles?.length) return message.titles.length;
  if (message.qids?.length) return message.qids.length;
  if (message.searchLimit) return message.searchLimit;
  return fetched;
}

async function fetchWikipediaDocuments(message: CorpusIngestQueueMessage): Promise<CorpusDocument[]> {
  if (message.strategy === "random") {
    const titles = await fetchWikipediaRandomTitles(clamp(message.randomLimit ?? CORPUS_QUEUE_BATCH_SIZE, 1, 50));
    return fetchWikipediaDocuments({ ...message, strategy: "ids", titles });
  }

  if ((!message.titles || message.titles.length === 0) && message.query) {
    const titles = await fetchWikipediaSearchTitles(message.query, clamp(message.searchLimit ?? 10, 1, 50));
    return fetchWikipediaDocuments({ ...message, strategy: "ids", titles });
  }

  const api = new URL("https://en.wikipedia.org/w/api.php");
  api.searchParams.set("action", "query");
  api.searchParams.set("format", "json");
  api.searchParams.set("utf8", "1");
  api.searchParams.set("origin", "*");
  api.searchParams.set("prop", "extracts|info|pageprops");
  api.searchParams.set("explaintext", "1");
  api.searchParams.set("exsectionformat", "plain");
  api.searchParams.set("inprop", "url");

  api.searchParams.set("titles", (message.titles ?? []).join("|"));

  const data = await fetchJson<{
    query?: {
      pages?: Record<string, {
        pageid?: number;
        title?: string;
        extract?: string;
        fullurl?: string;
        lastrevid?: number;
        pageprops?: { wikibase_item?: string };
      }>;
    };
  }>(api.toString());

  const documents: CorpusDocument[] = [];
  for (const page of Object.values(data.query?.pages ?? {})) {
    if (!page.pageid || !page.title) continue;
    let text = page.extract ?? "";
    let url = page.fullurl ?? `https://en.wikipedia.org/wiki/${encodeURIComponent(String(page.title).replaceAll(" ", "_"))}`;
    if (text.length < 160) {
      const summary = await fetchWikipediaSummary(String(page.title));
      text = summary?.extract ?? text;
      url = summary?.content_urls?.desktop?.page ?? url;
    }
    if (text.length < 160) continue;
    documents.push({
      docId: `wikipedia:${page.pageid}`,
      source: "wikipedia",
      sourceId: String(page.pageid),
      title: page.title,
      url,
      lang: "en",
      revision: page.lastrevid ? String(page.lastrevid) : undefined,
      text,
      license: "CC BY-SA 4.0",
      metadata: {
        wikidata_id: page.pageprops?.wikibase_item,
        ingest_strategy: message.strategy,
        run_id: message.runId,
      },
    });
  }
  return documents;
}

async function fetchWikipediaRandomTitles(limit: number): Promise<string[]> {
  const api = new URL("https://en.wikipedia.org/w/api.php");
  api.searchParams.set("action", "query");
  api.searchParams.set("list", "random");
  api.searchParams.set("rnnamespace", "0");
  api.searchParams.set("rnfilterredir", "nonredirects");
  api.searchParams.set("rnlimit", String(limit));
  api.searchParams.set("format", "json");
  api.searchParams.set("origin", "*");
  const data = await fetchJson<{ query?: { random?: Array<{ title?: string }> } }>(api.toString());
  return (data.query?.random ?? []).map((item) => item.title).filter((title): title is string => Boolean(title));
}

async function fetchWikipediaSearchTitles(query: string, limit: number): Promise<string[]> {
  const api = new URL("https://en.wikipedia.org/w/api.php");
  api.searchParams.set("action", "query");
  api.searchParams.set("list", "search");
  api.searchParams.set("srsearch", query);
  api.searchParams.set("srlimit", String(limit));
  api.searchParams.set("srnamespace", "0");
  api.searchParams.set("format", "json");
  api.searchParams.set("origin", "*");
  const data = await fetchJson<{ query?: { search?: Array<{ title?: string }> } }>(api.toString());
  return (data.query?.search ?? []).map((item) => item.title).filter((title): title is string => Boolean(title));
}

async function fetchWikipediaSummary(title: string): Promise<{
  extract?: string;
  content_urls?: { desktop?: { page?: string } };
} | null> {
  try {
    return await fetchJson<{
      extract?: string;
      content_urls?: { desktop?: { page?: string } };
    }>(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`);
  } catch {
    return null;
  }
}

async function fetchWikidataDocuments(message: CorpusIngestQueueMessage): Promise<CorpusDocument[]> {
  let qids = message.qids ?? [];
  if (qids.length === 0 && message.query) {
    qids = await searchWikidataIds(message.query, message.randomLimit ?? 10);
  }
  if (qids.length === 0) return [];

  const api = new URL("https://www.wikidata.org/w/api.php");
  api.searchParams.set("action", "wbgetentities");
  api.searchParams.set("ids", qids.join("|"));
  api.searchParams.set("props", "labels|descriptions|aliases|claims");
  api.searchParams.set("languages", "en");
  api.searchParams.set("format", "json");
  api.searchParams.set("origin", "*");

  const data = await fetchJson<{ entities?: Record<string, WikidataEntity> }>(api.toString());
  return Object.values(data.entities ?? {})
    .filter((entity) => entity.id && entity.id !== "-1")
    .map((entity) => {
      const label = entity.labels?.en?.value ?? entity.id;
      const description = entity.descriptions?.en?.value ?? "";
      const aliases = (entity.aliases?.en ?? []).map((alias) => alias.value).filter(Boolean);
      const claims = compactClaims(entity.claims ?? {});
      const text = [
        `${label} (${entity.id})`,
        description,
        aliases.length ? `Aliases: ${aliases.join(", ")}` : "",
        claims.length ? `Claims: ${claims.map((claim) => `${claim.property}=${claim.value}`).join("; ")}` : "",
      ].filter(Boolean).join("\n");
      return {
        docId: `wikidata:${entity.id}`,
        source: "wikidata",
        sourceId: entity.id,
        title: label,
        url: `https://www.wikidata.org/wiki/${entity.id}`,
        lang: "en",
        text,
        license: "CC0 1.0",
        metadata: {
          description,
          aliases,
          claims,
          run_id: message.runId,
        },
      };
    });
}

async function indexDocument(env: Env, runId: string, doc: CorpusDocument): Promise<number> {
  const now = secondsNow();
  const hash = await sha256Hex(doc.text);
  const chunks = makeChunks(doc);
  const embeddings = await embedTexts(env, chunks.map((chunk) => chunk.text));
  const vectors: VectorizeVector[] = [];

  for (let index = 0; index < chunks.length; index += 1) {
    const values = embeddings[index];
    if (!values) continue;
    const chunk = chunks[index];
    vectors.push({
      id: chunk.vectorId,
      values,
      metadata: {
        source: doc.source,
        source_id: doc.sourceId,
        doc_id: doc.docId,
        chunk_id: chunk.chunkId,
        title: doc.title,
        source_uri: doc.url,
        snippet: chunk.text.slice(0, 320),
        content: chunk.text.slice(0, 2_000),
        run_id: runId,
      },
    });
  }

  for (const vectorBatch of chunkArray(vectors, 16)) {
    await env.OHI_VECTOR.upsert(vectorBatch);
  }

  const statements: D1PreparedStatement[] = [
    env.OHI_DB.prepare(
      `INSERT OR REPLACE INTO corpus_documents
        (doc_id, source, source_id, title, url, lang, revision, content_hash, license, indexed_at, metadata_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(
      doc.docId,
      doc.source,
      doc.sourceId,
      doc.title,
      doc.url,
      doc.lang,
      doc.revision ?? null,
      hash,
      doc.license,
      now,
      JSON.stringify(doc.metadata),
    ),
  ];

  for (const chunk of chunks) {
    statements.push(
      env.OHI_DB.prepare(
        `INSERT OR REPLACE INTO corpus_chunks
          (chunk_id, doc_id, chunk_index, text, char_count, embedding_model, vector_id, indexed_at, metadata_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).bind(
        chunk.chunkId,
        doc.docId,
        chunk.chunkIndex,
        chunk.text,
        chunk.text.length,
        EMBEDDING_MODEL,
        chunk.vectorId,
        now,
        JSON.stringify(chunk.metadata),
      ),
    );
  }

  if (doc.source === "wikidata") {
    const metadata = doc.metadata as { aliases?: string[]; claims?: Array<{ property: string; value: string }> };
    statements.push(
      env.OHI_DB.prepare(
        `INSERT OR REPLACE INTO wikidata_entities
          (qid, label, description, aliases_json, claims_json, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ).bind(
        doc.sourceId,
        doc.title,
        String(doc.metadata.description ?? ""),
        JSON.stringify(metadata.aliases ?? []),
        JSON.stringify(metadata.claims ?? []),
        now,
      ),
    );
    for (const claim of metadata.claims ?? []) {
      statements.push(
        env.OHI_DB.prepare(
          `INSERT OR REPLACE INTO corpus_graph_edges
            (edge_id, source_type, source_id, target_type, target_id, edge_type, weight, metadata_json, created_at)
           VALUES (?, 'wikidata_entity', ?, 'wikidata_value', ?, ?, 1, ?, ?)`,
        ).bind(
          `wd:${doc.sourceId}:${claim.property}:${claim.value}`.slice(0, 240),
          doc.sourceId,
          claim.value,
          claim.property,
          JSON.stringify(claim),
          now,
        ),
      );
    }
  } else {
    const qid = typeof doc.metadata.wikidata_id === "string" ? doc.metadata.wikidata_id : "";
    if (qid) {
      statements.push(
        env.OHI_DB.prepare(
          `INSERT OR REPLACE INTO document_entities
            (doc_id, qid, relation, score, metadata_json, created_at)
           VALUES (?, ?, 'about', 1, ?, ?)`,
        ).bind(doc.docId, qid, JSON.stringify({ source: "wikipedia_pageprops" }), now),
      );
    }
  }

  await env.OHI_DB.batch(statements);
  await putCorpusObject(env, `runs/${runId}/documents/${doc.source}/${doc.sourceId}.json`, {
    ...doc,
    text: doc.text,
    hash,
    chunks: chunks.map((chunk) => ({ ...chunk, text: chunk.text.slice(0, 2_000) })),
  });

  return vectors.length;
}

function makeChunks(doc: CorpusDocument): CorpusChunk[] {
  const chunks: CorpusChunk[] = [];
  const text = doc.text.replace(/\s+/g, " ").trim();
  const step = DEFAULT_CHUNK_CHARS - DEFAULT_CHUNK_OVERLAP;
  for (let start = 0; start < text.length && chunks.length < 200; start += step) {
    const raw = text.slice(start, start + DEFAULT_CHUNK_CHARS).trim();
    if (raw.length < 160) continue;
    const chunkIndex = chunks.length;
    const chunkId = `${doc.docId}:chunk:${chunkIndex}`;
    chunks.push({
      chunkId,
      vectorId: `corpus:${chunkId}`,
      docId: doc.docId,
      chunkIndex,
      text: `${doc.title}\n\n${raw}`,
      metadata: {
        source: doc.source,
        source_id: doc.sourceId,
        title: doc.title,
        url: doc.url,
      },
    });
  }
  return chunks;
}

async function embedTexts(env: Env, texts: string[]): Promise<Array<number[] | null>> {
  const output: Array<number[] | null> = [];
  for (const batch of chunkArray(texts, 8)) {
    try {
      const result = await env.AI.run(
        EMBEDDING_MODEL,
        { text: batch.map((text) => text.slice(0, 8_000)) },
        { gateway: { id: "default", skipCache: false, cacheTtl: 86400 } },
      );
      output.push(...extractEmbeddings(result, batch.length));
    } catch {
      output.push(...batch.map(() => null));
    }
  }
  return output;
}

function extractEmbeddings(result: unknown, expected: number): Array<number[] | null> {
  const obj = result as Record<string, unknown>;
  const data = Array.isArray(obj?.data)
    ? obj.data
    : obj?.result && typeof obj.result === "object" && Array.isArray((obj.result as Record<string, unknown>).data)
      ? ((obj.result as Record<string, unknown>).data as unknown[])
      : [];
  const embeddings = data.map((item) => {
    if (Array.isArray(item)) return item as number[];
    if (item && typeof item === "object") {
      const record = item as Record<string, unknown>;
      if (Array.isArray(record.embedding)) return record.embedding as number[];
      if (Array.isArray(record.values)) return record.values as number[];
    }
    return null;
  });
  while (embeddings.length < expected) embeddings.push(null);
  return embeddings.slice(0, expected);
}

async function searchWikidataIds(query: string, limit: number): Promise<string[]> {
  const api = new URL("https://www.wikidata.org/w/api.php");
  api.searchParams.set("action", "wbsearchentities");
  api.searchParams.set("search", query);
  api.searchParams.set("language", "en");
  api.searchParams.set("format", "json");
  api.searchParams.set("limit", String(clamp(limit, 1, 50)));
  api.searchParams.set("origin", "*");
  const data = await fetchJson<{ search?: Array<{ id?: string }> }>(api.toString());
  return (data.search ?? []).map((item) => item.id).filter((id): id is string => Boolean(id));
}

function compactClaims(claims: Record<string, Array<{ mainsnak?: { datavalue?: { value?: unknown } } }>>): Array<{ property: string; value: string }> {
  const out: Array<{ property: string; value: string }> = [];
  for (const [property, rows] of Object.entries(claims).slice(0, 40)) {
    for (const row of rows.slice(0, 3)) {
      const value = row.mainsnak?.datavalue?.value;
      const normalized = normalizeClaimValue(value);
      if (normalized) out.push({ property, value: normalized });
    }
  }
  return out.slice(0, 100);
}

function normalizeClaimValue(value: unknown): string | null {
  if (typeof value === "string") return value.slice(0, 200);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record.id === "string") return record.id;
    if (typeof record["numeric-id"] === "number") return `Q${record["numeric-id"]}`;
    if (typeof record.time === "string") return record.time;
    if (typeof record.text === "string") return record.text.slice(0, 200);
  }
  return null;
}

async function maybeCompleteRun(env: Env, runId: string): Promise<void> {
  const row = await env.OHI_DB.prepare(
    "SELECT batches_total, batches_completed FROM corpus_ingestion_runs WHERE run_id = ?",
  )
    .bind(runId)
    .first<{ batches_total: number; batches_completed: number }>();
  const batchesTotal = Number(row?.batches_total ?? 0);
  const batchesCompleted = Number(row?.batches_completed ?? 0);
  if (batchesTotal > 0 && batchesCompleted >= batchesTotal) {
    await env.OHI_DB.prepare(
      "UPDATE corpus_ingestion_runs SET status = 'complete', updated_at = ?, completed_at = ? WHERE run_id = ? AND status != 'complete'",
    )
      .bind(secondsNow(), secondsNow(), runId)
      .run();
  }
}

async function putCorpusObject(env: Env, key: string, value: unknown): Promise<void> {
  const bucket = (env as Env & { OHI_CORPUS?: R2Bucket }).OHI_CORPUS;
  if (!bucket) return;
  await bucket.put(key, JSON.stringify(value), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });
}

function hasCorpusBucket(env: Env): boolean {
  return Boolean((env as Env & { OHI_CORPUS?: R2Bucket }).OHI_CORPUS);
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: {
      accept: "application/json",
      "user-agent": "OpenHallucinationIndex/0.1 (https://ohi.shiftbloom.studio; corpus-ingestion)",
    },
  });
  if (!response.ok) throw new Error(`fetch failed: ${response.status}`);
  return response.json() as Promise<T>;
}

function cleanList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => String(item ?? "").trim()).filter(Boolean).slice(0, MAX_INGESTION_LIMIT)
    : [];
}

function chunkArray<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string" || !value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function secondsNow(): number {
  return Date.now() / 1000;
}

async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
