# Research Ingestion Sources

This shortlist focuses on sources that fit the current OHI split:

- `Neo4j`: entities, identifiers, citation edges, trial-paper links, funders, institutions.
- `Qdrant`: chunked abstracts, full text, trial descriptions, outcome sections, and dataset descriptions.

## Recommended order

1. `PubMed + PMC Open Access + OpenAlex`
2. `ClinicalTrials.gov`
3. `Crossref + OpenCitations`
4. `DataCite`
5. `OpenAIRE Graph`

This ordering keeps the first phase practical: strong biomedical coverage, real abstracts/full text, and enough identifiers to join records across sources.

## Source map

| Source | Best fit for OHI | Write mostly to | Why it matters | Official entry point |
| --- | --- | --- | --- | --- |
| OpenAlex snapshot | Works, authors, institutions, concepts, funders, cited/related-work metadata | Neo4j first, Qdrant second | Broad research graph across disciplines, monthly snapshot, strong identifier coverage | <https://developers.openalex.org/download/snapshot-format> |
| OpenAlex CLI | Filtered metadata plus PDFs/TEI XML where available | Qdrant | Good way to bootstrap domain-specific research corpora instead of downloading the entire snapshot first | <https://docs.openalex.org/download-all-data/openalex-cli> |
| PubMed baseline + daily updates | Abstract-heavy biomedical paper corpus | Neo4j and Qdrant | Stable PMID-centered ingest with baseline + incremental updates | <https://pubmed.ncbi.nlm.nih.gov/download/> |
| PMC Open Access Subset | Full biomedical article text, XML, supplementary files | Qdrant first, Neo4j second | Best open full-text source for study passages, methods, results, and discussion chunks | <https://pmc.ncbi.nlm.nih.gov/tools/ftp/> |
| Europe PMC downloads | Open-access articles, preprints, author manuscripts, article metadata | Qdrant and Neo4j | Adds Europe PMC OA text, preprints, and useful identifier mapping files | <https://europepmc.org/downloads> |
| ClinicalTrials.gov API / CSV / FHIR | Trial registry, eligibility, arms, outcomes, recruitment, linked publications | Neo4j and Qdrant | High-value study registry data with structured fields and frequent updates | <https://clinicaltrials.gov/data-api/api> |
| Crossref public data file | DOI metadata, references, funders, journal metadata, retractions | Neo4j first | Strong DOI backbone for joining scholarly records across providers | <https://www.crossref.org/documentation/retrieve-metadata/bulk-downloads/> |
| OpenCitations Meta + Index | Open citation graph and bibliographic metadata | Neo4j | Useful citation-edge source when you want an open citation network without depending only on Crossref references | <https://download.opencitations.net/> |
| DataCite public data file | Dataset, software, repository, and non-article DOI metadata | Neo4j first, Qdrant second | Important if you want evidence to include datasets and research outputs beyond papers | <https://support.datacite.org/docs/datacite-public-data-file> |
| OpenAIRE Graph | Publications, datasets, projects, organizations, funders, relations | Neo4j | Large cross-source research graph; good later-stage federation layer | <https://graph.openaire.eu/docs/10.5.0/cloud-access/> |

## How to use them in this repo

### 1. PubMed + PMC Open Access

Use `PubMed` as the metadata and abstract backbone:

- `(:Paper {pmid, doi, pmcid, title, journal, pub_year, mesh_terms, abstract})`
- `(:Author)`, `(:Journal)`, `(:MeSHTerm)` nodes
- `[:HAS_MESH]`, `[:PUBLISHED_IN]`, `[:AUTHORED]` edges

Use `PMC Open Access` for the actual evidence text:

- chunk `title`, `abstract`, `methods`, `results`, `discussion`, `conclusion`
- keep section labels in Aura
- send only vectors plus minimal payload to Qdrant, same as the current passage pattern

Avoid indexing low-signal sections such as references, acknowledgements, and supplementary link lists into Qdrant.

### 2. OpenAlex

Use `OpenAlex` to enrich the scholarly graph:

- normalize work identifiers across `doi`, `pmid`, `pmcid`
- add `concepts`, `institutions`, `funders`, and `cited_by_count`
- treat OpenAlex as graph enrichment, not as your primary full-text corpus

The full snapshot is best for offline graph loads. The CLI is better when you want filtered topic-first harvesting for a specific domain.

### 3. ClinicalTrials.gov

Model trials separately from papers:

- `(:Trial {nct_id, status, phase, study_type, enrollment, start_date, completion_date})`
- `[:STUDIES_CONDITION]`, `[:TESTS_INTERVENTION]`, `[:HAS_OUTCOME]`, `[:SPONSORED_BY]`
- `[:MENTIONED_IN]` or `[:LINKED_PUBLICATION]` edges to papers when PMIDs or citations exist

For Qdrant, chunk:

- brief summary
- detailed description
- eligibility criteria
- primary and secondary outcomes
- adverse events / posted results when available

This is one of the best future sources for claim checking around medical studies because it provides pre-publication and structured-study context.

### 4. Crossref + OpenCitations

Use `Crossref` as the DOI authority layer and `OpenCitations` as the open citation-edge layer:

- Crossref fills metadata gaps and funder/journal coverage
- OpenCitations gives you a large open citation graph for `[:CITES]`

If both are present, prefer Crossref metadata for the DOI record and OpenCitations for citation edge expansion.

### 5. DataCite

Use `DataCite` when you want study evidence to include:

- datasets
- software packages
- repositories
- supplementary research outputs

This is especially useful if you want claims about data release, dataset provenance, or study artifacts to resolve against first-class nodes instead of being flattened into paper text.

## Practical ingestion strategy

### Neo4j

Prioritize stable identifiers and joins:

- `PMID`
- `PMCID`
- `DOI`
- `NCT`
- `OpenAlex ID`
- `DataCite DOI`

Keep source-native IDs on every node and build explicit same-record joins, for example:

- `(:Paper)-[:SAME_AS]->(:OpenAlexWork)`
- `(:Paper)-[:HAS_PMC_VERSION]->(:FullTextArticle)`
- `(:Trial)-[:LINKED_PUBLICATION]->(:Paper)`

### Qdrant

Keep the vector corpus text-heavy and evidence-oriented:

- abstracts
- result summaries
- methods and outcome sections
- trial eligibility criteria
- dataset descriptions

Do not index every metadata field as text. Use metadata for filtering, and use evidence-bearing sections for embeddings.

## First extensions worth building next

1. A `PubMed` parser that writes paper metadata into Neo4j and abstract passages into Aura/Qdrant.
2. A `PMC OA` parser that maps `PMCID -> PMID/DOI`, chunks section-aware XML, and links back to `Paper`.
3. A `ClinicalTrials.gov` ingester that writes `Trial` nodes plus chunked summary/eligibility/outcome passages.
4. An `OpenAlex` enrichment job that attaches concepts, institutions, funders, and citation counts after the primary ingest.
