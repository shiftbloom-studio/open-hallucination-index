"""
External Knowledge Source Adapters
===================================

HTTP-based adapters for external knowledge APIs.
These are NOT MCP servers - they connect directly to public APIs.

Organized by domain:
- Linked Data: Wikidata, DBpedia
- Wiki: MediaWiki Action API, Wikimedia REST
- Academic: OpenAlex, Crossref, Europe PMC, OpenCitations
- Medical: NCBI E-utilities, ClinicalTrials.gov
## News/Events: GDELT (pending)
## Economic: World Bank Indicators (pending)
## Security: OSV (Open Source Vulnerabilities) (pending)
"""

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    HTTPKnowledgeSource,
    HTTPKnowledgeSourceError,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.clinicaltrials import (
    ClinicalTrialsAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.crossref import (
    CrossrefAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.dbpedia import (
    DBpediaAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.europepmc import (
    EuropePMCAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.mediawiki import (
    MediaWikiAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.ncbi import (
    NCBIAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.openalex import (
    OpenAlexAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.opencitations import (
    OpenCitationsAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.wikidata import (
    WikidataAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.wikimedia_rest import (
    WikimediaRESTAdapter,
)

__all__ = [
    # Base
    "HTTPKnowledgeSource",
    "HTTPKnowledgeSourceError",
    "ClinicalTrialsAdapter",
    "CrossrefAdapter",
    "DBpediaAdapter",
    "EuropePMCAdapter",
    "MediaWikiAdapter",
    "NCBIAdapter",
    "OpenAlexAdapter",
    "OpenCitationsAdapter",
    "WikidataAdapter",
    "WikimediaRESTAdapter",
]
