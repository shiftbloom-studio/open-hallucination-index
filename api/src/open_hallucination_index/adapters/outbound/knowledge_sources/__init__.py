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
- News/Events: GDELT
- Economic: World Bank Indicators
- Security: OSV (Open Source Vulnerabilities)
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
from open_hallucination_index.adapters.outbound.knowledge_sources.gdelt import (
    GDELTAdapter,
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
from open_hallucination_index.adapters.outbound.knowledge_sources.osv import (
    OSVAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.wikidata import (
    WikidataAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.wikimedia_rest import (
    WikimediaRESTAdapter,
)
from open_hallucination_index.adapters.outbound.knowledge_sources.worldbank import (
    WorldBankAdapter,
)

__all__ = [
    # Base
    "HTTPKnowledgeSource",
    "HTTPKnowledgeSourceError",
    "ClinicalTrialsAdapter",
    "CrossrefAdapter",
    "DBpediaAdapter",
    "EuropePMCAdapter",
    "GDELTAdapter",
    "MediaWikiAdapter",
    "NCBIAdapter",
    "OpenAlexAdapter",
    "OpenCitationsAdapter",
    "OSVAdapter",
    "WikidataAdapter",
    "WikimediaRESTAdapter",
    "WorldBankAdapter",
]
