"""
ClinicalTrials.gov v2 API Adapter
=================================

Queries ClinicalTrials.gov for clinical study data.
https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from open_hallucination_index.adapters.outbound.knowledge_sources.base import (
    HTTPKnowledgeSource,
)
from open_hallucination_index.domain.entities import Evidence, EvidenceSource

if TYPE_CHECKING:
    from open_hallucination_index.domain.entities import Claim

logger = logging.getLogger(__name__)


class ClinicalTrialsAdapter(HTTPKnowledgeSource):
    """
    Adapter for ClinicalTrials.gov v2 API.

    Provides access to clinical trial registrations,
    protocols, and results.
    """

    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)

    @property
    def source_name(self) -> str:
        return "ClinicalTrials.gov"

    @property
    def evidence_source(self) -> EvidenceSource:
        return EvidenceSource.CLINICALTRIALS

    async def health_check(self) -> bool:
        """Check ClinicalTrials.gov API health."""
        if not self._client:
            return False
        try:
            response = await self._client.get(
                "/studies",
                params={"pageSize": 1},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def find_evidence(self, claim: Claim) -> list[Evidence]:
        """Find clinical trial evidence for a claim."""
        if not self._available:
            return []

        evidences: list[Evidence] = []
        search_term = claim.subject or claim.text[:100]

        try:
            studies = await self._search_studies(search_term, limit=5)

            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                desc_module = protocol.get("descriptionModule", {})

                nct_id = id_module.get("nctId", "")
                title = id_module.get("briefTitle", "")

                if not title:
                    continue

                content = f"{title}"

                # Add study status
                status = status_module.get("overallStatus", "")
                if status:
                    content += f"\n\nStatus: {status}"

                # Add brief summary
                brief_summary = desc_module.get("briefSummary", "")
                if brief_summary:
                    content += f"\n\nSummary: {brief_summary[:600]}"

                # Add conditions
                conditions_module = protocol.get("conditionsModule", {})
                conditions = conditions_module.get("conditions", [])
                if conditions:
                    content += f"\n\nConditions: {', '.join(conditions[:5])}"

                # Add interventions
                arms_module = protocol.get("armsInterventionsModule", {})
                interventions = arms_module.get("interventions", [])
                if interventions:
                    intervention_names = [i.get("name", "") for i in interventions[:3]]
                    content += f"\nInterventions: {', '.join(intervention_names)}"

                # Add dates
                start_date = status_module.get("startDateStruct", {}).get("date", "")
                if start_date:
                    content += f"\n\nStart Date: {start_date}"

                evidences.append(
                    self._create_evidence(
                        content=content,
                        source_id=f"clinicaltrials:{nct_id}",
                        source_uri=f"https://clinicaltrials.gov/study/{nct_id}",
                        similarity_score=0.85,
                        structured_data={
                            "nctId": nct_id,
                            "title": title,
                            "status": status,
                            "conditions": conditions[:5],
                            "phase": status_module.get("phases", []),
                        },
                    )
                )

            logger.debug(f"Found {len(evidences)} ClinicalTrials evidences")
            return evidences

        except Exception as e:
            logger.warning(f"ClinicalTrials search failed: {e}")
            return []

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search clinical trials."""
        if not self._available:
            return []
        return await self._search_studies(query, limit)

    async def _search_studies(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for clinical studies."""
        try:
            response = await self._client.get(
                "/studies",
                params={
                    "query.term": query,
                    "pageSize": limit,
                    "sort": "@relevance",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("studies", [])
        except Exception as e:
            logger.warning(f"ClinicalTrials search error: {e}")
            return []

    async def get_study(self, nct_id: str) -> dict[str, Any] | None:
        """Get study by NCT ID."""
        try:
            response = await self._client.get(f"/studies/{nct_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def search_by_condition(self, condition: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search studies by medical condition."""
        try:
            response = await self._client.get(
                "/studies",
                params={
                    "query.cond": condition,
                    "pageSize": limit,
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("studies", [])
        except Exception:
            return []

    async def search_by_intervention(
        self, intervention: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search studies by intervention (drug, device, etc.)."""
        try:
            response = await self._client.get(
                "/studies",
                params={
                    "query.intr": intervention,
                    "pageSize": limit,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("studies", [])
        except Exception:
            return []
