/**
 * ClinicalTrials.gov Source
 * =========================
 * 
 * Clinical study data.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface ClinicalTrialStudy {
  protocolSection?: {
    identificationModule?: {
      nctId?: string;
      briefTitle?: string;
    };
    statusModule?: {
      overallStatus?: string;
      startDateStruct?: { date?: string };
      phases?: string[];
    };
    descriptionModule?: {
      briefSummary?: string;
    };
    conditionsModule?: {
      conditions?: string[];
    };
    armsInterventionsModule?: {
      interventions?: Array<{ name?: string; type?: string }>;
    };
  };
}

interface ClinicalTrialsResponse {
  studies?: ClinicalTrialStudy[];
}

export class ClinicalTrialsSource extends BaseSource {
  name = "clinicaltrials";
  description = "ClinicalTrials.gov clinical studies";

  constructor() {
    super("https://clinicaltrials.gov/api/v2");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/studies`, {
        params: { pageSize: 1 },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  async search(query: string, limit = 5): Promise<SearchResult[]> {
    // Extract NCT ID if present (e.g. NCT04368728) to improve search precision
    const nctMatch = query.match(/NCT\d{8}/i);
    const cleaned = (nctMatch ? nctMatch[0] : query)
      .replace(/["'`]/g, "")
      .replace(/[^\w\s-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    if (!cleaned || cleaned.length < 3) {
      return [];
    }

    try {
      const response = await httpClient.get<ClinicalTrialsResponse>(`${this.baseUrl}/studies`, {
        params: {
          "query.term": cleaned,
          pageSize: limit,
          // Removed sort: "@relevance" as it causes issues with v2 API
        },
      });

      return (response.data.studies || []).map((study) => {
      const protocol = study.protocolSection || {};
      const id = protocol.identificationModule || {};
      const status = protocol.statusModule || {};
      const desc = protocol.descriptionModule || {};
      const conditions = protocol.conditionsModule || {};
      const interventions = protocol.armsInterventionsModule?.interventions || [];

      const nctId = id.nctId || "";
      const title = id.briefTitle || nctId;

      const content = [
        `Status: ${status.overallStatus || "Unknown"}`,
        status.phases?.length ? `Phase: ${status.phases.join(", ")}` : "",
        conditions.conditions?.length ? `Conditions: ${conditions.conditions.slice(0, 3).join(", ")}` : "",
        interventions.length ? `Interventions: ${interventions.slice(0, 3).map((i) => i.name).join(", ")}` : "",
        "",
        desc.briefSummary?.slice(0, 600) || "",
      ]
        .filter(Boolean)
        .join("\n");

        return {
        source: this.name,
        title,
        content,
        url: `https://clinicaltrials.gov/study/${nctId}`,
        metadata: {
          nctId,
          status: status.overallStatus,
          phases: status.phases,
          startDate: status.startDateStruct?.date,
        },
        };
      });
    } catch (error) {
      console.warn("ClinicalTrials search failed", error);
      return [];
    }
  }

  async getStudy(nctId: string): Promise<SearchResult | null> {
    try {
      const response = await httpClient.get<ClinicalTrialStudy>(`${this.baseUrl}/studies/${nctId}`);
      const protocol = response.data.protocolSection;
      if (!protocol) return null;

      return {
        source: this.name,
        title: protocol.identificationModule?.briefTitle || nctId,
        content: protocol.descriptionModule?.briefSummary || "",
        url: `https://clinicaltrials.gov/study/${nctId}`,
        metadata: {
          nctId,
          status: protocol.statusModule?.overallStatus,
        },
      };
    } catch {
      return null;
    }
  }
}
