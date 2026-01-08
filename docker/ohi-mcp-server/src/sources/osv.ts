/**
 * OSV Source
 * ==========
 * 
 * Open Source Vulnerabilities database.
 */

import { BaseSource, SearchResult } from "./base.js";
import { httpClient } from "../utils/http-client.js";

interface OSVVulnerability {
  id?: string;
  summary?: string;
  details?: string;
  severity?: Array<{ type?: string; score?: string }>;
  affected?: Array<{
    package?: { name?: string; ecosystem?: string };
    ranges?: Array<{ events?: Array<{ introduced?: string; fixed?: string }> }>;
  }>;
  references?: Array<{ type?: string; url?: string }>;
  published?: string;
  modified?: string;
}

interface OSVQueryResponse {
  vulns?: OSVVulnerability[];
}

export class OSVSource extends BaseSource {
  name = "osv";
  description = "Open Source Vulnerabilities database";

  constructor() {
    super("https://api.osv.dev/v1");
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await httpClient.get(`${this.baseUrl}/vulns/GHSA-jfh8-c2jp-5v3q`);
      return response.status === 200 || response.status === 404;
    } catch {
      return false;
    }
  }

  async search(query: string, _limit = 5): Promise<SearchResult[]> {
    // Check if it's a vulnerability ID
    if (query.startsWith("GHSA-") || query.startsWith("CVE-") || query.startsWith("PYSEC-")) {
      const vuln = await this.getVulnerability(query);
      return vuln ? [vuln] : [];
    }

    // Search by package
    return this.queryPackage(query);
  }

  async queryPackage(packageName: string, ecosystem?: string): Promise<SearchResult[]> {
    try {
      const body: Record<string, unknown> = {
        package: { name: packageName },
      };
      if (ecosystem) {
        (body.package as Record<string, string>).ecosystem = ecosystem;
      }

      const response = await httpClient.post<OSVQueryResponse>(`${this.baseUrl}/query`, body);

      return (response.data.vulns || []).slice(0, 10).map((vuln) => this.formatVulnerability(vuln));
    } catch {
      return [];
    }
  }

  async getVulnerability(id: string): Promise<SearchResult | null> {
    try {
      const response = await httpClient.get<OSVVulnerability>(`${this.baseUrl}/vulns/${id}`);
      return this.formatVulnerability(response.data);
    } catch {
      return null;
    }
  }

  private formatVulnerability(vuln: OSVVulnerability): SearchResult {
    const id = vuln.id || "Unknown";
    const severity = vuln.severity?.[0];
    const affected = vuln.affected?.[0];
    const pkg = affected?.package;

    const content = [
      vuln.summary || "",
      "",
      severity ? `Severity: ${severity.type} ${severity.score}` : "",
      pkg ? `Package: ${pkg.ecosystem}/${pkg.name}` : "",
      vuln.published ? `Published: ${vuln.published.split("T")[0]}` : "",
      "",
      vuln.details?.slice(0, 500) || "",
    ]
      .filter(Boolean)
      .join("\n");

    const reference = vuln.references?.find((r) => r.type === "ADVISORY" || r.type === "WEB");

    return {
      source: this.name,
      title: `${id}: ${vuln.summary?.slice(0, 80) || "Security vulnerability"}`,
      content,
      url: reference?.url || `https://osv.dev/vulnerability/${id}`,
      metadata: {
        id,
        severity: severity?.score,
        package: pkg ? `${pkg.ecosystem}/${pkg.name}` : undefined,
        published: vuln.published,
      },
    };
  }
}
