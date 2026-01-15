/**
 * Context7 Client Helpers
 * =======================
 *
 * Lightweight helpers for Context7 REST endpoints. These helpers return
 * plain text outputs for compatibility with MCP clients that expect
 * text responses.
 */

import { httpClient } from "./http-client.js";

interface ResolveLibraryResponse {
  libraryId?: string;
  library_id?: string;
  name?: string;
  alternatives?: string[];
}

interface Context7CodeSnippet {
  codeTitle?: string;
  codeDescription?: string;
  codeLanguage?: string;
  codeId?: string;
  pageTitle?: string;
  codeList?: Array<{ language?: string; code?: string }>;
}

interface Context7InfoSnippet {
  pageId?: string;
  breadcrumb?: string;
  content?: string;
}

interface Context7ContextResponse {
  codeSnippets?: Context7CodeSnippet[];
  infoSnippets?: Context7InfoSnippet[];
}

const DEFAULT_BASE_URL = "https://context7.com";

function getBaseUrl(): string {
  return (process.env.CONTEXT7_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, "");
}

function getAuthHeaders(): Record<string, string> {
  const apiKey = process.env.CONTEXT7_API_KEY || "";
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

function formatResolveResponse(
  data: ResolveLibraryResponse,
  fallbackLibraryName: string
): string {
  const libraryId = data.libraryId || data.library_id;
  if (!libraryId) {
    return `No libraries found matching "${fallbackLibraryName}".`;
  }

  const title = data.name || fallbackLibraryName;
  const lines = [
    `- Title: ${title}`,
    `- Context7-compatible library ID: ${libraryId}`,
  ];

  if (data.alternatives && data.alternatives.length > 0) {
    lines.push(`- Alternatives: ${data.alternatives.join(", ")}`);
  }

  return lines.join("\n");
}

function formatContextResponse(data: Context7ContextResponse): string {
  const lines: string[] = [];

  if (data.infoSnippets && data.infoSnippets.length > 0) {
    lines.push("## Context Information");
    for (const snippet of data.infoSnippets) {
      if (snippet.breadcrumb) {
        lines.push(`### ${snippet.breadcrumb}`);
      }
      if (snippet.pageId) {
        lines.push(`Source: ${snippet.pageId}`);
      }
      if (snippet.content) {
        lines.push(snippet.content);
      }
      lines.push("");
    }
  }

  if (data.codeSnippets && data.codeSnippets.length > 0) {
    lines.push("## Code Snippets");
    for (const snippet of data.codeSnippets) {
      const title = snippet.codeTitle || snippet.pageTitle || "Code Snippet";
      lines.push(`### ${title}`);
      if (snippet.codeDescription) {
        lines.push(snippet.codeDescription);
      }
      if (snippet.codeId) {
        lines.push(`Source: ${snippet.codeId}`);
      }
      if (snippet.codeList) {
        for (const block of snippet.codeList) {
          const language = block.language || snippet.codeLanguage || "";
          const code = block.code || "";
          lines.push("```" + language);
          lines.push(code);
          lines.push("```");
        }
      }
      lines.push("");
    }
  }

  if (lines.length === 0) {
    return "No documentation snippets available.";
  }

  return lines.join("\n").trim();
}

export async function resolveLibraryId(
  query: string,
  libraryName: string
): Promise<string> {
  const baseUrl = getBaseUrl();
  const headers = getAuthHeaders();

  try {
    const response = await httpClient.post<string | ResolveLibraryResponse>(
      `${baseUrl}/context7/resolve-library-id`,
      { query, libraryName },
      { headers }
    );

    if (response.status === 404) {
      return `No libraries found matching "${libraryName}".`;
    }

    if (typeof response.data === "string") {
      return response.data.trim();
    }

    return formatResolveResponse(response.data, libraryName);
  } catch {
    const fallback = await httpClient.get<string | ResolveLibraryResponse>(
      `${baseUrl}/resolve-library-id`,
      { headers, params: { query, libraryName } }
    );

    if (fallback.status === 404) {
      return `No libraries found matching "${libraryName}".`;
    }

    if (typeof fallback.data === "string") {
      return fallback.data.trim();
    }

    return formatResolveResponse(fallback.data, libraryName);
  }
}

export async function queryDocs(
  libraryId: string,
  query: string
): Promise<string> {
  const baseUrl = getBaseUrl();
  const headers = getAuthHeaders();

  const response = await httpClient.get<string | Context7ContextResponse>(
    `${baseUrl}/api/v2/context`,
    {
      headers,
      params: {
        libraryId,
        query,
        type: "txt",
      },
    }
  );

  if (typeof response.data === "string") {
    return response.data.trim();
  }

  return formatContextResponse(response.data);
}
