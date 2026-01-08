/**
 * OHI Unified MCP Server
 * ======================
 * 
 * High-performance MCP server aggregating 13+ knowledge sources:
 * - Wikidata SPARQL
 * - MediaWiki Action API
 * - Wikimedia REST
 * - DBpedia SPARQL
 * - OpenAlex
 * - Crossref
 * - Europe PMC
 * - NCBI E-utilities
 * - ClinicalTrials.gov
 * - OpenCitations
 * - GDELT
 * - World Bank
 * - OSV (Open Source Vulnerabilities)
 * 
 * Features:
 * - Async parallel queries
 * - Connection pooling
 * - Rate limiting per source
 * - Response caching
 * - Intelligent source routing
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import express, { Request, Response } from "express";

import { toolAggregator } from "./aggregator.js";
import { sourceRegistry } from "./sources/registry.js";
import { RateLimiter } from "./utils/rate-limiter.js";
import { ResponseCache } from "./utils/cache.js";

// Configuration
const PORT = parseInt(process.env.PORT || "8080", 10);
const CACHE_TTL = parseInt(process.env.CACHE_TTL || "300", 10); // 5 minutes
const ENABLE_CACHE = process.env.ENABLE_CACHE !== "false";

// Initialize components
const rateLimiter = new RateLimiter();
const cache = new ResponseCache(CACHE_TTL, ENABLE_CACHE);
// Registry and aggregator are singleton instances from modules

// Define MCP tools
const tools: Tool[] = [
  {
    name: "search_all",
    description: "Search across ALL knowledge sources simultaneously. Returns aggregated results from Wikipedia, Wikidata, academic databases, and more.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results per source (default: 3)" },
        sources: { 
          type: "array", 
          items: { type: "string" },
          description: "Specific sources to query (optional, defaults to all)" 
        },
      },
      required: ["query"],
    },
  },
  {
    name: "search_wikipedia",
    description: "Search Wikipedia via MediaWiki API and Wikimedia REST",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "get_wikipedia_summary",
    description: "Get Wikipedia article summary",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string", description: "Article title" },
      },
      required: ["title"],
    },
  },
  {
    name: "search_wikidata",
    description: "Search Wikidata entities and get structured facts",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Entity search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "query_wikidata_sparql",
    description: "Execute SPARQL query against Wikidata",
    inputSchema: {
      type: "object",
      properties: {
        sparql: { type: "string", description: "SPARQL query" },
      },
      required: ["sparql"],
    },
  },
  {
    name: "search_dbpedia",
    description: "Search DBpedia for structured Wikipedia data",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "search_academic",
    description: "Search academic literature across OpenAlex, Crossref, and Europe PMC",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results per source (default: 3)" },
      },
      required: ["query"],
    },
  },
  {
    name: "search_openalex",
    description: "Search OpenAlex for academic works, authors, institutions",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "search_crossref",
    description: "Search Crossref for DOI metadata and publications",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "get_doi_metadata",
    description: "Get metadata for a specific DOI",
    inputSchema: {
      type: "object",
      properties: {
        doi: { type: "string", description: "DOI (e.g., 10.1038/nature12373)" },
      },
      required: ["doi"],
    },
  },
  {
    name: "search_pubmed",
    description: "Search PubMed/NCBI for biomedical literature",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "search_europepmc",
    description: "Search Europe PMC for life sciences literature",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "search_clinical_trials",
    description: "Search ClinicalTrials.gov for clinical studies",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query (condition, drug, etc.)" },
        limit: { type: "number", description: "Max results (default: 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "get_citations",
    description: "Get citation data from OpenCitations for a DOI",
    inputSchema: {
      type: "object",
      properties: {
        doi: { type: "string", description: "DOI to get citations for" },
      },
      required: ["doi"],
    },
  },
  {
    name: "search_gdelt",
    description: "Search GDELT for global news and events",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        mode: { 
          type: "string", 
          enum: ["artlist", "timelinevol", "tonechart"],
          description: "GDELT mode (default: artlist)" 
        },
        limit: { type: "number", description: "Max results (default: 10)" },
      },
      required: ["query"],
    },
  },
  {
    name: "get_world_bank_indicator",
    description: "Get World Bank economic indicator data",
    inputSchema: {
      type: "object",
      properties: {
        indicator: { type: "string", description: "Indicator code (e.g., NY.GDP.MKTP.CD)" },
        country: { type: "string", description: "Country code (e.g., US, DE) or 'all'" },
        year: { type: "string", description: "Year or range (e.g., 2020 or 2015:2020)" },
      },
      required: ["indicator"],
    },
  },
  {
    name: "search_vulnerabilities",
    description: "Search OSV for open source vulnerabilities",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Package name or CVE" },
        ecosystem: { 
          type: "string", 
          description: "Package ecosystem (npm, PyPI, Go, etc.)" 
        },
      },
      required: ["query"],
    },
  },
  {
    name: "get_vulnerability",
    description: "Get details for a specific vulnerability ID",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Vulnerability ID (e.g., GHSA-xxx or CVE-xxx)" },
      },
      required: ["id"],
    },
  },
];

// Create MCP server
const server = new Server(
  {
    name: "ohi-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools,
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const result = await toolAggregator.callTool(name, args || {});
    return {
      content: [
        {
          type: "text",
          text: typeof result === "string" ? result : JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      content: [{ type: "text", text: `Error: ${message}` }],
      isError: true,
    };
  }
});

// Start server based on transport mode
async function main() {
  // Initialize all sources
  await sourceRegistry.initialize();
  
  const transportMode = process.env.TRANSPORT || "sse";

  if (transportMode === "stdio") {
    // STDIO transport for CLI usage
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("OHI MCP Server running on STDIO");
  } else {
    // SSE transport for HTTP usage
    const app = express();
    app.use(express.json());

    // Health check
    app.get("/health", (_req: Request, res: Response) => {
      res.json({ 
        status: "healthy", 
        sources: sourceRegistry.getAll().map(s => s.name),
        cache: cache.getStats(),
      });
    });

    // SSE endpoint
    const transports = new Map<string, SSEServerTransport>();

    app.get("/sse", async (req: Request, res: Response) => {
      const transport = new SSEServerTransport("/messages", res);
      const sessionId = crypto.randomUUID();
      transports.set(sessionId, transport);

      res.on("close", () => {
        transports.delete(sessionId);
      });

      await server.connect(transport);
    });

    app.post("/messages", async (req: Request, res: Response) => {
      // Find the transport for this session
      const sessionId = req.headers["x-session-id"] as string;
      const transport = transports.get(sessionId);
      
      if (transport) {
        await transport.handlePostMessage(req, res);
      } else {
        res.status(404).json({ error: "Session not found" });
      }
    });

    // Stats endpoint
    app.get("/stats", (_req: Request, res: Response) => {
      res.json({
        sources: sourceRegistry.getStats(),
        cache: cache.getStats(),
        rateLimiter: rateLimiter.getStats(),
      });
    });

    const sourceNames = sourceRegistry.getAll().map(s => s.name);
    app.listen(PORT, "0.0.0.0", () => {
      console.log(`OHI MCP Server running on http://0.0.0.0:${PORT}`);
      console.log(`Available sources: ${sourceNames.join(", ")}`);
    });
  }
}

main().catch(console.error);
