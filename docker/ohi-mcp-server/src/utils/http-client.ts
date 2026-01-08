/**
 * HTTP Client
 * ===========
 * 
 * Shared HTTP client with connection pooling and retry logic.
 */

interface RequestOptions {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean>;
  body?: string | Record<string, unknown>;
  timeout?: number;
}

interface HttpResponse<T = unknown> {
  status: number;
  data: T;
  headers: Headers;
}

const DEFAULT_TIMEOUT = 30000;
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000;

export class HttpClient {
  private userAgent: string;

  constructor(userAgent: string = "OHI-MCP-Server/1.0") {
    this.userAgent = userAgent;
  }

  async request<T = unknown>(url: string, options: RequestOptions = {}): Promise<HttpResponse<T>> {
    const {
      method = "GET",
      headers = {},
      params,
      body,
      timeout = DEFAULT_TIMEOUT,
    } = options;

    // Build URL with params
    let fullUrl = url;
    if (params && Object.keys(params).length > 0) {
      const searchParams = new URLSearchParams();
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      }
      fullUrl += (url.includes("?") ? "&" : "?") + searchParams.toString();
    }

    const requestHeaders: Record<string, string> = {
      "User-Agent": this.userAgent,
      Accept: "application/json",
      ...headers,
    };

    let requestBody: string | undefined;
    if (body) {
      if (typeof body === "string") {
        requestBody = body;
      } else {
        requestBody = JSON.stringify(body);
        if (!requestHeaders["Content-Type"]) {
          requestHeaders["Content-Type"] = "application/json";
        }
      }
    }

    // Retry logic
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(fullUrl, {
          method,
          headers: requestHeaders,
          body: requestBody,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Handle non-2xx responses
        if (!response.ok && response.status !== 404) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const contentType = response.headers.get("content-type") || "";
        let data: T;

        if (contentType.includes("application/json")) {
          data = await response.json() as T;
        } else if (contentType.includes("application/sparql-results+json")) {
          data = await response.json() as T;
        } else {
          data = await response.text() as T;
        }

        return {
          status: response.status,
          data,
          headers: response.headers,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        // Don't retry on client errors
        if (lastError.message.includes("HTTP 4")) {
          throw lastError;
        }

        if (attempt < MAX_RETRIES) {
          await this.sleep(RETRY_DELAY * (attempt + 1));
        }
      }
    }

    throw lastError || new Error("Request failed");
  }

  async get<T = unknown>(url: string, options?: Omit<RequestOptions, "method" | "body">): Promise<HttpResponse<T>> {
    return this.request<T>(url, { ...options, method: "GET" });
  }

  async post<T = unknown>(url: string, body?: RequestOptions["body"], options?: Omit<RequestOptions, "method" | "body">): Promise<HttpResponse<T>> {
    return this.request<T>(url, { ...options, method: "POST", body });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Shared client instance
export const httpClient = new HttpClient(
  "OHI-MCP-Server/1.0 (https://github.com/open-hallucination-index)"
);
