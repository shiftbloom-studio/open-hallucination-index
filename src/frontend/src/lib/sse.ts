import { OhiError } from "./ohi-client";
import type { SseEvent, SseEventName, VerifyRequest, ApiErrorBody } from "./ohi-types";
import { SSE_EVENT_NAMES } from "./ohi-types";

const EVENT_NAME_SET: ReadonlySet<SseEventName> = new Set(SSE_EVENT_NAMES);

export interface StreamHandlers {
  onEvent: (event: SseEvent) => void;
  onError: (err: unknown) => void;
  onComplete: () => void;
}

/**
 * Parse a single SSE frame (lines separated by \n, frame boundary is
 * caller-controlled). Returns null on unknown event name or malformed JSON.
 */
export function parseFrame(raw: string): SseEvent | null {
  let event: string | null = null;
  const dataLines: string[] = [];

  for (const rawLine of raw.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line.startsWith(":") || line === "") continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (!event || dataLines.length === 0) return null;
  if (!EVENT_NAME_SET.has(event as SseEventName)) return null;

  const joined = dataLines.join("\n");
  try {
    const parsed = JSON.parse(joined) as unknown;
    return { event, data: parsed } as SseEvent;
  } catch {
    return null;
  }
}

/**
 * POST-SSE client for /verify/stream. Sends the request body, reads the
 * response body as a stream, and emits parsed SseEvents to handlers.
 */
export async function streamVerify(
  body: VerifyRequest,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    handlers.onError(new Error("NEXT_PUBLIC_API_BASE is not set"));
    return;
  }

  let res: Response;
  try {
    res = await fetch(`${base.replace(/\/$/, "")}/verify/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    handlers.onError(err);
    return;
  }

  if (!res.ok || !res.body) {
    const retryAfter = Number(res.headers.get("Retry-After") ?? "") || undefined;
    let errBody: ApiErrorBody;
    try {
      errBody = (await res.json()) as ApiErrorBody;
    } catch {
      errBody = { detail: res.statusText };
    }
    handlers.onError(new OhiError(res.status, errBody, retryAfter));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Frames are terminated by \n\n (tolerating \r\n\r\n).
      // Scan for either; split on the first match each iteration.
      let boundary: number;
      while (true) {
        const lf = buf.indexOf("\n\n");
        const crlf = buf.indexOf("\r\n\r\n");
        if (lf === -1 && crlf === -1) {
          boundary = -1;
          break;
        }
        if (crlf !== -1 && (lf === -1 || crlf < lf)) {
          boundary = crlf;
          const frame = buf.slice(0, boundary);
          buf = buf.slice(boundary + 4);
          const evt = parseFrame(frame);
          if (evt) handlers.onEvent(evt);
        } else {
          boundary = lf;
          const frame = buf.slice(0, boundary);
          buf = buf.slice(boundary + 2);
          const evt = parseFrame(frame);
          if (evt) handlers.onEvent(evt);
        }
      }
    }
    // Flush final frame if stream ended without trailing blank line
    if (buf.trim().length > 0) {
      const evt = parseFrame(buf);
      if (evt) handlers.onEvent(evt);
    }
    handlers.onComplete();
  } catch (err) {
    handlers.onError(err);
  }
}
