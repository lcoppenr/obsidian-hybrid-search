import type { SearchOptions, SearchResult } from './searcher.js';

export type SearchFunction = (query: string, options?: SearchOptions) => Promise<SearchResult[]>;

interface StdioRequest {
  id?: string;
  query?: string;
  options?: SearchOptions;
}

export interface StdioResponse {
  id: string;
  results?: SearchResult[];
  error?: string;
}

/**
 * Process a single newline-delimited JSON request for the stdio IPC server.
 * Exported for unit testing — called by `ohs serve --stdio` in a loop.
 *
 * Protocol:
 *   Request:  {"id":"1","query":"zettelkasten","options":{...}}
 *   Response: {"id":"1","results":[...]}
 *   Error:    {"id":"1","error":"message"}
 */
export async function handleStdioLine(
  line: string,
  searchFn: SearchFunction,
  writeLine: (s: string) => void,
): Promise<void> {
  const trimmed = line.trim();
  if (!trimmed) return;

  let id = 'unknown';
  try {
    const req = JSON.parse(trimmed) as StdioRequest;
    id = req.id ?? 'unknown';

    if (req.query === undefined) {
      writeLine(JSON.stringify({ id, error: 'missing required field: query' }));
      return;
    }

    const results = await searchFn(req.query, req.options ?? {});
    writeLine(JSON.stringify({ id, results }));
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    writeLine(JSON.stringify({ id, error }));
  }
}
