import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { timingSafeEqual } from 'node:crypto';
import { readFileSync } from 'node:fs';
import http, {
  type IncomingMessage,
  type Server as NodeHttpServer,
  type ServerResponse,
} from 'node:http';
import { config, validateServiceConfig } from './config.js';
import { closeDb } from './db.js';
import { createMcpRuntime, createMcpServer, startMcpBackgroundServices } from './mcp-runtime.js';
import { registerProcessHandlers } from './process-resilience.js';

export interface HttpMcpServerOptions {
  host: string;
  port: number;
  allowedHosts?: string[];
  allowAnyHost?: boolean;
  startBackgroundServices?: boolean;
}

export interface HttpMcpServerHandle {
  host: string;
  port: number;
  url: string;
  healthUrl: string;
  close(): Promise<void>;
}

export async function runHttpMcpServer(
  options: HttpMcpServerOptions,
): Promise<HttpMcpServerHandle> {
  validateServiceConfig();
  const bearerToken = loadBearerToken(config.mcpTokenFile);
  const runtime = await createMcpRuntime();
  let actualPort = options.port;

  const nodeServer = http.createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      writeJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
    });
  });

  async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const hostHeader = req.headers.host ?? `${options.host}:${actualPort}`;
    if (!isAllowedHostHeader(hostHeader, options.host, actualPort, options)) {
      writeJson(res, 403, { error: 'Invalid Host header' });
      return;
    }
    const url = new URL(req.url ?? '/', `http://${hostHeader}`);

    if (req.method === 'GET' && url.pathname === '/health') {
      writeJson(res, 200, {
        ok: true,
        name: 'obsidian-hybrid-search',
        version: runtime.version,
        transport: 'streamable-http',
        vaultPath: config.vaultPath,
      });
      return;
    }

    // /health stays open (liveness for the supervisor); everything else needs the token.
    if (bearerToken !== undefined && !hasValidBearer(req.headers.authorization, bearerToken)) {
      writeJson(res, 401, { error: 'unauthorized' }, { 'www-authenticate': 'Bearer' });
      return;
    }

    if (url.pathname !== '/mcp') {
      writeJson(res, 404, { error: 'not found' });
      return;
    }

    if (req.method !== 'POST') {
      // Stateless Streamable HTTP only answers POST: there is no SSE stream to
      // GET and no session to DELETE. RFC 9110 requires Allow on every 405.
      writeJson(res, 405, { error: 'method not allowed' }, { allow: 'POST' });
      return;
    }

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableDnsRebindingProtection: options.allowAnyHost !== true,
      allowedHosts: allowedHosts(options.host, actualPort, options.allowedHosts),
    });
    const mcpServer = createMcpServer(runtime);
    res.once('close', () => {
      void Promise.allSettled([transport.close(), mcpServer.close()]);
    });
    await mcpServer.connect(transport);
    await transport.handleRequest(req, res);
  }

  await listen(nodeServer, options.host, options.port);
  const address = nodeServer.address();
  if (!address || typeof address === 'string') {
    throw new Error('HTTP MCP server did not bind to a TCP address');
  }
  actualPort = address.port;

  if (options.startBackgroundServices) {
    startMcpBackgroundServices(runtime);
  }

  const url = `http://${options.host}:${actualPort}/mcp`;
  return {
    host: options.host,
    port: actualPort,
    url,
    healthUrl: `http://${options.host}:${actualPort}/health`,
    close: async () => {
      await closeNodeServer(nodeServer);
      closeDb();
    },
  };
}

export async function runHttpMcpServerCli(options: {
  host: string;
  port: number;
  allowedHosts?: string[];
  allowAnyHost?: boolean;
}): Promise<void> {
  registerProcessHandlers();
  const handle = await runHttpMcpServer({ ...options, startBackgroundServices: true });
  console.log(`[mcp-http] listening on ${handle.url}`);

  const shutdown = () => {
    void handle
      .close()
      .catch((err: unknown) => {
        console.error('[mcp-http] shutdown error:', err);
        process.exit(1);
      })
      .then(() => process.exit(0));
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

function listen(server: NodeHttpServer, host: string, port: number): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      server.off('error', reject);
      resolve();
    });
  });
}

function closeNodeServer(server: NodeHttpServer): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function writeJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): void {
  if (res.headersSent) return;
  res.writeHead(status, { 'content-type': 'application/json', ...extraHeaders });
  res.end(JSON.stringify(body));
}

function isAllowedHostHeader(
  hostHeader: string,
  host: string,
  port: number,
  options: Pick<HttpMcpServerOptions, 'allowedHosts' | 'allowAnyHost'>,
): boolean {
  if (options.allowAnyHost === true) return true;
  return allowedHosts(host, port, options.allowedHosts).includes(hostHeader.trim());
}

export function normalizeAllowedHosts(hosts: readonly string[] | undefined): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const host of hosts ?? []) {
    const value = host.trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    normalized.push(value);
  }
  return normalized;
}

function allowedHosts(
  host: string,
  port: number,
  extraHosts: readonly string[] | undefined,
): string[] {
  return normalizeAllowedHosts(
    [host, 'localhost', '127.0.0.1', ...normalizeAllowedHosts(extraHosts)].flatMap((allowedHost) =>
      expandAllowedHostForPort(allowedHost, port),
    ),
  );
}

function expandAllowedHostForPort(host: string, port: number): string[] {
  if (hasExplicitPort(host)) return [host];
  return [host, `${host}:${port}`];
}

function hasExplicitPort(host: string): boolean {
  if (host.startsWith('[')) return /\]:\d+$/.test(host);
  const colonMatches = host.match(/:/g);
  return colonMatches?.length === 1 && /:\d+$/.test(host);
}

/**
 * OBSIDIAN_MCP_TOKEN_FILE: read once at startup. A missing, unreadable or empty file
 * stops the server rather than silently serving without authentication.
 */
export function loadBearerToken(tokenFile: string | undefined): string | undefined {
  if (tokenFile === undefined) return undefined;
  let token: string;
  try {
    token = readFileSync(tokenFile, 'utf-8').trim();
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    throw new Error(`OBSIDIAN_MCP_TOKEN_FILE could not be read (${tokenFile}): ${reason}`);
  }
  if (!token) throw new Error(`OBSIDIAN_MCP_TOKEN_FILE is empty (${tokenFile})`);
  return token;
}

function hasValidBearer(header: string | undefined, token: string): boolean {
  if (!header) return false;
  const value = header.trim();
  if (value.slice(0, 7).toLowerCase() !== 'bearer ') return false;
  const given = Buffer.from(value.slice(7).trim());
  const expected = Buffer.from(token);
  return given.length === expected.length && timingSafeEqual(given, expected);
}
