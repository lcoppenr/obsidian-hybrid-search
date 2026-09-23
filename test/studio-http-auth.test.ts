/**
 * OBSIDIAN_MCP_TOKEN_FILE: bearer token on `serve` over HTTP.
 */
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, describe, it } from 'vitest';
import type { HttpMcpServerHandle } from '../src/mcp-http-server.js';

const { closeDb, initVecTable, openDb } = await import('../src/db.js');
const { runHttpMcpServer, loadBearerToken } = await import('../src/mcp-http-server.js');

let server: HttpMcpServerHandle | undefined;
const cleanup: string[] = [];

afterEach(async () => {
  await server?.close();
  server = undefined;
  for (const dir of cleanup.splice(0)) rmSync(dir, { recursive: true, force: true });
  delete process.env.OBSIDIAN_VAULT_PATH;
  delete process.env.OBSIDIAN_MCP_TOKEN_FILE;
  delete process.env.OBSIDIAN_MIN_SCAN_RATIO;
});

function createTempVault(): string {
  const dir = mkdtempSync(path.join(tmpdir(), 'ohs-http-auth-test-'));
  cleanup.push(dir);
  mkdirSync(path.join(dir, '.obsidian'), { recursive: true });
  writeFileSync(path.join(dir, 'alpha.md'), '# Alpha\n\nAlpha note content.\n');
  process.env.OBSIDIAN_VAULT_PATH = dir;
  openDb();
  initVecTable(4);
  closeDb();
  return dir;
}

function writeToken(token: string): string {
  const dir = mkdtempSync(path.join(tmpdir(), 'ohs-token-'));
  cleanup.push(dir);
  const file = path.join(dir, 'token');
  writeFileSync(file, `${token}\n`);
  return file;
}

function initialize(url: string, headers: Record<string, string> = {}): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'application/json, text/event-stream',
      ...headers,
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-06-18',
        capabilities: {},
        clientInfo: { name: 'vitest', version: '1.0.0' },
      },
    }),
  });
}

describe('OBSIDIAN_MCP_TOKEN_FILE', () => {
  it('unset (default): no authentication, as upstream', async () => {
    createTempVault();
    server = await runHttpMcpServer({ host: '127.0.0.1', port: 0 });
    assert.equal((await initialize(server.url)).status, 200);
  });

  it('set: 401 without a token, 401 with a wrong one, 200 with the right one', async () => {
    createTempVault();
    process.env.OBSIDIAN_MCP_TOKEN_FILE = writeToken('s3cret-token');
    server = await runHttpMcpServer({ host: '127.0.0.1', port: 0 });

    const none = await initialize(server.url);
    assert.equal(none.status, 401);
    assert.equal(none.headers.get('www-authenticate'), 'Bearer');

    assert.equal((await initialize(server.url, { authorization: 'Bearer wrong' })).status, 401);
    assert.equal((await initialize(server.url, { authorization: 's3cret-token' })).status, 401);
    assert.equal(
      (await initialize(server.url, { authorization: 'Bearer s3cret-token' })).status,
      200,
    );
  });

  it('set: /health stays open for liveness', async () => {
    createTempVault();
    process.env.OBSIDIAN_MCP_TOKEN_FILE = writeToken('s3cret-token');
    server = await runHttpMcpServer({ host: '127.0.0.1', port: 0 });
    const res = await fetch(server.healthUrl);
    assert.equal(res.status, 200);
  });

  it('set: unknown paths are 401, not 404, without the token', async () => {
    createTempVault();
    process.env.OBSIDIAN_MCP_TOKEN_FILE = writeToken('s3cret-token');
    server = await runHttpMcpServer({ host: '127.0.0.1', port: 0 });
    const res = await fetch(server.url.replace('/mcp', '/other'));
    assert.equal(res.status, 401);
  });

  it('a missing or empty token file stops startup, naming the setting', async () => {
    createTempVault();
    process.env.OBSIDIAN_MCP_TOKEN_FILE = '/nonexistent/ohs-token';
    await assert.rejects(
      runHttpMcpServer({ host: '127.0.0.1', port: 0 }),
      /OBSIDIAN_MCP_TOKEN_FILE/,
    );
    process.env.OBSIDIAN_MCP_TOKEN_FILE = writeToken('   ');
    await assert.rejects(
      runHttpMcpServer({ host: '127.0.0.1', port: 0 }),
      /OBSIDIAN_MCP_TOKEN_FILE/,
    );
  });

  it('an invalid service setting stops startup, naming the setting', async () => {
    createTempVault();
    process.env.OBSIDIAN_MIN_SCAN_RATIO = '2';
    await assert.rejects(
      runHttpMcpServer({ host: '127.0.0.1', port: 0 }),
      /OBSIDIAN_MIN_SCAN_RATIO/,
    );
  });

  it('loadBearerToken trims the trailing newline', () => {
    assert.equal(loadBearerToken(writeToken('abc')), 'abc');
    assert.equal(loadBearerToken(undefined), undefined);
  });
});
