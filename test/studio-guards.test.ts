/**
 * Unreadable-vault guards, scan ratio, scheduled rescan and status fields
 * (OBSIDIAN_UNREADABLE_VAULT, OBSIDIAN_UNREADABLE_SUBTREE, OBSIDIAN_MIN_SCAN_RATIO,
 * OBSIDIAN_RESCAN_MINUTES).
 *
 * The suite may run as root, where chmod cannot make a folder unreadable, so
 * readdirSync/opendirSync are wrapped to fail with EACCES for the paths in `failing`.
 */
import assert from 'node:assert/strict';
import * as realFs from 'node:fs';
import { mkdtempSync, rmSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, afterEach, beforeEach, describe, it, vi, type MockInstance } from 'vitest';

const failing = new Set<string>();

function eacces(p: unknown): Error {
  return Object.assign(new Error(`EACCES: permission denied, scandir '${String(p)}'`), {
    code: 'EACCES',
  });
}

vi.doMock('node:fs', async (importOriginal) => {
  const mod = await importOriginal<typeof import('node:fs')>();
  const readdirSync = ((p: realFs.PathLike, options?: unknown) => {
    if (failing.has(String(p))) throw eacces(p);
    return (mod.readdirSync as (p: realFs.PathLike, o?: unknown) => unknown)(p, options);
  }) as typeof mod.readdirSync;
  const opendirSync = ((p: realFs.PathLike, options?: realFs.OpenDirOptions) => {
    if (failing.has(String(p))) throw eacces(p);
    return mod.opendirSync(p, options);
  }) as typeof mod.opendirSync;
  return { ...mod, default: { ...mod, readdirSync, opendirSync }, readdirSync, opendirSync };
});

const failingReads = new Set<string>();

vi.doMock('node:fs/promises', async (importOriginal) => {
  const mod = await importOriginal<typeof import('node:fs/promises')>();
  const readFile = ((p: Parameters<typeof mod.readFile>[0], ...rest: unknown[]) => {
    const key = typeof p === 'string' ? p : p instanceof URL ? p.pathname : '';
    if (failingReads.has(key)) {
      return Promise.reject(
        Object.assign(new Error(`ETIMEDOUT: operation timed out, open '${key}'`), {
          code: 'ETIMEDOUT',
        }),
      );
    }
    return (mod.readFile as (...a: unknown[]) => Promise<unknown>)(p, ...rest);
  }) as typeof mod.readFile;
  return { ...mod, default: { ...mod, readFile }, readFile };
});

vi.mock('chokidar', () => ({
  watch: vi.fn().mockReturnValue({
    add: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
  }),
}));

vi.resetModules();

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-studio-guards-vault-'));
const dbDir = mkdtempSync(path.join(tmpdir(), 'ohs-studio-guards-db-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;
process.env.OBSIDIAN_DB_DIR = dbDir;

const { closeDb, getDb, initVecTable, openDb, wipeDatabaseFiles } = await import('../src/db.js');

const embedder = await import('../src/embedder.js');
vi.spyOn(embedder, 'embedDetailed').mockImplementation(async (texts: string[]) =>
  texts.map(() => ({ ok: true as const, embedding: new Float32Array([0.1, 0.2, 0.3, 0.4]) })),
);
vi.spyOn(embedder, 'getContextLength').mockResolvedValue(512);
vi.spyOn(embedder, 'getDocumentTokenPolicy').mockResolvedValue({
  limit: 508,
  count: (text) => Math.ceil(Array.from(text).length / 4),
});

const {
  getScanStatus,
  indexVaultSync,
  resetIndexingState,
  startBackgroundIndexing,
  startWatcher,
  stopRescanTimer,
} = await import('../src/indexer.js');
const { buildStatusPayload } = await import('../src/status-payload.js');

let stderrSpy: MockInstance<typeof process.stderr.write>;

const SETTINGS = [
  'OBSIDIAN_UNREADABLE_VAULT',
  'OBSIDIAN_UNREADABLE_SUBTREE',
  'OBSIDIAN_MIN_SCAN_RATIO',
  'OBSIDIAN_RESCAN_MINUTES',
] as const;

function write(rel: string, body: string): void {
  const full = path.join(vaultDir, rel);
  realFs.mkdirSync(path.dirname(full), { recursive: true });
  writeFileSync(full, body);
}

function notePaths(): string[] {
  return (getDb().prepare('SELECT path FROM notes ORDER BY path').all() as { path: string }[]).map(
    (r) => r.path,
  );
}

async function seedVault(): Promise<void> {
  write('a.md', '# A\n\nAlpha body text.\n');
  write('b.md', '# B\n\nBeta body text.\n');
  write('c.md', '# C\n\nGamma body text.\n');
  write('sub/d.md', '# D\n\nDelta body text.\n');
  write('sub/e.md', '# E\n\nEpsilon body text.\n');
  await indexVaultSync(false, 'seed');
}

async function watcherHandler(event: string): Promise<(p: string) => void> {
  const chokidar = await import('chokidar');
  const watchMock = chokidar.watch as unknown as ReturnType<typeof vi.fn>;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const instance = watchMock.mock.results.at(-1)?.value;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const calls = instance.on.mock.calls as unknown[][];
  return calls.filter((c) => c[0] === event).at(-1)![1] as (p: string) => void;
}

async function settle(ms = 50): Promise<void> {
  await new Promise((r) => setTimeout(r, ms));
}

beforeEach(async () => {
  failing.clear();
  failingReads.clear();
  for (const name of SETTINGS) delete process.env[name];
  resetIndexingState();
  closeDb();
  wipeDatabaseFiles();
  rmSync(vaultDir, { recursive: true, force: true });
  realFs.mkdirSync(vaultDir, { recursive: true });
  openDb();
  initVecTable(4);
  await seedVault();
  stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
});

afterEach(() => {
  stopRescanTimer();
  stderrSpy.mockRestore();
});

afterAll(() => {
  failing.clear();
  failingReads.clear();
  closeDb();
  for (const name of SETTINGS) delete process.env[name];
  delete process.env.OBSIDIAN_DB_DIR;
  rmSync(vaultDir, { recursive: true, force: true });
  rmSync(dbDir, { recursive: true, force: true });
  vi.doUnmock('node:fs');
  vi.doUnmock('node:fs/promises');
  vi.resetModules();
});

describe('OBSIDIAN_UNREADABLE_VAULT', () => {
  it('keep (default): an unreadable vault root serves the last index', async () => {
    assert.equal(notePaths().length, 5);
    failing.add(vaultDir);

    await startBackgroundIndexing(512);

    assert.equal(notePaths().length, 5);
    assert.equal(getScanStatus().vault_reachable, false);
  });

  it('keep: a full reindex against an unreadable root fails and leaves the index alone', async () => {
    failing.add(vaultDir);
    await assert.rejects(indexVaultSync(false, 'x'), /not readable/);
    assert.equal(notePaths().length, 5);
  });

  it('purge with the ratio guard off: legacy behaviour empties the index', async () => {
    process.env.OBSIDIAN_UNREADABLE_VAULT = 'purge';
    process.env.OBSIDIAN_MIN_SCAN_RATIO = '0';
    failing.add(vaultDir);

    await startBackgroundIndexing(512);

    assert.deepEqual(notePaths(), []);
  });

  it('purge with the default ratio: the ratio guard still stops the wipe', async () => {
    process.env.OBSIDIAN_UNREADABLE_VAULT = 'purge';
    failing.add(vaultDir);

    await startBackgroundIndexing(512);

    assert.equal(notePaths().length, 5);
  });

  it('keep: watcher unlink events are ignored while the root is unreadable', async () => {
    startWatcher(512);
    await settle();
    const unlink = await watcherHandler('unlink');

    failing.add(vaultDir);
    unlink(path.join(vaultDir, 'a.md'));
    await settle();
    assert.ok(notePaths().includes('a.md'));

    failing.clear();
    unlink(path.join(vaultDir, 'a.md'));
    await settle();
    assert.ok(!notePaths().includes('a.md'));
  });

  it('purge: watcher unlink events delete even when the root is unreadable', async () => {
    process.env.OBSIDIAN_UNREADABLE_VAULT = 'purge';
    startWatcher(512);
    await settle();
    const unlink = await watcherHandler('unlink');

    failing.add(vaultDir);
    unlink(path.join(vaultDir, 'b.md'));
    await settle();
    assert.ok(!notePaths().includes('b.md'));
  });
});

describe('OBSIDIAN_UNREADABLE_SUBTREE', () => {
  it('keep (default): notes under an unreadable folder stay, the rest is cleaned', async () => {
    unlinkSync(path.join(vaultDir, 'a.md'));
    failing.add(path.join(vaultDir, 'sub'));

    await startBackgroundIndexing(512);

    assert.deepEqual(notePaths(), ['b.md', 'c.md', 'sub/d.md', 'sub/e.md']);
    assert.equal(getScanStatus().last_scan_failed_dirs, 1);
  });

  it('purge: notes under an unreadable folder are removed', async () => {
    process.env.OBSIDIAN_UNREADABLE_SUBTREE = 'purge';
    failing.add(path.join(vaultDir, 'sub'));

    await startBackgroundIndexing(512);

    assert.deepEqual(notePaths(), ['a.md', 'b.md', 'c.md']);
  });
});

describe('OBSIDIAN_MIN_SCAN_RATIO', () => {
  it('0.5 (default): a scan finding under half the indexed notes deletes nothing', async () => {
    for (const f of ['a.md', 'b.md', 'c.md']) unlinkSync(path.join(vaultDir, f));

    await startBackgroundIndexing(512);

    assert.equal(notePaths().length, 5);
    const logged = stderrSpy.mock.calls.some((c) =>
      String(c[0]).includes('OBSIDIAN_MIN_SCAN_RATIO'),
    );
    assert.ok(logged, 'the skip is logged');
  });

  it('a scan at or above the ratio deletes normally', async () => {
    unlinkSync(path.join(vaultDir, 'a.md'));
    await startBackgroundIndexing(512);
    assert.deepEqual(notePaths(), ['b.md', 'c.md', 'sub/d.md', 'sub/e.md']);
  });

  it('0 turns the guard off', async () => {
    process.env.OBSIDIAN_MIN_SCAN_RATIO = '0';
    for (const f of ['a.md', 'b.md', 'c.md']) unlinkSync(path.join(vaultDir, f));

    await startBackgroundIndexing(512);

    assert.deepEqual(notePaths(), ['sub/d.md', 'sub/e.md']);
  });

  it('1 skips deletion on any shortfall', async () => {
    process.env.OBSIDIAN_MIN_SCAN_RATIO = '1';
    unlinkSync(path.join(vaultDir, 'a.md'));
    await startBackgroundIndexing(512);
    assert.equal(notePaths().length, 5);
  });
});

describe('OBSIDIAN_RESCAN_MINUTES', () => {
  it('the scheduled re-check picks up an edit the watcher never reported', async () => {
    process.env.OBSIDIAN_RESCAN_MINUTES = '0.005'; // 300 ms
    startWatcher(512);
    await settle();

    // chokidar is mocked, so no watcher event fires for this edit or the new file.
    write('a.md', '# A\n\nAlpha body rewritten with zanzibar.\n');
    write('new.md', '# New\n\nAdded while nobody was watching.\n');
    const later = Date.now() + 1_000;
    realFs.utimesSync(path.join(vaultDir, 'a.md'), later / 1000, later / 1000);

    const deadline = Date.now() + 5_000;
    let content = '';
    while (Date.now() < deadline) {
      await settle(100);
      const row = getDb().prepare("SELECT content FROM notes WHERE path = 'a.md'").get() as {
        content: string;
      };
      content = row.content;
      if (content.includes('zanzibar') && notePaths().includes('new.md')) break;
    }
    assert.match(content, /zanzibar/);
    assert.ok(notePaths().includes('new.md'));
    assert.ok(getScanStatus().last_scan_ok_at !== null);
  });

  it('0 (default) schedules nothing', async () => {
    startWatcher(512);
    await settle();
    write('a.md', '# A\n\nChanged but never rescanned: quokka.\n');
    const later = Date.now() + 1_000;
    realFs.utimesSync(path.join(vaultDir, 'a.md'), later / 1000, later / 1000);
    await settle(600);
    const row = getDb().prepare("SELECT content FROM notes WHERE path = 'a.md'").get() as {
      content: string;
    };
    assert.doesNotMatch(row.content, /quokka/);
  });
});

describe('status', () => {
  it('reports reachability, the last good scan and every effective setting', async () => {
    process.env.OBSIDIAN_RESCAN_MINUTES = '15';
    await startBackgroundIndexing(512);
    const status = buildStatusPayload({ contextLength: 512, version: 'test' });

    assert.equal(status.vault_reachable, true);
    assert.equal(typeof status.last_scan_ok_at, 'string');
    assert.equal(status.last_scan_notes, 5);
    assert.equal(status.unreadable_vault, 'keep');
    assert.equal(status.unreadable_subtree, 'keep');
    assert.equal(status.min_scan_ratio, 0.5);
    assert.equal(status.rescan_minutes, 15);
    assert.equal(status.mcp_token_file, null);
    assert.deepEqual(status.text_extensions, []);
    assert.equal(status.html_mode, 'text');
    assert.equal(status.text_max_kb, 512);

    failing.add(vaultDir);
    const down = buildStatusPayload({ contextLength: 512, version: 'test' });
    assert.equal(down.vault_reachable, false);
    assert.equal(down.last_scan_ok_at, status.last_scan_ok_at, 'last good scan is kept');
  });
});

describe('background indexing failures', () => {
  it('logs each failed file and a summary, and reports them in status', async () => {
    write('cloud-a.md', '# A2\n\nNot downloaded.\n');
    write('cloud-b.md', '# B2\n\nNot downloaded either.\n');
    failingReads.add(path.join(vaultDir, 'cloud-a.md'));
    failingReads.add(path.join(vaultDir, 'cloud-b.md'));

    await startBackgroundIndexing(512);

    const lines = stderrSpy.mock.calls.map((c) => String(c[0]));
    assert.ok(lines.some((l) => /^\[indexer\] failed cloud-a\.md: ETIMEDOUT/.test(l)));
    assert.ok(lines.some((l) => /^\[indexer\] failed cloud-b\.md: ETIMEDOUT/.test(l)));
    assert.ok(
      lines.some((l) => /Indexing complete in .* — 0 indexed, 5 skipped, 2 errors\n$/.test(l)),
    );

    const status = getScanStatus();
    assert.equal(status.last_index_pass_errors, 2);
    assert.deepEqual(
      status.last_index_pass_error_samples.map((e) => e.path).sort((a, b) => a.localeCompare(b)),
      ['cloud-a.md', 'cloud-b.md'],
    );
    assert.equal(typeof status.last_index_pass_at, 'string');
  });

  it('caps per-file lines at 10 per pass and counts the rest', async () => {
    for (let i = 0; i < 13; i++) {
      write(`cloud-${i}.md`, `# C${i}\n\nNot downloaded.\n`);
      failingReads.add(path.join(vaultDir, `cloud-${i}.md`));
    }

    await startBackgroundIndexing(512);

    const lines = stderrSpy.mock.calls.map((c) => String(c[0]));
    assert.equal(lines.filter((l) => l.startsWith('[indexer] failed ')).length, 10);
    assert.ok(lines.some((l) => l.includes('13 errors (3 more not listed)')));
    assert.equal(getScanStatus().last_index_pass_errors, 13);
    assert.equal(getScanStatus().last_index_pass_error_samples.length, 5);
  });

  it('a clean pass reports zero errors', async () => {
    await startBackgroundIndexing(512);
    assert.equal(getScanStatus().last_index_pass_errors, 0);
    assert.deepEqual(getScanStatus().last_index_pass_error_samples, []);
  });
});
