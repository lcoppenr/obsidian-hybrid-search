/**
 * Plain-text indexing (OBSIDIAN_TEXT_EXTENSIONS, OBSIDIAN_HTML_MODE, OBSIDIAN_TEXT_MAX_KB).
 */
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeEach, describe, it, vi, type MockInstance } from 'vitest';

vi.mock('chokidar', () => ({
  watch: vi.fn().mockReturnValue({
    add: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
  }),
}));

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-studio-text-vault-'));
const dbDir = mkdtempSync(path.join(tmpdir(), 'ohs-studio-text-db-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;
process.env.OBSIDIAN_DB_DIR = dbDir;

const { closeDb, getDb, getNoteByPath, initVecTable, openDb, resolveNotePath, wipeDatabaseFiles } =
  await import('../src/db.js');

const embedder = await import('../src/embedder.js');
vi.spyOn(embedder, 'embedDetailed').mockImplementation(async (texts: string[]) =>
  texts.map(() => ({ ok: true as const, embedding: new Float32Array([0.1, 0.2, 0.3, 0.4]) })),
);
vi.spyOn(embedder, 'embed').mockImplementation(async (texts: string[]) =>
  texts.map(() => new Float32Array([0.1, 0.2, 0.3, 0.4])),
);
vi.spyOn(embedder, 'getContextLength').mockResolvedValue(512);
vi.spyOn(embedder, 'getDocumentTokenPolicy').mockResolvedValue({
  limit: 508,
  count: (text) => Math.ceil(Array.from(text).length / 4),
});

const { htmlToText, indexVaultSync, resetIndexingState } = await import('../src/indexer.js');
const { search } = await import('../src/searcher.js');

let stderrSpy: MockInstance<typeof process.stderr.write>;

const SETTINGS = ['OBSIDIAN_TEXT_EXTENSIONS', 'OBSIDIAN_HTML_MODE', 'OBSIDIAN_TEXT_MAX_KB'];

const PS1 = [
  '#Requires -Version 5.1',
  '#region Helpers',
  '# Deploy the widget service to every node',
  'function Invoke-WidgetDeploy {',
  '  param([string]$Target)',
  '  Write-Output "deploying $Target"',
  '}',
  '#endregion',
].join('\n');

const BASH = [
  '#!/usr/bin/env bash',
  '# Rotate the backup snapshots',
  'if [[ -f /tmp/lock ]]; then exit 1; fi',
  'rotate_snapshots() { echo rotating; }',
].join('\n');

const YAML = ['---', 'service: backup', 'schedule: nightly', 'retention_days: 30'].join('\n');

const HTML = [
  '<!doctype html><html><head><title>Briefing &amp; Status</title>',
  '<style>.x { color: #ff0000; }</style></head>',
  '<body><h1>Morning briefing</h1><p>Server <b>ubuntuserver</b> healthy.</p>',
  '<script>function trackVisitor() { return 1; }</script>',
  '<!-- hidden comment --></body></html>',
].join('\n');

function write(rel: string, body: string): void {
  const full = path.join(vaultDir, rel);
  mkdirSync(path.dirname(full), { recursive: true });
  writeFileSync(full, body);
}

function notePaths(): string[] {
  return (getDb().prepare('SELECT path FROM notes ORDER BY path').all() as { path: string }[]).map(
    (r) => r.path,
  );
}

function tagsOf(p: string): string[] {
  return (
    getDb()
      .prepare(
        'SELECT t.tag FROM note_tags t JOIN notes n ON n.id = t.note_id WHERE n.path = ? ORDER BY t.tag',
      )
      .all(p) as { tag: string }[]
  ).map((r) => r.tag);
}

function linksFrom(p: string): string[] {
  return (
    getDb().prepare('SELECT to_path FROM links WHERE from_path = ? ORDER BY to_path').all(p) as {
      to_path: string;
    }[]
  ).map((r) => r.to_path);
}

async function reindex(): Promise<void> {
  await indexVaultSync(false, 'test');
}

beforeEach(async () => {
  for (const name of SETTINGS) delete process.env[name];
  resetIndexingState();
  closeDb();
  wipeDatabaseFiles();
  rmSync(vaultDir, { recursive: true, force: true });
  mkdirSync(vaultDir, { recursive: true });
  openDb();
  initVecTable(4);
  write('Note.md', '# Note\n\nSee [[Deploy.ps1]] and [[Other]].\n');
  write('Other.md', '# Other\n\nPlain note.\n');
  write('Scripts/Deploy.ps1', PS1);
  write('Scripts/rotate.sh', BASH);
  write('config/backup.yaml', YAML);
  write('site/index.html', HTML);
  write('data/settings.json', '{"theme": "dark", "fontSize": 14}');
  stderrSpy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
});

afterAll(() => {
  stderrSpy.mockRestore();
  closeDb();
  for (const name of SETTINGS) delete process.env[name];
  delete process.env.OBSIDIAN_DB_DIR;
  rmSync(vaultDir, { recursive: true, force: true });
  rmSync(dbDir, { recursive: true, force: true });
});

describe('OBSIDIAN_TEXT_EXTENSIONS', () => {
  it('empty (default): only Markdown is indexed', async () => {
    await reindex();
    assert.deepEqual(notePaths(), ['Note.md', 'Other.md']);
  });

  it('indexes the listed types as plain text', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1, .SH ,yaml,html,json';
    await reindex();
    assert.deepEqual(notePaths(), [
      'Note.md',
      'Other.md',
      'Scripts/Deploy.ps1',
      'Scripts/rotate.sh',
      'config/backup.yaml',
      'data/settings.json',
      'site/index.html',
    ]);
    const ps1 = getNoteByPath('Scripts/Deploy.ps1')!;
    assert.equal(ps1.title, 'Deploy.ps1');
    assert.equal(ps1.content, PS1);
  });

  it('bash [[ ]] is not a link and #Requires / #region are not tags', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,sh';
    await reindex();
    assert.deepEqual(linksFrom('Scripts/rotate.sh'), []);
    assert.deepEqual(tagsOf('Scripts/Deploy.ps1'), []);
    assert.deepEqual(tagsOf('Scripts/rotate.sh'), []);
  });

  it('# comment lines are not treated as headings', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1';
    await reindex();
    const rows = getDb()
      .prepare(
        "SELECT c.heading_path FROM chunks c JOIN notes n ON n.id = c.note_id WHERE n.path = 'Scripts/Deploy.ps1'",
      )
      .all() as { heading_path: string | null }[];
    assert.ok(rows.length > 0);
    assert.ok(rows.every((r) => r.heading_path === null));
  });

  it('YAML starting with --- keeps its content (no front-matter stripping)', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'yaml';
    await reindex();
    assert.equal(getNoteByPath('config/backup.yaml')!.content, YAML);
  });

  it('finds a function by exact name', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,sh';
    await reindex();
    const results = await search('Invoke-WidgetDeploy', { mode: 'fulltext', limit: 3 });
    assert.equal(results[0]?.path, 'Scripts/Deploy.ps1');
    const bash = await search('rotate_snapshots', { mode: 'fulltext', limit: 3 });
    assert.equal(bash[0]?.path, 'Scripts/rotate.sh');
  });

  it('file_ext front matter filters by type', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,sh,html';
    await reindex();
    const onlyPs1 = await search('', { mode: 'fulltext', frontmatter: 'file_ext:ps1', limit: 0 });
    assert.deepEqual(
      onlyPs1.map((r) => r.path),
      ['Scripts/Deploy.ps1'],
    );
    const noHtml = await search('', { mode: 'fulltext', frontmatter: '-file_ext:html', limit: 0 });
    const paths = noHtml.map((r) => r.path).sort((a, b) => a.localeCompare(b));
    assert.ok(!paths.includes('site/index.html'));
    assert.ok(paths.includes('Note.md'), 'Markdown notes have no file_ext and are not excluded');
    assert.ok(paths.includes('Scripts/rotate.sh'));
  });

  it('removing an extension purges those notes only; adding it back re-indexes them', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,json';
    await reindex();
    assert.ok(notePaths().includes('data/settings.json'));

    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1';
    await reindex();
    assert.deepEqual(notePaths(), ['Note.md', 'Other.md', 'Scripts/Deploy.ps1']);
    const logged = stderrSpy.mock.calls.some((c) =>
      /text extensions removed \(json\); purged 1 note/.test(String(c[0])),
    );
    assert.ok(logged, 'the purge is logged with its count');

    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,json';
    await reindex();
    assert.deepEqual(notePaths(), [
      'Note.md',
      'Other.md',
      'Scripts/Deploy.ps1',
      'data/settings.json',
    ]);
  });

  it('the extension purge is exempt from the scan-ratio guard', async () => {
    // 3 json files of 5 notes: removing json drops the scan well below 0.5.
    write('data/a.json', '{"a": 1}');
    write('data/b.json', '{"b": 2}');
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'json';
    await reindex();
    assert.equal(notePaths().length, 5);

    delete process.env.OBSIDIAN_TEXT_EXTENSIONS;
    await reindex();
    assert.deepEqual(notePaths(), ['Note.md', 'Other.md']);
  });

  it('[[Deploy.ps1]] resolves to the indexed script', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1';
    await reindex();
    assert.deepEqual(linksFrom('Note.md'), ['Other.md', 'Scripts/Deploy.ps1']);
  });

  it('read resolves a bare text file name', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1';
    await reindex();
    assert.deepEqual(resolveNotePath('Deploy.ps1'), {
      type: 'resolved',
      path: 'Scripts/Deploy.ps1',
    });
  });

  it('rejects md and malformed entries, naming the setting', () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,md';
    return import('../src/config.js').then(({ validateServiceConfig }) => {
      assert.throws(() => validateServiceConfig(), /OBSIDIAN_TEXT_EXTENSIONS/);
      process.env.OBSIDIAN_TEXT_EXTENSIONS = 'ps1,*.sh';
      assert.throws(() => validateServiceConfig(), /OBSIDIAN_TEXT_EXTENSIONS/);
    });
  });
});

describe('OBSIDIAN_HTML_MODE', () => {
  it('text (default): strips tags, script, style and comments; title from <title>', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'html';
    await reindex();
    const note = getNoteByPath('site/index.html')!;
    assert.equal(note.title, 'Briefing & Status');
    assert.match(note.content, /Morning briefing/);
    assert.match(note.content, /Server ubuntuserver healthy\./);
    assert.doesNotMatch(note.content, /trackVisitor|#ff0000|hidden comment|<b>/);
  });

  it('raw indexes the markup', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'html';
    process.env.OBSIDIAN_HTML_MODE = 'raw';
    await reindex();
    const note = getNoteByPath('site/index.html')!;
    assert.equal(note.content, HTML);
    assert.equal(note.title, 'Briefing & Status');
  });

  it('htmlToText decodes entities and keeps block breaks', () => {
    assert.equal(htmlToText('<p>a &lt; b</p><p>c</p>'), 'a < b\nc');
  });

  it('rejects other values, naming the setting', async () => {
    process.env.OBSIDIAN_HTML_MODE = 'markdown';
    const { validateServiceConfig } = await import('../src/config.js');
    assert.throws(() => validateServiceConfig(), /OBSIDIAN_HTML_MODE/);
  });
});

describe('OBSIDIAN_TEXT_MAX_KB', () => {
  it('skips and logs text files over the limit; Markdown is not limited', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'txt';
    process.env.OBSIDIAN_TEXT_MAX_KB = '1';
    write('big.txt', 'x'.repeat(2048));
    write('small.txt', 'small file');
    write('Big.md', '# Big\n\n' + 'word '.repeat(600));
    await reindex();
    const paths = notePaths();
    assert.ok(!paths.includes('big.txt'));
    assert.ok(paths.includes('small.txt'));
    assert.ok(paths.includes('Big.md'));
    const logged = stderrSpy.mock.calls.some((c) => String(c[0]).includes('skipped big.txt'));
    assert.ok(logged);
  });

  it('drops a file that grew past the limit', async () => {
    process.env.OBSIDIAN_TEXT_EXTENSIONS = 'txt';
    process.env.OBSIDIAN_TEXT_MAX_KB = '1';
    write('grow.txt', 'short');
    await reindex();
    assert.ok(notePaths().includes('grow.txt'));
    write('grow.txt', 'y'.repeat(4096));
    await reindex();
    assert.ok(!notePaths().includes('grow.txt'));
  });

  it('rejects values below 1, naming the setting', async () => {
    process.env.OBSIDIAN_TEXT_MAX_KB = '0';
    const { validateServiceConfig } = await import('../src/config.js');
    assert.throws(() => validateServiceConfig(), /OBSIDIAN_TEXT_MAX_KB/);
  });
});
