import matter from 'gray-matter';
import { createHash } from 'node:crypto';
import { opendirSync, readdirSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  createDocumentTextProjector,
  embedChunksWithRecovery,
  type EmbeddedChunk,
} from './chunk-embedding.js';
import {
  chunkNote,
  createChunkBoundaryIndex,
  refineChunksToFit,
  slidingWindow,
} from './chunker.js';
import { config } from './config.js';
import {
  deleteNote,
  getDb,
  getNoteMeta,
  getPathsToRemoveForIgnoreChange,
  getStoredEmbeddingDim,
  initVecTable,
  isLikelyDatabaseCorruption,
  openDb,
  updateLastIndexed,
  upsertLinks,
  upsertMarkdownLinks,
  upsertNote,
  upsertNoteUrls,
  wipeDatabaseSidecars,
} from './db.js';
import { embedDetailed, getContextLength, getDocumentTokenPolicy } from './embedder.js';
import { createIgnorePolicy, type IgnorePolicy } from './ignore.js';
import { extractMarkdownReferences, resolveMarkdownNoteLinks } from './markdown-references.js';
import { bumpIndexVersion } from './searcher.js';

export interface IndexResult {
  indexed: number;
  skipped: number;
  errors: Array<{ path: string; error: string }>;
}

function findDatabaseCorruptionError(
  errors: ReadonlyArray<{ path: string; error: string }>,
): { path: string; error: string } | undefined {
  return errors.find((error) => isLikelyDatabaseCorruption(error.error));
}

function recoverDatabaseSidecarsForIndexing(): void {
  wipeDatabaseSidecars();
  openDb();
  const embeddingDim = getStoredEmbeddingDim();
  if (embeddingDim !== null) {
    initVecTable(embeddingDim);
  }
}

async function runWithDatabaseRecovery<T>(
  label: string,
  operation: () => T | Promise<T>,
  recoverDatabase: (() => void) | undefined,
): Promise<T> {
  try {
    return await operation();
  } catch (err) {
    if (!recoverDatabase || !isLikelyDatabaseCorruption(err)) throw err;
    process.stderr.write(
      `[auto-heal] SQLite index corruption detected during ${label}; removing WAL/SHM sidecars and retrying once.\n`,
    );
    recoverDatabase();
    return operation();
  }
}

async function indexBatch(
  files: readonly string[],
  contextLength: number,
  force: boolean,
): Promise<IndexResult> {
  const result: IndexResult = { indexed: 0, skipped: 0, errors: [] };
  const policy = createIgnorePolicy();
  await Promise.all(
    files.map(async (f) => {
      const status = await indexFile(f, contextLength, force, policy);
      if (status === 'indexed') result.indexed++;
      else if (status === 'skipped') result.skipped++;
      else {
        result.errors.push({
          path: f,
          error: typeof status === 'object' ? status.error : 'indexing failed',
        });
      }
    }),
  );
  return result;
}

async function indexBatchWithRecovery(
  batch: readonly string[],
  contextLength: number,
  force: boolean,
  recoverDatabase: (() => void) | undefined,
): Promise<IndexResult> {
  const batchResult = await indexBatch(batch, contextLength, force);
  const corruption = findDatabaseCorruptionError(batchResult.errors);
  if (!corruption) return batchResult;

  if (!recoverDatabase) {
    throw new Error(`${corruption.path}: ${corruption.error}`);
  }

  process.stderr.write(
    `[auto-heal] SQLite index corruption detected; removing WAL/SHM sidecars and retrying ${batch.length} file${batch.length > 1 ? 's' : ''} from the affected batch.\n`,
  );
  recoverDatabase();

  const retryResult = await indexBatch(batch, contextLength, true);
  const retryCorruption = findDatabaseCorruptionError(retryResult.errors);
  if (retryCorruption) {
    throw new Error(`${retryCorruption.path}: ${retryCorruption.error}`);
  }

  return retryResult;
}

export async function withIndexingDbLock<T>(operation: () => T | Promise<T>): Promise<T> {
  const run = _indexingDbLock.then(
    () => Promise.resolve(operation()),
    () => Promise.resolve(operation()),
  );
  _indexingDbLock = run.then(
    () => {},
    () => {},
  );
  return run;
}

function toVaultRelativePath(fullPath: string): string {
  return path.relative(config.vaultPath, fullPath).split(path.sep).join('/').normalize('NFD');
}

function getExistingNotePathSet(): Set<string> {
  const rows = getDb().prepare('SELECT path FROM notes').all() as { path: string }[];
  return new Set(rows.map((row) => row.path));
}

function resolveMarkdownReferencesForNote(
  fromPath: string,
  content: string,
  existingPaths = getExistingNotePathSet(),
): { links: string[]; urls: string[] } {
  const references = extractMarkdownReferences(content);
  const links = resolveMarkdownNoteLinks(
    fromPath,
    references.localDestinations.map((link) => link.destination),
    existingPaths,
  );
  return { links, urls: references.urls };
}

export function isMarkdownPath(p: string): boolean {
  return p.endsWith('.md');
}

/** Lowercase extension without the dot; '' when there is none. */
export function fileExtension(p: string): string {
  return path.extname(p).slice(1).toLowerCase();
}

/** True for a file name the indexer picks up: Markdown, or one of the configured text types. */
export function isIndexableFileName(
  name: string,
  textExtensions: ReadonlySet<string> = new Set(config.textExtensions),
): boolean {
  if (name.endsWith('.md')) return true;
  const ext = fileExtension(name);
  return ext !== '' && textExtensions.has(ext);
}

export interface VaultScan {
  /** Absolute paths of indexable, non-ignored files. */
  files: string[];
  /** False when the vault root itself could not be listed. */
  rootReadable: boolean;
  /** Vault-relative folders (trailing '/') that failed to list, excluding the root. */
  failedDirs: string[];
}

function* walkDir(
  dir: string,
  policy: IgnorePolicy,
  failedDirs: string[],
  textExtensions: ReadonlySet<string>,
): Generator<string> {
  let entries: { name: string; isDirectory(): boolean; isFile(): boolean }[];
  try {
    entries = readdirSync(dir, {
      withFileTypes: true,
      encoding: 'utf-8',
    });
  } catch {
    failedDirs.push(dir);
    return;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const rel = toVaultRelativePath(full);
      if (!policy.isIgnored(rel + '/')) {
        yield* walkDir(full, policy, failedDirs, textExtensions);
      }
    } else if (entry.isFile() && isIndexableFileName(entry.name, textExtensions)) {
      yield full;
    }
  }
}

export function scanVaultDetailed(): VaultScan {
  const files: string[] = [];
  const failed: string[] = [];
  const policy = createIgnorePolicy();
  const textExtensions = new Set(config.textExtensions);
  for (const fullPath of walkDir(config.vaultPath, policy, failed, textExtensions)) {
    const rel = toVaultRelativePath(fullPath);
    if (!policy.isIgnored(rel)) {
      files.push(fullPath);
    }
  }
  const rootReadable = !failed.includes(config.vaultPath);
  const failedDirs = failed
    .filter((dir) => dir !== config.vaultPath)
    .map((dir) => toVaultRelativePath(dir) + '/');
  return { files, rootReadable, failedDirs };
}

export function scanVault(): string[] {
  return scanVaultDetailed().files;
}

/** Whether the vault root can be listed right now. */
export function isVaultRootReadable(): boolean {
  try {
    const dir = opendirSync(config.vaultPath);
    try {
      dir.readSync();
    } finally {
      dir.closeSync();
    }
    return true;
  } catch {
    return false;
  }
}

interface ScanRecord {
  lastScanOkAt: string | null;
  lastScanNotes: number | null;
  lastScanFailedDirs: number | null;
}

let _lastScan: ScanRecord = { lastScanOkAt: null, lastScanNotes: null, lastScanFailedDirs: null };

function recordScan(scan: VaultScan): void {
  if (!scan.rootReadable) return;
  _lastScan = {
    lastScanOkAt: new Date().toISOString(),
    lastScanNotes: scan.files.length,
    lastScanFailedDirs: scan.failedDirs.length,
  };
}

/** Vault reachability and the last scan that could read the vault root. */
export function getScanStatus(): {
  vault_reachable: boolean;
  last_scan_ok_at: string | null;
  last_scan_notes: number | null;
  last_scan_failed_dirs: number | null;
  last_index_pass_at: string | null;
  last_index_pass_errors: number;
  last_index_pass_error_samples: Array<{ path: string; error: string }>;
} {
  return {
    vault_reachable: isVaultRootReadable(),
    last_scan_ok_at: _lastScan.lastScanOkAt,
    last_scan_notes: _lastScan.lastScanNotes,
    last_scan_failed_dirs: _lastScan.lastScanFailedDirs,
    last_index_pass_at: _lastPass.at,
    last_index_pass_errors: _lastPass.errors,
    last_index_pass_error_samples: _lastPass.errorSamples,
  };
}

const TEXT_EXTENSIONS_KEY = 'text_extensions';

/**
 * Compare the configured text-extension list with the one the index was built with.
 * Notes whose extension was removed are purged here, deliberately and with a log line;
 * this purge is exempt from OBSIDIAN_MIN_SCAN_RATIO. Added extensions need nothing:
 * the next scan picks the files up. Returns the number of notes purged.
 */
export function applyTextExtensionChange(): number {
  const db = getDb();
  const current = config.textExtensions;
  const currentJson = JSON.stringify(current);
  const row = db.prepare('SELECT value FROM settings WHERE key = ?').get(TEXT_EXTENSIONS_KEY) as
    { value: string } | undefined;
  if (row?.value === currentJson) return 0;

  let removed: string[] = [];
  if (row) {
    try {
      const previous = JSON.parse(row.value) as unknown;
      if (Array.isArray(previous)) {
        removed = previous.map(String).filter((ext) => !current.includes(ext));
      }
    } catch {
      /* malformed setting: treat as first run */
    }
  }

  let purged = 0;
  if (removed.length > 0) {
    const removedSet = new Set(removed);
    const paths = (db.prepare('SELECT path FROM notes').all() as { path: string }[]).map(
      (r) => r.path,
    );
    for (const p of paths) {
      if (!isMarkdownPath(p) && removedSet.has(fileExtension(p))) {
        deleteNote(p);
        purged++;
      }
    }
  }

  db.prepare('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)').run(
    TEXT_EXTENSIONS_KEY,
    currentJson,
  );
  if (removed.length > 0) {
    process.stderr.write(
      `[indexer] text extensions removed (${removed.join(', ')}); purged ${purged} note${purged === 1 ? '' : 's'}\n`,
    );
  }
  if (purged > 0) bumpIndexVersion();
  return purged;
}

/**
 * Stale-note cleanup behind the unreadable-vault guards. Every scan-then-cleanup
 * path goes through here so a vault that cannot be read serves the last index
 * instead of emptying it.
 */
async function cleanupAfterScan(
  scan: VaultScan,
  label: string,
  recoverDatabase: (() => void) | undefined,
): Promise<'cleaned' | 'vault-unreadable'> {
  recordScan(scan);
  await runWithDatabaseRecovery(
    `${label} (text extensions)`,
    () => applyTextExtensionChange(),
    recoverDatabase,
  );

  if (!scan.rootReadable && config.unreadableVault === 'keep') {
    process.stderr.write(
      `[indexer] vault root is not readable (${config.vaultPath}); keeping the existing index, cleanup skipped\n`,
    );
    return 'vault-unreadable';
  }

  const keepSubtrees = config.unreadableSubtree === 'keep';
  if (scan.failedDirs.length > 0) {
    const shown = scan.failedDirs.slice(0, 5).join(', ');
    const more = scan.failedDirs.length > 5 ? ` and ${scan.failedDirs.length - 5} more` : '';
    process.stderr.write(
      `[indexer] ${scan.failedDirs.length} folder${scan.failedDirs.length === 1 ? '' : 's'} not readable (${shown}${more}); ${keepSubtrees ? 'keeping' : 'purging'} their notes\n`,
    );
  }

  const fsPaths = new Set(scan.files.map(toVaultRelativePath));
  await runWithDatabaseRecovery(
    label,
    () =>
      cleanupStaleNotes(fsPaths, {
        protectedPrefixes: keepSubtrees ? scan.failedDirs : [],
        minScanRatio: config.minScanRatio,
      }),
    recoverDatabase,
  );
  return 'cleaned';
}

export async function indexFile(
  fullPath: string,
  contextLength?: number,
  force = false,
  policy = createIgnorePolicy(),
): Promise<'indexed' | 'skipped' | { error: string }> {
  try {
    const stat = statSync(fullPath);
    const mtime = stat.mtimeMs;

    const relPath = toVaultRelativePath(fullPath);
    if (policy.isIgnored(relPath)) return 'skipped';
    const existing = force ? undefined : getNoteMeta(relPath);

    // Fast skip: mtime unchanged
    if (existing && existing.mtime === mtime) return 'skipped';

    const isText = !isMarkdownPath(relPath);
    if (isText && stat.size > config.textMaxKb * 1024) {
      process.stderr.write(
        `[indexer] skipped ${relPath}: ${Math.ceil(stat.size / 1024)} KB exceeds OBSIDIAN_TEXT_MAX_KB=${config.textMaxKb}\n`,
      );
      // A file that grew past the limit no longer qualifies; drop its stale entry.
      if (getNoteMeta(relPath)) {
        deleteNote(relPath);
        bumpIndexVersion();
      }
      return 'skipped';
    }

    const raw = await readFile(fullPath, 'utf-8');
    const hash = createHash('md5').update(raw).digest('hex');

    // Slow skip: content unchanged, only update mtime
    if (existing && existing.hash === hash) {
      getDb().prepare('UPDATE notes SET mtime = ? WHERE path = ?').run(mtime, relPath);
      return 'skipped';
    }

    if (isText) {
      await indexTextFile(relPath, fullPath, raw, mtime, hash, contextLength);
      return 'indexed';
    }

    const parsed = matter(raw);
    const { data: frontmatter, content } = parsed;
    const frontmatterRaw: string = parsed.matter;
    const title =
      frontmatter.title == null ? path.basename(fullPath, '.md') : String(frontmatter.title);
    const frontmatterTags: string[] = Array.isArray(frontmatter.tags)
      ? frontmatter.tags.map(String)
      : typeof frontmatter.tags === 'string'
        ? frontmatter.tags.split(',').map((t: string) => t.trim())
        : [];
    const inlineTags = parseInlineTags(content);
    const tags = [...new Set([...frontmatterTags, ...inlineTags])];
    const aliases = parseAliasField(frontmatter.aliases as unknown);

    const ctxLen = contextLength ?? (await getContextLength());
    const semanticChunks = chunkNote(content, ctxLen).filter((c) => c.text.trim().length > 0);
    let embeddedChunks: EmbeddedChunk[] = [];
    // Notes with no body content (only frontmatter) still get indexed so that
    // tag/frontmatter filters and title search can find them.
    if (semanticChunks.length > 0) {
      const tokenPolicy = await getDocumentTokenPolicy();
      const boundaryIndex = createChunkBoundaryIndex(content);
      const project = createDocumentTextProjector(title, tokenPolicy);
      const chunks = refineChunksToFit(
        content,
        semanticChunks,
        tokenPolicy.limit,
        (body, headingChain) => tokenPolicy.count(project(body, headingChain)),
        config.chunkOverlap,
        boundaryIndex,
      );
      embeddedChunks = await embedChunksWithRecovery({
        source: content,
        chunks,
        boundaryIndex,
        project,
        embed: (texts) => embedDetailed(texts, 'document'),
      });
    }

    upsertNote({
      path: relPath,
      title,
      tags,
      aliases,
      content,
      frontmatter: frontmatter,
      mtime: stat.mtimeMs,
      hash,
      chunks: embeddedChunks.map(({ chunk, embedding }) => ({
        text: chunk.text,
        headingPath: chunk.headingChain.length > 0 ? chunk.headingChain.join(' > ') : null,
        embedding,
        charStart: chunk.charStart,
        charEnd: chunk.charEnd,
      })),
    });

    const resolvedLinks = resolveWikilinks(frontmatterRaw + '\n' + content, relPath);
    upsertLinks(relPath, resolvedLinks);
    const markdownReferences = resolveMarkdownReferencesForNote(relPath, content);
    upsertMarkdownLinks(relPath, markdownReferences.links);
    upsertNoteUrls(relPath, markdownReferences.urls);

    bumpIndexVersion();
    return 'indexed';
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (process.env.LOG_LEVEL === 'debug') {
      console.error('[indexer] error indexing', fullPath, err);
    }
    return { error: msg };
  }
}

/**
 * Plain-text branch (OBSIDIAN_TEXT_EXTENSIONS): scripts, HTML, config. None of the
 * Markdown parsing applies — no front matter, no tags, no wikilinks or Markdown links
 * (bash `[[ ]]`, `#Requires`, CSS colours and `# comment` lines would all be misread).
 * Chunked by sliding window only. A synthetic `file_ext` front-matter field lets
 * searches filter by type (`frontmatter: "file_ext:ps1"`, `"-file_ext:html"`).
 */
async function indexTextFile(
  relPath: string,
  fullPath: string,
  raw: string,
  mtime: number,
  hash: string,
  contextLength: number | undefined,
): Promise<void> {
  const ext = fileExtension(relPath);
  const isHtml = ext === 'html' || ext === 'htm';
  const content = isHtml && config.htmlMode === 'text' ? htmlToText(raw) : raw;
  const title = (isHtml ? extractHtmlTitle(raw) : undefined) ?? path.basename(fullPath);

  const ctxLen = contextLength ?? (await getContextLength());
  let embeddedChunks: EmbeddedChunk[] = [];
  if (content.trim().length > 0) {
    const windows = slidingWindow(content, ctxLen, config.chunkOverlap).filter(
      (c) => c.text.trim().length > 0,
    );
    if (windows.length > 0) {
      const tokenPolicy = await getDocumentTokenPolicy();
      const boundaryIndex = createChunkBoundaryIndex(content);
      const project = createDocumentTextProjector(title, tokenPolicy);
      const chunks = refineChunksToFit(
        content,
        windows,
        tokenPolicy.limit,
        (body, headingChain) => tokenPolicy.count(project(body, headingChain)),
        config.chunkOverlap,
        boundaryIndex,
      );
      embeddedChunks = await embedChunksWithRecovery({
        source: content,
        chunks,
        boundaryIndex,
        project,
        embed: (texts) => embedDetailed(texts, 'document'),
      });
    }
  }

  upsertNote({
    path: relPath,
    title,
    tags: [],
    aliases: [],
    content,
    frontmatter: { file_ext: ext },
    mtime,
    hash,
    chunks: embeddedChunks.map(({ chunk, embedding }) => ({
      text: chunk.text,
      headingPath: null,
      embedding,
      charStart: chunk.charStart,
      charEnd: chunk.charEnd,
    })),
  });
  upsertLinks(relPath, []);
  upsertMarkdownLinks(relPath, []);
  upsertNoteUrls(relPath, []);
  bumpIndexVersion();
}

const HTML_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
};

function decodeHtmlEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, entity: string) => {
    if (entity.startsWith('#x') || entity.startsWith('#X')) {
      const code = Number.parseInt(entity.slice(2), 16);
      return Number.isFinite(code) && code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
    if (entity.startsWith('#')) {
      const code = Number.parseInt(entity.slice(1), 10);
      return Number.isFinite(code) && code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
    return HTML_ENTITIES[entity.toLowerCase()] ?? match;
  });
}

/** HTML to searchable text: drops script/style/comments and tags, keeps block breaks. */
export function htmlToText(html: string): string {
  const withoutCode = html
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<script\b[\s\S]*?<\/script\s*>/gi, ' ')
    .replace(/<style\b[\s\S]*?<\/style\s*>/gi, ' ')
    .replace(/<head\b[\s\S]*?<\/head\s*>/gi, (head) => {
      // Keep the <title> text; the rest of <head> is metadata.
      const title = extractHtmlTitle(head);
      return title ? `${title}\n` : ' ';
    });
  const text = withoutCode
    .replace(/<(br|\/p|\/div|\/li|\/tr|\/h[1-6]|\/section|\/article|\/table)\b[^>]*>/gi, '\n')
    .replace(/<[^<>]*>/g, ' ');
  return decodeHtmlEntities(text)
    .split('\n')
    .map((line) => line.replace(/[ \t\f\v\r]+/g, ' ').trim())
    .filter((line, i, lines) => line !== '' || (i > 0 && lines[i - 1] !== ''))
    .join('\n')
    .trim();
}

export function extractHtmlTitle(html: string): string | undefined {
  const match = /<title\b[^>]*>([\s\S]*?)<\/title\s*>/i.exec(html);
  if (!match) return undefined;
  const title = decodeHtmlEntities(match[1]!).replace(/\s+/g, ' ').trim();
  return title || undefined;
}

export async function indexFileWithRecovery(
  fullPath: string,
  contextLength: number,
  force = false,
  recoverDatabase: () => void = recoverDatabaseSidecarsForIndexing,
): Promise<'indexed' | 'skipped' | { error: string }> {
  const policy = createIgnorePolicy();
  const status = await indexFile(fullPath, contextLength, force, policy);
  if (status === 'indexed' || status === 'skipped' || !isLikelyDatabaseCorruption(status.error)) {
    return status;
  }

  process.stderr.write(
    '[auto-heal] SQLite index corruption detected while indexing one file; removing WAL/SHM sidecars and retrying once.\n',
  );
  recoverDatabase();
  return indexFile(fullPath, contextLength, true, createIgnorePolicy());
}

/**
 * One-time migration: populate links from stored note content for all notes
 * that were indexed before the links feature was added. No API calls — just
 * wikilink parsing and DB writes.
 */
// eslint-disable-next-line @typescript-eslint/require-await
export async function populateMissingLinks(): Promise<void> {
  const db = getDb();
  const done = (
    db.prepare("SELECT value FROM settings WHERE key = 'links_v1'").get() as
      { value: string } | undefined
  )?.value;
  if (done) return;

  const notes = db
    .prepare(
      "SELECT path, content, frontmatter FROM notes WHERE content IS NOT NULL AND path LIKE '%.md'",
    )
    .all() as {
    path: string;
    content: string;
    frontmatter: string;
  }[];
  for (const note of notes) {
    const links = resolveWikilinks((note.frontmatter || '') + '\n' + note.content, note.path);
    upsertLinks(note.path, links);
  }

  db.prepare("INSERT OR REPLACE INTO settings(key, value) VALUES('links_v1', '1')").run();
}

/**
 * One-time migration: populate resolved Markdown file links and external URLs
 * from stored note content for all notes indexed before this feature existed.
 */
// eslint-disable-next-line @typescript-eslint/require-await
export async function populateMissingMarkdownReferences(): Promise<void> {
  const db = getDb();
  const done = (
    db.prepare("SELECT value FROM settings WHERE key = 'markdown_links_v1'").get() as
      { value: string } | undefined
  )?.value;
  if (done) return;

  const notes = db
    .prepare("SELECT path, content FROM notes WHERE content IS NOT NULL AND path LIKE '%.md'")
    .all() as {
    path: string;
    content: string;
  }[];
  const existingPaths = getExistingNotePathSet();
  const tx = db.transaction(() => {
    for (const note of notes) {
      const references = resolveMarkdownReferencesForNote(note.path, note.content, existingPaths);
      upsertMarkdownLinks(note.path, references.links);
      upsertNoteUrls(note.path, references.urls);
    }
    db.prepare(
      "INSERT OR REPLACE INTO settings(key, value) VALUES('markdown_links_v1', '1')",
    ).run();
    db.prepare(
      "INSERT OR REPLACE INTO settings(key, value) VALUES('db_version', CAST(COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = 'db_version'), 0) + 1 AS TEXT))",
    ).run();
  });
  tx();
}

/**
 * Re-resolve wikilinks for ALL indexed notes unconditionally.
 * Called after every full vault reindex so that notes whose targets
 * didn't exist at index time get their links backfilled.
 */
// eslint-disable-next-line @typescript-eslint/require-await
async function resolveAllLinks(): Promise<void> {
  const db = getDb();
  const notes = db
    .prepare(
      "SELECT path, content, frontmatter FROM notes WHERE content IS NOT NULL AND path LIKE '%.md'",
    )
    .all() as {
    path: string;
    content: string;
    frontmatter: string;
  }[];
  for (const note of notes) {
    const links = resolveWikilinks((note.frontmatter || '') + '\n' + note.content, note.path);
    upsertLinks(note.path, links);
  }
}

// eslint-disable-next-line @typescript-eslint/require-await
async function resolveAllMarkdownReferences(): Promise<void> {
  const db = getDb();
  const notes = db
    .prepare("SELECT path, content FROM notes WHERE content IS NOT NULL AND path LIKE '%.md'")
    .all() as {
    path: string;
    content: string;
  }[];
  const existingPaths = getExistingNotePathSet();
  for (const note of notes) {
    const references = resolveMarkdownReferencesForNote(note.path, note.content, existingPaths);
    upsertMarkdownLinks(note.path, references.links);
    upsertNoteUrls(note.path, references.urls);
  }
}

/**
 * Remove notes that no longer belong in the index:
 * - notes matching updated ignore patterns
 * - notes whose files were deleted from disk
 * Called on server startup and during full reindex.
 */
export function cleanupStaleNotes(
  fsPaths?: Set<string>,
  options: { protectedPrefixes?: readonly string[]; minScanRatio?: number } = {},
): void {
  let deleted = 0;

  // Newly ignored notes: file still exists on disk, keep their link entries
  // so backlinks from ignored notes remain visible in search results
  const policy = createIgnorePolicy();
  const pathsToRemove = getPathsToRemoveForIgnoreChange(
    config.ignorePatterns,
    policy.signature(),
    (p) => policy.isIgnored(p),
  );
  for (const p of pathsToRemove) {
    if (policy.isIgnored(p)) {
      deleteNote(p, true); // keepLinks=true
      deleted++;
    }
  }

  // Notes deleted from filesystem: remove everything including links (broken links)
  if (fsPaths) {
    const db = getDb();
    const dbPaths = (db.prepare('SELECT path FROM notes').all() as { path: string }[]).map(
      (r) => r.path,
    );
    // Scan-driven callers pass config.minScanRatio; a bare call keeps upstream behaviour.
    const ratio = options.minScanRatio ?? 0;
    const protectedPrefixes = options.protectedPrefixes ?? [];
    const isProtected = (p: string) => protectedPrefixes.some((prefix) => p.startsWith(prefix));
    const candidates = dbPaths.filter((p) => !fsPaths.has(p) && !isProtected(p));
    // Notes kept because their folder could not be read are not missing; only the
    // deletion candidates count against the scan.
    const accounted = dbPaths.length - candidates.length;
    if (ratio > 0 && candidates.length > 0 && accounted < ratio * dbPaths.length) {
      process.stderr.write(
        `[indexer] scan found ${accounted} of ${dbPaths.length} indexed notes (below OBSIDIAN_MIN_SCAN_RATIO=${ratio}); deletion pass skipped\n`,
      );
    } else {
      for (const dbPath of candidates) {
        deleteNote(dbPath); // keepLinks=false
        deleted++;
      }
    }
  }

  if (deleted > 0) bumpIndexVersion();
}

export function formatDuration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem > 0 ? `${m}m ${rem}s` : `${m}m`;
}

const PROGRESS_BAR_WIDTH = 20;
// Number of completed batches to wait before showing ETA (lets the rate stabilise)
const ETA_WARMUP_BATCHES = 3;

export function renderProgressLine(processed: number, total: number, etaStr: string): string {
  const pct = total > 0 ? processed / total : 1;
  const filled = Math.round(pct * PROGRESS_BAR_WIDTH);
  const bar = '█'.repeat(filled) + '░'.repeat(PROGRESS_BAR_WIDTH - filled);
  const pctLabel = `${Math.round(pct * 100)}%`.padStart(4);
  return `  ${bar}  ${pctLabel} (${processed}/${total} notes)${etaStr}`;
}

export async function indexVaultSync(
  force = false,
  header = 'Indexing vault...',
  options: { requireClean?: boolean; recoverDatabase?: () => void } = {},
): Promise<IndexResult> {
  const scan = scanVaultDetailed();
  const files = scan.files;
  const cleanup = await cleanupAfterScan(scan, 'stale-note cleanup', options.recoverDatabase);
  if (cleanup === 'vault-unreadable') {
    throw new Error(
      `Vault path is not readable: ${config.vaultPath}; the existing index was left unchanged.`,
    );
  }

  const contextLength = await getContextLength();
  const result: IndexResult = { indexed: 0, skipped: 0, errors: [] };

  if (files.length === 0) {
    await runWithDatabaseRecovery(
      'link resolution',
      () => resolveAllLinks(),
      options.recoverDatabase,
    );
    await runWithDatabaseRecovery(
      'markdown reference resolution',
      () => resolveAllMarkdownReferences(),
      options.recoverDatabase,
    );
    await runWithDatabaseRecovery(
      'freshness update',
      () => updateLastIndexed(),
      options.recoverDatabase,
    );
    return result;
  }

  const isTTY = process.stderr.isTTY === true;
  const logEvery = Math.max(config.batchSize, Math.floor(files.length / 10));

  process.stderr.write(`${header}\n`);
  if (isTTY) {
    // Print initial empty bar without newline — will be overwritten in-place
    process.stderr.write(renderProgressLine(0, files.length, ''));
  }
  const startTime = Date.now();

  for (let i = 0; i < files.length; i += config.batchSize) {
    const batch = files.slice(i, i + config.batchSize);
    const batchResult = await indexBatchWithRecovery(
      batch,
      contextLength,
      force,
      options.recoverDatabase,
    );
    result.indexed += batchResult.indexed;
    result.skipped += batchResult.skipped;
    result.errors.push(...batchResult.errors);

    const processed = Math.min(i + config.batchSize, files.length);
    const completedBatches = Math.floor(i / config.batchSize) + 1;
    const elapsedSec = (Date.now() - startTime) / 1000;
    const rate = elapsedSec > 0 ? processed / elapsedSec : 0;
    const remainingSec =
      completedBatches >= ETA_WARMUP_BATCHES && rate > 0 && processed < files.length
        ? (files.length - processed) / rate
        : 0;
    const etaStr = remainingSec > 5 ? ` — ${formatDuration(remainingSec)} remaining` : '';

    if (isTTY) {
      // \r\x1b[2K: return to line start and clear it, then redraw
      process.stderr.write(`\r\x1b[2K${renderProgressLine(processed, files.length, etaStr)}`);
    } else if (processed % logEvery < config.batchSize || processed >= files.length) {
      const pct = Math.round((processed / files.length) * 100);
      process.stderr.write(`${processed}/${files.length} (${pct}%)${etaStr}\n`);
    }
  }

  if (isTTY) {
    process.stderr.write('\n'); // finalise the progress bar line
  }
  const elapsed = formatDuration((Date.now() - startTime) / 1000);
  const summaryParts = [`${result.indexed} indexed`, `${result.skipped} skipped`];
  if (result.errors.length > 0) {
    summaryParts.push(`${result.errors.length} error${result.errors.length > 1 ? 's' : ''}`);
  }
  process.stderr.write(`Done in ${elapsed} — ${summaryParts.join(', ')}\n`);
  for (const e of result.errors) {
    process.stderr.write(`  ${e.path}: ${e.error}\n`);
  }
  if (options.requireClean === true && result.errors.length > 0) {
    throw new Error(
      `Full-vault reindex failed with ${result.errors.length} error${
        result.errors.length > 1 ? 's' : ''
      }; index freshness was not updated.`,
    );
  }

  await runWithDatabaseRecovery(
    'link resolution',
    () => resolveAllLinks(),
    options.recoverDatabase,
  );
  await runWithDatabaseRecovery(
    'markdown reference resolution',
    () => resolveAllMarkdownReferences(),
    options.recoverDatabase,
  );
  await runWithDatabaseRecovery(
    'freshness update',
    () => updateLastIndexed(),
    options.recoverDatabase,
  );
  return result;
}

const _indexQueue: string[] = [];
let _isIndexing = false;
let _totalExpected = 0;
let _processedCount = 0;
let _indexingDbLock: Promise<void> = Promise.resolve();

/** @internal Reset module-level queue state for test isolation. */
export function resetIndexingState(): void {
  _indexQueue.length = 0;
  _isIndexing = false;
  _totalExpected = 0;
  _processedCount = 0;
  _indexingDbLock = Promise.resolve();
  _lastScan = { lastScanOkAt: null, lastScanNotes: null, lastScanFailedDirs: null };
  _lastPass = emptyPassRecord();
}

/**
 * Returns the current background-indexing progress.
 * queued  — files still waiting in the queue (not yet processed)
 * total   — total files enqueued at the start of the current run
 * processed — files already processed in the current run
 * isRunning — whether a background indexing pass is active
 *
 * Used by the `status` tool/command to report correct `pending` counts
 * even before files have been written to the DB (S-19 fix).
 */
export function getIndexingStatus(): {
  queued: number;
  total: number;
  processed: number;
  isRunning: boolean;
} {
  return {
    queued: _indexQueue.length,
    total: _totalExpected,
    processed: _processedCount,
    isRunning: _isIndexing,
  };
}

/** Per-file error lines written to the log per indexing pass; the rest are counted. */
const MAX_ERROR_LOG_LINES = 10;
/** Failed files kept for `status`. */
const MAX_ERROR_SAMPLES = 5;

interface IndexPassRecord {
  at: string | null;
  indexed: number;
  skipped: number;
  errors: number;
  errorSamples: Array<{ path: string; error: string }>;
}

const emptyPassRecord = (): IndexPassRecord => ({
  at: null,
  indexed: 0,
  skipped: 0,
  errors: 0,
  errorSamples: [],
});

let _lastPass: IndexPassRecord = emptyPassRecord();

function recordPassResult(pass: IndexPassRecord, result: IndexResult): void {
  pass.indexed += result.indexed;
  pass.skipped += result.skipped;
  for (const failure of result.errors) {
    const rel = toVaultRelativePath(failure.path);
    if (pass.errors < MAX_ERROR_LOG_LINES) {
      process.stderr.write(`[indexer] failed ${rel}: ${failure.error}\n`);
    }
    if (pass.errorSamples.length < MAX_ERROR_SAMPLES) {
      pass.errorSamples.push({ path: rel, error: failure.error });
    }
    pass.errors++;
  }
}

async function processQueue(contextLength: number): Promise<void> {
  if (_isIndexing) return;
  await withIndexingDbLock(async () => {
    if (_isIndexing) return;
    _isIndexing = true;
    const total = _totalExpected;
    const startTime = Date.now();

    if (total > 0) {
      process.stderr.write(`Indexing vault...\n`);
    }

    const pass = emptyPassRecord();
    try {
      const logEvery = Math.max(config.batchSize, Math.floor(total / 10));
      while (_indexQueue.length > 0) {
        const batch = _indexQueue.splice(0, config.batchSize);
        const batchResult = await indexBatchWithRecovery(
          batch,
          contextLength,
          false,
          recoverDatabaseSidecarsForIndexing,
        );
        _processedCount += batch.length;
        recordPassResult(pass, batchResult);

        if (
          total > 0 &&
          (_processedCount % logEvery < config.batchSize || _indexQueue.length === 0)
        ) {
          const pct = Math.round((_processedCount / total) * 100);
          const elapsedSec = (Date.now() - startTime) / 1000;
          const rate = elapsedSec > 0 ? _processedCount / elapsedSec : 0;
          const remainingSec = rate > 0 && _indexQueue.length > 0 ? _indexQueue.length / rate : 0;
          const eta = remainingSec > 5 ? ` — ${formatDuration(remainingSec)} remaining` : '';
          process.stderr.write(`${_processedCount}/${total} (${pct}%)${eta}\n`);
        }
      }

      await runWithDatabaseRecovery(
        'background freshness update',
        () => updateLastIndexed(),
        recoverDatabaseSidecarsForIndexing,
      );

      if (total > 0 || pass.errors > 0) {
        const elapsed = formatDuration((Date.now() - startTime) / 1000);
        const parts = [`${pass.indexed} indexed`, `${pass.skipped} skipped`];
        if (pass.errors > 0) {
          parts.push(`${pass.errors} error${pass.errors === 1 ? '' : 's'}`);
        }
        const hidden =
          pass.errors > MAX_ERROR_LOG_LINES
            ? ` (${pass.errors - MAX_ERROR_LOG_LINES} more not listed)`
            : '';
        process.stderr.write(`Indexing complete in ${elapsed} — ${parts.join(', ')}${hidden}\n`);
      }
    } finally {
      pass.at = new Date().toISOString();
      _lastPass = pass;
      _isIndexing = false;
    }
  });
}

export async function startBackgroundIndexing(contextLength: number): Promise<void> {
  const scan = scanVaultDetailed();
  const files = scan.files;
  await withIndexingDbLock(() =>
    cleanupAfterScan(scan, 'background stale-note cleanup', recoverDatabaseSidecarsForIndexing),
  );
  _totalExpected = files.length;
  _processedCount = 0;
  _indexQueue.push(...files);
  try {
    await processQueue(contextLength);
  } catch (err) {
    console.warn('[indexer] background indexing error:', err);
  }
}

let _rescanTimer: ReturnType<typeof setInterval> | undefined;
let _rescanQueued = false;

/** Stop the OBSIDIAN_RESCAN_MINUTES timer, if one is running. */
export function stopRescanTimer(): void {
  if (_rescanTimer) clearInterval(_rescanTimer);
  _rescanTimer = undefined;
  _rescanQueued = false;
}

const fileDelays = new Map<string, ReturnType<typeof setTimeout>>();
const pendingUnlinks = new Set<string>();

export function startWatcher(contextLength: number): void {
  import('chokidar')
    .then(({ watch }) => {
      let watcherPolicy = createIgnorePolicy();
      let watcherTextExtensions: ReadonlySet<string> = new Set(config.textExtensions);
      const watcher = watch(config.vaultPath, {
        ignored: (filePath: string) => {
          const base = path.basename(filePath);
          if (base === '.gitignore') return false;
          try {
            if (statSync(filePath).isDirectory()) {
              const rel = toVaultRelativePath(filePath);
              // chokidar v5 consults `ignored` for the watch root itself, where
              // rel is ''. isIgnored('/') throws → the catch below would fall
              // through to `return true`, marking the whole tree ignored and
              // silently disabling all file watching. Never ignore the vault root.
              if (rel === '') return false;
              return watcherPolicy.isIgnored(rel + '/');
            }
          } catch {
            // File doesn't exist — fall through to extension check below
          }
          if (!isIndexableFileName(base, watcherTextExtensions)) return true;
          const rel = toVaultRelativePath(filePath);
          return watcherPolicy.isIgnored(rel);
        },
        persistent: true,
        ignoreInitial: true,
      });

      const handleFileChange = (filePath: string) => {
        const existing = fileDelays.get(filePath);
        if (existing) clearTimeout(existing);
        const timer = setTimeout(() => {
          fileDelays.delete(filePath);
          const normalizedPath = path.normalize(filePath).normalize('NFD');
          if (!_indexQueue.includes(normalizedPath)) {
            _indexQueue.push(normalizedPath);
            _totalExpected++;
          }
          processQueue(contextLength).catch((err) => {
            console.warn('[watcher] queue processing error:', err);
          });
        }, config.debounce);
        fileDelays.set(filePath, timer);
      };

      const handleGitignoreChange = () => {
        void withIndexingDbLock(async () => {
          try {
            watcherPolicy = createIgnorePolicy();
            watcherTextExtensions = new Set(config.textExtensions);
            const scan = scanVaultDetailed();
            const files = scan.files;
            await cleanupAfterScan(
              scan,
              'watcher gitignore cleanup',
              recoverDatabaseSidecarsForIndexing,
            );
            if (!_isIndexing && _indexQueue.length === 0) {
              // A rescan re-queues every file; start the progress counters afresh so the
              // log and `status` report this pass rather than a running total.
              _totalExpected = 0;
              _processedCount = 0;
            }
            for (const file of files) {
              const normalizedPath = path.normalize(file).normalize('NFD');
              if (!_indexQueue.includes(normalizedPath)) {
                _indexQueue.push(normalizedPath);
                _totalExpected++;
              }
            }
            watcher.add(config.vaultPath);
            void processQueue(contextLength).catch((err) => {
              console.warn('[watcher] queue processing error:', err);
            });
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.warn(`[watcher] gitignore change error: ${msg}`);
          }
        });
      };

      const safeHandleFileChange = (filePath: string) => {
        try {
          if (path.basename(filePath) === '.gitignore') {
            handleGitignoreChange();
            return;
          }
          handleFileChange(filePath);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.warn(`[watcher] file change error for ${filePath}: ${msg}`);
        }
      };

      const safeHandleUnlink = (filePath: string) => {
        if (pendingUnlinks.has(filePath)) return;
        pendingUnlinks.add(filePath);
        void withIndexingDbLock(async () => {
          try {
            if (path.basename(filePath) === '.gitignore') {
              handleGitignoreChange();
              return;
            }
            if (config.unreadableVault === 'keep' && !isVaultRootReadable()) {
              process.stderr.write(
                `[watcher] ignoring unlink of ${filePath}: vault root is not readable\n`,
              );
              return;
            }
            const rel = toVaultRelativePath(filePath);
            const existing = fileDelays.get(filePath);
            if (existing) {
              clearTimeout(existing);
              fileDelays.delete(filePath);
            }
            await runWithDatabaseRecovery(
              'watcher unlink',
              () => {
                deleteNote(rel);
                bumpIndexVersion();
              },
              recoverDatabaseSidecarsForIndexing,
            );
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.warn(`[watcher] unlink error for ${filePath}: ${msg}`);
          } finally {
            pendingUnlinks.delete(filePath);
          }
        });
      };

      watcher.on('add', safeHandleFileChange);
      watcher.on('change', safeHandleFileChange);
      watcher.on('unlink', safeHandleUnlink);

      const rescanMinutes = config.rescanMinutes;
      if (rescanMinutes > 0) {
        stopRescanTimer();
        _rescanTimer = setInterval(() => {
          // One rescan at a time: a slow pass must not stack more behind the lock.
          if (_rescanQueued) return;
          _rescanQueued = true;
          handleGitignoreChange();
          // Queued behind the rescan on the same lock, so it clears once the rescan ran.
          void withIndexingDbLock(() => {
            _rescanQueued = false;
          });
        }, rescanMinutes * 60_000);
        _rescanTimer.unref();
      }
    })
    .catch((err) => {
      console.warn('[watcher] chokidar load error:', err);
    });
}

/**
 * Extract inline tags from note body: #tag, #tag/subtag
 * Matches # preceded by start-of-string or whitespace, followed by
 * a letter/underscore and then any word chars, hyphens, or slashes.
 */
/**
 * Parse the `aliases` field from gray-matter frontmatter.
 * Handles three formats emitted by Obsidian:
 *   1. YAML list  → gray-matter produces a JS array
 *   2. JSON-stringified array → gray-matter produces a string like '["a","b"]'
 *   3. Plain string → single alias
 */
export function parseAliasField(raw: unknown): string[] {
  const nonempty = (s: string) => s.trim().length > 0;
  if (Array.isArray(raw)) return raw.map(String).filter(nonempty);
  if (typeof raw !== 'string' || !raw.trim()) return [];
  const trimmed = raw.trim();
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.map(String).filter(nonempty);
  } catch {
    // not valid JSON — treat as a single alias
  }
  return [trimmed];
}

export function parseInlineTags(content: string): string[] {
  const seen = new Set<string>();
  // Strip code blocks to avoid matching # inside them
  const stripped = content.replace(/```[\s\S]*?```/g, '').replace(/`[^`]*`/g, '');
  for (const match of stripped.matchAll(
    /(?:^|[\s,;(])#([a-zA-Z_\u00C0-\u024F][a-zA-Z0-9_\-/\u00C0-\u024F]*)/gm,
  )) {
    seen.add(match[1]!);
  }
  return [...seen];
}

export function parseWikilinks(content: string): string[] {
  const seen = new Set<string>();
  let index = 0;
  while (index < content.length) {
    const start = content.indexOf('[[', index);
    if (start === -1) break;
    const end = content.indexOf(']]', start + 2);
    if (end === -1) break;

    const inner = content.slice(start + 2, end);
    const pipeIndex = inner.indexOf('|');
    const headingIndex = inner.indexOf('#');
    const targetEnd = minPositiveIndex(pipeIndex, headingIndex, inner.length);
    const target = inner.slice(0, targetEnd).trim();
    if (target) seen.add(target);
    index = end + 2;
  }
  return [...seen];
}

function minPositiveIndex(first: number, second: number, fallback: number): number {
  const indexes = [first, second].filter((candidate) => candidate >= 0);
  return indexes.length > 0 ? Math.min(...indexes) : fallback;
}

// eslint-disable-next-line sonarjs/cognitive-complexity -- wikilink resolution requires O(N) alias/title lookups
export function resolveWikilinks(content: string, fromPath: string): string[] {
  const db = getDb();
  const raw = parseWikilinks(content);
  if (raw.length === 0) return [];

  // Load all paths, titles and aliases once — O(1) lookups instead of N queries
  const allNotes = db.prepare('SELECT path, title, aliases FROM notes').all() as {
    path: string;
    title: string;
    aliases: string | null;
  }[];

  const pathSet = new Set(allNotes.map((n) => n.path));

  // titleMap: NFD-normalized + lowercased for reliable cross-platform matching.
  // Titles from frontmatter may be NFC; titles derived from filenames on macOS
  // are NFD. Normalising to NFD before lowercasing ensures both forms match.
  const titleMap = new Map(allNotes.map((n) => [n.title.normalize('NFD').toLowerCase(), n.path]));

  // basenameMap: case-insensitive (lowercase key) — Obsidian wikilinks are
  // case-insensitive with respect to the note filename.
  // suffixMap: for partial-path wikilinks like [[sub/note]] that don't match
  // the exact vault-relative path but share a trailing path segment.
  const basenameMap = new Map<string, string>();
  const suffixMap = new Map<string, string>();
  // aliasMap: NFD-normalized + lowercased for the same reason as titleMap.
  const aliasMap = new Map<string, string>();

  for (const n of allNotes) {
    const base = path.basename(n.path).toLowerCase();
    if (!basenameMap.has(base)) basenameMap.set(base, n.path);

    // Build all trailing sub-paths so [[sub/note]] matches 'folder/sub/note.md'
    const parts = n.path.split('/');
    for (let i = 1; i < parts.length; i++) {
      const suffix = parts.slice(i).join('/').toLowerCase();
      if (!suffixMap.has(suffix)) suffixMap.set(suffix, n.path);
    }

    if (n.aliases) {
      try {
        const aliases = JSON.parse(n.aliases) as string[];
        for (const alias of aliases) {
          const key = alias.normalize('NFD').toLowerCase();
          if (alias && !aliasMap.has(key)) {
            aliasMap.set(key, n.path);
          }
        }
      } catch {
        /* ignore malformed aliases */
      }
    }
  }

  const resolved: string[] = [];
  for (const rawTarget of raw) {
    const target = rawTarget.normalize('NFD');

    // 0. Indexed text files ([[Script.ps1]]): the target already carries its extension.
    const targetExt = fileExtension(target);
    if (targetExt !== '' && targetExt !== 'md') {
      const byTextPath =
        (pathSet.has(target) ? target : undefined) ??
        (target.includes('/') ? suffixMap.get(target.toLowerCase()) : undefined) ??
        basenameMap.get(path.basename(target).toLowerCase());
      if (byTextPath && byTextPath !== fromPath) {
        resolved.push(byTextPath);
        continue;
      }
    }

    const withMd = target.endsWith('.md') ? target : target + '.md';
    const base = path.basename(withMd);

    // 1. Exact vault-relative path match (already NFD-normalised)
    if (pathSet.has(withMd) && withMd !== fromPath) {
      resolved.push(withMd);
      continue;
    }

    // 2. Suffix/partial-path match: [[sub/note]] → 'folder/sub/note.md'
    //    Only applied when the target contains a directory separator so we
    //    don't accidentally use this for plain note-name wikilinks.
    if (withMd.includes('/')) {
      const bySuffix = suffixMap.get(withMd.toLowerCase());
      if (bySuffix && bySuffix !== fromPath) {
        resolved.push(bySuffix);
        continue;
      }
    }

    // 3. Basename match — case-insensitive so [[My Note]] finds 'my note.md'
    const byBasename = basenameMap.get(base.toLowerCase());
    if (byBasename && byBasename !== fromPath) {
      resolved.push(byBasename);
      continue;
    }

    // 4. Alias match (NFD-normalised, case-insensitive)
    const byAlias = aliasMap.get(target.toLowerCase());
    if (byAlias && byAlias !== fromPath) {
      resolved.push(byAlias);
      continue;
    }

    // 5. Title match (NFD-normalised, case-insensitive)
    const byTitle = titleMap.get(target.toLowerCase());
    if (byTitle && byTitle !== fromPath) {
      resolved.push(byTitle);
    }
  }

  return [...new Set(resolved)];
}
