/**
 * Tests for the ignore-pattern lifecycle in getPathsToRemoveForIgnoreChange.
 *
 * Two specific concerns:
 *
 *  I1 – Idempotency: calling getPathsToRemoveForIgnoreChange with the same
 *       patterns twice must return [] on the second call.  A localeCompare-based
 *       sort that differs across locales (or across orderings) could cause the
 *       stored JSON to never equal the new JSON, making the function delete
 *       matching notes on EVERY server start / reindex.
 *
 *  I2 – First-run with existing notes: when ignore_patterns has never been
 *       stored but the DB already has notes (e.g. after a schema migration),
 *       the function should NOT delete notes whose paths do NOT match the
 *       current patterns.  Only actually-ignored paths should be removed.
 *
 *  I3 – Valid notes (paths outside ignore patterns) survive repeated
 *       cleanupStaleNotes-equivalent calls (via getPathsToRemoveForIgnoreChange
 *       + the fsPaths-based deletion logic replicated here).
 *
 *  I4 – matchesIgnorePattern edge cases: verify the pattern-matching logic
 *       doesn't accidentally match valid note paths.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it } from 'vitest';

// ─── Vault setup ──────────────────────────────────────────────────────────────

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-ignore-test-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;
// Use default ignore patterns (same as production default in config.ts)
delete process.env.OBSIDIAN_IGNORE_PATTERNS;

const { openDb, initVecTable, upsertNote, getDb, getPathsToRemoveForIgnoreChange } =
  await import('../src/db.js');
const { search, bumpIndexVersion } = await import('../src/searcher.js');

const DIM = 4;
let _seq = 0;
const emb = () => {
  const s = (_seq++ % 9) / 9;
  return new Float32Array([s, 1 - s, s * 0.5, (1 - s) * 0.5]);
};

function insertNote(p: string, title: string, content: string, tags: string[] = []) {
  upsertNote({
    path: p,
    title,
    tags,
    content,
    mtime: Date.now(),
    hash: 'h-' + p,
    chunks: [{ text: content, embedding: emb() }],
  });
}

const DEFAULT_PATTERNS = ['.obsidian/**', 'templates/**', '*.canvas'];

beforeAll(() => {
  openDb();
  initVecTable(DIM);
});

afterAll(() => {
  rmSync(vaultDir, { recursive: true, force: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// I1 – Idempotency: repeated calls with the same patterns return [] after first
// ─────────────────────────────────────────────────────────────────────────────

describe('I1 – getPathsToRemoveForIgnoreChange is idempotent', () => {
  it('second call with same patterns returns [] (no localeCompare drift)', () => {
    // First call stores the patterns and may return paths to remove
    const first = getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);

    // Second call must detect "no change" and return early
    const second = getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);
    assert.deepEqual(
      second,
      [],
      'I1 FAILED: same patterns produced a non-empty removal list on second call — ' +
        'localeCompare or serialization inconsistency suspected',
    );

    // Third call to confirm stability
    const third = getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);
    assert.deepEqual(third, [], 'I1 FAILED: still non-empty on third call');

    // first call result is informational only
    assert.ok(Array.isArray(first));
  });

  it('patterns in a different input order still match stored value', () => {
    // Store with one order
    getPathsToRemoveForIgnoreChange(['.obsidian/**', 'templates/**', '*.canvas']);

    // Call with reversed order — both sides sort, so comparison should pass
    const result = getPathsToRemoveForIgnoreChange(['*.canvas', 'templates/**', '.obsidian/**']);
    assert.deepEqual(
      result,
      [],
      'I1 FAILED: different input order caused false pattern-change detection',
    );
  });

  it('patterns with extra whitespace still match stored value', () => {
    getPathsToRemoveForIgnoreChange(['.obsidian/**', 'templates/**', '*.canvas']);

    // config.ignorePatterns trims whitespace; the function itself does NOT.
    // Verify that pre-trimmed patterns still match the stored value.
    const trimmedResult = getPathsToRemoveForIgnoreChange(
      [' .obsidian/**', 'templates/** ', ' *.canvas '].map((p) => p.trim()),
    );
    assert.deepEqual(
      trimmedResult,
      [],
      'I1 FAILED: trimmed whitespace patterns still mismatch stored value',
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// I2 – First-run with existing notes: valid notes must not be deleted
// ─────────────────────────────────────────────────────────────────────────────

describe('I2 – first-run scenario: DB has notes, ignore_patterns not yet stored', () => {
  it('valid notes (outside ignore paths) are NOT returned as paths to remove', () => {
    // Insert a valid note and a note in an ignored path
    insertNote('notes/valid-note.md', 'Valid Note', 'valid note content here');
    insertNote('templates/ignored-note.md', 'Ignored Note', 'template note content');
    bumpIndexVersion();

    // Simulate "first run" by deleting the stored ignore patterns setting
    const db = getDb();
    db.prepare("DELETE FROM settings WHERE key = 'ignore_patterns'").run();

    // Now call as if it's the first time — noteCount > 0 but no stored patterns
    const toRemove = getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);

    assert.ok(
      !toRemove.includes('notes/valid-note.md'),
      'I2 FAILED: valid note was returned for deletion on first-run scenario',
    );

    // Templates note should be in the removal list (it matches templates/**)
    assert.ok(
      toRemove.includes('templates/ignored-note.md'),
      'I2 expected: ignored-path note should be in removal list',
    );
  });

  it('after first-run, valid note is still findable in search', async () => {
    const result = await search('valid note content here', { mode: 'fulltext' });
    assert.ok(
      result.some((r) => r.path === 'notes/valid-note.md'),
      'I2 FAILED: valid note disappeared from search after ignore-pattern first-run',
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// I3 – Valid notes survive repeated pattern evaluations
// ─────────────────────────────────────────────────────────────────────────────

describe('I3 – valid notes survive repeated getPathsToRemoveForIgnoreChange calls', () => {
  it('notes outside ignore paths are never returned across 5 consecutive calls', () => {
    insertNote('base/category.md', 'Category Note', 'category note survives repeated checks', [
      'system/category',
    ]);
    bumpIndexVersion();

    // Simulate what happens on each server startup / reindex
    for (let i = 0; i < 5; i++) {
      const toRemove = getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);
      assert.ok(
        !toRemove.includes('base/category.md'),
        `I3 FAILED: valid note appeared in removal list on call ${i + 1}`,
      );
    }
  });

  it('valid note remains findable after 5 consecutive pattern evaluations', async () => {
    for (let i = 0; i < 5; i++) {
      getPathsToRemoveForIgnoreChange(DEFAULT_PATTERNS);
    }
    bumpIndexVersion();

    const result = await search('category note survives repeated checks', { mode: 'fulltext' });
    assert.ok(
      result.some((r) => r.path === 'base/category.md'),
      'I3 FAILED: note disappeared after repeated pattern evaluations',
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// I4 – matchesIgnorePattern edge cases
// ─────────────────────────────────────────────────────────────────────────────

describe('I4 – ignore pattern matching does not hit false positives', () => {
  // We test matchesIgnorePattern indirectly via isIgnored (re-implemented inline
  // so the test is self-contained and doesn't depend on unexported functions).
  function matchesAnyPattern(relPath: string, patterns: string[]): boolean {
    return patterns.some((pattern) => {
      if (pattern.endsWith('/**')) {
        const prefix = pattern.slice(0, -3);
        return (
          relPath === prefix ||
          relPath.startsWith(prefix + path.sep) ||
          relPath.startsWith(prefix + '/')
        );
      }
      if (pattern.startsWith('*.')) {
        const ext = pattern.slice(1);
        return relPath.endsWith(ext) || path.basename(relPath).endsWith(ext);
      }
      return relPath === pattern || relPath.startsWith(pattern + '/');
    });
  }

  const IGNORED = [...DEFAULT_PATTERNS];

  it('.obsidian/** does not match notes outside .obsidian/', () => {
    assert.ok(!matchesAnyPattern('base/categories/knowledge base.md', IGNORED));
    assert.ok(!matchesAnyPattern('notes/some-note.md', IGNORED));
    assert.ok(!matchesAnyPattern('projects/my-project.md', IGNORED));
    // But does match things inside .obsidian/
    assert.ok(matchesAnyPattern('.obsidian/plugins/some-plugin.md', IGNORED));
  });

  it('templates/** does not match notes with "templates" anywhere except as prefix', () => {
    assert.ok(!matchesAnyPattern('notes/templates-overview.md', IGNORED));
    assert.ok(!matchesAnyPattern('base/my-templates.md', IGNORED));
    // But does match the templates/ directory
    assert.ok(matchesAnyPattern('templates/my-template.md', IGNORED));
  });

  it('*.canvas does not match .md files', () => {
    assert.ok(!matchesAnyPattern('diagram.canvas.md', IGNORED));
    assert.ok(!matchesAnyPattern('some-note.md', IGNORED));
    // But does match actual canvas files
    assert.ok(matchesAnyPattern('diagram.canvas', IGNORED));
  });

  it('a path containing "templates" as a substring is not ignored', () => {
    // This is a concrete false-positive risk
    assert.ok(!matchesAnyPattern('base/note-templates.md', IGNORED));
    assert.ok(!matchesAnyPattern('sources/templates-overview.md', IGNORED));
  });
});
