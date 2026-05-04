import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it, vi } from 'vitest';

// Mock db.js BEFORE any imports that use it (ESM live-binding rule)
const deleteNoteMock = vi.fn(() => {
  throw new Error('FK constraint');
});

vi.mock('../src/db.js', async (importOriginal) => {
  const mod = await importOriginal<typeof import('../src/db.js')>();
  return {
    ...mod,
    deleteNote: deleteNoteMock,
  };
});

// Mock chokidar before importing indexer
vi.mock('chokidar', () => ({
  watch: vi.fn().mockReturnValue({
    on: vi.fn().mockReturnThis(),
  }),
}));

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-watcher-test-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

const { startWatcher } = await import('../src/indexer');

describe('watcher error resilience', () => {
  beforeAll(() => {
    deleteNoteMock.mockClear();
  });

  afterAll(() => {
    rmSync(vaultDir, { recursive: true, force: true });
  });

  it('watcher unlink swallows deleteNote errors without crashing', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    startWatcher(512);
    await new Promise((r) => setTimeout(r, 50));

    const chokidar = await import('chokidar');
    const watchMock = chokidar.watch as ReturnType<typeof vi.fn>;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const watcherInstance = watchMock.mock.results.at(-1)?.value;
    assert.ok(watcherInstance, 'watcher instance should exist');

    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    const eventHandlers = watcherInstance.on.mock.calls as unknown[][];
    const unlinkCall = eventHandlers.find((c: unknown[]) => c[0] === 'unlink');
    assert.ok(unlinkCall, 'unlink handler should be registered');

    const unlinkHandler = unlinkCall[1] as (p: string) => void;

    assert.doesNotThrow(() => {
      unlinkHandler(path.join(vaultDir, 'nonexistent.md'));
    });

    assert.equal(deleteNoteMock.mock.calls.length, 1, 'deleteNote should have been called');
    assert.ok(
      warnSpy.mock.calls.some((c) => String(c[0]).includes('[watcher] unlink error')),
      'error should be logged to console.warn',
    );

    warnSpy.mockRestore();
  });
});
