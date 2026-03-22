# GPU Local Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `@huggingface/transformers` with `node-llama-cpp` to get GPU-accelerated local embeddings (BGE-M3 Q4_K_M) and reranking (BGE-reranker-v2-m3 Q4_K_M) with zero user configuration.

**Architecture:** New `src/llama-backend.ts` isolates all `node-llama-cpp` code and exposes two functions: `llamaEmbed()` and `llamaRerank()`. `embedder.ts` and `reranker.ts` become thin adapters. API mode (`OPENAI_API_KEY` / `OPENAI_BASE_URL`) is completely untouched. Models auto-download to `~/.cache/llama-models/` on first use via lazy loading.

**Tech Stack:** node-llama-cpp (llama.cpp Node.js bindings), TypeScript, vitest, better-sqlite3

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `package.json` | Modify | Add `node-llama-cpp` dep, remove `@huggingface/transformers` |
| `src/llama-backend.ts` | **Create** | All llama.cpp code: singleton, download, lazy-load, embed, rerank |
| `src/embedder.ts` | Modify | Remove Xenova code, update `LOCAL_MODEL`, delegate to `llamaEmbed()` |
| `src/reranker.ts` | Modify | Slim down to pass-through; remove `_loadModel`, `ensureLoaded`, `pipeline` |
| `src/config.ts` | Modify | Remove `RERANKER_MODEL` getter |
| `test/llama-backend.test.ts` | **Create** | Unit tests for `llama-backend.ts` with mocked `node-llama-cpp` |
| `test/reranker.test.ts` | Modify | Replace pipeline mock injection with `vi.mock('../src/llama-backend.js')` |
| `vitest.integration.config.ts` | Modify | Increase timeout to 600s for BGE-M3 first-run download |
| `eval/results/baseline-no-rerank.json` | Maybe update | Only if nDCG@5 improves after migration |

---

### Task 1: Run pre-migration eval to save baseline

**Files:**
- Create: `eval/results/before-gpu-migration.json`

- [ ] **Step 1: Run eval and save pre-migration baseline**

```bash
npm run eval -- --vault fixtures/obsidian-help/en --output eval/results/before-gpu-migration.json
```

Expected: JSON file created. nDCG@5 should be ≈ 0.780 (matches committed baseline). If it differs significantly, note it before continuing.

- [ ] **Step 2: Verify the numbers are reasonable**

```bash
grep -E '"ndcg|"mrr' eval/results/before-gpu-migration.json
```

---

### Task 2: Update package.json and install node-llama-cpp

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Update package.json**

Make these edits to `package.json`:
1. In `"dependencies"`, add: `"node-llama-cpp": "latest"`
2. Remove the entire `"optionalDependencies"` section (it only contained `@huggingface/transformers`)

The result should look like:
```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "latest",
    "better-sqlite3": "latest",
    "chokidar": "^4.0.0",
    "cli-table3": "^0.6.3",
    "commander": "^12.0.0",
    "gray-matter": "^4.0.3",
    "node-llama-cpp": "latest",
    "picocolors": "^1.1.1",
    "sqlite-vec": "latest"
  }
}
```

> **After the spike (Task 3):** Pin `node-llama-cpp` to a concrete major version (e.g., `"^3.0.0"`) once you confirm which API version is in use. The `"latest"` specifier here is a placeholder — `node-llama-cpp` has had breaking API changes across major versions.

- [ ] **Step 2: Install**

```bash
npm install
```

Expected: `node-llama-cpp` postinstall downloads native GPU backend binaries. On macOS this downloads Metal binaries (~150 MB). On Linux with CUDA it downloads CUDA binaries (~400 MB). This takes 1–5 minutes — do not interrupt. CPU fallback binary is smaller and faster.

- [ ] **Step 3: Verify node-llama-cpp is available**

```bash
node -e "import('node-llama-cpp').then(m => console.log('OK', Object.keys(m).slice(0,5)))"
```

Expected: prints `OK` followed by exported names like `['getLlama', ...]`

- [ ] **Step 4: Commit the dependency change**

```bash
git add package.json package-lock.json
git commit -m "feat: add node-llama-cpp dependency, remove @huggingface/transformers"
```

---

### Task 3: Spike — validate node-llama-cpp APIs

**Purpose:** Before writing any production code, validate the exact API shape for loading GGUF models, generating embeddings, and scoring reranker pairs. The implementation in Task 4 uses what you discover here. **Do not skip this task.**

**Files:**
- Create (temporary, not committed): `scripts/spike-llama.ts`

- [ ] **Step 1: Create the spike script**

```typescript
// scripts/spike-llama.ts — DELETE BEFORE COMMITTING
import { getLlama } from 'node-llama-cpp';
import os from 'node:os';
import path from 'node:path';

const cacheDir = path.join(os.homedir(), '.cache', 'llama-models');

async function main() {
  console.log('=== Initializing Llama ===');
  const llama = await getLlama();
  console.log('getLlama() returned:', typeof llama, Object.keys(llama as object).slice(0, 8));

  // ── Test 1: Load embedding model ─────────────────────────────────────────
  console.log('\n=== Loading BGE-M3 embedding model ===');
  console.log('(first run downloads ~370 MB — be patient)');

  // Try loading with hf: URI — node-llama-cpp may auto-download
  // If this fails with "hf: not supported", download separately first
  const embedModel = await (llama as any).loadModel({
    modelPath: 'hf:BAAI/bge-m3/bge-m3-Q4_K_M.gguf',
  });
  console.log('embedModel type:', typeof embedModel);
  console.log('embedModel keys:', Object.keys(embedModel as object).slice(0, 10));

  const embedCtx = await (embedModel as any).createEmbeddingContext();
  console.log('embedCtx type:', typeof embedCtx);
  console.log('embedCtx keys:', Object.keys(embedCtx as object).slice(0, 10));

  // Test: what does getEmbeddingFor return?
  const result = await (embedCtx as any).getEmbeddingFor('passage: This is a test sentence.');
  console.log('\n--- Embedding result ---');
  console.log('result type:', typeof result, result?.constructor?.name);
  console.log('result keys (if object):', result && typeof result === 'object' ? Object.keys(result as object) : 'N/A');
  // Check if it has .vector property
  const vec = (result as any)?.vector ?? result;
  console.log('vector type:', typeof vec, vec?.constructor?.name);
  console.log('vector length:', (vec as any)?.length);
  console.log('vector[0..4]:', Array.from((vec as any)?.slice?.(0, 5) ?? []));

  // ── Test 2: Load reranker model ───────────────────────────────────────────
  console.log('\n=== Loading BGE-reranker-v2-m3 model ===');
  console.log('(first run downloads ~320 MB)');

  const rerankerModel = await (llama as any).loadModel({
    modelPath: 'hf:BAAI/bge-reranker-v2-m3/bge-reranker-v2-m3-Q4_K_M.gguf',
  });
  const rerankerCtx = await (rerankerModel as any).createEmbeddingContext();

  // Test: pass query+doc pair to reranker
  const pair = 'What is Obsidian?\n\nObsidian is a knowledge base app that works on local Markdown files.';
  const rerankerResult = await (rerankerCtx as any).getEmbeddingFor(pair);
  console.log('\n--- Reranker result ---');
  const rerankerVec = (rerankerResult as any)?.vector ?? rerankerResult;
  console.log('result type:', typeof rerankerVec, rerankerVec?.constructor?.name);
  console.log('result length:', (rerankerVec as any)?.length);
  console.log('result values:', Array.from((rerankerVec as any) ?? []));
  // IMPORTANT: if length=1, that single value IS the relevance score
  // if length>1, note which index corresponds to relevance

  // Cleanup
  await (embedCtx as any).dispose?.();
  await (rerankerCtx as any).dispose?.();

  console.log('\n=== SPIKE COMPLETE ===');
  console.log('Answer these before implementing llama-backend.ts:');
  console.log('1. Does "hf:" URI work for loadModel, or do we need createModelDownloader?');
  console.log('2. Does getEmbeddingFor return Float32Array directly or {vector: Float32Array}?');
  console.log('3. For reranker: length of output vector and which index is the relevance score?');
  console.log('4. Any other API shapes that differ from the skeleton in Task 4?');
}

main().catch(console.error);
```

- [ ] **Step 2: Run the spike**

```bash
npx tsx scripts/spike-llama.ts
```

Expected: Models download on first run (progress output from node-llama-cpp). Then prints the API shape information. Read the output carefully.

- [ ] **Step 3: Write down your findings** (answer in a comment at the top of `src/llama-backend.ts` when you create it)

Key questions:
1. **Download:** Does `hf:` URI work in `loadModel()` directly, or must you use `createModelDownloader` first?
2. **Embedding output:** Does `getEmbeddingFor()` return `Float32Array` directly, or `{ vector: Float32Array }`, or something else?
3. **Reranker output:** How long is the output vector? Is index `[0]` the relevance logit, or another index?
4. **Any imports needed** beyond `getLlama`? (e.g., `createModelDownloader`, `LlamaEmbeddingContext`)
5. **Reranker input format:** How does the reranker expect query + document as input? Is it a single concatenated string (e.g., `"query\n\ndocument"`)? Or does `node-llama-cpp` have a dedicated pair-encoding API (e.g., separate `query` and `document` arguments)? The correct format affects relevance score quality — note the exact format used and whether it produces scores that correlate with relevance.

- [ ] **Step 4: Delete the spike script**

```bash
rm scripts/spike-llama.ts
```

---

### Task 4: Implement src/llama-backend.ts (TDD)

**Files:**
- Create: `test/llama-backend.test.ts`
- Create: `src/llama-backend.ts`

Write the tests first, then implement. Use findings from the spike to write accurate mocks.

- [ ] **Step 1: Write failing tests**

Create `test/llama-backend.test.ts`. The mock shape must match the real `node-llama-cpp` API you discovered in the spike:

```typescript
import assert from 'node:assert/strict';
import { beforeEach, describe, it, vi } from 'vitest';

process.env.OBSIDIAN_VAULT_PATH = '/tmp/ohs-llama-test';

// ─── Mock node-llama-cpp ──────────────────────────────────────────────────────
// UPDATE this mock shape to match what the spike revealed.
// The structure below assumes:
//   - getLlama() → { loadModel() }
//   - loadModel() → { createEmbeddingContext() }
//   - createEmbeddingContext() → { getEmbeddingFor() }
//   - getEmbeddingFor() → Float32Array (or { vector: Float32Array } — update as needed)

const mockGetEmbeddingFor = vi.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3]));
const mockCreateEmbeddingContext = vi.fn().mockResolvedValue({
  getEmbeddingFor: mockGetEmbeddingFor,
  dispose: vi.fn(),
});
const mockLoadModel = vi.fn().mockResolvedValue({
  createEmbeddingContext: mockCreateEmbeddingContext,
  dispose: vi.fn(),
});
const mockGetLlama = vi.fn().mockResolvedValue({
  loadModel: mockLoadModel,
});

vi.mock('node-llama-cpp', () => ({
  getLlama: mockGetLlama,
  // Add other exports discovered in spike if needed (e.g., createModelDownloader)
}));

const { llamaEmbed, llamaRerank, _resetForTest } = await import('../src/llama-backend.js');
import type { RerankCandidate } from '../src/reranker.js';

// ─── llamaEmbed ───────────────────────────────────────────────────────────────
describe('llamaEmbed', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // CRITICAL: reset module-level singletons between tests.
    // Without this, the first test that loads a model warms up the singleton,
    // and subsequent tests (including the deduplication test) never trigger loadModel
    // because embedCtx is already populated — making the test vacuously pass.
    _resetForTest();
    mockGetEmbeddingFor.mockResolvedValue(new Float32Array([0.1, 0.2, 0.3]));
  });

  it('returns one Float32Array per input text', async () => {
    const results = await llamaEmbed(['hello', 'world'], 'document');
    assert.strictEqual(results.length, 2);
    assert.ok(results[0] instanceof Float32Array, 'Expected Float32Array');
    assert.ok(results[1] instanceof Float32Array, 'Expected Float32Array');
  });

  it('prepends "passage: " prefix for document type', async () => {
    await llamaEmbed(['Obsidian is great'], 'document');
    const calledWith = mockGetEmbeddingFor.mock.calls[0]?.[0] as string;
    assert.ok(
      calledWith.startsWith('passage: '),
      `Expected "passage: " prefix, got: "${calledWith}"`,
    );
  });

  it('prepends "query: " prefix for query type', async () => {
    await llamaEmbed(['What is Obsidian?'], 'query');
    const calledWith = mockGetEmbeddingFor.mock.calls[0]?.[0] as string;
    assert.ok(
      calledWith.startsWith('query: '),
      `Expected "query: " prefix, got: "${calledWith}"`,
    );
  });

  it('returns null for a text when getEmbeddingFor throws', async () => {
    mockGetEmbeddingFor.mockRejectedValueOnce(new Error('inference failed'));
    const results = await llamaEmbed(['bad text'], 'document');
    assert.strictEqual(results[0], null);
  });

  it('loads embedding model only once for concurrent calls', async () => {
    let loadCount = 0;
    mockLoadModel.mockImplementation(async () => {
      loadCount++;
      await new Promise((r) => setTimeout(r, 20));
      return { createEmbeddingContext: mockCreateEmbeddingContext, dispose: vi.fn() };
    });

    await Promise.all([
      llamaEmbed(['a'], 'document'),
      llamaEmbed(['b'], 'document'),
      llamaEmbed(['c'], 'document'),
    ]);

    assert.strictEqual(loadCount, 1, 'embedding model should be loaded exactly once');
  });

  it('does not load reranker model when only llamaEmbed is called', async () => {
    // Reset call count
    vi.clearAllMocks();
    mockGetEmbeddingFor.mockResolvedValue(new Float32Array([0.1, 0.2, 0.3]));
    let loadCallCount = 0;
    mockLoadModel.mockImplementation(async () => {
      loadCallCount++;
      return { createEmbeddingContext: mockCreateEmbeddingContext, dispose: vi.fn() };
    });

    await llamaEmbed(['test'], 'document');

    // Only the embedding model should have been loaded (1 call), not the reranker (which would be a 2nd call)
    assert.strictEqual(loadCallCount, 1, 'only one model should be loaded (embedder, not reranker)');
  });
});

// ─── llamaRerank ──────────────────────────────────────────────────────────────
describe('llamaRerank', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetForTest();
    // Reranker returns a 1-element vector; [0] is the relevance logit
    // UPDATE the index if the spike revealed a different index
    mockGetEmbeddingFor.mockResolvedValue(new Float32Array([0.9]));
  });

  it('returns one score per candidate', async () => {
    const candidates: RerankCandidate[] = [
      { title: 'A', chunkText: 'text a', snippet: '' },
      { title: 'B', chunkText: 'text b', snippet: '' },
    ];
    const scores = await llamaRerank('query', candidates);
    assert.strictEqual(scores.length, 2);
    assert.ok(typeof scores[0] === 'number');
    assert.ok(typeof scores[1] === 'number');
  });

  it('returns empty array for empty candidates without calling model', async () => {
    const scores = await llamaRerank('q', []);
    assert.deepEqual(scores, []);
    assert.strictEqual(mockGetEmbeddingFor.mock.calls.length, 0);
  });

  it('uses chunkText when available, falls back to snippet', async () => {
    const seenTexts: string[] = [];
    mockGetEmbeddingFor.mockImplementation(async (text: string) => {
      seenTexts.push(text);
      return new Float32Array([0.5]);
    });

    await llamaRerank('q', [
      { title: 'T1', chunkText: 'chunk body', snippet: 'snippet text' },
      { title: 'T2', chunkText: undefined, snippet: 'fallback snippet' },
    ]);

    assert.ok(seenTexts[0]?.includes('chunk body'), `Expected "chunk body" in text, got: ${seenTexts[0]}`);
    assert.ok(seenTexts[1]?.includes('fallback snippet'), `Expected "fallback snippet" in text, got: ${seenTexts[1]}`);
  });

  it('includes title and content separated by double newline', async () => {
    const seenTexts: string[] = [];
    mockGetEmbeddingFor.mockImplementation(async (text: string) => {
      seenTexts.push(text);
      return new Float32Array([0.5]);
    });

    await llamaRerank('q', [{ title: 'My Title', chunkText: 'chunk body', snippet: '' }]);

    assert.ok(
      seenTexts[0]?.includes('My Title\n\nchunk body'),
      `Expected "My Title\\n\\nchunk body" in text, got: ${seenTexts[0]}`,
    );
  });

  it('returns 0 for a candidate when getEmbeddingFor throws', async () => {
    mockGetEmbeddingFor.mockRejectedValueOnce(new Error('reranker failed'));
    const scores = await llamaRerank('q', [{ title: 'A', chunkText: 'x', snippet: '' }]);
    assert.strictEqual(scores[0], 0);
  });
});
```

- [ ] **Step 2: Run tests — verify they FAIL (module not found)**

```bash
npm test -- --reporter=verbose test/llama-backend.test.ts
```

Expected: FAIL — `Cannot find module '../src/llama-backend.js'`

- [ ] **Step 3: Implement src/llama-backend.ts**

Fill in based on spike findings. The `// UPDATE` comments mark where spike results determine the exact code:

```typescript
import os from 'node:os';
import path from 'node:path';
// UPDATE: add any other imports discovered in spike (e.g., createModelDownloader)
import { getLlama } from 'node-llama-cpp';

import { config } from './config.js';
import type { RerankCandidate } from './reranker.js';

// Spike findings (fill in after Task 3):
// - getLlama() API: ...
// - loadModel() modelPath format: hf: URI works / need separate download / ...
// - getEmbeddingFor() return type: Float32Array directly / { vector: Float32Array } / ...
// - Reranker logit index: output[0] / output[N] = relevance score

// ── Constants ─────────────────────────────────────────────────────────────────
// UPDATE: these URIs may need adjustment if hf: scheme is not supported
const EMBED_MODEL_URI = 'hf:BAAI/bge-m3/bge-m3-Q4_K_M.gguf';
const RERANKER_MODEL_URI = 'hf:BAAI/bge-reranker-v2-m3/bge-reranker-v2-m3-Q4_K_M.gguf';

function getCacheDir(): string {
  return path.join(os.homedir(), '.cache', 'llama-models');
}

// ── Singleton Llama instance ──────────────────────────────────────────────────
// One getLlama() per process — required by llama.cpp
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let llamaInstance: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let llamaPromise: Promise<any> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getLlamaInstance(): Promise<any> {
  if (llamaInstance) return llamaInstance;
  if (!llamaPromise) {
    llamaPromise = getLlama().then((l) => {
      llamaInstance = l;
      return l;
    });
  }
  return llamaPromise;
}

// ── Embedding model (BGE-M3) ──────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let embedCtx: any = null;
let embedLoadPromise: Promise<void> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getEmbedContext(): Promise<any> {
  if (embedCtx) return embedCtx;
  if (!embedLoadPromise) {
    embedLoadPromise = (async () => {
      const llama = await getLlamaInstance();
      // UPDATE: adjust modelPath loading based on spike findings
      // If hf: URI is supported:
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
      const model = await llama.loadModel({ modelPath: EMBED_MODEL_URI });
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
      embedCtx = await model.createEmbeddingContext();
    })();
  }
  await embedLoadPromise;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return embedCtx;
}

// ── Reranker model (BGE-reranker-v2-m3) ──────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let rerankerCtx: any = null;
let rerankerLoadPromise: Promise<void> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRerankerContext(): Promise<any> {
  if (rerankerCtx) return rerankerCtx;
  if (!rerankerLoadPromise) {
    rerankerLoadPromise = (async () => {
      const llama = await getLlamaInstance();
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
      const model = await llama.loadModel({ modelPath: RERANKER_MODEL_URI });
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
      rerankerCtx = await model.createEmbeddingContext();
    })();
  }
  await rerankerLoadPromise;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return rerankerCtx;
}

// ── Helper: extract Float32Array from whatever getEmbeddingFor returns ─────────
// UPDATE: adjust based on spike findings for the actual return type
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toFloat32Array(raw: any): Float32Array {
  if (raw instanceof Float32Array) return raw;
  // If result has a .vector property (LlamaEmbedding object)
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  if (raw?.vector instanceof Float32Array) return raw.vector as Float32Array;
  // Fallback: coerce
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  return new Float32Array(raw as number[]);
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Reset all module-level singletons.
 * ONLY for use in tests — allows beforeEach to start each test with a clean state
 * so that deduplication and lazy-loading tests are not vacuously true.
 */
export function _resetForTest(): void {
  llamaInstance = null;
  llamaPromise = null;
  embedCtx = null;
  embedLoadPromise = null;
  rerankerCtx = null;
  rerankerLoadPromise = null;
}

export async function llamaEmbed(
  texts: string[],
  type: 'query' | 'document',
): Promise<(Float32Array | null)[]> {
  const ctx = await getEmbedContext();
  const prefix = type === 'query' ? 'query: ' : 'passage: ';
  const results: (Float32Array | null)[] = [];

  for (let i = 0; i < texts.length; i += config.batchSize) {
    const batch = texts.slice(i, i + config.batchSize);
    const batchResults = await Promise.all(
      batch.map(async (text) => {
        try {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
          const raw = await ctx.getEmbeddingFor(prefix + text);
          return toFloat32Array(raw);
        } catch {
          return null;
        }
      }),
    );
    results.push(...batchResults);
  }

  return results;
}

export async function llamaRerank(
  query: string,
  candidates: RerankCandidate[],
): Promise<number[]> {
  if (candidates.length === 0) return [];
  const ctx = await getRerankerContext();

  return Promise.all(
    candidates.map(async (c) => {
      const content = c.chunkText ?? c.snippet;
      // Pass query + doc to the reranker as a combined text
      // UPDATE: format may need adjustment based on spike findings
      const text = `${query}\n\n${c.title}\n\n${content}`;
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
        const raw = await ctx.getEmbeddingFor(text);
        const vec = toFloat32Array(raw);
        // UPDATE: verify index based on spike findings
        // For a 1-label cross-encoder, vec[0] is the relevance logit
        return (vec[0] as number | undefined) ?? 0;
      } catch {
        return 0;
      }
    }),
  );
}
```

- [ ] **Step 4: Run the tests — they should pass**

```bash
npm test -- --reporter=verbose test/llama-backend.test.ts
```

Expected: All tests pass. If a test fails due to mock shape mismatch, update the mock in `test/llama-backend.test.ts` to match the real API from the spike.

- [ ] **Step 5: Run full test suite to ensure nothing is broken**

```bash
npm test
```

Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/llama-backend.ts test/llama-backend.test.ts
git commit -m "feat: add llama-backend.ts with GPU-accelerated BGE-M3 embeddings and reranker"
```

---

### Task 5: Update src/embedder.ts

**Files:**
- Modify: `src/embedder.ts`

- [ ] **Step 1: Apply all changes to embedder.ts**

Make the following changes:

**Remove (search by content, not line number — lines may have shifted):**
- `import os from 'node:os';` — only used by `getCacheDir`, now dead
- `import path from 'node:path';` — only used by `getCacheDir`, now dead
- The entire `getCacheDir()` function (3 lines starting with `function getCacheDir()`)
- `let localPipeline: any = null` — module-level var, no longer needed
- The entire `getLocalPipeline()` function (the ~33-line `async function getLocalPipeline()` block)

**Keep as-is:** `let cachedContextLength: number | null = null` and `let cachedDim: number | null = null` — these are still used by `getContextLength()` and `getEmbeddingDim()` in both API and local mode.

**Update `LOCAL_MODEL` constant (line 5):**
```typescript
// Before:
export const LOCAL_MODEL = 'Xenova/multilingual-e5-small';
// After:
export const LOCAL_MODEL = 'BAAI/bge-m3';
```

**Add to `KNOWN_CONTEXT_LENGTHS`** (after the existing `'baai/bge-m3': 8192` entry around line 77):
```typescript
'BAAI/bge-m3': 8192,          // uppercase — matches LOCAL_MODEL constant exactly
```
Both `'baai/bge-m3'` (lowercase, for Ollama/OpenRouter) and `'BAAI/bge-m3'` (uppercase, for LOCAL_MODEL lookup) should coexist.

**Update `getContextLength()` local branch** (around lines 139–161). Remove the `getLocalPipeline()` fallback block. The local branch becomes:
```typescript
} else {
  if (KNOWN_CONTEXT_LENGTHS[LOCAL_MODEL]) {
    cachedContextLength = KNOWN_CONTEXT_LENGTHS[LOCAL_MODEL]!;
    return cachedContextLength;
  }
  // falls through to chunkContextFallback below
}
```

**Replace `embedLocal()` function body entirely:**
```typescript
async function embedLocal(
  texts: string[],
  type: 'query' | 'document',
): Promise<(Float32Array | null)[]> {
  const { llamaEmbed } = await import('./llama-backend.js');
  return llamaEmbed(texts, type);
}
```
(Dynamic import: keeps node-llama-cpp lazy, avoids module-level side effects at embedder import time.)

**Also remove** the `Xenova/` prefix entries from `KNOWN_CONTEXT_LENGTHS` (lines 110–116):
```typescript
// Remove these lines:
'Xenova/multilingual-e5-small': 512,
'Xenova/multilingual-e5-base': 512,
'Xenova/nomic-embed-text-v1.5': 8192,
'Xenova/all-MiniLM-L6-v2': 256,
'Xenova/all-MiniLM-L12-v2': 256,
'Xenova/bge-small-en-v1.5': 512,
```
These were only relevant when using Xenova models. Remove the entire `// ── Xenova-prefix models` section.

- [ ] **Step 2: Build**

```bash
npm run build 2>&1
```

Expected: No TypeScript errors. Fix any that appear.

- [ ] **Step 3: Run all tests**

```bash
npm test
```

Expected: All tests pass. The existing embedder unit tests mock `embed()` at a higher level and remain unaffected.

- [ ] **Step 4: Check lint and knip**

```bash
npm run lint && npm run knip
```

Expected: 0 errors. If knip flags any dead code, remove it.

- [ ] **Step 5: Commit**

```bash
git add src/embedder.ts
git commit -m "feat: migrate embedder to BGE-M3 via llama-backend, remove Xenova code"
```

---

### Task 6: Update src/reranker.ts

**Files:**
- Modify: `src/reranker.ts`

- [ ] **Step 1: Replace the entire contents of reranker.ts**

```typescript
import { llamaRerank } from './llama-backend.js';

export interface RerankCandidate {
  title: string;
  chunkText?: string;
  snippet: string;
}

export class CrossEncoderReranker {
  constructor(public readonly modelName: string) {}

  /**
   * Score all candidates against the query.
   * Returns scores in the same order as the input (NOT reordered).
   * Caller is responsible for sorting and slicing.
   * Returns all-zeros on llamaRerank error (graceful degradation).
   */
  async scoreAll(query: string, candidates: RerankCandidate[]): Promise<number[]> {
    if (candidates.length === 0) return [];
    try {
      return await llamaRerank(query, candidates);
    } catch (err) {
      process.stderr.write(
        `Reranking failed: ${err instanceof Error ? err.message : String(err)}. Returning original order.\n`,
      );
      return candidates.map(() => 0);
    }
  }
}

/** Module-level singleton — imported by searcher.ts */
export const reranker = new CrossEncoderReranker('BAAI/bge-reranker-v2-m3');
```

- [ ] **Step 2: Build**

```bash
npm run build 2>&1
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/reranker.ts
git commit -m "feat: migrate reranker to GPU-accelerated llama-backend"
```

---

### Task 7: Remove RERANKER_MODEL from config.ts

**Files:**
- Modify: `src/config.ts`

- [ ] **Step 1: Delete the rerankerModel getter**

In `src/config.ts`, delete these three lines:
```typescript
  get rerankerModel(): string {
    return process.env.RERANKER_MODEL ?? 'onnx-community/bge-reranker-v2-m3-ONNX';
  },
```

- [ ] **Step 2: Build and verify no remaining references**

```bash
npm run build 2>&1 && npm run knip
```

Expected: No errors, no knip issues. `config.rerankerModel` is no longer referenced anywhere (the old `reranker.ts` used it at the singleton call site, but that's been replaced with the hardcoded string in Task 6).

- [ ] **Step 3: Commit**

```bash
git add src/config.ts
git commit -m "chore: remove RERANKER_MODEL config — reranker model is now hardcoded in llama-backend"
```

---

### Task 8: Update test/reranker.test.ts

**Files:**
- Modify: `test/reranker.test.ts`

The old tests injected a mock `pipeline` field directly into the class instance. That field no longer exists. We now mock `llamaRerank` at the module level. The formatting tests (chunkText vs snippet, title format) have moved to `llama-backend.test.ts` — they are not duplicated here.

- [ ] **Step 1: Rewrite reranker.test.ts**

```typescript
import assert from 'node:assert/strict';
import { beforeEach, describe, it, vi } from 'vitest';

process.env.OBSIDIAN_VAULT_PATH = '/tmp/ohs-reranker-test';

// Mock llamaRerank so CrossEncoderReranker never touches node-llama-cpp
vi.mock('../src/llama-backend.js', () => ({
  llamaEmbed: vi.fn(),
  llamaRerank: vi.fn(),
}));

const { CrossEncoderReranker } = await import('../src/reranker.js');
const { llamaRerank } = await import('../src/llama-backend.js');
const mockLlamaRerank = vi.mocked(llamaRerank);

describe('CrossEncoderReranker.scoreAll', () => {
  beforeEach(() => {
    mockLlamaRerank.mockClear();
  });
  it('delegates to llamaRerank and returns scores in input order', async () => {
    mockLlamaRerank.mockResolvedValueOnce([0.9, 0.5, 0.1]);
    const r = new CrossEncoderReranker('test-model');
    const candidates = [
      { title: 'A', chunkText: 'a', snippet: '' },
      { title: 'B', chunkText: 'b', snippet: '' },
      { title: 'C', chunkText: 'c', snippet: '' },
    ];
    const scores = await r.scoreAll('query', candidates);
    assert.deepEqual(scores, [0.9, 0.5, 0.1]);
  });

  it('returns all-zeros when llamaRerank throws (graceful degradation)', async () => {
    mockLlamaRerank.mockRejectedValueOnce(new Error('reranker exploded'));
    const r = new CrossEncoderReranker('test-model');
    const candidates = [
      { title: 'A', chunkText: 'x', snippet: '' },
      { title: 'B', chunkText: 'y', snippet: '' },
    ];
    const scores = await r.scoreAll('q', candidates);
    assert.strictEqual(scores.length, 2);
    assert.strictEqual(scores[0], 0);
    assert.strictEqual(scores[1], 0);
  });

  it('returns empty array for empty candidates without calling llamaRerank', async () => {
    const r = new CrossEncoderReranker('test-model');
    const scores = await r.scoreAll('q', []);
    assert.deepEqual(scores, []);
    assert.strictEqual(mockLlamaRerank.mock.calls.length, 0);
  });
});
```

- [ ] **Step 2: Run the reranker tests**

```bash
npm test -- --reporter=verbose test/reranker.test.ts
```

Expected: All 3 tests pass.

- [ ] **Step 3: Run the full test suite**

```bash
npm test
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add test/reranker.test.ts
git commit -m "test: rewrite reranker tests — mock llamaRerank instead of injecting pipeline"
```

---

### Task 9: Increase integration test timeout

**Files:**
- Modify: `vitest.integration.config.ts`

- [ ] **Step 1: Update timeout**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/integration.test.ts'],
    testTimeout: 600_000, // 10 minutes — covers BGE-M3 first-run download (~370 MB)
  },
});
```

- [ ] **Step 2: Commit**

```bash
git add vitest.integration.config.ts
git commit -m "test: increase integration test timeout to 10 min for BGE-M3 cold-start download"
```

---

### Task 10: Full verification pass

- [ ] **Step 1: Run the full check suite**

```bash
npm run format && npm run build && npm test && npm run lint && npm run knip
```

Expected: All pass with 0 errors. Fix any issues before continuing.

- [ ] **Step 2: Run integration tests (validates full local GPU path)**

```bash
unset OPENAI_API_KEY
unset OPENAI_BASE_URL
npm run test:integration
```

Expected: BGE-M3 loads from cache (or downloads on first run). All integration tests pass. This is the end-to-end validation of the GPU local path.

---

### Task 11: Eval — measure ranking quality

- [ ] **Step 1: Run eval with new BGE-M3 model**

```bash
npm run eval -- --vault fixtures/obsidian-help/en --output eval/results/after-gpu-migration.json
```

- [ ] **Step 2: Compare against pre-migration baseline**

```bash
npm run eval:compare -- eval/results/before-gpu-migration.json eval/results/after-gpu-migration.json
```

Expected: nDCG@5 should be higher than 0.780. BGE-M3 is significantly better than multilingual-e5-small.

- [ ] **Step 3: Update committed baseline (only if metrics improved)**

If the new nDCG@5 is higher than the current floor in `test/eval/regression.test.ts`:

```bash
cp eval/results/after-gpu-migration.json eval/results/baseline-no-rerank.json
```

Then open `test/eval/regression.test.ts` and raise (never lower) the `FLOOR` constants to match the new run. Update the "Measured baseline" comment to show the new nDCG@5 value.

- [ ] **Step 4: Verify regression tests pass with new floors**

```bash
npm test -- --reporter=verbose test/eval/regression.test.ts
```

- [ ] **Step 5: Commit updated baseline**

```bash
git add eval/results/baseline-no-rerank.json test/eval/regression.test.ts
git commit -m "test: update eval baseline — BGE-M3 nDCG@5 = <insert actual value>"
```

---

### Task 12: Update docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (check for Xenova/e5-small references)

- [ ] **Step 1: Search for stale references**

```bash
grep -rn "Xenova\|e5-small\|multilingual-e5\|RERANKER_MODEL\|@huggingface/transformers" CLAUDE.md README.md
```

- [ ] **Step 2: Update CLAUDE.md**

In the Environment Variables table:
- Remove the `RERANKER_MODEL` row entirely
- Update any mention of `Xenova/multilingual-e5-small` → `BAAI/bge-m3`
- Update the `RERANKER_MODEL` default note in any architecture section

In the "Testing the Local Embedding Model" section:
- Update model name reference
- Note that first run downloads ~370 MB for BGE-M3 and ~320 MB for BGE-reranker-v2-m3

- [ ] **Step 3: Update README.md (if applicable)**

Fix any stale model names, env var references, or system requirements.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update model references for BGE-M3 GPU migration"
```

---

## Final sanity check

```bash
git log --oneline feat/gpu-local-inference ^master
npm run format && npm run build && npm test && npm run lint && npm run knip
```
