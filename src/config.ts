import os from 'node:os';
import path from 'node:path';

export type UnreadableMode = 'keep' | 'purge';
export type HtmlMode = 'text' | 'raw';

function readEnum<T extends string>(name: string, allowed: readonly T[], fallback: T): T {
  const raw = process.env[name]?.trim().toLowerCase();
  if (raw === undefined || raw === '') return fallback;
  if ((allowed as readonly string[]).includes(raw)) return raw as T;
  throw new Error(`${name} must be one of ${allowed.join(', ')} (got "${process.env[name]}")`);
}

function readNumber(
  name: string,
  fallback: number,
  { min, max, integer = false }: { min: number; max?: number; integer?: boolean },
): number {
  const raw = process.env[name]?.trim();
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  const outOfRange = value < min || (max !== undefined && value > max);
  if (!Number.isFinite(value) || outOfRange || (integer && !Number.isInteger(value))) {
    const range = max === undefined ? `>= ${min}` : `between ${min} and ${max}`;
    throw new Error(
      `${name} must be ${integer ? 'an integer' : 'a number'} ${range} (got "${raw}")`,
    );
  }
  return value;
}

function expandHome(p: string): string {
  if (p === '~') return os.homedir();
  if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2));
  return p;
}

export const config = {
  get obsidianPrefix(): string {
    return process.env.OBSIDIAN_PREFIX ?? '';
  },
  get vaultPath(): string {
    const v = process.env.OBSIDIAN_VAULT_PATH;
    if (!v) throw new Error('OBSIDIAN_VAULT_PATH environment variable is required');
    return v;
  },
  get ignorePatterns(): string[] {
    return (process.env.OBSIDIAN_IGNORE_PATTERNS ?? '.obsidian/**,templates/**,*.canvas')
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean);
  },
  get includePatterns(): string[] {
    return (process.env.OBSIDIAN_INCLUDE_PATTERNS ?? '')
      .split(',')
      .map((p) => p.trim())
      .filter(Boolean);
  },
  get respectGitignore(): boolean {
    const raw = process.env.OBSIDIAN_RESPECT_GITIGNORE?.trim().toLowerCase();
    return raw !== 'false' && raw !== '0' && raw !== 'no';
  },
  get apiKey(): string | undefined {
    return process.env.OPENAI_API_KEY;
  },
  get apiBaseUrl(): string {
    return process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1';
  },
  get apiModel(): string {
    return process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small';
  },
  get localModel(): string {
    return process.env.LOCAL_EMBEDDING_MODEL ?? 'Xenova/multilingual-e5-small';
  },
  get rerankerModel(): string {
    return process.env.RERANKER_MODEL ?? 'onnx-community/bge-reranker-v2-m3-ONNX';
  },
  get dbPath(): string {
    const dbDir = process.env.OBSIDIAN_DB_DIR;
    if (dbDir) return path.join(dbDir, '.obsidian-hybrid-search.db');
    const v = process.env.OBSIDIAN_VAULT_PATH;
    if (!v) throw new Error('OBSIDIAN_VAULT_PATH environment variable is required');
    return path.join(v, '.obsidian-hybrid-search.db');
  },
  /** Vault root unreadable: `keep` serves the last index, `purge` is the legacy wipe. */
  get unreadableVault(): UnreadableMode {
    return readEnum('OBSIDIAN_UNREADABLE_VAULT', ['keep', 'purge'], 'keep');
  },
  /** A folder inside the vault failed to read: `keep` its notes, or `purge` them. */
  get unreadableSubtree(): UnreadableMode {
    return readEnum('OBSIDIAN_UNREADABLE_SUBTREE', ['keep', 'purge'], 'keep');
  },
  /** Skip the deletion pass when a scan finds fewer than this fraction of indexed notes. 0 = off. */
  get minScanRatio(): number {
    return readNumber('OBSIDIAN_MIN_SCAN_RATIO', 0.5, { min: 0, max: 1 });
  },
  /** Scheduled vault re-check interval in minutes. 0 = off. */
  get rescanMinutes(): number {
    return readNumber('OBSIDIAN_RESCAN_MINUTES', 0, { min: 0 });
  },
  /** File holding the bearer token required by `serve` over HTTP. Unset = no auth. */
  get mcpTokenFile(): string | undefined {
    const raw = process.env.OBSIDIAN_MCP_TOKEN_FILE?.trim();
    return raw ? expandHome(raw) : undefined;
  },
  /** Extra file extensions indexed as plain text (lowercase, no dot). `.md` is always indexed. */
  get textExtensions(): string[] {
    const raw = process.env.OBSIDIAN_TEXT_EXTENSIONS ?? '';
    const seen = new Set<string>();
    for (const part of raw.split(',')) {
      const ext = part.trim().toLowerCase().replace(/^\./, '');
      if (!ext) continue;
      if (!/^[a-z0-9_-]+$/.test(ext)) {
        throw new Error(`OBSIDIAN_TEXT_EXTENSIONS contains an invalid extension "${part.trim()}"`);
      }
      if (ext === 'md') {
        throw new Error('OBSIDIAN_TEXT_EXTENSIONS must not list md (Markdown is always indexed)');
      }
      seen.add(ext);
    }
    return [...seen].sort((a, b) => a.localeCompare(b));
  },
  /** HTML indexing: `text` strips tags, scripts and styles; `raw` indexes the markup. */
  get htmlMode(): HtmlMode {
    return readEnum('OBSIDIAN_HTML_MODE', ['text', 'raw'], 'text');
  },
  /** Text files larger than this (KiB) are skipped and logged. */
  get textMaxKb(): number {
    return readNumber('OBSIDIAN_TEXT_MAX_KB', 512, { min: 1 });
  },
  // internal defaults
  chunkContextFallback: 512,
  chunkOverlap: 64,
  chunkMinLength: 50,
  chunkHeadingLevel: 0,
  batchSize: 10,
  debounce: 5_000,
};

/**
 * Read every service setting once so an invalid value stops startup with a message
 * that names the setting, instead of surfacing later from a background task.
 */
export function validateServiceConfig(): void {
  effectiveServiceSettings();
}

/** The service settings as `status` reports them. Never includes the token itself. */
export function effectiveServiceSettings(): Record<string, unknown> {
  return {
    unreadable_vault: config.unreadableVault,
    unreadable_subtree: config.unreadableSubtree,
    min_scan_ratio: config.minScanRatio,
    rescan_minutes: config.rescanMinutes,
    mcp_token_file: config.mcpTokenFile ?? null,
    text_extensions: config.textExtensions,
    html_mode: config.htmlMode,
    text_max_kb: config.textMaxKb,
  };
}
