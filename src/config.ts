import path from 'node:path'

if (!process.env.VAULT_PATH) {
  throw new Error('VAULT_PATH environment variable is required')
}

export const config = {
  vaultPath: process.env.VAULT_PATH,
  ignorePatterns: (process.env.IGNORE_PATTERNS ?? '.obsidian/**,templates/**,*.canvas').split(','),
  apiKey: process.env.API_KEY,
  apiBaseUrl: process.env.API_BASE_URL ?? 'https://openrouter.ai/api/v1',
  apiModel: process.env.API_EMBEDDING_MODEL ?? 'openai/text-embedding-3-small',
  dbPath: path.join(process.env.VAULT_PATH, '.obsidian-hybrid-search.db'),
  // internal defaults
  chunkContextFallback: 512,
  chunkOverlap: 64,
  chunkMinLength: 50,
  chunkHeadingLevel: 0,
  batchSize: 10,
  debounce: 30_000,
} as const
