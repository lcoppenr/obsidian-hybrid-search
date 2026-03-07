#!/usr/bin/env node
import { Command } from 'commander'
import Table from 'cli-table3'
import path from 'node:path'
import { openDb, initVecTable, getStats } from './db.js'
import { getContextLength, getEmbeddingDim } from './embedder.js'
import { search } from './searcher.js'
import { indexVaultSync, indexFile } from './indexer.js'
import { config } from './config.js'

async function init() {
  openDb()
  const [contextLength, embeddingDim] = await Promise.all([
    getContextLength(),
    getEmbeddingDim(),
  ])
  initVecTable(embeddingDim)
  return contextLength
}

const program = new Command()
  .name('obsidian-hybrid-search')
  .description('Hybrid search for your Obsidian vault')
  .version('0.1.0')

program
  .command('search [query]', { isDefault: true })
  .description('Search the vault (default command)')
  .option('--mode <mode>', 'Search mode: hybrid|semantic|fulltext|title', 'hybrid')
  .option('--scope <scope>', 'Limit to a subfolder, e.g. notes/pkm/')
  .option('--limit <n>', 'Maximum results', '10')
  .option('--threshold <n>', 'Minimum score threshold 0..1', '0')
  .option('--json', 'Output as JSON')
  .action(async (query: string | undefined, opts) => {
    if (!query) {
      program.help()
      return
    }

    await init()

    const results = await search(query, {
      mode: opts.mode,
      scope: opts.scope,
      limit: parseInt(opts.limit),
      threshold: parseFloat(opts.threshold),
    })

    if (opts.json) {
      console.log(JSON.stringify(results, null, 2))
      return
    }

    if (results.length === 0) {
      console.log('No results found.')
      return
    }

    const table = new Table({
      head: ['SCORE', 'PATH', 'SNIPPET'],
      colWidths: [7, 45, 60],
      wordWrap: true,
    })

    for (const r of results) {
      table.push([r.score.toFixed(2), r.path, (r.snippet ?? '').slice(0, 120)])
    }

    console.log(table.toString())
  })

program
  .command('reindex [path]')
  .description('Reindex the vault or a specific file')
  .option('--force', 'Force reindex even if unchanged')
  .action(async (filePath: string | undefined, opts) => {
    const contextLength = await init()

    if (filePath) {
      const fullPath = path.join(config.vaultPath, filePath)
      const status = await indexFile(fullPath, contextLength, opts.force)
      console.log(JSON.stringify(
        status === 'indexed'
          ? { indexed: 1, skipped: 0, errors: [] }
          : status === 'skipped'
            ? { indexed: 0, skipped: 1, errors: [] }
            : { indexed: 0, skipped: 0, errors: [{ path: filePath, error: 'indexing failed' }] },
        null, 2
      ))
    } else {
      console.error('Indexing vault...')
      const result = await indexVaultSync(opts.force)
      console.log(JSON.stringify(result, null, 2))
    }
  })

program
  .command('status')
  .description('Show indexing status')
  .action(async () => {
    const contextLength = await init()
    const stats = getStats()
    console.log(JSON.stringify({
      total: stats.total,
      indexed: stats.indexed,
      pending: stats.pending,
      last_indexed: stats.lastIndexed,
      model: config.apiKey ? config.apiModel : 'Xenova/all-MiniLM-L6-v2 (local)',
      context_length: contextLength,
    }, null, 2))
  })

program.parseAsync(process.argv).catch(err => {
  console.error('Error:', err.message)
  process.exit(1)
})
