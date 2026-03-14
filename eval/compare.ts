/**
 * eval/compare.ts — A/B comparison of two eval result JSON files.
 *
 * Usage:
 *   npm run eval:compare -- eval/results/baseline.json eval/results/after-s9.json
 */

import fs from 'node:fs';
import path from 'node:path';

interface EvalMeta {
  date: string;
  ohs_version: string;
  model: string;
  vault: string;
  note_count: number;
  golden_set: string;
  golden_set_size: number;
  k: number;
}

interface AggregatedMetrics {
  ndcg_5: number;
  ndcg_k: number;
  mrr: number;
  hit_1: number;
  hit_3: number;
  hit_5: number;
  recall_k: number;
}

interface EvalResult {
  meta: EvalMeta;
  summary: AggregatedMetrics;
  by_category: Record<string, AggregatedMetrics>;
  per_query: {
    id: string;
    query: string;
    category: string;
    ndcg_5: number;
  }[];
}

function loadResult(filePath: string): EvalResult {
  const absPath = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
  if (!fs.existsSync(absPath)) {
    console.error(`ERROR: file not found: ${absPath}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(absPath, 'utf-8')) as EvalResult;
}

function fmt(n: number): string {
  return n.toFixed(3);
}

function delta(a: number, b: number): string {
  const d = b - a;
  const sign = d >= 0 ? '+' : '';
  const marker = Math.abs(d) >= 0.01 ? ' ✓' : '';
  return `${sign}${d.toFixed(3)}${marker}`;
}

function pad(s: string, width: number): string {
  return s.padEnd(width);
}

function row(label: string, a: number, b: number, labelWidth = 12): string {
  return `${pad(label, labelWidth)}   ${fmt(a)}      ${fmt(b)}      ${delta(a, b)}`;
}

function main(): void {
  const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
  if (args.length < 2) {
    console.error('Usage: npm run eval:compare -- <baseline.json> <after.json>');
    process.exit(1);
  }

  const [fileA, fileB] = args as [string, string];
  const a = loadResult(fileA);
  const b = loadResult(fileB);

  const labelA = `baseline (${a.meta.date.slice(0, 10)})`;
  const labelB = path.basename(fileB).replace('.json', '') + ` (${b.meta.date.slice(0, 10)})`;

  console.log(`\nComparing: ${labelA} vs ${labelB}`);

  if (a.meta.model !== b.meta.model) {
    console.log(`⚠  Model differs: ${a.meta.model} vs ${b.meta.model}`);
  }
  if (a.meta.vault !== b.meta.vault) {
    console.log(`⚠  Vault differs: ${a.meta.vault} vs ${b.meta.vault}`);
  }
  if (a.meta.golden_set !== b.meta.golden_set) {
    console.log(`⚠  Golden set differs: ${a.meta.golden_set} vs ${b.meta.golden_set}`);
  }

  console.log();
  console.log(`Metric        Baseline   After      Delta`);
  console.log(`${'─'.repeat(50)}`);
  console.log(row('nDCG@5', a.summary.ndcg_5, b.summary.ndcg_5));
  console.log(row(`nDCG@${a.meta.k}`, a.summary.ndcg_k, b.summary.ndcg_k));
  console.log(row('MRR', a.summary.mrr, b.summary.mrr));
  console.log(row('Hit@1', a.summary.hit_1, b.summary.hit_1));
  console.log(row('Hit@3', a.summary.hit_3, b.summary.hit_3));
  console.log(row('Hit@5', a.summary.hit_5, b.summary.hit_5));
  console.log(row(`Recall@${a.meta.k}`, a.summary.recall_k, b.summary.recall_k));

  const allCategories = new Set([...Object.keys(a.by_category), ...Object.keys(b.by_category)]);

  if (allCategories.size > 0) {
    console.log();
    console.log('By category (nDCG@5):');
    for (const cat of allCategories) {
      const catA = a.by_category[cat]?.ndcg_5 ?? 0;
      const catB = b.by_category[cat]?.ndcg_5 ?? 0;
      const d = catB - catA;
      const sign = d >= 0 ? '+' : '';
      const marker = Math.abs(d) >= 0.01 ? '  ✓ biggest gain' : '';
      console.log(`  ${pad(cat, 14)} ${fmt(catA)} → ${fmt(catB)}  ${sign}${d.toFixed(3)}${marker}`);
    }
  }

  console.log();
}

main();
