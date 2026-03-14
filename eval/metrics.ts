/**
 * Pure metric functions for eval (nDCG, MRR, Hit@k, Recall@k).
 * No side effects, no DB dependency.
 */

/**
 * Discounted Cumulative Gain at k.
 * relevanceScores[i] is the relevance of the i-th result (0-based).
 * Relevance: 1.0 = fully relevant, 0.5 = partial, 0.0 = irrelevant.
 */
function dcgAtK(relevanceScores: number[], k: number): number {
  let dcg = 0;
  const limit = Math.min(k, relevanceScores.length);
  for (let i = 0; i < limit; i++) {
    const rel = relevanceScores[i] ?? 0;
    dcg += rel / Math.log2(i + 2); // i+2 because log2(1) = 0 and i is 0-based
  }
  return dcg;
}

/**
 * Normalized Discounted Cumulative Gain at k.
 * Normalized by the ideal DCG (perfect ranking of relevantPaths + partialPaths).
 *
 * @param resultPaths - Ordered list of returned note paths
 * @param relevantPaths - Paths that are fully relevant (rel=1.0)
 * @param partialPaths - Paths that are partially relevant (rel=0.5)
 * @param k - Cutoff rank
 */
export function ndcg(
  resultPaths: string[],
  relevantPaths: string[],
  partialPaths: string[],
  k: number,
): number {
  const relevantSet = new Set(relevantPaths);
  const partialSet = new Set(partialPaths);

  const actualScores = resultPaths.slice(0, k).map((p) => {
    if (relevantSet.has(p)) return 1.0;
    if (partialSet.has(p)) return 0.5;
    return 0.0;
  });

  const idealScores: number[] = [
    ...Array<number>(relevantPaths.length).fill(1.0),
    ...Array<number>(partialPaths.length).fill(0.5),
  ].slice(0, k);

  const idealDcg = dcgAtK(idealScores, k);
  if (idealDcg === 0) return 0;

  return dcgAtK(actualScores, k) / idealDcg;
}

/**
 * Mean Reciprocal Rank.
 * Returns 1/rank of the first fully-relevant result (1-based), or 0 if none found.
 */
export function mrr(resultPaths: string[], relevantPaths: string[]): number {
  const relevantSet = new Set(relevantPaths);
  for (let i = 0; i < resultPaths.length; i++) {
    if (relevantSet.has(resultPaths[i] ?? '')) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Hit@k: returns true if any of the top-k results is in relevantPaths.
 */
export function hitAtK(resultPaths: string[], relevantPaths: string[], k: number): boolean {
  const relevantSet = new Set(relevantPaths);
  return resultPaths.slice(0, k).some((p) => relevantSet.has(p));
}

/**
 * Recall@k: fraction of relevantPaths found in the top-k results.
 */
export function recallAtK(resultPaths: string[], relevantPaths: string[], k: number): number {
  if (relevantPaths.length === 0) return 0;
  const relevantSet = new Set(relevantPaths);
  const found = resultPaths.slice(0, k).filter((p) => relevantSet.has(p)).length;
  return found / relevantPaths.length;
}
