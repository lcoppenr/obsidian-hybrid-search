import { describe, expect, it } from 'vitest';
import { hitAtK, mrr, ndcg, recallAtK } from '../../eval/metrics.js';

describe('ndcg()', () => {
  it('returns 1.0 when perfect ranking', () => {
    const results = ['a.md', 'b.md', 'c.md'];
    const relevant = ['a.md', 'b.md'];
    expect(ndcg(results, relevant, [], 5)).toBeCloseTo(1.0);
  });

  it('returns 0.0 when no relevant results in top-k', () => {
    const results = ['x.md', 'y.md', 'z.md'];
    const relevant = ['a.md'];
    expect(ndcg(results, relevant, [], 5)).toBe(0.0);
  });

  it('returns 0.0 when relevant set is empty', () => {
    expect(ndcg(['a.md'], [], [], 5)).toBe(0.0);
  });

  it('discounts lower-ranked results', () => {
    // first result relevant: higher score
    const scoreFirst = ndcg(['a.md', 'x.md'], ['a.md'], [], 5);
    // second result relevant: lower score
    const scoreSecond = ndcg(['x.md', 'a.md'], ['a.md'], [], 5);
    expect(scoreFirst).toBeGreaterThan(scoreSecond);
  });

  it('handles partial relevance (0.5 score)', () => {
    // p.md is partial, q.md is fully relevant but missing from results → nDCG < 1.0 > 0.0
    const results = ['p.md', 'x.md'];
    const score = ndcg(results, ['q.md'], ['p.md'], 5);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it('full hit at rank 1 scores higher than partial hit at rank 1 (with same ideal set)', () => {
    // When there are 2 relevant docs (a + b) and we return [a, x]:
    //   full:    a is fully relevant → higher DCG
    //   partial: a is partial, b is fully relevant but missing → lower DCG
    const full = ndcg(['a.md', 'x.md'], ['a.md', 'b.md'], [], 5);
    const partial = ndcg(['a.md', 'x.md'], ['b.md'], ['a.md'], 5);
    expect(full).toBeGreaterThan(partial);
  });

  it('respects k cutoff — result beyond k is ignored', () => {
    const results = ['x.md', 'x.md', 'x.md', 'x.md', 'x.md', 'a.md'];
    const relevant = ['a.md'];
    expect(ndcg(results, relevant, [], 5)).toBe(0.0);
  });
});

describe('mrr()', () => {
  it('returns 1.0 when first result is relevant', () => {
    expect(mrr(['a.md', 'b.md', 'c.md'], ['a.md'])).toBe(1.0);
  });

  it('returns 0.5 when second result is first relevant', () => {
    expect(mrr(['x.md', 'a.md', 'b.md'], ['a.md'])).toBe(0.5);
  });

  it('returns 1/3 when third result is first relevant', () => {
    expect(mrr(['x.md', 'y.md', 'a.md'], ['a.md'])).toBeCloseTo(1 / 3);
  });

  it('returns 0.0 when no relevant result found', () => {
    expect(mrr(['x.md', 'y.md'], ['a.md'])).toBe(0.0);
  });

  it('returns 0.0 for empty results', () => {
    expect(mrr([], ['a.md'])).toBe(0.0);
  });
});

describe('hitAtK()', () => {
  it('returns true when relevant doc is in top-3', () => {
    expect(hitAtK(['x.md', 'y.md', 'a.md'], ['a.md'], 3)).toBe(true);
  });

  it('returns false when relevant doc is outside top-3', () => {
    expect(hitAtK(['x.md', 'y.md', 'z.md', 'a.md'], ['a.md'], 3)).toBe(false);
  });

  it('returns false when no relevant docs', () => {
    expect(hitAtK(['x.md', 'y.md'], ['a.md'], 5)).toBe(false);
  });

  it('returns true for hit at exactly k', () => {
    expect(hitAtK(['x.md', 'y.md', 'a.md'], ['a.md'], 3)).toBe(true);
  });
});

describe('recallAtK()', () => {
  it('returns 1.0 when all relevant docs are in top-5', () => {
    const results = ['a.md', 'b.md', 'c.md', 'x.md', 'y.md'];
    const relevant = ['a.md', 'b.md', 'c.md'];
    expect(recallAtK(results, relevant, 5)).toBe(1.0);
  });

  it('returns 0.5 when half of relevant docs are in top-5', () => {
    const results = ['a.md', 'x.md', 'y.md', 'z.md', 'w.md'];
    const relevant = ['a.md', 'b.md'];
    expect(recallAtK(results, relevant, 5)).toBe(0.5);
  });

  it('returns 0.0 when no relevant docs in top-k', () => {
    expect(recallAtK(['x.md', 'y.md'], ['a.md', 'b.md'], 5)).toBe(0.0);
  });

  it('returns 0.0 when relevant set is empty', () => {
    expect(recallAtK(['a.md'], [], 5)).toBe(0.0);
  });

  it('respects k cutoff', () => {
    const results = ['a.md', 'b.md', 'c.md'];
    const relevant = ['a.md', 'b.md', 'c.md'];
    // only top-2 checked → recall = 2/3
    expect(recallAtK(results, relevant, 2)).toBeCloseTo(2 / 3);
  });
});
