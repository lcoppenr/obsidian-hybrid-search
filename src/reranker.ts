import { config } from './config.js';

export interface RerankCandidate {
  title: string;
  chunkText?: string;
  snippet: string;
}

export class CrossEncoderReranker {
  private pipeline: ((inputs: unknown[], opts?: unknown) => Promise<unknown>) | null = null;
  private loadPromise: Promise<void> | null = null;

  constructor(public readonly modelName: string) {}

  /**
   * Score all candidates against the query.
   * Returns scores in the same order as the input (NOT reordered).
   * Caller is responsible for sorting and slicing.
   * Returns all-zeros on pipeline error (graceful degradation).
   */
  async scoreAll(query: string, candidates: RerankCandidate[]): Promise<number[]> {
    if (candidates.length === 0) return [];

    // Distinguish load failure from scoring failure — different user-facing messages
    try {
      await this.ensureLoaded();
    } catch (loadErr) {
      process.stderr.write(
        `Reranker model failed to load: ${loadErr instanceof Error ? loadErr.message : String(loadErr)}. Falling back to hybrid results.\n`,
      );
      return candidates.map(() => 0);
    }

    try {
      const inputs = candidates.map((c) => ({
        text: query,
        text_pair: `${c.title}\n\n${c.chunkText ?? c.snippet}`,
      }));

      // @xenova/transformers text-classification batch output:
      // Array<Array<{label: string; score: number}>> — one array of labels per input
      // BGE reranker labels: LABEL_0 = not relevant, LABEL_1 = relevant
      // Do not cast to a concrete type — noUncheckedIndexedAccess must stay active
      const outputs: Array<Array<{ label: string; score: number }> | undefined> = (await (
        this.pipeline as (i: unknown[], o?: unknown) => Promise<unknown>
      )(inputs, {
        truncation: true,
      })) as Array<Array<{ label: string; score: number }> | undefined>;

      return candidates.map((_, i) => outputs[i]?.find((x) => x.label === 'LABEL_1')?.score ?? 0);
    } catch (err) {
      process.stderr.write(
        `Reranking failed: ${err instanceof Error ? err.message : String(err)}. Returning original order.\n`,
      );
      return candidates.map(() => 0);
    }
  }

  private async ensureLoaded(): Promise<void> {
    if (this.pipeline) return;
    if (!this.loadPromise) {
      // Assign loadPromise BEFORE awaiting — prevents race where two concurrent
      // callers both see loadPromise === null and load the model twice.
      process.stderr.write(`Loading reranker model ${this.modelName}, please wait...\n`);
      this.loadPromise = this._loadModel().then((p) => {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- _loadModel returns any (xenova has no types)
        this.pipeline = p;
      });
    }
    await this.loadPromise;
  }

  /** Separated for testability — tests can override _loadModel to count invocations. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- @xenova/transformers has no types
  protected async _loadModel(): Promise<any> {
    const { pipeline } = await import('@xenova/transformers');
    return pipeline('text-classification', this.modelName);
  }
}

/** Module-level singleton — imported by searcher.ts */
export const reranker = new CrossEncoderReranker(config.rerankerModel);
