import { AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';

const MODEL = 'onnx-community/bge-reranker-v2-m3-ONNX';
console.log(`Downloading ${MODEL} (~32MB int8 quantized)...`);

// Load tokenizer + model directly (same path as reranker.ts uses)
const [tokenizer, model] = await Promise.all([
  (AutoTokenizer as any).from_pretrained(MODEL),
  (AutoModelForSequenceClassification as any).from_pretrained(MODEL, { quantized: true }),
]);

// Quick smoke test
const query = 'what is obsidian?';
const doc = 'Obsidian is a note-taking app.';
const encoded = (tokenizer as any)([query], { text_pair: [doc], padding: true, truncation: true });
const { logits } = await (model as any)(encoded);
console.log('Smoke test logit:', logits.data[0]);
console.log('Done! Model cached in ~/.cache/');
