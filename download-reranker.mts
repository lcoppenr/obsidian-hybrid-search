import { pipeline } from '@xenova/transformers';
console.log('Downloading Xenova/bge-reranker-large (~563MB)...');
await pipeline('text-classification', 'Xenova/bge-reranker-large');
console.log('Done!');
