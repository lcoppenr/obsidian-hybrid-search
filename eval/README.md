# Eval System

Инструмент для измерения качества поиска: запускает golden set запросов против
проиндексированного хранилища и считает nDCG, MRR, Hit@k, Recall@k.

## Быстрый старт

```bash
# Запустить eval (первый раз скачивает локальную модель ~30 сек)
npm run eval -- \
  --vault fixtures/obsidian-help/en \
  --golden-set eval/golden-sets/obsidian-help.json \
  --output eval/results/baseline-$(date +%Y%m%d).json

# Сравнить два запуска A/B
npm run eval:compare -- eval/results/baseline.json eval/results/after-change.json
```

## Ориентиры по метрикам (из S-16)

Основная метрика — **nDCG@5** и **nDCG@10**.

| Конфигурация             | nDCG      | Статус            |
| ------------------------ | --------- | ----------------- |
| BM25-only baseline       | 0.45–0.55 | отправная точка   |
| Hybrid (BM25 + semantic) | 0.58–0.65 | хороший результат |
| Hybrid + cross-encoder   | 0.65–0.72 | цель после S-9    |

## Измеренный baseline

Хранилище: `fixtures/obsidian-help/en` (171 заметка)
Модель: `Xenova/multilingual-e5-small` (локальная, без API)
Golden set: `eval/golden-sets/obsidian-help.json` (20 запросов)

| Метрика   | Значение  | По категориям                                                        |
| --------- | --------- | -------------------------------------------------------------------- |
| nDCG@5    | **0.603** | keyword=0.714 / conceptual=0.352 / multilingual=0.580 / syntax=0.715 |
| nDCG@10   | 0.672     |                                                                      |
| MRR       | 0.688     |                                                                      |
| Hit@1     | 0.600     |                                                                      |
| Hit@3     | 0.750     |                                                                      |
| Hit@5     | 0.750     |                                                                      |
| Recall@10 | 0.900     |                                                                      |

nDCG@5=0.603 — в диапазоне «хороший hybrid», как и ожидалось.
Слабое место: **conceptual запросы** (0.352) — перефразированные запросы без ключевых слов.

## Структура файлов

```
eval/
├── metrics.ts                  # ndcg(), mrr(), hitAtK(), recallAtK()
├── evaluate.ts                 # индексирует vault + прогоняет golden set → JSON
├── compare.ts                  # читает два JSON → таблица дельт
├── golden-sets/
│   ├── obsidian-help.json      # 20 запросов против fixtures/obsidian-help/en
│   └── personal.json           # твой личный golden set (gitignored)
└── results/
    └── *.json                  # gitignored, создаются локально
```

## Формат golden set

```json
{
  "id": "q001",
  "query": "как делать atomic notes",
  "relevant_paths": ["notes/zettelkasten.md"],
  "partial_paths": ["notes/pkm/overview.md"],
  "category": "conceptual",
  "notes": "пользователь пишет 'atomic', заметка содержит 'атомарные'"
}
```

Категории: `keyword`, `conceptual`, `multilingual`, `syntax`.
Пути — относительно корня vault.

## Как читать вывод compare

```
Metric     Baseline   After      Delta
nDCG@5     0.603      0.648      +0.045  ✓   ← улучшение ≥0.01 помечается ✓
MRR        0.688      0.650      -0.038      ← регрессия
```

Изменение `|delta| ≥ 0.01` считается значимым при 20 запросах.
Для статистически уверенных выводов нужно 50+ запросов.

## Личный golden set

Создай `eval/golden-sets/personal.json` по тому же формату с запросами из
реальной практики (метод A из S-16: история реальных запросов → релевантные заметки).
Файл gitignored — не попадёт в репозиторий.
