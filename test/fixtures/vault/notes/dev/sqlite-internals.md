---
tags:
  - dev
  - database
---

# SQLite Internals

SQLite stores data in a B-tree structure. Each table is a separate B-tree, and indexes are B-trees as well. Fixed-size pages (4096 bytes by default) make up the database file. WAL mode enables concurrent reads during writes. FTS5 implements an inverted index on top of regular SQLite tables, using BM25 for ranking results. The sqlite-vec extension adds support for float32 vectors and KNN operations via virtual tables.
