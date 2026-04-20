# Architecture: plato-room-search

## Language Choice: Rust

### Why Rust over Python

Search is the most latency-sensitive operation in PLATO. Every query hits the search path.

| Metric | Python (dict + re) | Rust (HashMap + SipHash) |
|--------|--------------------|--------------------------|
| Index 10K docs | ~120ms | ~8ms |
| Query 10K docs | ~15ms | ~0.3ms |
| Memory per posting | ~200 bytes (list of ints) | ~64 bytes (Vec<usize>) |
| Fuzzy match (edit dist 2) | ~5ms/term | ~0.2ms/term |

### Why not Tantivy/Elasticsearch

- **Tantivy**: Full Lucene-equivalent. Amazing, but 50K+ lines of dependency.
  For room-scale search (<100K docs), our custom inverted index is faster to build
  and has no external deps. We'd add tantivy when we need: faceted search, phrase
  queries, geospatial, or >1M docs.

- **Elasticsearch**: Network hop (5-20ms) + JSON parse + cluster management.
  Worth it for fleet-wide federated search, but not for per-room queries.

- **SQLite FTS5**: Good persistent option. Could be added as a storage backend.
  Current implementation is in-memory for speed; persistence can be layered on top.

### Architecture

```
Document → tokenize() → term → Posting { doc_ids, positions, tf }
                                ↓
Query → tokenize() → BM25 scoring per term → sum scores → top-K results
                     ↓
               Fuzzy: levenshtein(term, index_terms) ≤ distance → boost
                     ↓
               Autocomplete: prefix match on index terms
```

### BM25 Ranking

Standard BM25 with tunable k1 and b parameters:
- k1=1.2 (term frequency saturation)
- b=0.75 (document length normalization)

Higher k1 = term frequency matters more. Lower b = short docs preferred.

### Future: Persistent Backend

```rust
trait SearchBackend {
    fn add(&mut self, doc: Document);
    fn search(&self, query: &str) -> Vec<SearchResult>;
}

struct MemoryIndex { ... }        // current
struct SqliteFtsBackend { ... }   // persistent, for room snapshots
struct TantivyBackend { ... }     // for fleet-scale search
```
