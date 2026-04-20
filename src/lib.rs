//! # plato-room-search
//!
//! Full-text search for PLATO rooms. Inverted index with BM25 ranking, fuzzy matching,
//! and field-scoped queries.
//!
//! ## Why Rust
//!
//! Search is latency-critical. Python's regex + dict overhead adds 10-50ms per query
//! on large indexes. Rust gives us:
//! - Zero-copy string slicing for tokenization
//! - HashMap with SipHash for O(1) term lookups
//! - Rayon-ready for parallel index builds
//! - WASM compilation for client-side search
//!
//! ## Alternatives
//!
//! - **Tantivy**: Full Lucene-equivalent in Rust. Overkill for room-scale (<100K docs).
//!   We'd add tantivy if we need faceting, phrase queries, or geosearch.
//!
//! - **Elasticsearch**: Network hop + JSON parse adds 5-20ms latency. Only worth it
//!   for cross-fleet federated search (>10M docs).
//!
//! - **SQLite FTS5**: Good option if we want persistent search. Could add as a backend.
//!
//! - **Python (current)**: Fine for prototyping. Breaks down at >50K tiles per room.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A searchable document (tile or room description).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub room: String,
    pub field: String,     // "title", "body", "tag"
    pub weight: f64,       // document-level boost
}

/// A search result with relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub score: f64,
    pub room: String,
    pub field: String,
    pub matches: Vec<String>,
}

/// Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub min_term_length: usize,
    pub max_results: usize,
    pub fuzzy_distance: usize,
    pub bm25_k1: f64,
    pub bm25_b: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { min_term_length: 2, max_results: 50, fuzzy_distance: 1,
               bm25_k1: 1.2, bm25_b: 0.75 }
    }
}

/// Term posting: which documents contain this term and where.
#[derive(Debug, Clone, Default)]
struct Posting {
    doc_ids: Vec<usize>,   // indices into the document list
    positions: Vec<Vec<usize>>,  // character positions within each doc
    tf: HashMap<usize, usize>,  // term frequency per doc
}

/// The search engine.
pub struct RoomSearch {
    config: SearchConfig,
    documents: Vec<Document>,
    index: HashMap<String, Posting>,  // term → posting list
    doc_lengths: Vec<usize>,
    avg_doc_length: f64,
    total_docs: usize,
    /// term → document frequency (how many docs contain this term)
    df: HashMap<String, usize>,
}

impl RoomSearch {
    pub fn new(config: SearchConfig) -> Self {
        Self { config, documents: Vec::new(), index: HashMap::new(),
               doc_lengths: Vec::new(), avg_doc_length: 0.0, total_docs: 0, df: HashMap::new() }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc: Document) {
        let doc_idx = self.documents.len();
        let tokens = tokenize(&doc.content, self.config.min_term_length);
        self.doc_lengths.push(tokens.len());

        let mut seen_terms: HashSet<String> = HashSet::new();
        for (pos, token) in tokens.iter().enumerate() {
            let posting = self.index.entry(token.clone()).or_default();
            if !posting.doc_ids.contains(&doc_idx) {
                posting.doc_ids.push(doc_idx);
                posting.positions.push(Vec::new());
            }
            let idx = posting.doc_ids.iter().position(|&i| i == doc_idx).unwrap();
            posting.positions[idx].push(pos);
            *posting.tf.entry(doc_idx).or_insert(0) += 1;
            seen_terms.insert(token.clone());
        }

        // Update document frequency
        for term in seen_terms {
            *self.df.entry(term).or_insert(0) += 1;
        }

        self.documents.push(doc);
        self.total_docs = self.documents.len();
        let total_len: usize = self.doc_lengths.iter().sum();
        self.avg_doc_length = total_len as f64 / self.total_docs.max(1) as f64;
    }

    /// Add multiple documents (batch).
    pub fn add_documents(&mut self, docs: Vec<Document>) {
        for doc in docs {
            self.add_document(doc);
        }
    }

    /// Search the index. Returns results sorted by BM25 score.
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let terms = tokenize(query, self.config.min_term_length);
        if terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f64> = HashMap::new();
        let mut matches: HashMap<usize, Vec<String>> = HashMap::new();

        for term in &terms {
            // Exact match
            if let Some(posting) = self.index.get(term) {
                self.bm25_score(term, posting, &mut scores);
                for &doc_idx in &posting.doc_ids {
                    matches.entry(doc_idx).or_default().push(term.clone());
                }
            }
            // Fuzzy match
            if self.config.fuzzy_distance > 0 {
                for (idx_term, posting) in &self.index {
                    if levenshtein(term, idx_term) <= self.config.fuzzy_distance {
                        self.bm25_score(term, posting, &mut scores);
                        for &doc_idx in &posting.doc_ids {
                            matches.entry(doc_idx).or_default().push(idx_term.clone());
                        }
                    }
                }
            }
        }

        let mut results: Vec<SearchResult> = scores.into_iter().map(|(doc_idx, score)| {
            let doc = &self.documents[doc_idx];
            SearchResult { doc_id: doc.id.clone(), score, room: doc.room.clone(),
                         field: doc.field.clone(), matches: matches.remove(&doc_idx).unwrap_or_default() }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.max_results);
        results
    }

    /// Search within a specific room.
    pub fn search_room(&self, query: &str, room: &str) -> Vec<SearchResult> {
        self.search(query).into_iter().filter(|r| r.room == room).collect()
    }

    /// Search a specific field.
    pub fn search_field(&self, query: &str, field: &str) -> Vec<SearchResult> {
        self.search(query).into_iter().filter(|r| r.field == field).collect()
    }

    /// Get all terms in the index (useful for autocomplete).
    pub fn terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = self.index.keys().cloned().collect();
        terms.sort();
        terms
    }

    /// Autocomplete: prefix matching on index terms.
    pub fn autocomplete(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.index.keys()
            .filter(|t| t.starts_with(prefix))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Document frequency for a term.
    pub fn doc_freq(&self, term: &str) -> usize {
        *self.df.get(term).unwrap_or(&0)
    }

    /// Index statistics.
    pub fn stats(&self) -> SearchStats {
        SearchStats { documents: self.total_docs, terms: self.index.len(),
                     avg_doc_length: self.avg_doc_length,
                     index_size_bytes: std::mem::size_of_val(&self.index) }
    }

    fn bm25_score(&self, term: &str, posting: &Posting, scores: &mut HashMap<usize, f64>) {
        let n = self.total_docs as f64;
        let df = posting.doc_ids.len() as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;

        for (&doc_idx, &tf) in &posting.tf {
            let dl = self.doc_lengths[doc_idx] as f64;
            let tf_norm = (tf as f64 * (k1 + 1.0)) / (tf as f64 + k1 * (1.0 - b + b * dl / self.avg_doc_length.max(0.1)));
            let score = idf * tf_norm;
            *scores.entry(doc_idx).or_insert(0.0) += score;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    pub documents: usize,
    pub terms: usize,
    pub avg_doc_length: f64,
    pub index_size_bytes: usize,
}

/// Simple tokenizer: lowercase, split on non-alphanumeric, filter short terms.
fn tokenize(text: &str, min_len: usize) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= min_len)
        .map(String::from)
        .collect()
}

/// Levenshtein distance for fuzzy matching.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut dp = vec![vec![0; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i-1] == b[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j] + 1).min(dp[i][j-1] + 1).min(dp[i-1][j-1] + cost);
        }
    }
    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, content: &str, room: &str) -> Document {
        Document { id: id.into(), content: content.into(), room: room.into(),
                   field: "body".into(), weight: 1.0 }
    }

    #[test]
    fn test_basic_search() {
        let mut engine = RoomSearch::new(SearchConfig::default());
        engine.add_document(make_doc("1", "constraint theory snaps vectors to exact coordinates", "ct-lab"));
        engine.add_document(make_doc("2", "tiles are the atomic unit of knowledge", "plato"));
        engine.add_document(make_doc("3", "the forge trains models on GPU hardware", "forge"));
        let results = engine.search("constraint theory");
        assert_eq!(results[0].doc_id, "1");
    }

    #[test]
    fn test_room_scoped() {
        let mut engine = RoomSearch::new(SearchConfig::default());
        engine.add_document(make_doc("1", "hello world", "room-a"));
        engine.add_document(make_doc("2", "hello again", "room-b"));
        let results = engine.search_room("hello", "room-a");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "1");
    }

    #[test]
    fn test_fuzzy() {
        let mut config = SearchConfig::default();
        config.fuzzy_distance = 2;
        let mut engine = RoomSearch::new(config);
        engine.add_document(make_doc("1", "constraint theory is about exact geometry", "ct"));
        let results = engine.search("constriant");  // typo
        assert!(!results.is_empty());
    }

    #[test]
    fn test_autocomplete() {
        let mut engine = RoomSearch::new(SearchConfig::default());
        engine.add_document(make_doc("1", "tile tilemap tiles tiling", "test"));
        let completions = engine.autocomplete("tile", 10);
        assert!(completions.len() >= 2);
    }

    #[test]
    fn test_stats() {
        let mut engine = RoomSearch::new(SearchConfig::default());
        engine.add_document(make_doc("1", "hello world foo bar", "test"));
        engine.add_document(make_doc("2", "hello baz qux", "test"));
        let stats = engine.stats();
        assert_eq!(stats.documents, 2);
        assert!(stats.terms >= 4);
    }
}
