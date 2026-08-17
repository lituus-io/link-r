// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The BM25 sparse index — the lexical half of hybrid retrieval.
//!
//! Captures exact terms (a resource name, `WIF`, `PSC`) that dense embeddings
//! blur. Built fresh from per-document token lists (so incremental upserts never
//! have to mutate an inverted index in place), serialized with roaring posting
//! lists, and decoded zero-allocation-per-query at search time.

use crate::error::{Error, Result};
use crate::index::bytesio::{put_f32, put_len_prefixed, put_u16, put_u32, Reader};
use crate::resource::DocId;
use compact_str::CompactString;
use roaring::RoaringBitmap;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Standard BM25 term-saturation parameter.
pub const DEFAULT_K1: f32 = 1.2;
/// Standard BM25 length-normalization parameter.
pub const DEFAULT_B: f32 = 0.75;
/// Corpus size below which the `parallel` build path isn't worth rayon's overhead.
#[cfg(feature = "parallel")]
const PAR_MIN_DOCS: usize = 512;

/// One term's posting list: the doc set plus aligned term frequencies.
#[derive(Debug)]
struct TermPosting {
    term: CompactString,
    docs: RoaringBitmap,
    /// Term frequencies, in the same ascending doc order roaring iterates.
    tf: Vec<u16>,
}

/// A queryable, serializable BM25 index.
#[derive(Debug)]
pub struct Bm25 {
    k1: f32,
    b: f32,
    avgdl: f32,
    doc_count: u32,
    doc_len: Vec<u32>,
    terms: Vec<TermPosting>,
    term_index: HashMap<CompactString, usize>,
}

/// Count the term frequencies within one document.
fn local_counts(terms: &[CompactString]) -> HashMap<CompactString, u16> {
    let mut counts: HashMap<CompactString, u16> = HashMap::with_capacity(terms.len());
    for term in terms {
        let entry = counts.entry(term.clone()).or_insert(0);
        *entry = entry.saturating_add(1);
    }
    counts
}

/// Accumulate `term -> (doc -> tf)` across the corpus. Dispatches to a rayon path
/// for large corpora; both paths produce equal counts (see the equivalence test).
fn accumulate(doc_terms: &[Vec<CompactString>]) -> HashMap<CompactString, HashMap<DocId, u16>> {
    #[cfg(feature = "parallel")]
    if doc_terms.len() >= PAR_MIN_DOCS {
        return accumulate_parallel(doc_terms);
    }
    accumulate_sequential(doc_terms)
}

/// Sequential accumulation: per-document local counts merged in doc order.
fn accumulate_sequential(
    doc_terms: &[Vec<CompactString>],
) -> HashMap<CompactString, HashMap<DocId, u16>> {
    let mut acc: HashMap<CompactString, HashMap<DocId, u16>> = HashMap::new();
    for (doc, terms) in doc_terms.iter().enumerate() {
        let doc = doc as DocId;
        for (term, tf) in local_counts(terms) {
            acc.entry(term).or_default().insert(doc, tf);
        }
    }
    acc
}

/// Parallel accumulation: per-document local counts computed with rayon, then
/// merged sequentially in doc order — deterministic, so serialization is
/// byte-identical to the sequential path.
#[cfg(feature = "parallel")]
fn accumulate_parallel(
    doc_terms: &[Vec<CompactString>],
) -> HashMap<CompactString, HashMap<DocId, u16>> {
    use rayon::prelude::*;
    let locals: Vec<HashMap<CompactString, u16>> =
        doc_terms.par_iter().map(|t| local_counts(t)).collect();
    let mut acc: HashMap<CompactString, HashMap<DocId, u16>> = HashMap::new();
    for (doc, local) in locals.into_iter().enumerate() {
        let doc = doc as DocId;
        for (term, tf) in local {
            acc.entry(term).or_default().insert(doc, tf);
        }
    }
    acc
}

impl Bm25 {
    /// Build from each document's normalized token list (token order irrelevant;
    /// frequencies are counted). `doc_terms[d]` are the tokens of document `d`.
    #[must_use]
    pub fn build(doc_terms: &[Vec<CompactString>], k1: f32, b: f32) -> Self {
        let doc_count = doc_terms.len() as u32;
        let doc_len: Vec<u32> = doc_terms.iter().map(|t| t.len() as u32).collect();
        let total_len: u64 = doc_len.iter().map(|&l| u64::from(l)).sum();
        let avgdl = if doc_count == 0 {
            0.0
        } else {
            total_len as f32 / doc_count as f32
        };

        // term -> (doc -> tf), accumulated across the corpus. The accumulation
        // strategy (sequential or rayon-parallel) does not affect the result: the
        // finalize step below sorts terms and postings, so output is byte-identical.
        let acc = accumulate(doc_terms);

        // Finalize into sorted term postings for deterministic serialization.
        let mut term_strings: Vec<CompactString> = acc.keys().cloned().collect();
        term_strings.sort_unstable();
        let mut terms = Vec::with_capacity(term_strings.len());
        let mut term_index = HashMap::with_capacity(term_strings.len());
        for (i, term) in term_strings.into_iter().enumerate() {
            let postings = &acc[&term];
            let mut docs_sorted: Vec<(DocId, u16)> =
                postings.iter().map(|(&d, &f)| (d, f)).collect();
            docs_sorted.sort_unstable_by_key(|(d, _)| *d);
            let mut docs = RoaringBitmap::new();
            let mut tf = Vec::with_capacity(docs_sorted.len());
            for (d, f) in docs_sorted {
                docs.insert(d);
                tf.push(f);
            }
            term_index.insert(term.clone(), i);
            terms.push(TermPosting { term, docs, tf });
        }

        Self {
            k1,
            b,
            avgdl,
            doc_count,
            doc_len,
            terms,
            term_index,
        }
    }

    /// Number of documents.
    #[must_use]
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Average document length (for the index header).
    #[must_use]
    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    /// BM25 `k1`.
    #[must_use]
    pub fn k1(&self) -> f32 {
        self.k1
    }

    /// BM25 `b`.
    #[must_use]
    pub fn b(&self) -> f32 {
        self.b
    }

    /// Score documents against `query_terms` (already normalized), optionally
    /// restricted to `allowed`. Returns the top `limit` `(DocId, score)` descending.
    #[must_use]
    pub fn score(
        &self,
        query_terms: &[CompactString],
        allowed: Option<&RoaringBitmap>,
        limit: usize,
    ) -> Vec<(DocId, f32)> {
        if self.doc_count == 0 || limit == 0 {
            return Vec::new();
        }
        let n = self.doc_count as f32;
        let mut scores: HashMap<DocId, f32> = HashMap::new();

        // De-duplicate query terms so a repeated term doesn't double-count idf.
        let mut seen: std::collections::HashSet<&CompactString> = std::collections::HashSet::new();
        for term in query_terms {
            if !seen.insert(term) {
                continue;
            }
            let Some(&ti) = self.term_index.get(term) else {
                continue;
            };
            let posting = &self.terms[ti];
            let df = posting.docs.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (doc, &tf) in posting.docs.iter().zip(&posting.tf) {
                if allowed.is_some_and(|a| !a.contains(doc)) {
                    continue;
                }
                let dl = self.doc_len[doc as usize] as f32;
                let tf = f32::from(tf);
                let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl.max(1.0));
                *scores.entry(doc).or_insert(0.0) += idf * (tf * (self.k1 + 1.0)) / denom;
            }
        }

        let mut ranked: Vec<(DocId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(limit);
        ranked
    }

    /// Reconstruct each document's term multiset (term repeated by frequency).
    ///
    /// Term *order* within a document is not preserved, but BM25 is order-free, so
    /// this losslessly seeds a rebuild after an incremental upsert.
    #[must_use]
    pub fn to_doc_terms(&self) -> Vec<Vec<CompactString>> {
        let mut docs: Vec<Vec<CompactString>> = vec![Vec::new(); self.doc_count as usize];
        for posting in &self.terms {
            for (doc, &tf) in posting.docs.iter().zip(&posting.tf) {
                let bucket = &mut docs[doc as usize];
                for _ in 0..tf {
                    bucket.push(posting.term.clone());
                }
            }
        }
        docs
    }

    /// Serialize to the BM25 section payload.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        put_f32(&mut buf, self.k1);
        put_f32(&mut buf, self.b);
        put_f32(&mut buf, self.avgdl);
        put_u32(&mut buf, self.doc_count);
        put_u32(&mut buf, self.terms.len() as u32);
        for &len in &self.doc_len {
            put_u32(&mut buf, len);
        }
        for posting in &self.terms {
            put_len_prefixed(&mut buf, posting.term.as_bytes());
            put_u32(&mut buf, posting.docs.len() as u32);
            let mut roaring_bytes = Vec::new();
            posting
                .docs
                .serialize_into(&mut roaring_bytes)
                .expect("roaring serialize into Vec is infallible");
            put_len_prefixed(&mut buf, &roaring_bytes);
            for &tf in &posting.tf {
                put_u16(&mut buf, tf);
            }
        }
        buf
    }

    /// Decode from a BM25 section payload.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let k1 = r.f32()?;
        let b = r.f32()?;
        let avgdl = r.f32()?;
        let doc_count = r.u32()?;
        let term_count = r.u32()?;
        // Cap every pre-allocation by remaining bytes so a malicious count cannot
        // trigger a huge allocation before the reader runs out of input.
        let mut doc_len = Vec::with_capacity((doc_count as usize).min(r.remaining()));
        for _ in 0..doc_count {
            doc_len.push(r.u32()?);
        }
        let mut terms = Vec::with_capacity((term_count as usize).min(r.remaining()));
        let mut term_index = HashMap::with_capacity((term_count as usize).min(r.remaining()));
        for i in 0..term_count as usize {
            let term = CompactString::from(r.str()?);
            let df = r.u32()? as usize;
            let roaring_bytes = r.len_prefixed()?;
            let docs = RoaringBitmap::deserialize_from(roaring_bytes)
                .map_err(|e| Error::format(format!("roaring decode: {e}")))?;
            if docs.len() as usize != df {
                return Err(Error::format("BM25 df mismatch"));
            }
            let mut tf = Vec::with_capacity(df.min(r.remaining()));
            for _ in 0..df {
                tf.push(r.u16()?);
            }
            term_index.insert(term.clone(), i);
            terms.push(TermPosting { term, docs, tf });
        }
        Ok(Self {
            k1,
            b,
            avgdl,
            doc_count,
            doc_len,
            terms,
            term_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(words: &[&str]) -> Vec<CompactString> {
        words.iter().map(|w| CompactString::from(*w)).collect()
    }

    fn sample() -> Bm25 {
        // doc0: about cats; doc1: about dogs; doc2: cats and dogs
        let docs = vec![
            terms(&["the", "cat", "sat", "cat"]),
            terms(&["the", "dog", "ran"]),
            terms(&["cat", "and", "dog"]),
        ];
        Bm25::build(&docs, DEFAULT_K1, DEFAULT_B)
    }

    #[test]
    fn sequential_and_parallel_accumulate_agree() {
        // A corpus large enough to engage the parallel path; both accumulation
        // strategies must yield identical counts (byte-reproducibility follows).
        let mut docs: Vec<Vec<CompactString>> = Vec::new();
        for i in 0..600usize {
            docs.push(vec![
                CompactString::from("common"),
                CompactString::from(format!("t{}", i % 23)),
                CompactString::from(format!("u{}", i % 7)),
            ]);
        }
        let seq = accumulate_sequential(&docs);
        #[cfg(feature = "parallel")]
        {
            let par = accumulate_parallel(&docs);
            assert_eq!(seq, par, "parallel accumulation diverged from sequential");
        }
        // The full build is deterministic regardless of the path taken.
        let bytes_a = Bm25::build(&docs, DEFAULT_K1, DEFAULT_B).to_bytes();
        let bytes_b = Bm25::build(&docs, DEFAULT_K1, DEFAULT_B).to_bytes();
        assert_eq!(bytes_a, bytes_b);
        assert_eq!(seq.get("common").map(HashMap::len), Some(600));
    }

    #[test]
    fn scores_rank_relevant_docs_first() {
        let bm = sample();
        let hits = bm.score(&terms(&["cat"]), None, 10);
        // doc0 has tf=2 for "cat", doc2 has tf=1 → doc0 should rank first.
        assert_eq!(hits[0].0, 0);
        assert!(hits.iter().any(|(d, _)| *d == 2));
        assert!(!hits.iter().any(|(d, _)| *d == 1)); // doc1 has no "cat"
    }

    #[test]
    fn idf_decreases_with_document_frequency() {
        let bm = sample();
        // "cat" appears in 2 docs, a unique rare term would score higher idf.
        let common = bm.score(&terms(&["the"]), None, 10); // appears in 2 docs
        let rare = bm.score(&terms(&["sat"]), None, 10); // appears in 1 doc
                                                         // rare term's top score should exceed the common term's top score.
        assert!(rare[0].1 > common[0].1);
    }

    #[test]
    fn allowed_filter_restricts_results() {
        let bm = sample();
        let mut allowed = RoaringBitmap::new();
        allowed.insert(2);
        let hits = bm.score(&terms(&["cat"]), Some(&allowed), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn repeated_query_term_not_double_counted() {
        let bm = sample();
        let once = bm.score(&terms(&["cat"]), None, 10);
        let twice = bm.score(&terms(&["cat", "cat"]), None, 10);
        assert!((once[0].1 - twice[0].1).abs() < f32::EPSILON);
    }

    #[test]
    fn unknown_terms_score_nothing() {
        let bm = sample();
        assert!(bm.score(&terms(&["zebra"]), None, 10).is_empty());
    }

    #[test]
    fn serialization_roundtrips() {
        let bm = sample();
        let bytes = bm.to_bytes();
        let decoded = Bm25::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.doc_count, bm.doc_count);
        assert!((decoded.avgdl - bm.avgdl).abs() < 1e-6);
        // identical scoring after roundtrip
        let a = bm.score(&terms(&["cat", "dog"]), None, 10);
        let b = decoded.score(&terms(&["cat", "dog"]), None, 10);
        assert_eq!(a, b);
    }

    #[test]
    fn decode_rejects_truncated_bytes() {
        let bytes = sample().to_bytes();
        assert!(Bm25::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    }

    #[test]
    fn decode_rejects_huge_counts_without_oom() {
        // Regression for the fuzz-found OOM: huge doc/term counts with a truncated
        // body must error via capped pre-allocation, not allocate gigabytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DEFAULT_K1.to_bits().to_le_bytes());
        bytes.extend_from_slice(&DEFAULT_B.to_bits().to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // doc_count
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // term_count
        assert!(Bm25::from_bytes(&bytes).is_err());
    }

    #[test]
    fn empty_corpus_scores_nothing() {
        let bm = Bm25::build(&[], DEFAULT_K1, DEFAULT_B);
        assert!(bm.score(&terms(&["cat"]), None, 10).is_empty());
    }
}
