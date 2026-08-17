// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The persisted link graph — the "paths" of the knowledge graph.
//!
//! Each document carries the canonical-URL keys ([`UrlKey`]) of its outbound
//! links. Edges are keyed by `UrlKey`, not internal `DocId`, because ids are not
//! stable across removals; a target that is not (yet) indexed simply resolves to
//! nothing and becomes a live path once the knowledge base accumulates it. On open
//! the keyed edges are resolved into a [`Graph`] of both out- and in-adjacency,
//! which powers `related()` and the optional one-hop score boost during search.

use crate::error::{Error, Result};
use crate::index::bytesio::{put_u32, put_u64, Reader};
use crate::resource::DocId;
use crate::url_key::UrlKey;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Encode per-document outbound edges (`UrlKey`s) into the `Edges` section payload.
#[must_use]
pub fn encode(edges: &[Vec<UrlKey>]) -> Vec<u8> {
    let mut buf = Vec::new();
    put_u32(&mut buf, edges.len() as u32);
    for doc_edges in edges {
        put_u32(&mut buf, doc_edges.len() as u32);
        for key in doc_edges {
            put_u64(&mut buf, key.raw());
        }
    }
    buf
}

/// Decode the `Edges` section payload into per-document `UrlKey` lists. Every
/// count is capped by the remaining input so adversarial input cannot force a huge
/// allocation (mirrors the other section decoders / the loader fuzz invariant).
pub fn decode(bytes: &[u8]) -> Result<Vec<Vec<UrlKey>>> {
    let mut r = Reader::new(bytes);
    let doc_count = r.u32()? as usize;
    let mut out = Vec::with_capacity(doc_count.min(r.remaining()));
    for _ in 0..doc_count {
        let edge_count = r.u32()? as usize;
        // Each edge is 8 bytes; cap by remaining/8 so a hostile count errors out.
        let mut list = Vec::with_capacity(edge_count.min(r.remaining() / 8 + 1));
        for _ in 0..edge_count {
            list.push(UrlKey(r.u64()?));
        }
        out.push(list);
    }
    if r.remaining() != 0 {
        return Err(Error::format("trailing bytes in Edges section"));
    }
    Ok(out)
}

/// Resolved link graph: out- and in-adjacency by internal `DocId`. Dangling edges
/// (targets not currently indexed) are dropped from the resolved adjacency but the
/// keyed form is retained on disk, so they heal as the corpus grows.
#[derive(Debug, Default)]
pub struct Graph {
    out: Vec<SmallVec<[DocId; 8]>>,
    inc: Vec<SmallVec<[DocId; 8]>>,
}

impl Graph {
    /// Resolve `UrlKey` edges into a `DocId` adjacency using the URL→id map.
    #[must_use]
    pub fn resolve(edges: &[Vec<UrlKey>], by_url: &HashMap<UrlKey, DocId>) -> Self {
        let n = edges.len();
        let mut out: Vec<SmallVec<[DocId; 8]>> = vec![SmallVec::new(); n];
        let mut inc: Vec<SmallVec<[DocId; 8]>> = vec![SmallVec::new(); n];
        for (src, targets) in edges.iter().enumerate() {
            let src = src as DocId;
            for key in targets {
                if let Some(&dst) = by_url.get(key) {
                    if (dst as usize) < n && dst != src {
                        out[src as usize].push(dst);
                        inc[dst as usize].push(src);
                    }
                }
            }
        }
        Self { out, inc }
    }

    /// Whether the graph has no resolved edges (e.g. an index without a graph section).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.out.iter().all(SmallVec::is_empty)
    }

    /// Outbound neighbors of `doc`.
    #[must_use]
    pub fn out_neighbors(&self, doc: DocId) -> &[DocId] {
        self.out.get(doc as usize).map_or(&[], |v| v)
    }

    /// Inbound neighbors of `doc` (documents that link to it).
    #[must_use]
    pub fn in_neighbors(&self, doc: DocId) -> &[DocId] {
        self.inc.get(doc as usize).map_or(&[], |v| v)
    }

    /// The `k` documents most related to `doc`: outbound targets and co-cited
    /// siblings (documents sharing an inbound link), ranked by connection count.
    #[must_use]
    pub fn related(&self, doc: DocId, k: usize) -> Vec<(DocId, u32)> {
        if k == 0 {
            return Vec::new();
        }
        let mut score: HashMap<DocId, u32> = HashMap::new();
        // Direct outbound links weigh most.
        for &t in self.out_neighbors(doc) {
            *score.entry(t).or_insert(0) += 2;
        }
        // Co-citation: documents pointed to by the same parents as `doc`.
        for &parent in self.in_neighbors(doc) {
            for &sibling in self.out_neighbors(parent) {
                if sibling != doc {
                    *score.entry(sibling).or_insert(0) += 1;
                }
            }
        }
        score.remove(&doc);
        let mut ranked: Vec<(DocId, u32)> = score.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
    }

    /// Apply a one-hop score boost within a candidate set: each candidate gains
    /// `weight * mean(neighbor score)` over its in/out neighbors that are also
    /// candidates. Rewards hub/authority pages the flat scores under-rank. Returns
    /// the re-sorted `(DocId, score)` list.
    #[must_use]
    pub fn boosted(&self, ranked: &[(DocId, f32)], weight: f32) -> Vec<(DocId, f32)> {
        if weight <= 0.0 || self.is_empty() {
            return ranked.to_vec();
        }
        let base: HashMap<DocId, f32> = ranked.iter().copied().collect();
        let mut out: Vec<(DocId, f32)> = ranked
            .iter()
            .map(|&(id, s)| {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for &nb in self
                    .out_neighbors(id)
                    .iter()
                    .chain(self.in_neighbors(id))
                {
                    if let Some(&ns) = base.get(&nb) {
                        sum += ns;
                        count += 1;
                    }
                }
                let boost = if count > 0 {
                    weight * (sum / count as f32)
                } else {
                    0.0
                };
                (id, s + boost)
            })
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> UrlKey {
        UrlKey(n)
    }

    #[test]
    fn encode_decode_roundtrips() {
        let edges = vec![
            vec![key(10), key(20)],
            vec![],
            vec![key(30), key(40), key(50)],
        ];
        let bytes = encode(&edges);
        assert_eq!(decode(&bytes).unwrap(), edges);
    }

    #[test]
    fn decode_rejects_huge_count_without_oom() {
        let bytes = [0xff, 0xff, 0xff, 0xff]; // 4B doc count, no bodies
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn resolve_drops_dangling_and_self() {
        // doc0 -> doc1 (key 100) and -> dangling (key 999); doc1 -> doc0.
        let mut by_url = HashMap::new();
        by_url.insert(key(100), 1);
        by_url.insert(key(200), 0);
        let edges = vec![vec![key(100), key(999)], vec![key(200)]];
        let g = Graph::resolve(&edges, &by_url);
        assert_eq!(g.out_neighbors(0), &[1]); // dangling 999 dropped
        assert_eq!(g.in_neighbors(1), &[0]);
        assert_eq!(g.out_neighbors(1), &[0]);
    }

    #[test]
    fn related_prefers_direct_then_cocited() {
        // parent(0) -> a(1), b(2); a(1) -> b(2). related(1) should surface b(2).
        let mut by_url = HashMap::new();
        by_url.insert(key(1), 1);
        by_url.insert(key(2), 2);
        let edges = vec![vec![key(1), key(2)], vec![key(2)], vec![]];
        let g = Graph::resolve(&edges, &by_url);
        let rel = g.related(1, 5);
        assert_eq!(rel[0].0, 2, "b is both a direct target and co-cited");
    }

    #[test]
    fn boost_zero_is_identity() {
        let g = Graph::default();
        let ranked = vec![(0u32, 1.0f32), (1, 0.5)];
        assert_eq!(g.boosted(&ranked, 0.0), ranked);
    }
}
