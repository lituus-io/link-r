// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Property-based tests: invariants that must hold for arbitrary inputs.

use compact_str::CompactString;
use link_r::index::IndexBuilder;
use link_r::metric::Metric;
use link_r::query::{PreparedQuery, RankParams};
use link_r::resource::ResourceKind;
use link_r::url_key::{canonicalize, UrlKey};
use link_r::{Document, Filter, Index};
use proptest::prelude::*;
use url::Url;

const DIM: usize = 8;

/// A single document: a unique path, some terms, and a vector.
fn document_strategy() -> impl Strategy<Value = (String, Vec<String>, Vec<f32>)> {
    (
        "[a-z][a-z0-9]{0,7}",
        prop::collection::vec("[a-z]{1,6}", 1..6),
        prop::collection::vec(-1.0f32..1.0, DIM..=DIM),
    )
}

fn corpus_strategy() -> impl Strategy<Value = Vec<(String, Vec<String>, Vec<f32>)>> {
    prop::collection::vec(document_strategy(), 1..12)
}

fn build_index(corpus: &[(String, Vec<String>, Vec<f32>)]) -> Index {
    let mut builder = IndexBuilder::new(DIM, Metric::Cosine, 1);
    for (i, (slug, terms, vector)) in corpus.iter().enumerate() {
        // Make URLs unique even if slugs collide.
        let url = Url::parse(&format!("https://x.dev/{slug}/{i}")).unwrap();
        let doc = Document {
            url,
            kind: ResourceKind::Text,
            content_hash: i as u64,
            title: None,
            snippet: CompactString::from("s"),
            lang: None,
            tags: smallvec_from(&[]),
            terms: terms.iter().map(CompactString::from).collect(),
            vector: vector.clone(),
            edges: Vec::new(),
            fetched_at_ms: 0,
            etag: None,
            pinned: false,
        };
        builder.upsert(doc).unwrap();
    }
    builder.build()
}

fn smallvec_from(items: &[&str]) -> smallvec::SmallVec<[CompactString; 4]> {
    items.iter().map(|s| CompactString::from(*s)).collect()
}

fn url_strategy() -> impl Strategy<Value = String> {
    (
        "[a-z]{1,6}",
        prop::collection::vec("[a-z0-9]{1,5}", 0..4),
        prop::option::of("[a-z]{1,3}=[a-z0-9]{1,4}"),
        prop::option::of("[a-z]{1,5}"),
    )
        .prop_map(|(host, segs, query, frag)| {
            let mut url = format!("https://{host}.dev/{}", segs.join("/"));
            if let Some(q) = query {
                url.push('?');
                url.push_str(&q);
            }
            if let Some(f) = frag {
                url.push('#');
                url.push_str(&f);
            }
            url
        })
}

proptest! {
    /// Building, saving, and reopening yields an index that resolves identically.
    #[test]
    fn index_save_open_roundtrip(corpus in corpus_strategy()) {
        let index = build_index(&corpus);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("idx.lnkr");
        index.save(&path).unwrap();
        let reopened = Index::open(&path).unwrap();

        prop_assert_eq!(index.len(), reopened.len());

        // A fixed query must return the same ranked URLs from both.
        let query = vec![0.5f32; DIM];
        let terms: Vec<CompactString> = vec![CompactString::from("a")];
        let filter = Filter::All;
        let pq = PreparedQuery {
            vector: &query,
            terms: &terms,
            filter: &filter,
            limit: 8,
            rank: RankParams::default(),
        };
        let a: Vec<String> = index.search_prepared(&pq).unwrap().iter().map(|h| h.url.to_owned()).collect();
        let b: Vec<String> = reopened.search_prepared(&pq).unwrap().iter().map(|h| h.url.to_owned()).collect();
        prop_assert_eq!(a, b);
    }

    /// Saving the same corpus twice yields byte-identical files (reproducible).
    #[test]
    fn index_is_byte_reproducible(corpus in corpus_strategy()) {
        let tmp = tempfile::tempdir().unwrap();
        let p1 = tmp.path().join("a.lnkr");
        let p2 = tmp.path().join("b.lnkr");
        build_index(&corpus).save(&p1).unwrap();
        build_index(&corpus).save(&p2).unwrap();
        prop_assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    }

    /// URL canonicalization is idempotent and key-stable.
    #[test]
    fn url_canonicalization_idempotent(url in url_strategy()) {
        let parsed = Url::parse(&url).unwrap();
        let once = canonicalize(&parsed);
        let twice = canonicalize(&Url::parse(&once).unwrap());
        prop_assert_eq!(&once, &twice);
        // The key derived from the canonical form is stable.
        prop_assert_eq!(UrlKey::parse(&url).unwrap(), UrlKey::parse(&once).unwrap());
    }

    /// Dropping a fragment never changes the canonical URL or its key.
    #[test]
    fn fragment_does_not_affect_key(base in url_strategy(), frag in "[a-z]{1,8}") {
        // Strip any existing fragment from `base` first.
        let base_no_frag = base.split('#').next().unwrap().to_owned();
        let with_frag = format!("{base_no_frag}#{frag}");
        prop_assert_eq!(
            UrlKey::parse(&base_no_frag).unwrap(),
            UrlKey::parse(&with_frag).unwrap()
        );
    }
}
