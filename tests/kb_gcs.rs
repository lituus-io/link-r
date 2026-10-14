// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! The knowledge base on Google Cloud Storage's JSON API:
//! - `OBJSTORE_FAKE_GCS=http://127.0.0.1:4443` — fake-gcs-server;
//! - `OBJSTORE_LIVE_BUCKET=<bucket>` — a real bucket via Application Default
//!   Credentials, under a unique `objstore-r-live/kb-…` prefix that is deleted
//!   afterwards (nothing else in the bucket is touched).
//!
//! Each test is skipped when its variable is unset.

#![cfg(feature = "gcs")]

#[path = "support/kb_site.rs"]
mod support;

use link_r::facade::{default_embedder, DefaultEmbedder};
use link_r::kb::{KnowledgeBase, LookupOpts, WriterOptions};
use objstore_r::gcs::Gcs;
use objstore_r::token::TokenProvider;
use objstore_r::ObjectStore;
use std::time::Instant;
use support::{Site, ROOT};

async fn exercise<T: TokenProvider>(gcs: &Gcs<T>, prefix: &str) {
    let site = Site::new();
    let dir = tempfile::tempdir().unwrap();
    let t = Instant::now();
    let mut kb = KnowledgeBase::<_, DefaultEmbedder>::writer(
        gcs,
        prefix,
        dir.path(),
        default_embedder().unwrap(),
        WriterOptions::new("live-writer"),
    )
    .await
    .unwrap();
    eprintln!("writer open (empty): {:?}", t.elapsed());
    let t = Instant::now();
    let r = kb.sync_crawl(&site, ROOT, |c| c.depth(3)).await.unwrap();
    eprintln!("crawl + bridge + publish: {:?} ({r:?})", t.elapsed());
    assert_eq!(r.added, 4);

    let t = Instant::now();
    let mut replica =
        KnowledgeBase::<_, DefaultEmbedder>::replica(gcs, prefix, default_embedder().unwrap())
            .await
            .unwrap()
            .unwrap();
    eprintln!(
        "replica hydration (graph image + link image, concurrent): {:?}",
        t.elapsed()
    );
    let t = Instant::now();
    let hits = replica
        .lookup("install linux", &LookupOpts::default())
        .await
        .unwrap();
    eprintln!("fused lookup: {:?}", t.elapsed());
    assert_eq!(hits[0].url, format!("{ROOT}install"));

    let t = Instant::now();
    assert!(!replica.refresh().await.unwrap());
    eprintln!("replica refresh, unchanged: {:?}", t.elapsed());

    site.edit(
        "faq",
        "FAQ",
        "Questions about quantum entanglement of widgets.",
    );
    let t = Instant::now();
    let r = kb.sync_crawl(&site, ROOT, |c| c.depth(3)).await.unwrap();
    eprintln!("incremental crawl + publish: {:?} ({r:?})", t.elapsed());
    assert_eq!(r.updated, 1);
    let t = Instant::now();
    assert!(replica.refresh().await.unwrap());
    eprintln!("replica refresh, changed: {:?}", t.elapsed());
    let hits = replica
        .lookup("quantum entanglement", &LookupOpts::default())
        .await
        .unwrap();
    assert_eq!(hits[0].url, format!("{ROOT}faq"));

    // A restarted writer resumes from the bucket.
    kb.release().await.unwrap();
    drop(kb);
    let dir2 = tempfile::tempdir().unwrap();
    let t = Instant::now();
    let kb = KnowledgeBase::<_, DefaultEmbedder>::writer(
        gcs,
        prefix,
        dir2.path(),
        default_embedder().unwrap(),
        WriterOptions::new("live-writer-2"),
    )
    .await
    .unwrap();
    eprintln!(
        "writer restart (pull graph + load links): {:?}",
        t.elapsed()
    );
    assert_eq!(kb.graph().len(), 4);
    kb.release().await.unwrap();

    for m in gcs.list(&format!("{prefix}/")).await.unwrap() {
        gcs.delete(&m.key).await.unwrap();
    }
    assert!(
        gcs.list(&format!("{prefix}/")).await.unwrap().is_empty(),
        "prefix cleaned up"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn knowledge_base_on_the_emulator() {
    let Some(gcs) = objstore_r::testing::emulator_bucket(4096).await else {
        eprintln!("OBJSTORE_FAKE_GCS unset: skipped");
        return;
    };
    exercise(&gcs, "kb/docs").await;
}

#[tokio::test(flavor = "current_thread")]
async fn knowledge_base_on_a_live_bucket() {
    let Some(gcs) = objstore_r::testing::live_bucket() else {
        eprintln!("OBJSTORE_LIVE_BUCKET unset: skipped");
        return;
    };
    exercise(&gcs, &objstore_r::testing::unique("objstore-r-live/kb")).await;
}
