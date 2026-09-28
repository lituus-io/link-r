// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Publishing and loading link-r indexes through objstore-r backends.

#[path = "../../tests/support/golden.rs"]
mod golden;

use link_r::{HashEmbedder, LinkIndex, SourceRef};
use link_r_objstore::{Error, Remote};
use objstore_r::mem::Fault;
use objstore_r::{Bytes, Cond, LocalDir, Mem, ObjectStore};

async fn corpus_index() -> LinkIndex {
    let mut index = LinkIndex::in_memory().unwrap();
    golden::ingest(&mut index).await;
    index
}

#[tokio::test(flavor = "current_thread")]
async fn publish_then_load_reads_identically_and_zero_copy() {
    let mem = Mem::default();
    let remote = Remote::new(&mem, "kb/docs").unwrap();
    assert!(remote.load().await.unwrap().is_none());
    let index = corpus_index().await;
    let p = remote.publish(&index).await.unwrap();
    assert!(p.uploaded);
    assert_eq!(p.generation, 1);
    let loaded = Remote::new(&mem, "kb/docs")
        .unwrap()
        .load()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(golden::dump(&loaded).await, golden::dump(&index).await);
    assert_eq!(loaded.to_bytes().unwrap(), index.to_bytes().unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn unchanged_images_are_not_republished_and_old_generations_are_removed() {
    let mem = Mem::default();
    let remote = Remote::new(&mem, "kb").unwrap();
    let mut index = corpus_index().await;
    remote.publish(&index).await.unwrap();
    let again = remote.publish(&index).await.unwrap();
    assert!(!again.uploaded);
    index.unpin("https://docs.example.com/guides/").unwrap();
    let p = remote.publish(&index).await.unwrap();
    assert_eq!(p.generation, 2);
    let keys = mem.keys();
    assert_eq!(keys.len(), 2, "HEAD + the current image only: {keys:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn two_publishers_from_one_state_exactly_one_wins() {
    let mem = Mem::default();
    let base = corpus_index().await;
    Remote::new(&mem, "kb")
        .unwrap()
        .publish(&base)
        .await
        .unwrap();
    let (a, b) = (
        Remote::new(&mem, "kb").unwrap(),
        Remote::new(&mem, "kb").unwrap(),
    );
    let mut ia = a.load().await.unwrap().unwrap();
    let mut ib = b.load().await.unwrap().unwrap();
    ia.unpin("https://docs.example.com/guides/").unwrap();
    ib.pin("https://docs.example.com/data/").unwrap();
    a.publish(&ia).await.unwrap();
    assert!(matches!(b.publish(&ib).await, Err(Error::Fenced { .. })));
    // After reloading, b can publish again.
    let mut ib = b.load().await.unwrap().unwrap();
    ib.pin("https://docs.example.com/data/").unwrap();
    assert_eq!(b.publish(&ib).await.unwrap().generation, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn lost_responses_and_failures_at_every_step_converge() {
    for lost in [false, true] {
        for step in 0..5 {
            let mem = Mem::default();
            let remote = Remote::new(&mem, "kb").unwrap();
            let mut index = corpus_index().await;
            remote.publish(&index).await.unwrap();
            let before = golden::dump(&index).await;
            index.unpin("https://docs.example.com/guides/").unwrap();
            let after = golden::dump(&index).await;
            let e = objstore_r::Error::Unavailable {
                message: "injected".into(),
                retriable: true,
            };
            mem.inject_after(
                step,
                if lost {
                    Fault::LostResponse(e)
                } else {
                    Fault::Fail(e)
                },
            );
            let first = remote.publish(&index).await;
            mem.clear_faults();
            let seen = golden::dump(
                &Remote::new(&mem, "kb")
                    .unwrap()
                    .load()
                    .await
                    .unwrap()
                    .unwrap(),
            )
            .await;
            assert!(seen == before || seen == after, "step {step} lost={lost}");
            if first.is_ok() {
                assert_eq!(seen, after);
            }
            remote
                .publish(&index)
                .await
                .unwrap_or_else(|e| panic!("step {step} lost={lost}: {e}"));
            let seen = golden::dump(
                &Remote::new(&mem, "kb")
                    .unwrap()
                    .load()
                    .await
                    .unwrap()
                    .unwrap(),
            )
            .await;
            assert_eq!(seen, after, "step {step} lost={lost}: retry converges");
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn tampering_is_detected() {
    let mem = Mem::default();
    let remote = Remote::new(&mem, "kb").unwrap();
    remote.publish(&corpus_index().await).await.unwrap();
    for key in mem.keys() {
        let original = mem.get(&key).await.unwrap().unwrap().bytes;
        let mut bad = original.to_vec();
        let at = bad.len() / 3;
        bad[at] ^= 0x10;
        mem.put(&key, Bytes::from(bad), Cond::None).await.unwrap();
        let r = Remote::new(&mem, "kb").unwrap().load().await;
        assert!(matches!(r, Err(Error::Corrupt { .. })), "{key}: {r:?}");
        mem.put(&key, original, Cond::None).await.unwrap();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn replicas_poll_with_one_metadata_request() {
    let dir = tempfile::tempdir().unwrap();
    let objects = LocalDir::open(dir.path()).unwrap();
    let writer = Remote::new(&objects, "kb").unwrap();
    let reader = Remote::new(&objects, "kb").unwrap();
    let mut index = corpus_index().await;
    writer.publish(&index).await.unwrap();
    assert!(reader.load_if_changed().await.unwrap().is_some());
    assert!(
        reader.load_if_changed().await.unwrap().is_none(),
        "unchanged"
    );
    index.unpin("https://docs.example.com/guides/").unwrap();
    writer.publish(&index).await.unwrap();
    let fresh = reader.load_if_changed().await.unwrap().unwrap();
    assert_eq!(golden::dump(&fresh).await, golden::dump(&index).await);
    assert_eq!(reader.generation().await.unwrap(), Some(2));
}

#[tokio::test(flavor = "current_thread")]
async fn a_custom_embedder_round_trips_and_mismatches_are_refused() {
    let mem = Mem::default();
    let remote = Remote::new(&mem, "wide").unwrap();
    let mut index = LinkIndex::in_memory_with(HashEmbedder::new(64));
    let root = SourceRef::http("https://docs.example.com/").unwrap();
    index
        .ingest_from(&golden::MemSource, &root, 2, false)
        .await
        .unwrap();
    remote.publish(&index).await.unwrap();
    let loaded = remote
        .load_with(HashEmbedder::new(64))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.len(), 4);
    assert!(
        matches!(remote.load().await, Err(Error::Link(_))),
        "default embedder has another dim"
    );
}

#[test]
fn prefixes_are_validated() {
    let mem = Mem::default();
    assert!(Remote::new(&mem, "../up").is_err());
    assert!(Remote::new(&mem, "a//b").is_err());
    assert!(Remote::new(&mem, "").is_ok());
}

#[cfg(feature = "gcs")]
#[tokio::test(flavor = "current_thread")]
async fn gcs_emulator_round_trip() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let Ok(endpoint) = std::env::var("OBJSTORE_FAKE_GCS") else {
        eprintln!("OBJSTORE_FAKE_GCS unset: skipped");
        return;
    };
    let bucket = format!("linkr-{}", std::process::id());
    let hostport = endpoint.trim_start_matches("http://").to_owned();
    let mut s = tokio::net::TcpStream::connect(&hostport).await.unwrap();
    let body = format!("{{\"name\":\"{bucket}\"}}");
    let req = format!(
        "POST /storage/v1/b?project=test HTTP/1.1\r\nhost: {hostport}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut sink = Vec::new();
    let _ = s.read_to_end(&mut sink).await;
    let cfg = objstore_r::GcsConfig {
        endpoint,
        resumable_threshold: 4096,
        ..objstore_r::GcsConfig::default()
    };
    let gcs = objstore_r::Gcs::new(&bucket, objstore_r::token::Static::none(), cfg).unwrap();
    let remote = Remote::new(&gcs, "kb").unwrap();
    let index = corpus_index().await;
    remote.publish(&index).await.unwrap();
    let loaded = Remote::new(&gcs, "kb")
        .unwrap()
        .load()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(golden::dump(&loaded).await, golden::dump(&index).await);
    for m in gcs.list("kb/").await.unwrap() {
        gcs.delete(&m.key).await.unwrap();
    }
}
