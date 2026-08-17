// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Crawl a GitHub repository's docs folder and resolve queries to doc links.
//!
//! Live-network example for `https://github.com/abyo-software/s4/tree/main/docs`.
//! GitHub serves directory listings as `/tree/...` pages and files as `/blob/...`
//! pages, so a plain path-prefix scope can't follow the file links. Instead we use
//! the path filter to confine the crawl to *any file under this repo's docs path*
//! (both `tree` and `blob`), the same-host scope to exclude external sites, and a
//! file-type filter to index only Markdown.
//!
//! Run (lexical hash embedder, no model download):
//!   cargo run --example crawl_github_docs
//! Run with semantic embeddings (downloads bge-small once):
//!   cargo run --example crawl_github_docs --features onnx

// This example drives the crawler, so it requires the `crawl` feature (on by
// default). Under a `--no-default-features` build without it, `main` is a no-op.
#[cfg(not(feature = "crawl"))]
fn main() {
    eprintln!("the `crawl_github_docs` example requires the `crawl` feature");
}

#[cfg(feature = "crawl")]
use link_r::prelude::*;
#[cfg(feature = "crawl")]
use std::time::Duration;

#[cfg(feature = "crawl")]
#[tokio::main]
async fn main() -> link_r::Result<()> {
    let parent = "https://github.com/abyo-software/s4/tree/main/docs";

    let mut index = LinkIndex::in_memory()?;
    let report = index
        .update(parent)
        .depth(3)
        .scope(CrawlScope::SameHost) // stay on github.com — never external sites
        .require_path("/abyo-software/s4") // crawl: stay inside this repository
        .require_path("/main/docs") // crawl: …inside its docs folder (tree + blob)
        .index_path("/blob/main/docs") // index: only the canonical file view —
        // collapses GitHub's /raw/ and /commits/ duplicates of each file
        .accept_extension("md") // index: only Markdown files
        .max_pages(200)
        .min_delay(Duration::from_millis(400)) // be polite to GitHub
        .run()
        .await?;

    println!(
        "indexed {} docs ({} added, {} unchanged, {} pages crawled)",
        index.len(),
        report.added,
        report.unchanged,
        report.pages_seen()
    );

    for query in [
        "deployment and configuration",
        "security threat model",
        "gpu benchmarks",
    ] {
        println!("\nquery: {query:?}");
        for (i, hit) in index.search(query, 3).await?.iter().enumerate() {
            println!("  {:>2}. {:.3}  {}", i + 1, hit.score, hit.url);
        }
    }
    Ok(())
}
