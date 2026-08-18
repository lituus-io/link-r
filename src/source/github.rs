// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The GitHub tree-API source: index a repository (or a subdirectory of one)
//! without crawling anything.
//!
//! One `GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=1` lists every
//! file with its blob SHA. That SHA is the change token: entries whose SHA
//! matches a caller-supplied validator are recorded as revalidated and **never
//! fetched**, so an unchanged repository costs exactly one HTTPS request per
//! sync. New or changed blobs are fetched from `raw.githubusercontent.com`,
//! classified by path (raw serves everything as `text/plain`), and yielded
//! with the blob SHA as their entity tag — which a persistent backend stores
//! and hands back as next sync's validators, closing the loop statelessly.
//!
//! Discovery does not depend on link structure: a corpus whose README links
//! via absolute URLs, or whose directories hold `.yaml`/`.py` with no markdown
//! connectivity at all, is enumerated completely. This is the property the
//! crawler cannot provide, and the reason this source exists.

use crate::auth::{AuthProvider, Credential};
use crate::error::{Error, Result};
use crate::fetch::{FetchOptions, Fetcher};
use crate::payload::DocPayload;
use crate::resource::{Page, Resource, ResourceKind, SourceRef};
use crate::source::Source;
use crate::url_key::UrlKey;
use compact_str::CompactString;
use futures::stream::{FuturesUnordered, StreamExt};
use secrecy::{ExposeSecret, SecretString};
use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Ready;
use url::Url;

/// Default per-file byte ceiling (matches the crawler's).
const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Default number of blob fetches in flight at once.
const DEFAULT_CONCURRENCY: usize = 8;

/// A parsed GitHub repository reference: `{owner, repo, ref, subdir}`.
///
/// Two spellings normalize here, so every consumer shares one parser:
///
/// - the URL form: `https://github.com/owner/repo[/tree/ref[/subdir…]]`
///   (e.g. `https://github.com/telus/bi-layer-docs/tree/main/stacks`);
/// - the ref grammar: `github:owner/repo@ref[//subdir]`.
///
/// A bare repo (no ref) leaves [`GithubSpec::ref_name`] empty; the source
/// resolves it to the repository's default branch with one extra API call.
///
/// Known limitation of the URL form: GitHub's own URLs cannot distinguish a
/// branch name containing `/` from the start of a path, so the first segment
/// after `/tree/` is taken as the ref. Branches like `feat/x` need the ref
/// grammar, where `@ref//subdir` is unambiguous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GithubSpec {
    /// Repository owner (user or organization).
    pub owner: CompactString,
    /// Repository name.
    pub repo: CompactString,
    /// Branch, tag, or commit SHA; empty = the repository's default branch.
    pub ref_name: CompactString,
    /// Subdirectory to index ("" = the whole repository).
    pub subdir: CompactString,
}

impl GithubSpec {
    /// Parse either spelling. See the type docs for the accepted forms.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("github:") {
            return Self::parse_ref_grammar(rest);
        }
        if s.starts_with("https://github.com/") || s.starts_with("http://github.com/") {
            return Self::parse_url(s);
        }
        Err(Error::invalid_url(format!(
            "not a GitHub spec: {s:?} (expected https://github.com/owner/repo[/tree/ref[/dir]] \
             or github:owner/repo@ref[//dir])"
        )))
    }

    /// Whether `s` looks like a GitHub spec at all (cheap, no allocation).
    /// Callers use this to route between the crawler and this source.
    #[must_use]
    pub fn matches(s: &str) -> bool {
        let s = s.trim();
        s.starts_with("github:")
            || s.starts_with("https://github.com/")
            || s.starts_with("http://github.com/")
    }

    fn parse_ref_grammar(rest: &str) -> Result<Self> {
        // owner/repo@ref//subdir — subdir split first so a ref may not
        // accidentally swallow the `//`.
        let bad = |m: &str| Error::invalid_url(format!("github:{rest}: {m}"));
        let (locator, subdir) = match rest.split_once("//") {
            Some((l, s)) => (l, s.trim_matches('/')),
            None => (rest, ""),
        };
        let (path, ref_name) = match locator.split_once('@') {
            Some((p, r)) if !r.is_empty() => (p, r),
            Some(_) => return Err(bad("empty ref after '@'")),
            None => (locator, ""),
        };
        let (owner, repo) = path
            .split_once('/')
            .ok_or_else(|| bad("expected owner/repo"))?;
        if owner.is_empty() || repo.is_empty() || repo.contains('/') {
            return Err(bad("expected exactly owner/repo"));
        }
        Ok(Self {
            owner: owner.into(),
            repo: repo.into(),
            ref_name: ref_name.into(),
            subdir: subdir.into(),
        })
    }

    fn parse_url(s: &str) -> Result<Self> {
        let url = Url::parse(s).map_err(|e| Error::invalid_url(format!("{s:?}: {e}")))?;
        let bad = |m: &str| Error::invalid_url(format!("{s:?}: {m}"));
        let mut segs = url
            .path_segments()
            .ok_or_else(|| bad("no path"))?
            .filter(|p| !p.is_empty());
        let owner = segs.next().ok_or_else(|| bad("missing owner"))?;
        let repo_raw = segs.next().ok_or_else(|| bad("missing repo"))?;
        let repo = repo_raw.strip_suffix(".git").unwrap_or(repo_raw);
        if repo.is_empty() {
            return Err(bad("missing repo"));
        }
        let (ref_name, subdir) = match segs.next() {
            None => (String::new(), String::new()),
            Some("tree" | "blob") => {
                let r = segs.next().ok_or_else(|| bad("missing ref after /tree/"))?;
                let rest: Vec<&str> = segs.collect();
                (r.to_owned(), rest.join("/"))
            }
            Some(other) => {
                return Err(bad(&format!(
                    "unexpected path segment {other:?} (only /tree/<ref>/<dir> is understood)"
                )));
            }
        };
        Ok(Self {
            owner: owner.into(),
            repo: repo.into(),
            ref_name: ref_name.into(),
            subdir: subdir.trim_matches('/').into(),
        })
    }
}

/// Bearer auth confined to GitHub's two content hosts.
///
/// [`crate::StaticTokenAuth::bearer_scoped`] covers one host, but a GitHub
/// sync talks to exactly two — the API host for the tree listing and the raw
/// host for blob content — and the token must reach both while never leaking
/// anywhere else (a crawled markdown file can link off-host; the fetch of that
/// link must be anonymous).
#[derive(Debug)]
pub struct GithubAuth {
    token: Option<SecretString>,
    api_host: CompactString,
    raw_host: CompactString,
}

impl GithubAuth {
    /// Auth for github.com. `token` = a PAT (works for public, private, and
    /// internal repositories — raw.githubusercontent.com accepts `Bearer`);
    /// `None` = anonymous, public repositories only.
    #[must_use]
    pub fn new(token: Option<String>) -> Self {
        Self {
            token: token.map(|t| SecretString::new(t.into_boxed_str())),
            api_host: "api.github.com".into(),
            raw_host: "raw.githubusercontent.com".into(),
        }
    }

    /// Override the trusted hosts (GitHub Enterprise).
    #[must_use]
    pub fn with_hosts(
        mut self,
        api_host: impl Into<CompactString>,
        raw_host: impl Into<CompactString>,
    ) -> Self {
        self.api_host = api_host.into();
        self.raw_host = raw_host.into();
        self
    }
}

impl AuthProvider for GithubAuth {
    type RefreshFuture<'a> = Ready<Result<()>>;

    fn credential(&self, resource: &Resource) -> Credential<'_> {
        let Some(token) = &self.token else {
            return Credential::Anonymous;
        };
        let trusted = resource
            .url
            .host_str()
            .is_some_and(|h| h == self.api_host || h == self.raw_host);
        if trusted {
            Credential::Bearer(Cow::Borrowed(token.expose_secret()))
        } else {
            Credential::Anonymous
        }
    }

    fn refresh(&self) -> Self::RefreshFuture<'_> {
        std::future::ready(Ok(()))
    }
}

/// One entry of the tree response (blobs only survive filtering).
#[derive(Debug, serde::Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    sha: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct TreeResponse {
    #[serde(default)]
    truncated: bool,
    tree: Vec<TreeEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct RepoResponse {
    default_branch: String,
}

/// A GitHub repository source over a [`Fetcher`].
///
/// The repository reference lives on the source itself; the `root` argument of
/// [`Source::discover`] is ignored (there is nothing to crawl *from* — the
/// tree call enumerates everything).
#[derive(Debug)]
pub struct GitHubSource<F: Fetcher> {
    fetcher: F,
    spec: GithubSpec,
    api_base: CompactString,
    raw_base: CompactString,
    /// Directory levels below `subdir` a file may sit at; `None` = unlimited.
    depth: Option<u16>,
    max_bytes: u64,
    concurrency: usize,
    /// Only index paths ending with these extensions (empty = every
    /// indexable kind). Mirrors the crawler's `accept_extension`.
    extensions: Vec<CompactString>,
    /// Known blob SHAs by canonical raw-URL key: a matching entry is recorded
    /// as revalidated and never fetched.
    validators: HashMap<UrlKey, CompactString>,
    /// Files revalidated (SHA unchanged) during the last discover.
    revalidated: std::sync::Mutex<Vec<Url>>,
    /// Files whose blob fetch failed during the last discover.
    failed: std::sync::atomic::AtomicU32,
}

impl<F: Fetcher> GitHubSource<F> {
    /// Create a source for `spec` over `fetcher` (pair it with [`GithubAuth`]).
    #[must_use]
    pub fn new(fetcher: F, spec: GithubSpec) -> Self {
        Self {
            fetcher,
            spec,
            api_base: "https://api.github.com".into(),
            raw_base: "https://raw.githubusercontent.com".into(),
            depth: None,
            max_bytes: DEFAULT_MAX_BYTES,
            concurrency: DEFAULT_CONCURRENCY,
            extensions: Vec::new(),
            validators: HashMap::new(),
            revalidated: std::sync::Mutex::new(Vec::new()),
            failed: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Override the API and raw base URLs (GitHub Enterprise, tests).
    #[must_use]
    pub fn bases(
        mut self,
        api_base: impl Into<CompactString>,
        raw_base: impl Into<CompactString>,
    ) -> Self {
        self.api_base = api_base.into();
        self.raw_base = raw_base.into();
        self
    }

    /// Cap how many directory levels below the subdir are indexed
    /// (0 = only files directly in it). Default: unlimited.
    #[must_use]
    pub fn depth(mut self, depth: u16) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Per-file byte ceiling. Oversized files are skipped **without a fetch**
    /// — the tree listing already carries each blob's size.
    #[must_use]
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Maximum blob fetches in flight at once (clamped to ≥ 1).
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Only index files whose path ends with `extension` (additive).
    #[must_use]
    pub fn accept_extension(mut self, extension: impl AsRef<str>) -> Self {
        let e = extension
            .as_ref()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        self.extensions.push(CompactString::from(e));
        self
    }

    /// Provide known blob SHAs (keyed by canonical raw-URL key) so unchanged
    /// files are revalidated without any fetch.
    #[must_use]
    pub fn validators(
        mut self,
        entries: impl IntoIterator<Item = (UrlKey, CompactString)>,
    ) -> Self {
        self.validators = entries.into_iter().collect();
        self
    }

    /// Files revalidated (blob SHA unchanged, nothing fetched) during the most
    /// recent discover. Same contract as the crawler's: this is the only place
    /// those checks are visible, and a freshness tracker consumes it.
    #[must_use]
    pub fn take_revalidated(&self) -> Vec<Url> {
        std::mem::take(&mut self.revalidated.lock().expect("revalidated lock poisoned"))
    }

    /// Files whose blob fetch failed during the most recent discover.
    #[must_use]
    pub fn failed_count(&self) -> u32 {
        self.failed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The raw-content URL for a repo-relative path at the resolved ref.
    fn raw_url(&self, ref_name: &str, path: &str) -> Result<Url> {
        let s = format!(
            "{}/{}/{}/{ref_name}/{path}",
            self.raw_base, self.spec.owner, self.spec.repo
        );
        Url::parse(&s).map_err(|e| Error::invalid_url(format!("{s:?}: {e}")))
    }

    /// One authenticated JSON GET against the API host.
    async fn api_json(&self, url: &str) -> Result<Vec<u8>> {
        let url = Url::parse(url).map_err(|e| Error::invalid_url(format!("{url:?}: {e}")))?;
        let resource = Resource::new(url);
        let opts = FetchOptions {
            if_none_match: None,
            // Tree responses for large repos run to a few MB of JSON; give
            // them their own generous ceiling independent of per-file limits.
            max_bytes: Some(64 * 1024 * 1024),
            user_agent: None,
        };
        let got = self.fetcher.fetch(&resource, opts).await?;
        Ok(got.payload.into_bytes().await?.to_vec())
    }

    /// Whether a tree entry survives the subdir/depth/kind/extension/size
    /// filters. Returns the repo-relative path when it does.
    fn admit<'e>(&self, entry: &'e TreeEntry) -> Option<&'e str> {
        if entry.kind != "blob" {
            return None;
        }
        let sub = self.spec.subdir.as_str();
        if !sub.is_empty() {
            let rest = entry.path.strip_prefix(sub)?;
            if !rest.starts_with('/') {
                return None; // "stacksfoo" must not match subdir "stacks"
            }
        }
        if let Some(depth) = self.depth {
            let rel_start = if sub.is_empty() { 0 } else { sub.len() + 1 };
            let rel = &entry.path[rel_start..];
            let dir_levels = rel.matches('/').count();
            if dir_levels > usize::from(depth) {
                return None;
            }
        }
        if entry.size.is_some_and(|s| s > self.max_bytes) {
            return None; // skipped for free: no fetch was ever issued
        }
        let kind = ResourceKind::from_path(&entry.path);
        if !kind.is_indexable() {
            return None;
        }
        if !self.extensions.is_empty() {
            let ext = entry.path.rsplit('/').next().unwrap_or("").rsplit_once('.');
            let matched = ext.is_some_and(|(_, e)| {
                self.extensions
                    .iter()
                    .any(|a| a.as_str().eq_ignore_ascii_case(e))
            });
            if !matched {
                return None;
            }
        }
        Some(&entry.path)
    }
}

/// Fetch one blob; the future owns its URL so many can run in a
/// `FuturesUnordered` (the same shape as the crawler's `fetch_one`).
async fn fetch_blob<F: Fetcher>(
    fetcher: &F,
    url: Url,
    sha: CompactString,
    max_bytes: u64,
) -> (Url, CompactString, Result<bytes::Bytes>) {
    let resource = Resource::new(url);
    let opts = FetchOptions {
        if_none_match: None,
        max_bytes: Some(max_bytes),
        user_agent: None,
    };
    let outcome = match fetcher.fetch(&resource, opts).await {
        Ok(got) => got.payload.into_bytes().await,
        Err(e) => Err(e),
    };
    (resource.url, sha, outcome)
}

impl<F: Fetcher> Source for GitHubSource<F> {
    type Pages<'a>
        = impl futures::Stream<Item = Result<Page<'a>>> + Send + 'a
    where
        Self: 'a;

    fn kind(&self) -> &'static str {
        "github"
    }

    #[allow(clippy::too_many_lines)] // one cohesive list-diff-fetch pass
    fn discover<'a>(&'a self, _root: &'a SourceRef) -> Self::Pages<'a> {
        async_stream::try_stream! {
            // Per-discover records, mirroring the crawler.
            self.failed.store(0, std::sync::atomic::Ordering::Relaxed);
            self.revalidated.lock().expect("revalidated lock poisoned").clear();

            // Resolve the ref: empty means "the repository's default branch".
            let ref_name = if self.spec.ref_name.is_empty() {
                let url = format!(
                    "{}/repos/{}/{}",
                    self.api_base, self.spec.owner, self.spec.repo
                );
                let body = self.api_json(&url).await?;
                let repo: RepoResponse = serde_json::from_slice(&body).map_err(|e| {
                    Error::format(format!("github repo response: {e}"))
                })?;
                CompactString::from(repo.default_branch)
            } else {
                self.spec.ref_name.clone()
            };

            // THE call: every blob in the tree, with its SHA, in one request.
            let url = format!(
                "{}/repos/{}/{}/git/trees/{}?recursive=1",
                self.api_base, self.spec.owner, self.spec.repo, ref_name
            );
            let body = self.api_json(&url).await?;
            let tree: TreeResponse = serde_json::from_slice(&body)
                .map_err(|e| Error::format(format!("github tree response: {e}")))?;
            if tree.truncated {
                Err(Error::crawl(
                    "github",
                    "tree listing truncated by GitHub (repository too large); \
                     narrow the spec to a subdirectory",
                ))?;
            }

            // Diff against the validators: unchanged SHAs are recorded and
            // skipped; the rest are fetched with bounded concurrency.
            let mut to_fetch: Vec<(Url, CompactString)> = Vec::new();
            for entry in &tree.tree {
                let Some(path) = self.admit(entry) else { continue };
                let url = self.raw_url(&ref_name, path)?;
                let sha = CompactString::from(entry.sha.as_str());
                if self
                    .validators
                    .get(&UrlKey::from_url(&url))
                    .is_some_and(|known| *known == sha)
                {
                    self.revalidated
                        .lock()
                        .expect("revalidated lock poisoned")
                        .push(url);
                    continue;
                }
                to_fetch.push((url, sha));
            }

            let mut queue = to_fetch.into_iter();
            let mut in_flight = FuturesUnordered::new();
            loop {
                while in_flight.len() < self.concurrency {
                    let Some((url, sha)) = queue.next() else { break };
                    in_flight.push(fetch_blob(&self.fetcher, url, sha, self.max_bytes));
                }
                let Some((url, sha, outcome)) = in_flight.next().await else { break };
                match outcome {
                    Ok(bytes) => {
                        let kind = ResourceKind::from_path(url.path());
                        let resource = Resource::new(url).with_kind(kind).with_etag(sha);
                        yield Page::new(resource, DocPayload::Owned(bytes));
                    }
                    Err(_) => {
                        // One dead blob must not sink the sync; it is counted
                        // and the caller reports it, exactly like the crawler.
                        self.failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(s: &str) -> GithubSpec {
        GithubSpec::parse(s).unwrap()
    }

    #[test]
    fn url_form_parses_the_default_corpus_shape() {
        let s = spec("https://github.com/telus/bi-layer-docs/tree/main/stacks");
        assert_eq!(s.owner, "telus");
        assert_eq!(s.repo, "bi-layer-docs");
        assert_eq!(s.ref_name, "main");
        assert_eq!(s.subdir, "stacks");
    }

    #[test]
    fn url_form_variants() {
        assert_eq!(
            spec("https://github.com/o/r"),
            GithubSpec {
                owner: "o".into(),
                repo: "r".into(),
                ref_name: "".into(),
                subdir: "".into()
            }
        );
        assert_eq!(spec("https://github.com/o/r.git").repo, "r");
        let deep = spec("https://github.com/o/r/tree/v1.2.3/a/b/c");
        assert_eq!(deep.ref_name, "v1.2.3");
        assert_eq!(deep.subdir, "a/b/c");
        assert!(GithubSpec::parse("https://github.com/o").is_err());
        assert!(GithubSpec::parse("https://github.com/o/r/pulls").is_err());
    }

    #[test]
    fn ref_grammar_parses_both_shapes() {
        let s = spec("github:telus/bi-layer-docs@main//stacks");
        assert_eq!(
            (s.owner.as_str(), s.repo.as_str()),
            ("telus", "bi-layer-docs")
        );
        assert_eq!((s.ref_name.as_str(), s.subdir.as_str()), ("main", "stacks"));
        // Slashed branch names are unambiguous here — the reason this grammar exists.
        let s = spec("github:o/r@feat/x//docs/examples");
        assert_eq!(s.ref_name, "feat/x");
        assert_eq!(s.subdir, "docs/examples");
        assert_eq!(spec("github:o/r").ref_name, "");
        assert!(GithubSpec::parse("github:r").is_err());
        assert!(GithubSpec::parse("github:o/r@").is_err());
        assert!(GithubSpec::parse("gitlab:o/r").is_err());
    }

    #[test]
    fn matches_routes_specs_not_ordinary_urls() {
        assert!(GithubSpec::matches("github:o/r@main"));
        assert!(GithubSpec::matches("https://github.com/o/r/tree/main/x"));
        assert!(!GithubSpec::matches("https://docs.example.com/"));
        assert!(!GithubSpec::matches(
            "https://raw.githubusercontent.com/o/r/main/a.md"
        ));
    }

    #[test]
    fn github_auth_confines_the_token_to_the_two_hosts() {
        let auth = GithubAuth::new(Some("sekrit".into()));
        let cred = |u: &str| auth.credential(&Resource::new(Url::parse(u).unwrap()));
        assert!(matches!(
            cred("https://api.github.com/repos/o/r"),
            Credential::Bearer(_)
        ));
        assert!(matches!(
            cred("https://raw.githubusercontent.com/o/r/main/a.md"),
            Credential::Bearer(_)
        ));
        // Off-host: a markdown file can link anywhere; the token must not follow.
        assert!(matches!(
            cred("https://evil.example.com/"),
            Credential::Anonymous
        ));
        assert!(matches!(
            cred("https://github.com/o/r"),
            Credential::Anonymous
        ));
        // Anonymous provider never sends anything.
        let anon = GithubAuth::new(None);
        assert!(matches!(
            anon.credential(&Resource::new(
                Url::parse("https://api.github.com/x").unwrap()
            )),
            Credential::Anonymous
        ));
        // The secret never appears in Debug output.
        assert!(!format!("{auth:?}").contains("sekrit"));
    }

    #[test]
    fn admit_applies_every_filter() {
        let src = GitHubSource::new(DummyFetcher, spec("github:o/r@main//stacks"))
            .depth(1)
            .max_bytes(1000)
            .accept_extension("md")
            .accept_extension(".YAML");
        let entry = |path: &str, size: u64| TreeEntry {
            path: path.into(),
            kind: "blob".into(),
            sha: "abc".into(),
            size: Some(size),
        };
        // In subdir, depth 1, small, .md → admitted.
        assert!(src.admit(&entry("stacks/api/README.md", 10)).is_some());
        // Depth 0 (directly in subdir) also fine.
        assert!(src.admit(&entry("stacks/Pulumi.yaml", 10)).is_some());
        // Too deep (2 levels below subdir).
        assert!(src.admit(&entry("stacks/a/b/x.md", 10)).is_none());
        // Outside the subdir, including the prefix-collision trap.
        assert!(src.admit(&entry("other/x.md", 10)).is_none());
        assert!(src.admit(&entry("stacksfoo/x.md", 10)).is_none());
        // Oversized: filtered from the tree size, no fetch.
        assert!(src.admit(&entry("stacks/big.md", 100_000)).is_none());
        // Extension filter.
        assert!(src.admit(&entry("stacks/query.sql", 10)).is_none());
        // Non-indexable kind is dropped before the extension filter matters.
        assert!(src.admit(&entry("stacks/logo.png", 10)).is_none());
        // Trees (directories) never admit.
        let dir = TreeEntry {
            path: "stacks/api".into(),
            kind: "tree".into(),
            sha: "d".into(),
            size: None,
        };
        assert!(src.admit(&dir).is_none());
    }

    /// `admit()` is pure over the entry; a fetcher is never touched in these tests.
    struct DummyFetcher;
    impl Fetcher for DummyFetcher {
        type FetchFuture<'a> = std::future::Ready<Result<crate::fetch::Fetched<'a>>>;
        fn fetch<'a>(
            &'a self,
            resource: &'a Resource,
            _opts: FetchOptions<'a>,
        ) -> Self::FetchFuture<'a> {
            std::future::ready(Err(Error::not_found(resource.url.as_str())))
        }
    }
}
