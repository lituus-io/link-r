// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The HTTP(S) fetcher (rustls), generic over a consumer-injected [`AuthProvider`].
//!
//! This is where async auth refresh flows through the GAT [`Fetcher`] unboxed:
//! `needs_refresh()` (atomic) → maybe `refresh().await` (the only async auth step)
//! → `credential()` (sync, lock-free) → apply header → GET.

use crate::auth::{AuthProvider, Credential};
use crate::error::{Error, Result};
use crate::fetch::{FetchMeta, FetchOptions, Fetched, Fetcher};
use crate::payload::DocPayload;
use crate::resource::{Resource, ResourceKind};
use bytes::BytesMut;
use compact_str::CompactString;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, ETAG, IF_NONE_MATCH, USER_AGENT};
use std::time::{Duration, Instant};

/// Default User-Agent sent with crawl requests.
pub(crate) const DEFAULT_UA: &str = concat!("link-r/", env!("CARGO_PKG_VERSION"));
/// Connection-establishment timeout for the default client.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-request timeout (headers + body) for the default client.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum redirects the default client follows before erroring.
const MAX_REDIRECTS: usize = 5;

/// An HTTP(S) fetcher over `reqwest` + rustls.
#[derive(Debug)]
pub struct HttpFetcher<A: AuthProvider> {
    client: reqwest::Client,
    auth: A,
    user_agent: CompactString,
}

impl<A: AuthProvider> HttpFetcher<A> {
    /// Build a fetcher with a default client: rustls, a bounded redirect policy,
    /// and connect/request timeouts.
    ///
    /// # Errors
    /// Returns [`Error::Backend`] if the HTTP client cannot be constructed.
    pub fn new(auth: A) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(DEFAULT_UA)
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .build()
            .map_err(|e| Error::backend("reqwest", e.to_string()))?;
        Ok(Self {
            client,
            auth,
            user_agent: CompactString::from(DEFAULT_UA),
        })
    }

    /// Build with a caller-provided client (e.g. custom timeouts/proxy).
    #[must_use]
    pub fn with_client(client: reqwest::Client, auth: A) -> Self {
        Self {
            client,
            auth,
            user_agent: CompactString::from(DEFAULT_UA),
        }
    }

    /// Override the User-Agent.
    #[must_use]
    pub fn user_agent(mut self, ua: impl Into<CompactString>) -> Self {
        self.user_agent = ua.into();
        self
    }
}

fn map_reqwest(e: &reqwest::Error, started: Instant) -> Error {
    if e.is_timeout() {
        Error::Timeout {
            duration_ms: started.elapsed().as_millis() as u64,
        }
    } else {
        Error::backend("reqwest", e.to_string())
    }
}

/// A body-stream error that can be turned into a crate [`Error`], carrying the
/// request start so a timeout mid-body reports a real elapsed duration. Static
/// dispatch keeps [`collect_capped`] free of `dyn`.
trait StreamErr {
    fn into_error(self, started: Instant) -> Error;
}

impl StreamErr for reqwest::Error {
    fn into_error(self, started: Instant) -> Error {
        map_reqwest(&self, started)
    }
}

#[cfg(test)]
impl StreamErr for std::convert::Infallible {
    fn into_error(self, _started: Instant) -> Error {
        match self {}
    }
}

/// Drain a byte stream into contiguous `Bytes`, enforcing an incremental size cap
/// so a chunked response (no `Content-Length`) cannot bypass `max_bytes`. Factored
/// out for direct unit testing over any chunk stream.
async fn collect_capped<S, E>(
    mut stream: S,
    hint: Option<u64>,
    max: Option<u64>,
    status: u16,
    started: Instant,
) -> Result<bytes::Bytes>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
    E: StreamErr,
{
    // Pre-size to the smaller of the length hint and the cap, so a hostile hint
    // can't force a huge allocation.
    let cap = match (hint, max) {
        (Some(h), Some(m)) => h.min(m),
        (Some(h), None) => h,
        (None, _) => 0,
    };
    let cap = usize::try_from(cap).unwrap_or(0).min(1 << 20);
    let mut body = BytesMut::with_capacity(cap);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.into_error(started))?;
        if let Some(max) = max {
            if body.len() as u64 + chunk.len() as u64 > max {
                return Err(Error::http(status, format!("body exceeds max {max} bytes")));
            }
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

impl<A: AuthProvider> Fetcher for HttpFetcher<A> {
    type FetchFuture<'a>
        = impl std::future::Future<Output = Result<Fetched<'a>>> + Send + 'a
    where
        Self: 'a;

    fn fetch<'a>(
        &'a self,
        resource: &'a Resource,
        opts: FetchOptions<'a>,
    ) -> Self::FetchFuture<'a> {
        async move {
            let started = Instant::now();
            // The only async auth step — its concrete future keeps this unboxed.
            if self.auth.needs_refresh() {
                self.auth.refresh().await?;
            }

            let ua = opts.user_agent.unwrap_or(self.user_agent.as_str());
            let mut req = self.client.get(resource.url.clone()).header(USER_AGENT, ua);

            match self.auth.credential(resource) {
                Credential::Bearer(token) => {
                    req = req.header(AUTHORIZATION, format!("Bearer {token}"));
                }
                Credential::Header { name, value } => {
                    req = req.header(name, value.as_ref());
                }
                Credential::Anonymous | Credential::Presigned => {}
            }
            if let Some(etag) = opts.if_none_match {
                req = req.header(IF_NONE_MATCH, etag);
            }

            let resp = req.send().await.map_err(|e| map_reqwest(&e, started))?;
            let status = resp.status().as_u16();
            match status {
                304 => return Err(Error::not_modified(resource.url.as_str())),
                401 => return Err(Error::unauthenticated(resource.url.as_str())),
                403 => return Err(Error::permission_denied(resource.url.as_str())),
                429 => return Err(rate_limited(&resp)),
                s if s >= 400 => {
                    return Err(Error::http(s, resource.url.as_str()));
                }
                _ => {}
            }

            // The URL after any redirects reqwest followed; only surfaced when it
            // actually differs, so the crawler keys/scopes on the real location.
            let final_url = (resp.url() != &resource.url).then(|| resp.url().clone());
            let classify_path = resp.url().path().to_owned();

            let kind = resp
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(ResourceKind::from_content_type)
                .filter(|k| *k != ResourceKind::Unknown)
                .unwrap_or_else(|| ResourceKind::from_path(&classify_path));

            let etag = resp
                .headers()
                .get(ETAG)
                .and_then(|v| v.to_str().ok())
                .map(CompactString::from);

            // Enforce max_bytes early via Content-Length when present, then stream
            // the body with an incremental cap so a chunked response can't bypass it.
            let hint = resp.content_length();
            if let (Some(max), Some(len)) = (opts.max_bytes, hint) {
                if len > max {
                    return Err(Error::http(status, format!("body {len} exceeds max {max}")));
                }
            }
            let bytes =
                collect_capped(resp.bytes_stream(), hint, opts.max_bytes, status, started).await?;

            Ok(Fetched {
                meta: FetchMeta {
                    kind,
                    etag,
                    status,
                    final_url,
                },
                payload: DocPayload::Owned(bytes),
            })
        }
    }
}

/// Floor and ceiling for a server-suggested retry delay. The ceiling keeps a
/// hostile or misconfigured `Retry-After: <far future date>` from parking a
/// refresh for hours; the crawl loop applies its own tighter clamp on top.
const RETRY_AFTER_MIN_MS: u64 = 1_000;
const RETRY_AFTER_MAX_MS: u64 = 300_000;

fn rate_limited(resp: &reqwest::Response) -> Error {
    // Honor Retry-After in both RFC 9110 forms (delay-seconds and IMF-fixdate),
    // then x-ratelimit-reset (epoch seconds); clamp into a sane window.
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let header = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok());
    let retry_after_ms = header("retry-after")
        .and_then(|s| {
            let s = s.trim();
            s.parse::<u64>()
                .ok()
                .map(|secs| secs.saturating_mul(1000))
                .or_else(|| parse_imf_fixdate_ms(s).map(|at| at.saturating_sub(now_ms)))
        })
        .or_else(|| {
            header("x-ratelimit-reset")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|epoch_s| epoch_s.saturating_mul(1000).saturating_sub(now_ms))
        })
        .unwrap_or(RETRY_AFTER_MIN_MS)
        .clamp(RETRY_AFTER_MIN_MS, RETRY_AFTER_MAX_MS);
    Error::rate_limited(retry_after_ms)
}

/// Parse an RFC 9110 IMF-fixdate ("Sun, 06 Nov 1994 08:49:37 GMT") into epoch
/// milliseconds. Returns `None` for anything else; callers fall back to the
/// default delay rather than erroring, matching the delay-seconds path.
fn parse_imf_fixdate_ms(s: &str) -> Option<u64> {
    let rest = s.split_once(", ").map_or(s, |(_, r)| r);
    let mut parts = rest.split_ascii_whitespace();
    let day: i64 = parts.next()?.parse().ok()?;
    let mon: i64 = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut hms = parts.next()?.split(':');
    let h: u64 = hms.next()?.parse().ok()?;
    let m: u64 = hms.next()?.parse().ok()?;
    let sec: u64 = hms.next()?.parse().ok()?;
    if parts.next() != Some("GMT") || !(1..=31).contains(&day) || h > 23 || m > 59 || sec > 60 {
        return None;
    }
    // Days-from-civil (Gregorian, proleptic); epoch day 0 = 1970-01-01.
    let y = year - i64::from(mon <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (mon + if mon > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    if days < 0 {
        return None;
    }
    Some((days as u64 * 86_400 + h * 3600 + m * 60 + sec) * 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AnonymousAuth;

    #[test]
    fn imf_fixdate_parses_known_instants() {
        assert_eq!(parse_imf_fixdate_ms("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        // RFC 9110's own example date.
        assert_eq!(
            parse_imf_fixdate_ms("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777_000)
        );
        // Weekday prefix is optional noise for us.
        assert_eq!(
            parse_imf_fixdate_ms("06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777_000)
        );
    }

    #[test]
    fn imf_fixdate_rejects_garbage() {
        for s in [
            "",
            "120",
            "tomorrow",
            "Sun, 32 Nov 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 25:00:00 GMT",
            "Sun, 06 Nov 1994 08:49:37 PST",
            "Sun, 06 Foo 1994 08:49:37 GMT",
        ] {
            assert_eq!(parse_imf_fixdate_ms(s), None, "should reject {s:?}");
        }
    }

    #[test]
    fn constructs_with_anonymous_auth() {
        let fetcher = HttpFetcher::new(AnonymousAuth).unwrap();
        assert_eq!(fetcher.user_agent, CompactString::from(DEFAULT_UA));
    }

    #[test]
    fn user_agent_override() {
        let fetcher = HttpFetcher::new(AnonymousAuth)
            .unwrap()
            .user_agent("my-bot/1.0");
        assert_eq!(fetcher.user_agent, "my-bot/1.0");
    }

    fn chunks(parts: &[&[u8]]) -> impl futures::Stream<Item = std::result::Result<bytes::Bytes, std::convert::Infallible>>
    {
        let owned: Vec<_> = parts
            .iter()
            .map(|p| Ok(bytes::Bytes::copy_from_slice(p)))
            .collect();
        futures::stream::iter(owned)
    }

    #[tokio::test]
    async fn collect_capped_under_cap_passes() {
        let started = Instant::now();
        let out = collect_capped(chunks(&[b"hello ", b"world"]), None, Some(1024), 200, started)
            .await
            .unwrap();
        assert_eq!(&out[..], b"hello world");
    }

    #[tokio::test]
    async fn collect_capped_over_cap_errors_without_length_hint() {
        // The chunked-bypass regression: no length hint, cap exceeded mid-stream.
        let started = Instant::now();
        let err = collect_capped(chunks(&[b"aaaa", b"bbbb", b"cccc"]), None, Some(8), 200, started)
            .await
            .unwrap_err();
        match err {
            Error::Http { status, .. } => assert_eq!(status, 200),
            other => panic!("expected Http cap error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn collect_capped_no_max_reads_all() {
        let started = Instant::now();
        let out = collect_capped(chunks(&[b"a", b"b", b"c"]), Some(3), None, 200, started)
            .await
            .unwrap();
        assert_eq!(&out[..], b"abc");
    }
}
