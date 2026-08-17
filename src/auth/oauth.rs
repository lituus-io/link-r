// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! An expiring-token provider with lock-free reads and single-flight refresh.
//!
//! The consumer implements only [`TokenSource`] (one async call into their
//! Google/GDrive/OAuth SDK); `link-r` owns caching, expiry, and de-duplicating
//! concurrent refreshes. The steady-state read ([`AuthProvider::credential`]) is
//! a lock-free `ArcSwap` load; only the rare refresh awaits, and because
//! [`AuthProvider::RefreshFuture`] is a concrete associated type the refresh flows
//! through the GAT [`Fetcher`](crate::fetch::Fetcher) with no boxing.

use crate::auth::{AuthProvider, Credential};
use crate::error::Result;
use crate::resource::Resource;
use arc_swap::ArcSwap;
use std::borrow::Cow;
use std::future::Future;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Sentinel meaning "no token has been fetched yet".
const NO_TOKEN: i64 = i64::MIN;
/// Default time before expiry at which a refresh is triggered.
const DEFAULT_BUFFER: Duration = Duration::from_secs(60);

/// A freshly minted access token from the consumer's auth backend.
#[derive(Clone, Debug)]
pub struct FreshToken {
    /// The bearer token value.
    pub bearer: String,
    /// How long until it expires.
    pub expires_in: Duration,
}

/// How to obtain a fresh token. Implemented by the embedding crate (its SDK call).
pub trait TokenSource: Send + Sync {
    /// The future returned by [`TokenSource::fetch`].
    type Fut<'a>: Future<Output = Result<FreshToken>> + Send + 'a
    where
        Self: 'a;

    /// Fetch a fresh token.
    fn fetch(&self) -> Self::Fut<'_>;
}

/// An [`AuthProvider`] that caches a bearer token and refreshes it before expiry.
#[derive(Debug)]
pub struct OAuthRefreshAuth<R: TokenSource> {
    token: ArcSwap<String>,
    expires_at_ms: AtomicI64,
    inflight: futures::lock::Mutex<()>,
    buffer: Duration,
    source: R,
}

impl<R: TokenSource> OAuthRefreshAuth<R> {
    /// Create a provider that refreshes 60s before expiry.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self::with_buffer(source, DEFAULT_BUFFER)
    }

    /// Create with an explicit pre-expiry refresh buffer.
    #[must_use]
    pub fn with_buffer(source: R, buffer: Duration) -> Self {
        Self {
            token: ArcSwap::from_pointee(String::new()),
            expires_at_ms: AtomicI64::new(NO_TOKEN),
            inflight: futures::lock::Mutex::new(()),
            buffer,
            source,
        }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

impl<R: TokenSource> AuthProvider for OAuthRefreshAuth<R> {
    type RefreshFuture<'a>
        = impl Future<Output = Result<()>> + Send + 'a
    where
        Self: 'a;

    fn credential(&self, _resource: &Resource) -> Credential<'_> {
        // Lock-free read; the token is cloned out (small) since it lives behind an
        // ArcSwap guard that cannot be borrowed past this call.
        let token = self.token.load();
        Credential::Bearer(Cow::Owned((**token).clone()))
    }

    fn needs_refresh(&self) -> bool {
        let exp = self.expires_at_ms.load(Ordering::Acquire);
        exp == NO_TOKEN || now_ms() + self.buffer.as_millis() as i64 >= exp
    }

    fn refresh(&self) -> Self::RefreshFuture<'_> {
        async move {
            // Single-flight: only one task refreshes; others wait then re-check.
            let _guard = self.inflight.lock().await;
            if !self.needs_refresh() {
                return Ok(());
            }
            let fresh = self.source.fetch().await?;
            let expires_at = now_ms() + fresh.expires_in.as_millis() as i64;
            self.token.store(Arc::new(fresh.bearer));
            self.expires_at_ms.store(expires_at, Ordering::Release);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Ready;
    use std::sync::atomic::AtomicUsize;
    use url::Url;

    /// A mock token source counting how many times it is asked for a token.
    struct MockSource {
        calls: AtomicUsize,
        ttl: Duration,
    }

    impl TokenSource for MockSource {
        type Fut<'a> = Ready<Result<FreshToken>>;
        fn fetch(&self) -> Self::Fut<'_> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(FreshToken {
                bearer: format!("token-{n}"),
                expires_in: self.ttl,
            }))
        }
    }

    fn resource() -> Resource {
        Resource::new(Url::parse("https://x.dev/a").unwrap())
    }

    #[tokio::test]
    async fn refreshes_then_caches() {
        let auth = OAuthRefreshAuth::new(MockSource {
            calls: AtomicUsize::new(0),
            ttl: Duration::from_secs(3600),
        });
        assert!(auth.needs_refresh(), "no token yet");
        auth.refresh().await.unwrap();

        match auth.credential(&resource()) {
            Credential::Bearer(v) => assert_eq!(v, "token-0"),
            other => panic!("expected bearer, got {other:?}"),
        }
        // Fresh 1-hour token → no refresh needed, and a second refresh is a no-op.
        assert!(!auth.needs_refresh());
        auth.refresh().await.unwrap();
        assert_eq!(auth.source.calls.load(Ordering::SeqCst), 1, "single fetch");
    }

    #[tokio::test]
    async fn expired_token_triggers_refresh() {
        let auth = OAuthRefreshAuth::with_buffer(
            MockSource {
                calls: AtomicUsize::new(0),
                ttl: Duration::from_millis(0), // already expired on arrival
            },
            Duration::from_secs(0),
        );
        auth.refresh().await.unwrap();
        assert!(auth.needs_refresh(), "zero-ttl token is immediately stale");
    }
}
