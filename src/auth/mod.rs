// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Consumer-injected authentication.
//!
//! `link-r` hardcodes no credentials. It is generic over an [`AuthProvider`]; the
//! crate embedding it supplies one (a PAT, an OAuth/GDrive token, a signed URL).
//! The per-fetch path — [`AuthProvider::credential`] — is synchronous and
//! lock-free; only [`AuthProvider::refresh`] is async (rare, for expiring tokens),
//! and because its future type is an associated type it flows through the GAT
//! [`Fetcher`](crate::fetch::Fetcher) with no boxing.
//!
//! For the rare case where the provider is chosen at runtime from config, the
//! object-safe [`DynAuthProvider`] escape hatch is the single justified `Box<dyn>`
//! in the crate (one boxed future per refresh, ~hourly, off the hot path).

pub mod anonymous;
#[cfg(feature = "oauth")]
pub mod oauth;
pub mod static_token;

pub use anonymous::AnonymousAuth;
#[cfg(feature = "oauth")]
pub use oauth::{FreshToken, OAuthRefreshAuth, TokenSource};
pub use static_token::StaticTokenAuth;

use crate::error::Result;
use crate::resource::Resource;
use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;

/// The credential to apply to a request, returned cheaply (often borrowed) per fetch.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Credential<'a> {
    /// No credential (public resource).
    Anonymous,
    /// `Authorization: Bearer <token>`.
    Bearer(Cow<'a, str>),
    /// A custom header `name: value` (e.g. `x-api-key`).
    Header {
        /// The header name.
        name: &'static str,
        /// The header value.
        value: Cow<'a, str>,
    },
    /// The URL is already signed; send it verbatim with no added credential.
    Presigned,
}

/// Supplies credentials for outbound requests.
pub trait AuthProvider: Send + Sync {
    /// The future returned by [`AuthProvider::refresh`]. Static providers use
    /// [`std::future::Ready`]; OAuth uses its token-exchange future.
    type RefreshFuture<'a>: Future<Output = Result<()>> + Send + 'a
    where
        Self: 'a;

    /// The credential to apply to a request for `resource`. Synchronous and
    /// lock-free — this is the per-fetch hot path.
    fn credential(&self, resource: &Resource) -> Credential<'_>;

    /// Whether the cached credential is missing or near expiry. Default: never.
    fn needs_refresh(&self) -> bool {
        false
    }

    /// Refresh an expiring credential. Static providers return a ready no-op.
    fn refresh(&self) -> Self::RefreshFuture<'_>;
}

/// An object-safe sibling of [`AuthProvider`] for runtime-selected providers.
///
/// This is the only `Box<dyn>` the crate condones, and only for the open-set
/// "provider chosen from config" case; a known set of providers should use a
/// consumer-side `enum` instead (zero allocation).
pub trait DynAuthProvider: Send + Sync {
    /// See [`AuthProvider::credential`].
    fn credential(&self, resource: &Resource) -> Credential<'_>;
    /// See [`AuthProvider::needs_refresh`].
    fn needs_refresh(&self) -> bool;
    /// See [`AuthProvider::refresh`]; the future is boxed here (the accepted cost).
    fn refresh_boxed(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

impl<A: AuthProvider> DynAuthProvider for A {
    fn credential(&self, resource: &Resource) -> Credential<'_> {
        AuthProvider::credential(self, resource)
    }
    fn needs_refresh(&self) -> bool {
        AuthProvider::needs_refresh(self)
    }
    fn refresh_boxed(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.refresh())
    }
}

/// Re-enter the static world: a boxed [`DynAuthProvider`] is itself an
/// [`AuthProvider`], so `Fetcher<Box<dyn DynAuthProvider>>` compiles.
impl AuthProvider for Box<dyn DynAuthProvider> {
    type RefreshFuture<'a>
        = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>
    where
        Self: 'a;

    fn credential(&self, resource: &Resource) -> Credential<'_> {
        (**self).credential(resource)
    }
    fn needs_refresh(&self) -> bool {
        (**self).needs_refresh()
    }
    fn refresh(&self) -> Self::RefreshFuture<'_> {
        (**self).refresh_boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn resource() -> Resource {
        Resource::new(Url::parse("https://x.dev/a").unwrap())
    }

    #[tokio::test]
    async fn dyn_auth_provider_roundtrips() {
        let boxed: Box<dyn DynAuthProvider> = Box::new(StaticTokenAuth::bearer("secret"));
        // used as a static AuthProvider via the blanket re-entry impl
        let cred = AuthProvider::credential(&boxed, &resource());
        assert!(matches!(cred, Credential::Bearer(_)));
        assert!(!AuthProvider::needs_refresh(&boxed));
        AuthProvider::refresh(&boxed).await.unwrap();
    }
}
