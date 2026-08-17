// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! A fixed-credential provider (PAT / bearer / custom header).
//!
//! The token is held in a [`SecretString`] so it is redacted in `Debug` and
//! zeroized on drop; it is exposed only when building a request header.

use crate::auth::{AuthProvider, Credential};
use crate::error::Result;
use crate::resource::Resource;
use compact_str::CompactString;
use secrecy::{ExposeSecret, SecretString};
use std::borrow::Cow;
use std::future::Ready;

#[derive(Clone, Copy, Debug)]
enum TokenKind {
    Bearer,
    Header(&'static str),
}

/// An [`AuthProvider`] holding one fixed secret credential.
///
/// A credential may be *host-scoped* ([`StaticTokenAuth::bearer_scoped`]): it is
/// then sent only to that host (or its subdomains), so a page that links to a
/// foreign host reached via an in-scope crawl can never exfiltrate the token.
#[derive(Debug)]
pub struct StaticTokenAuth {
    kind: TokenKind,
    token: SecretString,
    /// When set, the credential is only sent to this host or its subdomains.
    scope_host: Option<CompactString>,
}

impl StaticTokenAuth {
    /// `Authorization: Bearer <token>`, sent to any host.
    ///
    /// Prefer [`StaticTokenAuth::bearer_scoped`] for private crawls so the token is
    /// confined to the intended host.
    #[must_use]
    pub fn bearer(token: impl Into<String>) -> Self {
        Self {
            kind: TokenKind::Bearer,
            token: secret(token),
            scope_host: None,
        }
    }

    /// `Authorization: Bearer <token>`, sent only to `host` and its subdomains.
    #[must_use]
    pub fn bearer_scoped(token: impl Into<String>, host: impl Into<CompactString>) -> Self {
        Self {
            kind: TokenKind::Bearer,
            token: secret(token),
            scope_host: Some(host.into()),
        }
    }

    /// A GitHub personal access token (sent as a bearer token, which the GitHub
    /// REST API accepts).
    #[must_use]
    pub fn github_pat(token: impl Into<String>) -> Self {
        Self::bearer(token)
    }

    /// A custom header credential, e.g. `x-api-key: <token>`.
    #[must_use]
    pub fn header(name: &'static str, token: impl Into<String>) -> Self {
        Self {
            kind: TokenKind::Header(name),
            token: secret(token),
            scope_host: None,
        }
    }
}

fn secret(s: impl Into<String>) -> SecretString {
    SecretString::new(s.into().into_boxed_str())
}

/// Whether `host` equals `base` or is a subdomain of it (`a.b.dev` under `b.dev`).
fn host_in_scope(host: &str, base: &str) -> bool {
    host == base
        || (host.len() > base.len()
            && host.ends_with(base)
            && host.as_bytes()[host.len() - base.len() - 1] == b'.')
}

impl AuthProvider for StaticTokenAuth {
    type RefreshFuture<'a> = Ready<Result<()>>;

    fn credential(&self, resource: &Resource) -> Credential<'_> {
        // A scoped token is withheld from any host outside its scope.
        if let Some(scope) = &self.scope_host {
            let in_scope = resource
                .url
                .host_str()
                .is_some_and(|h| host_in_scope(h, scope));
            if !in_scope {
                return Credential::Anonymous;
            }
        }
        let value = Cow::Borrowed(self.token.expose_secret());
        match self.kind {
            TokenKind::Bearer => Credential::Bearer(value),
            TokenKind::Header(name) => Credential::Header { name, value },
        }
    }

    fn refresh(&self) -> Self::RefreshFuture<'_> {
        std::future::ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn resource() -> Resource {
        Resource::new(Url::parse("https://x.dev/a").unwrap())
    }

    #[test]
    fn bearer_exposes_token_only_in_credential() {
        let auth = StaticTokenAuth::bearer("s3cr3t");
        match auth.credential(&resource()) {
            Credential::Bearer(v) => assert_eq!(v, "s3cr3t"),
            other => panic!("expected bearer, got {other:?}"),
        }
        // Debug must not leak the secret.
        assert!(!format!("{auth:?}").contains("s3cr3t"));
    }

    #[test]
    fn custom_header_credential() {
        let auth = StaticTokenAuth::header("x-api-key", "abc");
        match auth.credential(&resource()) {
            Credential::Header { name, value } => {
                assert_eq!(name, "x-api-key");
                assert_eq!(value, "abc");
            }
            other => panic!("expected header, got {other:?}"),
        }
    }

    #[test]
    fn scoped_token_confined_to_host_and_subdomains() {
        let auth = StaticTokenAuth::bearer_scoped("s3cr3t", "github.com");
        let on = Resource::new(Url::parse("https://github.com/o/r").unwrap());
        let sub = Resource::new(Url::parse("https://api.github.com/x").unwrap());
        let off = Resource::new(Url::parse("https://evil.dev/x").unwrap());
        let lookalike = Resource::new(Url::parse("https://notgithub.com/x").unwrap());
        assert!(matches!(auth.credential(&on), Credential::Bearer(_)));
        assert!(matches!(auth.credential(&sub), Credential::Bearer(_)));
        assert!(matches!(auth.credential(&off), Credential::Anonymous));
        assert!(matches!(auth.credential(&lookalike), Credential::Anonymous));
    }
}
