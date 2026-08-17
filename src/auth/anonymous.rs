// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The no-credential provider for public resources.

use crate::auth::{AuthProvider, Credential};
use crate::error::Result;
use crate::resource::Resource;
use std::future::Ready;

/// An [`AuthProvider`] that applies no credential. The default for public crawls.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnonymousAuth;

impl AuthProvider for AnonymousAuth {
    type RefreshFuture<'a> = Ready<Result<()>>;

    fn credential(&self, _resource: &Resource) -> Credential<'_> {
        Credential::Anonymous
    }

    fn refresh(&self) -> Self::RefreshFuture<'_> {
        std::future::ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    #[test]
    fn anonymous_yields_no_credential() {
        let r = Resource::new(Url::parse("https://x.dev/a").unwrap());
        assert!(matches!(
            AnonymousAuth.credential(&r),
            Credential::Anonymous
        ));
        assert!(!AnonymousAuth.needs_refresh());
    }
}
