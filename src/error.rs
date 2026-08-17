// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Error and result types for `link-r`.
//!
//! Follows the house convention: a single `thiserror` enum, a `Result<T, E = Error>`
//! alias, `#[must_use]` constructor helpers, and an [`Error::is_retriable`] predicate
//! that drives backoff in the fetch/crawl layer.

/// Result type alias for `link-r` operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The unified error type for every `link-r` operation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A resource (page, index file) could not be found.
    #[error("not found: {uri}")]
    NotFound {
        /// The URI/path that was missing.
        uri: String,
    },

    /// The remote returned 304 Not Modified — used as the unchanged sentinel for
    /// incremental rebuilds.
    #[error("not modified: {uri}")]
    NotModified {
        /// The URI that was unchanged.
        uri: String,
    },

    /// Authentication was required and failed or was missing.
    #[error("authentication failed: {message}")]
    Unauthenticated {
        /// Human-readable detail.
        message: String,
    },

    /// The credential was valid but lacked permission.
    #[error("permission denied: {message}")]
    PermissionDenied {
        /// Human-readable detail.
        message: String,
    },

    /// The remote rate-limited us; retry after the given delay.
    #[error("rate limited; retry after {retry_after_ms}ms")]
    RateLimited {
        /// Suggested backoff in milliseconds (from `Retry-After` / `x-ratelimit-reset`).
        retry_after_ms: u64,
    },

    /// A non-success HTTP status that isn't modelled more specifically.
    #[error("HTTP {status}: {message}")]
    Http {
        /// The HTTP status code.
        status: u16,
        /// Human-readable detail.
        message: String,
    },

    /// A malformed or unsupported URL.
    #[error("invalid URL: {message}")]
    InvalidUrl {
        /// Human-readable detail.
        message: String,
    },

    /// Discovery/crawl failed for a source.
    #[error("crawl failed [{source_kind}]: {message}")]
    Crawl {
        /// The source kind (e.g. `"http"`, `"fs"`).
        source_kind: &'static str,
        /// Human-readable detail.
        message: String,
    },

    /// Content extraction failed.
    #[error("extraction failed: {message}")]
    Extract {
        /// Human-readable detail.
        message: String,
    },

    /// Embedding failed.
    #[error("embedding failed: {message}")]
    Embed {
        /// Human-readable detail.
        message: String,
    },

    /// The on-disk index format was invalid or corrupt.
    #[error("index format error: {message}")]
    Format {
        /// Human-readable detail.
        message: String,
    },

    /// A query/index dimension mismatch (e.g. searching with the wrong embedder).
    #[error("dimension mismatch: index={index}, query={query}")]
    DimMismatch {
        /// The index's embedding dimension.
        index: usize,
        /// The query vector's dimension.
        query: usize,
    },

    /// An operation timed out.
    #[error("operation timed out after {duration_ms}ms")]
    Timeout {
        /// The elapsed budget in milliseconds.
        duration_ms: u64,
    },

    /// An error originating from a named backend dependency.
    #[error("{backend} error: {message}")]
    Backend {
        /// The backend that produced the error (e.g. `"reqwest"`, `"fastembed"`).
        backend: &'static str,
        /// Human-readable detail.
        message: String,
    },

    /// An underlying I/O error.
    #[error("I/O error: {source}")]
    Io {
        /// The wrapped I/O error.
        #[from]
        source: std::io::Error,
    },
}

impl Error {
    /// Construct a [`Error::NotFound`].
    #[must_use]
    pub fn not_found(uri: impl Into<String>) -> Self {
        Self::NotFound { uri: uri.into() }
    }

    /// Construct a [`Error::NotModified`].
    #[must_use]
    pub fn not_modified(uri: impl Into<String>) -> Self {
        Self::NotModified { uri: uri.into() }
    }

    /// Construct a [`Error::Unauthenticated`].
    #[must_use]
    pub fn unauthenticated(message: impl Into<String>) -> Self {
        Self::Unauthenticated {
            message: message.into(),
        }
    }

    /// Construct a [`Error::PermissionDenied`].
    #[must_use]
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::PermissionDenied {
            message: message.into(),
        }
    }

    /// Construct a [`Error::RateLimited`].
    #[must_use]
    pub fn rate_limited(retry_after_ms: u64) -> Self {
        Self::RateLimited { retry_after_ms }
    }

    /// Construct a [`Error::Http`].
    #[must_use]
    pub fn http(status: u16, message: impl Into<String>) -> Self {
        Self::Http {
            status,
            message: message.into(),
        }
    }

    /// Construct a [`Error::InvalidUrl`].
    #[must_use]
    pub fn invalid_url(message: impl Into<String>) -> Self {
        Self::InvalidUrl {
            message: message.into(),
        }
    }

    /// Construct a [`Error::Crawl`].
    #[must_use]
    pub fn crawl(source_kind: &'static str, message: impl Into<String>) -> Self {
        Self::Crawl {
            source_kind,
            message: message.into(),
        }
    }

    /// Construct a [`Error::Extract`].
    #[must_use]
    pub fn extract(message: impl Into<String>) -> Self {
        Self::Extract {
            message: message.into(),
        }
    }

    /// Construct a [`Error::Embed`].
    #[must_use]
    pub fn embed(message: impl Into<String>) -> Self {
        Self::Embed {
            message: message.into(),
        }
    }

    /// Construct a [`Error::Format`].
    #[must_use]
    pub fn format(message: impl Into<String>) -> Self {
        Self::Format {
            message: message.into(),
        }
    }

    /// Construct a [`Error::Backend`].
    #[must_use]
    pub fn backend(backend: &'static str, message: impl Into<String>) -> Self {
        Self::Backend {
            backend,
            message: message.into(),
        }
    }

    /// Whether retrying the failed operation could plausibly succeed.
    ///
    /// Drives backoff in the fetch/crawl layer: timeouts, rate limits, transient
    /// HTTP 5xx/408/429, and recoverable I/O conditions are retriable; logical
    /// errors (not found, auth, format) are not.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Timeout { .. } | Self::RateLimited { .. } => true,
            Self::Http { status, .. } => matches!(status, 408 | 429 | 500 | 502 | 503 | 504),
            Self::Io { source } => matches!(
                source.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::WouldBlock
            ),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_build_expected_variants() {
        assert!(matches!(Error::not_found("x"), Error::NotFound { .. }));
        assert!(matches!(
            Error::http(503, "x"),
            Error::Http { status: 503, .. }
        ));
        assert!(matches!(
            Error::backend("reqwest", "boom"),
            Error::Backend {
                backend: "reqwest",
                ..
            }
        ));
    }

    #[test]
    fn retriable_classification() {
        assert!(Error::rate_limited(100).is_retriable());
        assert!(Error::Timeout { duration_ms: 1 }.is_retriable());
        assert!(Error::http(503, "").is_retriable());
        assert!(Error::http(500, "").is_retriable());
        assert!(!Error::http(404, "").is_retriable());
        assert!(!Error::not_found("x").is_retriable());
        assert!(!Error::unauthenticated("x").is_retriable());
    }

    #[test]
    fn io_errors_convert_via_from() {
        let io = std::io::Error::new(std::io::ErrorKind::TimedOut, "slow");
        let err: Error = io.into();
        assert!(err.is_retriable());
        assert!(matches!(err, Error::Io { .. }));
    }

    #[test]
    fn display_is_human_readable() {
        assert_eq!(
            Error::not_found("https://x/y").to_string(),
            "not found: https://x/y"
        );
    }
}
