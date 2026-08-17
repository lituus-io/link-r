// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Zero-copy document payloads.
//!
//! Mirrors the house `GetPayload` pattern: a fetch result can be a borrowed slice,
//! a memory-mapped view, an owned [`Bytes`] buffer, or a byte stream — whichever
//! avoids a copy for the originating source. Consumers that need contiguous bytes
//! call [`DocPayload::into_bytes`]; link-extraction and keyword distillation read
//! [`DocPayload::as_slice`] when the payload is already in memory.

use crate::error::Result;
use bytes::Bytes;
use futures::Stream;
use std::fmt;
use std::pin::Pin;

/// A boxed byte stream. The box wraps the *stream* once (not each item), keeping
/// the per-chunk path allocation-free; this matches the house `ByteStream` alias.
pub type ByteStream<'a> = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send + 'a>>;

/// A borrowed view into a memory-mapped region, tied to the lifetime of the owner.
///
/// The owner (an mmap or the index file view) guarantees the bytes outlive `'a`;
/// this newtype simply documents the zero-copy intent at call sites.
#[derive(Clone, Copy)]
pub struct MmapView<'a>(&'a [u8]);

impl<'a> MmapView<'a> {
    /// Wrap a borrowed slice as an mmap view.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    /// The underlying bytes.
    #[must_use]
    pub fn as_slice(&self) -> &'a [u8] {
        self.0
    }
}

impl fmt::Debug for MmapView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapView")
            .field("len", &self.0.len())
            .finish()
    }
}

impl AsRef<[u8]> for MmapView<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

/// The body of a fetched document, in whichever representation avoids a copy.
pub enum DocPayload<'a> {
    /// Borrowed directly from the caller's buffer — zero copy.
    Borrowed(&'a [u8]),
    /// A memory-mapped view (e.g. a local file) — zero copy.
    Mapped(MmapView<'a>),
    /// An owned, reference-counted buffer (e.g. a network fetch).
    Owned(Bytes),
    /// A lazily-produced byte stream for large bodies.
    Stream(ByteStream<'a>),
}

impl DocPayload<'_> {
    /// Borrow the bytes if they are already materialized in memory.
    ///
    /// Returns `None` for [`DocPayload::Stream`], which must be consumed via
    /// [`DocPayload::into_bytes`].
    #[must_use]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Self::Borrowed(b) => Some(b),
            Self::Mapped(m) => Some(m.as_slice()),
            Self::Owned(b) => Some(b),
            Self::Stream(_) => None,
        }
    }

    /// The byte length if known without consuming a stream.
    #[must_use]
    pub fn len(&self) -> Option<usize> {
        self.as_slice().map(<[u8]>::len)
    }

    /// Whether an in-memory payload is empty (always `None` for streams).
    #[must_use]
    pub fn is_empty(&self) -> Option<bool> {
        self.as_slice().map(<[u8]>::is_empty)
    }

    /// Collect the payload into a contiguous [`Bytes`], draining a stream if needed.
    pub async fn into_bytes(self) -> Result<Bytes> {
        match self {
            Self::Borrowed(b) => Ok(Bytes::copy_from_slice(b)),
            Self::Mapped(m) => Ok(Bytes::copy_from_slice(m.as_slice())),
            Self::Owned(b) => Ok(b),
            Self::Stream(mut s) => {
                use futures::StreamExt;
                let mut buf = Vec::new();
                while let Some(chunk) = s.next().await {
                    buf.extend_from_slice(&chunk?);
                }
                Ok(Bytes::from(buf))
            }
        }
    }
}

impl fmt::Debug for DocPayload<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Borrowed(b) => f.debug_tuple("Borrowed").field(&b.len()).finish(),
            Self::Mapped(m) => f.debug_tuple("Mapped").field(&m.as_slice().len()).finish(),
            Self::Owned(b) => f.debug_tuple("Owned").field(&b.len()).finish(),
            Self::Stream(_) => f.write_str("Stream(..)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_and_owned_expose_slices() {
        let data = b"hello world";
        let borrowed = DocPayload::Borrowed(data);
        assert_eq!(borrowed.as_slice(), Some(&data[..]));
        assert_eq!(borrowed.len(), Some(11));
        assert_eq!(borrowed.is_empty(), Some(false));

        let owned = DocPayload::Owned(Bytes::from_static(data));
        assert_eq!(owned.as_slice(), Some(&data[..]));
    }

    #[test]
    fn mapped_view_roundtrips() {
        let data = [1u8, 2, 3, 4];
        let view = MmapView::new(&data);
        assert_eq!(view.as_slice(), &data);
        assert_eq!(view.as_ref(), &data);
    }

    #[test]
    fn stream_has_no_in_memory_slice() {
        let s: ByteStream<'_> = Box::pin(futures::stream::empty());
        let payload = DocPayload::Stream(s);
        assert_eq!(payload.as_slice(), None);
        assert_eq!(payload.len(), None);
    }

    #[tokio::test]
    async fn into_bytes_drains_a_stream() {
        let chunks = vec![
            Ok(Bytes::from_static(b"foo")),
            Ok(Bytes::from_static(b"bar")),
        ];
        let s: ByteStream<'_> = Box::pin(futures::stream::iter(chunks));
        let bytes = DocPayload::Stream(s).into_bytes().await.unwrap();
        assert_eq!(&bytes[..], b"foobar");
    }

    #[tokio::test]
    async fn into_bytes_on_borrowed_copies() {
        let bytes = DocPayload::Borrowed(b"abc").into_bytes().await.unwrap();
        assert_eq!(&bytes[..], b"abc");
    }
}
