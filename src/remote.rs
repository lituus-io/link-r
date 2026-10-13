// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! Object storage for indexes (feature `remote`).
//!
//! Keep an index on an object store — Google Cloud Storage, a
//! local directory, or memory — through objstore-r's [`ObjectStore`] trait.
//!
//! ```text
//! {prefix}HEAD                          which generation is current (compare-and-swap only)
//! {prefix}index/{generation:016x}.lnkr  the .lnkr image, byte for byte (create-only)
//! ```
//!
//! - [`Remote::load`] fetches the current image and opens it with
//!   [`LinkIndex::from_bytes`]: searches then run zero-copy over the fetched
//!   buffer (no temp file, no second copy).
//! - [`Remote::publish`] uploads a new generation only when the image changed
//!   and swaps `HEAD` conditionally: of two publishers starting from the same
//!   state exactly one wins and the other gets [`Error::Fenced`].
//! - [`Remote::load_if_changed`] costs one metadata request when nothing
//!   changed — the replica refresh loop.
//!
//! Futures are unboxed (`impl Future` over borrowed state); no `Arc`, no
//! `dyn`; the internal `Mutex` is never held across an `.await`.

pub use objstore_r::ObjectStore;

use crate::facade::DefaultEmbedder;
use crate::{Embedder, LinkIndex};
use graph_r::hash::xxh3_64;
use objstore_r::{fenced, Bytes};
use std::future::Future;
use std::sync::{Mutex, PoisonError};

const MAGIC: u32 = 0x4D52_4E4C; // "LNRM"
const VERSION: u16 = 1;
/// `HEAD`'s sealed body: generation, length and checksum of the image.
const HEAD_BODY: usize = 24;

/// Result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What can go wrong.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The index failed to load or serialize.
    #[error(transparent)]
    Link(#[from] crate::Error),
    /// The object store failed.
    #[error("object store: {0}")]
    Store(objstore_r::Error),
    /// Another publisher swapped `HEAD` first. Reload before publishing.
    #[error("fenced: {reason}")]
    Fenced {
        /// Why.
        reason: &'static str,
    },
    /// Published data failed verification.
    #[error("corrupt remote index: {message}")]
    Corrupt {
        /// What was wrong.
        message: String,
    },
}

/// The fenced-publish outcomes keep their own variants.
impl From<objstore_r::Error> for Error {
    fn from(e: objstore_r::Error) -> Self {
        match e {
            objstore_r::Error::Fenced { reason } => Self::Fenced { reason },
            objstore_r::Error::Corrupt { message } => Self::Corrupt { message },
            other => Self::Store(other),
        }
    }
}

/// The published state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Published {
    /// Monotonic publish generation (1 for the first).
    pub generation: u64,
    /// Image length in bytes.
    pub len: u64,
    /// xxh3 of the image.
    pub sum: u64,
    /// The publish uploaded a new image (false: unchanged, nothing written).
    pub uploaded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Head {
    generation: u64,
    len: u64,
    sum: u64,
}

impl Head {
    fn encode(self) -> Vec<u8> {
        let mut body = Vec::with_capacity(HEAD_BODY);
        body.extend_from_slice(&self.generation.to_le_bytes());
        body.extend_from_slice(&self.len.to_le_bytes());
        body.extend_from_slice(&self.sum.to_le_bytes());
        fenced::seal(MAGIC, VERSION, &body)
    }

    fn decode(b: &[u8]) -> Result<Self> {
        let body = fenced::unseal(b, MAGIC, VERSION)?;
        if body.len() != HEAD_BODY {
            return Err(Error::Corrupt {
                message: "HEAD has the wrong length".into(),
            });
        }
        let u64_at = |i: usize| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&body[i..i + 8]);
            u64::from_le_bytes(a)
        };
        let head = Self {
            generation: u64_at(0),
            len: u64_at(8),
            sum: u64_at(16),
        };
        if head.generation == 0 {
            return Err(Error::Corrupt {
                message: "HEAD names generation 0".into(),
            });
        }
        Ok(head)
    }
}

/// The CAS token plus what it named.
#[derive(Clone, Copy, Debug)]
struct Seen {
    head_gen: u64,
    head: Head,
}

/// An index location on an object store. See the crate docs.
#[derive(Debug)]
pub struct Remote<'o, O: ObjectStore> {
    objects: &'o O,
    prefix: String,
    seen: Mutex<Option<Seen>>,
}

impl<'o, O: ObjectStore> Remote<'o, O> {
    /// An index under `prefix` (empty, or a key prefix; `/` implied).
    pub fn new(objects: &'o O, prefix: &str) -> Result<Self> {
        let prefix = fenced::prefix(prefix)?;
        Ok(Self {
            objects,
            prefix,
            seen: Mutex::new(None),
        })
    }

    fn head_key(&self) -> String {
        format!("{}HEAD", self.prefix)
    }
    fn index_key(&self, generation: u64) -> String {
        format!("{}index/{generation:016x}.lnkr", self.prefix)
    }
    fn seen(&self) -> Option<Seen> {
        *self.seen.lock().unwrap_or_else(PoisonError::into_inner)
    }
    fn remember(&self, s: Option<Seen>) {
        *self.seen.lock().unwrap_or_else(PoisonError::into_inner) = s;
    }

    async fn read_head(&self) -> Result<Option<Seen>> {
        let Some(obj) = self.objects.get(&self.head_key()).await? else {
            self.remember(None);
            return Ok(None);
        };
        let seen = Seen {
            head_gen: obj.generation,
            head: Head::decode(&obj.bytes)?,
        };
        Ok(Some(seen))
    }

    async fn fetch_image(&self, head: Head) -> Result<Bytes> {
        let key = self.index_key(head.generation);
        Ok(fenced::fetch_verified(self.objects, &key, head.len, head.sum).await?)
    }

    /// The published generation, if any (reads `HEAD`).
    pub fn generation(&self) -> impl Future<Output = Result<Option<u64>>> + '_ {
        async move { Ok(self.read_head().await?.map(|s| s.head.generation)) }
    }

    /// Load the current index with a caller-supplied embedder (`None` when
    /// nothing is published). The image is verified, then opened zero-copy.
    pub fn load_with<'a, E: Embedder + 'a>(
        &'a self,
        embedder: E,
    ) -> impl Future<Output = Result<Option<LinkIndex<E>>>> + 'a {
        async move {
            let Some(seen) = self.read_head().await? else {
                return Ok(None);
            };
            let image = self.fetch_image(seen.head).await?;
            let index = LinkIndex::from_bytes_with(image, embedder)?;
            self.remember(Some(seen));
            Ok(Some(index))
        }
    }

    /// [`Remote::load_with`], or an empty in-memory index with the same
    /// embedder when nothing is published yet (a writer's first start).
    /// The flag says whether an image was loaded.
    pub fn load_or_empty_with<'a, E: Embedder + 'a>(
        &'a self,
        embedder: E,
    ) -> impl Future<Output = Result<(LinkIndex<E>, bool)>> + 'a {
        async move {
            let Some(seen) = self.read_head().await? else {
                return Ok((LinkIndex::in_memory_with(embedder), false));
            };
            let image = self.fetch_image(seen.head).await?;
            let index = LinkIndex::from_bytes_with(image, embedder)?;
            self.remember(Some(seen));
            Ok((index, true))
        }
    }

    /// Like [`Remote::load_with`], but only when `HEAD` moved since this
    /// `Remote` last loaded or published — one metadata request otherwise.
    pub fn load_if_changed_with<'a, E: Embedder + 'a>(
        &'a self,
        embedder: E,
    ) -> impl Future<Output = Result<Option<LinkIndex<E>>>> + 'a {
        async move {
            let meta = self.objects.head(&self.head_key()).await?;
            match (meta, self.seen()) {
                (None, _) => Ok(None),
                (Some(m), Some(s)) if m.generation == s.head_gen => Ok(None),
                _ => self.load_with(embedder).await,
            }
        }
    }

    /// Replica refresh in place: when `HEAD` moved since this `Remote` last
    /// loaded or published, swap the new image into `index` (keeping its
    /// embedder; see [`LinkIndex::reload_from_bytes`]). One metadata request
    /// when nothing changed. Returns whether `index` changed.
    pub fn load_if_changed_into<'a, E: Embedder>(
        &'a self,
        index: &'a mut LinkIndex<E>,
    ) -> impl Future<Output = Result<bool>> + 'a {
        async move {
            let meta = self.objects.head(&self.head_key()).await?;
            match (meta, self.seen()) {
                (None, _) => return Ok(false),
                (Some(m), Some(s)) if m.generation == s.head_gen => return Ok(false),
                _ => {}
            }
            let Some(seen) = self.read_head().await? else {
                return Ok(false);
            };
            let image = self.fetch_image(seen.head).await?;
            index.reload_from_bytes(image)?;
            self.remember(Some(seen));
            Ok(true)
        }
    }

    /// Publish `index` if its image differs from the published one. Uploads a
    /// new generation create-only, then swaps `HEAD` conditionally; the
    /// previous image is removed afterwards (best effort).
    pub fn publish<'a, E: Embedder>(
        &'a self,
        index: &'a LinkIndex<E>,
    ) -> impl Future<Output = Result<Published>> + 'a {
        async move {
            let image = index.to_bytes()?;
            let sum = xxh3_64(&image);
            let len = image.len() as u64;
            let prev = match self.seen() {
                Some(s) => Some(s),
                None => self.read_head().await?,
            };
            if let Some(p) = prev.filter(|p| p.head.sum == sum && p.head.len == len) {
                self.remember(Some(p));
                return Ok(Published {
                    generation: p.head.generation,
                    len,
                    sum,
                    uploaded: false,
                });
            }
            let head = Head {
                generation: prev.map_or(1, |p| p.head.generation + 1),
                len,
                sum,
            };
            let key = self.index_key(head.generation);
            fenced::put_immutable(self.objects, &key, Bytes::from(image))
                .await
                .map_err(|e| self.forget(e))?;
            let body = Bytes::from(head.encode());
            let head_gen = fenced::cas(
                self.objects,
                &self.head_key(),
                prev.map(|p| p.head_gen),
                body,
            )
            .await
            .map_err(|e| self.forget(e))?;
            self.remember(Some(Seen { head_gen, head }));
            if let Some(p) = prev {
                let _ = self
                    .objects
                    .delete(&self.index_key(p.head.generation))
                    .await;
            }
            Ok(Published {
                generation: head.generation,
                len,
                sum,
                uploaded: true,
            })
        }
    }

    /// A failed publish leaves `HEAD` unknown: re-read before the next one.
    fn forget(&self, e: objstore_r::Error) -> Error {
        self.remember(None);
        e.into()
    }
}

impl<O: ObjectStore> Remote<'_, O> {
    /// [`Remote::load_with`] with link-r's default embedder.
    pub fn load(&self) -> impl Future<Output = Result<Option<LinkIndex>>> + '_ {
        async move {
            let embedder = default_embedder()?;
            self.load_with(embedder).await
        }
    }

    /// [`Remote::load_if_changed_with`] with link-r's default embedder.
    pub fn load_if_changed(&self) -> impl Future<Output = Result<Option<LinkIndex>>> + '_ {
        async move {
            let embedder = default_embedder()?;
            self.load_if_changed_with(embedder).await
        }
    }
}

fn default_embedder() -> Result<DefaultEmbedder> {
    Ok(crate::facade::default_embedder()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_round_trips_and_rejects_corruption() {
        let h = Head {
            generation: 9,
            len: 1234,
            sum: 77,
        };
        let b = h.encode();
        assert_eq!(Head::decode(&b).unwrap(), h);
        for i in 0..b.len() {
            let mut c = b.clone();
            c[i] ^= 1;
            assert!(Head::decode(&c).is_err(), "flip at {i}");
        }
        assert!(Head::decode(&b[..b.len() - 1]).is_err());
        assert!(Head::decode(&Head { generation: 0, ..h }.encode()).is_err());
    }
}
