// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! The vector distance metric, shared by the embedder and the index.

/// How dense vectors are compared. Cosine is the default; vectors are stored
/// L2-normalized so cosine reduces to a dot product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Metric {
    /// Cosine similarity (higher is closer). The default.
    #[default]
    Cosine,
    /// Raw dot product (higher is closer).
    Dot,
    /// Negative squared Euclidean distance (higher is closer).
    L2,
}

impl Metric {
    /// Stable on-disk tag.
    #[must_use]
    pub fn as_tag(self) -> u32 {
        match self {
            Self::Cosine => 0,
            Self::Dot => 1,
            Self::L2 => 2,
        }
    }

    /// Decode from an on-disk tag; unknown tags fall back to [`Metric::Cosine`].
    #[must_use]
    pub fn from_tag(tag: u32) -> Self {
        match tag {
            1 => Self::Dot,
            2 => Self::L2,
            _ => Self::Cosine,
        }
    }

    /// Whether vectors should be L2-normalized at build time for this metric.
    #[must_use]
    pub fn normalizes(self) -> bool {
        matches!(self, Self::Cosine)
    }
}
