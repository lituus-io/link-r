// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Shared text primitives: tokenization, normalization, char n-grams, stopwords.
//!
//! One implementation, three consumers — the deterministic hash embedder, the
//! BM25 sparse index, and RAKE keyword extraction all tokenize and normalize the
//! same way, so a query term and an indexed term always agree. Everything here is
//! allocation-light (borrowed slices; inline [`CompactString`]/[`SmallVec`]) and
//! fully deterministic so indexes are byte-reproducible.

use compact_str::CompactString;
use smallvec::SmallVec;

/// Split text into tokens: maximal runs of Unicode alphanumerics, borrowed
/// (zero-copy) from the input. Delimiters and empty runs are dropped.
///
/// Tokens are *not* lowercased here (that would force allocation); call
/// [`normalize`] when a case-folded term is needed.
pub fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// Case-fold a token to its canonical (lowercase) form.
///
/// Short ASCII tokens stay inline in the returned [`CompactString`] (no heap).
#[must_use]
pub fn normalize(token: &str) -> CompactString {
    if token.is_ascii() {
        if token.bytes().any(|b| b.is_ascii_uppercase()) {
            let mut out = CompactString::with_capacity(token.len());
            for b in token.bytes() {
                out.push(b.to_ascii_lowercase() as char);
            }
            out
        } else {
            CompactString::from(token)
        }
    } else {
        CompactString::from(token.to_lowercase())
    }
}

/// Yield normalized (lowercased) tokens.
pub fn normalized_tokens(text: &str) -> impl Iterator<Item = CompactString> + '_ {
    tokens(text).map(normalize)
}

/// Invoke `f` for each character n-gram of `token`, passing borrowed slices.
///
/// If the token has fewer than `n` characters, the whole token is emitted once.
/// Character boundaries (not byte boundaries) are respected, so this is safe for
/// multi-byte UTF-8. Allocation-free for tokens up to 24 characters.
pub fn for_each_char_ngram(token: &str, n: usize, mut f: impl FnMut(&str)) {
    if n == 0 || token.is_empty() {
        return;
    }
    // Char start offsets plus an end sentinel; inline for typical token lengths.
    let mut bounds: SmallVec<[usize; 24]> = token.char_indices().map(|(i, _)| i).collect();
    bounds.push(token.len());
    let char_count = bounds.len() - 1;
    if char_count <= n {
        f(token);
        return;
    }
    for i in 0..=char_count - n {
        f(&token[bounds[i]..bounds[i + n]]);
    }
}

/// A compact English stopword list for RAKE phrase splitting. Kept sorted for
/// binary search.
static STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your",
];

/// Whether a *normalized* (lowercase) token is a common English stopword.
#[must_use]
pub fn is_stopword(token_lower: &str) -> bool {
    STOPWORDS.binary_search(&token_lower).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_on_non_alphanumeric() {
        let got: Vec<&str> = tokens("Hello, world!  foo_bar.baz").collect();
        assert_eq!(got, vec!["Hello", "world", "foo", "bar", "baz"]);
    }

    #[test]
    fn tokenizer_is_zero_copy_and_unicode_aware() {
        let got: Vec<&str> = tokens("café—über").collect();
        assert_eq!(got, vec!["café", "über"]);
    }

    #[test]
    fn normalize_folds_case() {
        assert_eq!(normalize("Hello"), "hello");
        assert_eq!(normalize("already"), "already");
        assert_eq!(normalize("ÜBER"), "über");
    }

    #[test]
    fn normalized_tokens_match_query_and_doc() {
        let doc: Vec<CompactString> = normalized_tokens("The BigQuery Row").collect();
        assert_eq!(doc, vec!["the", "bigquery", "row"]);
    }

    #[test]
    fn char_ngrams_window_correctly() {
        let mut out = Vec::new();
        for_each_char_ngram("hello", 3, |g| out.push(g.to_string()));
        assert_eq!(out, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn char_ngrams_short_token_emits_whole() {
        let mut out = Vec::new();
        for_each_char_ngram("hi", 3, |g| out.push(g.to_string()));
        assert_eq!(out, vec!["hi"]);
    }

    #[test]
    fn char_ngrams_respect_utf8_boundaries() {
        let mut out = Vec::new();
        for_each_char_ngram("café", 2, |g| out.push(g.to_string()));
        assert_eq!(out, vec!["ca", "af", "fé"]);
    }

    #[test]
    fn stopwords_are_detected() {
        assert!(is_stopword("the"));
        assert!(is_stopword("and"));
        assert!(!is_stopword("bigquery"));
        // list must stay sorted for binary_search
        let mut sorted = STOPWORDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, STOPWORDS);
    }
}
