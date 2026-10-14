// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! Text primitives: graph-r's text rules (tokenization, case folding, char
//! n-grams, stopwords), so the hash embedder, BM25, keyword extraction and
//! graph-r's lexicon all agree on what a term is.

use compact_str::CompactString;

pub use graph_r::text::{estimate_tokens, for_each_char_ngram, is_stopword, normalize, tokens};

/// Case-folded tokens as owned terms (inline for short ASCII), the form the
/// BM25 index stores.
pub fn normalized_tokens(text: &str) -> impl Iterator<Item = CompactString> + '_ {
    tokens(text).map(|t| CompactString::from(normalize(t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_tokens_match_query_and_doc() {
        let doc: Vec<CompactString> = normalized_tokens("The BigQuery Row").collect();
        assert_eq!(doc, vec!["the", "bigquery", "row"]);
    }
}
