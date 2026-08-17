// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! HTML extraction: title, visible text, and outbound links.
//!
//! Pulls the `<title>` (falling back to the first `<h1>`), the text of
//! content-bearing elements (skipping `<script>`/`<style>`), and every `<a href>`
//! the crawler will resolve and follow. One parse serves both indexing and crawl
//! discovery.

use crate::error::Result;
use crate::extract::{distill, Descriptor};
use crate::resource::Resource;
use compact_str::CompactString;
use memchr::{memchr, memmem};
use scraper::{Html, Selector};
use smallvec::SmallVec;

/// Extract a descriptor (and outbound links) from an HTML document.
pub fn extract(resource: &Resource, bytes: &[u8]) -> Result<Descriptor> {
    let text = String::from_utf8_lossy(bytes);
    let doc = Html::parse_document(&text);

    let title = select_one_text(&doc, "title")
        .or_else(|| select_one_text(&doc, "h1"))
        .map(|t| CompactString::from(t.trim()));

    // Language from the root `<html lang="…">` attribute (BCP-47 primary subtag).
    let lang = doc
        .select(&parse_selector("html"))
        .next()
        .and_then(|el| el.value().attr("lang"))
        .map(|l| CompactString::from(l.split(['-', '_']).next().unwrap_or(l).trim()))
        .filter(|l| !l.is_empty());

    // Section headings (H1–H3) — high-signal for retrieval, with depths.
    let mut headings: SmallVec<[CompactString; 8]> = SmallVec::new();
    let mut heading_levels: SmallVec<[u8; 8]> = SmallVec::new();
    for el in doc.select(&parse_selector("h1,h2,h3")) {
        let text: String = el.text().collect();
        let text = text.trim();
        if !text.is_empty() {
            let level = match el.value().name() {
                "h1" => 1,
                "h2" => 2,
                _ => 3,
            };
            headings.push(CompactString::from(text));
            heading_levels.push(level);
            if headings.len() >= 8 {
                break;
            }
        }
    }

    // Visible text from content-bearing elements (script/style are excluded by
    // not selecting them).
    let content =
        parse_selector("p,h1,h2,h3,h4,h5,h6,li,td,th,blockquote,dd,dt,caption,figcaption,a");
    let mut body = String::new();
    for el in doc.select(&content) {
        for chunk in el.text() {
            let chunk = chunk.trim();
            if !chunk.is_empty() {
                body.push_str(chunk);
                body.push(' ');
            }
        }
    }

    // Outbound links share the fast byte scanner used by the crawl-discovery pass
    // (one implementation, no duplicated anchor walk).
    let links = extract_links(bytes);

    let mut d = distill(resource, title, &headings, &body, links, lang);
    d.heading_levels = heading_levels;
    Ok(d)
}

/// Parse only the outbound `<a href>` links from an HTML document.
///
/// A single forward byte scan — far cheaper than a full DOM parse — used by the
/// crawler's link-discovery pass and by [`extract`]. Skips comments and the raw
/// text of `<script>`/`<style>`, matches `<a>` case-insensitively (not `<abbr>`),
/// handles double/single/unquoted attribute values, and decodes HTML entities in
/// the href only when one is present. Returns raw, unresolved hrefs in document
/// order. Every index is bounds-checked; malformed input can never panic (guarded
/// by the `fuzz_html_links` target).
#[must_use]
pub fn extract_links(bytes: &[u8]) -> Vec<CompactString> {
    let n = bytes.len();
    let mut links = Vec::new();
    let mut i = 0usize;

    while i < n {
        // Hop to the next '<'.
        let Some(off) = memchr(b'<', &bytes[i..]) else {
            break;
        };
        i += off;
        let rest = &bytes[i..];
        if rest.len() < 2 {
            break;
        }

        // Comment: skip past the matching "-->".
        if rest.starts_with(b"<!--") {
            match memmem::find(&bytes[i + 4..], b"-->") {
                Some(end) => i += 4 + end + 3,
                None => break,
            }
            continue;
        }
        // Closing tag / declaration / processing instruction: skip to '>'.
        let b1 = rest[1];
        if b1 == b'/' || b1 == b'!' || b1 == b'?' {
            match memchr(b'>', &bytes[i + 1..]) {
                Some(gt) => i += 1 + gt + 1,
                None => break,
            }
            continue;
        }

        // Tag name: the maximal ASCII-alphanumeric run after '<'.
        let name_start = i + 1;
        let mut j = name_start;
        while j < n && bytes[j].is_ascii_alphanumeric() {
            j += 1;
        }
        let name = &bytes[name_start..j];
        if name.is_empty() {
            i += 1; // a bare '<' — treat as literal text
            continue;
        }
        // A real tag ends the name with whitespace, '>' or '/'; anything else
        // (e.g. a custom `<my-el>`) is skipped as a non-anchor tag.
        let boundary_ok = j >= n || {
            let c = bytes[j];
            c.is_ascii_whitespace() || c == b'>' || c == b'/'
        };

        if name.eq_ignore_ascii_case(b"script") || name.eq_ignore_ascii_case(b"style") {
            // Raw-text element: no anchors inside; skip to its close tag.
            let close: &[u8] = if name.eq_ignore_ascii_case(b"script") {
                b"</script"
            } else {
                b"</style"
            };
            match find_ci(&bytes[j..], close) {
                Some(end) => i = j + end, // land on the close tag's '<'
                None => break,
            }
            continue;
        }
        if boundary_ok && name.eq_ignore_ascii_case(b"a") {
            let (href, next) = scan_tag(bytes, j, true);
            if let Some(raw) = href {
                push_href(&mut links, raw);
            }
            i = next;
            continue;
        }
        // Any other tag: skip to '>', honoring quoted attribute values.
        i = scan_tag(bytes, j, false).1;
    }

    links
}

/// Scan a tag's attributes from `k` (just past the tag name) to just past its
/// `>`. When `capture_href`, returns the first `href` value slice found. Honors
/// quoted values so a `>` inside quotes doesn't end the tag. Never panics.
fn scan_tag(bytes: &[u8], mut pos: usize, capture_href: bool) -> (Option<&[u8]>, usize) {
    let len = bytes.len();
    let mut href: Option<&[u8]> = None;
    while pos < len {
        let byte = bytes[pos];
        if byte == b'>' {
            return (href, pos + 1);
        }
        if byte.is_ascii_whitespace() || byte == b'/' {
            pos += 1;
            continue;
        }
        // Attribute name: run until '=', '>', '/', or whitespace.
        let name_start = pos;
        while pos < len {
            let byte = bytes[pos];
            if byte == b'=' || byte == b'>' || byte == b'/' || byte.is_ascii_whitespace() {
                break;
            }
            pos += 1;
        }
        let name = &bytes[name_start..pos];

        // Optional '= value', tolerating whitespace around '='.
        let mut probe = pos;
        while probe < len && bytes[probe].is_ascii_whitespace() {
            probe += 1;
        }
        if probe < len && bytes[probe] == b'=' {
            probe += 1;
            while probe < len && bytes[probe].is_ascii_whitespace() {
                probe += 1;
            }
            if probe >= len {
                return (href, len);
            }
            let quote = bytes[probe];
            let (val, after) = if quote == b'"' || quote == b'\'' {
                let vstart = probe + 1;
                match memchr(quote, &bytes[vstart..]) {
                    Some(rel) => (&bytes[vstart..vstart + rel], vstart + rel + 1),
                    None => (&bytes[vstart..len], len),
                }
            } else {
                let vstart = probe;
                let mut end = vstart;
                while end < len && !bytes[end].is_ascii_whitespace() && bytes[end] != b'>' {
                    end += 1;
                }
                (&bytes[vstart..end], end)
            };
            if capture_href && href.is_none() && name.eq_ignore_ascii_case(b"href") {
                href = Some(val);
            }
            pos = after;
        } else {
            // Valueless attribute; ensure forward progress.
            pos = pos.max(name_start + 1);
        }
    }
    (href, len)
}

/// Case-insensitive search for the ASCII `needle` in `hay`, returning its start.
/// `needle` here always begins with `<`, so the first-byte scan is exact.
fn find_ci(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let first = *needle.first()?;
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        let off = memchr(first, &hay[i..])?;
        let pos = i + off;
        if pos + needle.len() <= hay.len()
            && hay[pos..pos + needle.len()].eq_ignore_ascii_case(needle)
        {
            return Some(pos);
        }
        i = pos + 1;
    }
    None
}

/// Trim, entity-decode (only if needed), and push a non-empty href.
fn push_href(links: &mut Vec<CompactString>, raw: &[u8]) {
    let s = String::from_utf8_lossy(raw);
    let s = s.trim();
    if s.is_empty() {
        return;
    }
    let out = if s.as_bytes().contains(&b'&') {
        decode_entities(s)
    } else {
        return links.push(CompactString::from(s));
    };
    let out = out.trim();
    if !out.is_empty() {
        links.push(CompactString::from(out));
    }
}

/// Decode the minimal set of HTML entities that matter for hrefs. Unknown
/// entities pass through verbatim.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(semi) = s[i + 1..].find(';') {
                if let Some(ch) = decode_entity(&s[i + 1..i + 1 + semi]) {
                    out.push(ch);
                    i += 1 + semi + 1;
                    continue;
                }
            }
        }
        // Copy one full char (i is always on a char boundary here).
        let ch = s[i..].chars().next().unwrap_or('\u{FFFD}');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Resolve a single entity body (the text between `&` and `;`) to a char.
fn decode_entity(ent: &str) -> Option<char> {
    match ent {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        _ => {
            let num = ent.strip_prefix('#')?;
            let code = match num.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => num.parse::<u32>().ok()?,
            };
            char::from_u32(code)
        }
    }
}

fn parse_selector(s: &str) -> Selector {
    // The selector strings here are compile-time constants known to be valid.
    Selector::parse(s).expect("static CSS selector must be valid")
}

fn select_one_text(doc: &Html, selector: &str) -> Option<String> {
    let sel = parse_selector(selector);
    doc.select(&sel)
        .next()
        .map(|e| e.text().collect::<String>())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceKind;
    use url::Url;

    fn resource(url: &str) -> Resource {
        Resource::new(Url::parse(url).unwrap()).with_kind(ResourceKind::Html)
    }

    const PAGE: &str = r#"<!doctype html>
        <html><head><title>Networking Guide</title>
        <style>.x{color:red}</style></head>
        <body>
          <h1>Private Service Connect</h1>
          <p>PSC lets you reach services privately.</p>
          <script>var secret = 1;</script>
          <a href="/docs/psc">PSC docs</a>
          <a href="https://other.dev/x">external</a>
        </body></html>"#;

    #[test]
    fn extracts_title_text_and_links() {
        let d = extract(&resource("https://x.dev/net"), PAGE.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("Networking Guide"));
        // body text present, script/style text absent
        assert!(d.bm25_terms.contains(&CompactString::from("psc")));
        assert!(!d.bm25_terms.contains(&CompactString::from("secret")));
        assert!(!d.bm25_terms.contains(&CompactString::from("color")));
        // links discovered
        assert!(d.links.iter().any(|l| l == "/docs/psc"));
        assert!(d.links.iter().any(|l| l == "https://other.dev/x"));
    }

    #[test]
    fn title_falls_back_to_h1() {
        let html = "<html><body><h1>Just an H1</h1><p>text</p></body></html>";
        let d = extract(&resource("https://x.dev/a"), html.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("Just an H1"));
    }

    #[test]
    fn extracts_lang_and_headings() {
        let html = r#"<html lang="en-US"><head><title>T</title></head><body>
            <h1>Alpha</h1><h2>Beta Section</h2><p>body</p></body></html>"#;
        let d = extract(&resource("https://x.dev/a"), html.as_bytes()).unwrap();
        assert_eq!(d.lang.as_deref(), Some("en")); // primary subtag
        assert!(d.headings.iter().any(|h| h == "Alpha"));
        assert!(d.headings.iter().any(|h| h == "Beta Section"));
        // Headings are folded into the embed text and BM25 terms.
        assert!(d.embed_text.contains("Beta Section"));
        assert!(d.bm25_terms.contains(&CompactString::from("beta")));
    }

    #[test]
    fn malformed_html_does_not_panic() {
        let html = b"<html><body><a href=>broken<p>unclosed";
        let d = extract(&resource("https://x.dev/a"), html).unwrap();
        let _ = d.bm25_terms; // just must not panic
    }

    #[test]
    fn scanner_handles_quote_styles_and_case() {
        let html = br#"<A HREF="/dq">d</A>
            <a href='/sq'>s</a>
            <a  href=/unquoted >u</a>
            <a class="x" href="/after-attr">y</a>"#;
        let links = extract_links(html);
        assert!(links.iter().any(|l| l == "/dq"));
        assert!(links.iter().any(|l| l == "/sq"));
        assert!(links.iter().any(|l| l == "/unquoted"));
        assert!(links.iter().any(|l| l == "/after-attr"));
    }

    #[test]
    fn scanner_ignores_script_style_comment_and_abbr() {
        let html = br#"
            <script>var s = '<a href="/inscript">x</a>';</script>
            <style>a[href="/instyle"]{}</style>
            <!-- <a href="/incomment">c</a> -->
            <abbr href="/inabbr">nope</abbr>
            <a href="/real">real</a>"#;
        let links = extract_links(html);
        assert_eq!(links, vec![CompactString::from("/real")]);
    }

    #[test]
    fn scanner_decodes_entities_and_skips_empty() {
        let html = br#"<a href="/s?a=1&amp;b=2">e</a><a href="">empty</a><a href="  ">ws</a>"#;
        let links = extract_links(html);
        assert_eq!(links, vec![CompactString::from("/s?a=1&b=2")]);
    }

    #[test]
    fn scanner_gt_inside_quoted_value_not_a_terminator() {
        let html = br#"<a title="a > b" href="/ok">x</a>"#;
        let links = extract_links(html);
        assert_eq!(links, vec![CompactString::from("/ok")]);
    }

    #[test]
    fn scanner_truncated_input_never_panics() {
        for cut in [
            &b"<a href=\"/x"[..],
            b"<!-- unterminated",
            b"<a href='",
            b"<scr",
            b"<a href=",
            b"<",
        ] {
            let _ = extract_links(cut); // must not panic
        }
    }
}
