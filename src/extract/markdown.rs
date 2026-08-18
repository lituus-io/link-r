// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Zero-dependency extractors for Markdown, code, and plain text.
//!
//! Deterministic and dependency-free: front-matter / ATX headings give the title,
//! the body feeds the shared [`distill`] pass. The tokenizer
//! ignores markup punctuation, so we keep the raw body rather than rendering it.

use crate::error::Result;
use crate::extract::{distill, Descriptor};
use crate::resource::Resource;
use compact_str::CompactString;

/// Maximum ATX headings collected from a Markdown body.
const MAX_HEADINGS: usize = 8;

/// Extract from a Markdown document (front-matter + ATX-heading aware).
pub fn extract_markdown(resource: &Resource, bytes: &[u8]) -> Result<Descriptor> {
    let text = String::from_utf8_lossy(bytes);
    let (front, body) = split_front_matter(&text);
    let title = front
        .and_then(front_matter_title)
        .or_else(|| first_atx_heading(body))
        .map(CompactString::from);
    let lang = front.and_then(front_matter_lang).map(CompactString::from);
    let (headings, levels) = atx_headings(body);
    let links = extract_md_links(body);
    let mut d = distill(resource, title, &headings, body, links, lang);
    d.heading_levels = levels.into_iter().collect();
    Ok(d)
}

/// Collect outbound link targets from a Markdown body: inline `[text](url)`
/// (images `![..](..)` excluded), reference definitions `[id]: url`, and
/// autolinks `<https://…>`. Zero-dependency line/byte scanning, matching the
/// module's no-renderer policy.
pub(crate) fn extract_md_links(body: &str) -> Vec<CompactString> {
    let mut links = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        match bytes[i] {
            b']' if bytes[i + 1] == b'(' => {
                // Backtrack: exclude images (preceding "![").
                let open = body[..i].rfind('[');
                let is_image = open.is_some_and(|o| o > 0 && bytes[o - 1] == b'!');
                if !is_image {
                    let rest = &body[i + 2..];
                    if let Some(end) = rest.find(')') {
                        let target = rest[..end].split_whitespace().next().unwrap_or("");
                        if !target.is_empty() && !target.starts_with('#') {
                            links.push(CompactString::from(target));
                        }
                    }
                }
                i += 2;
            }
            b'<' if body[i + 1..].starts_with("http") => {
                let rest = &body[i + 1..];
                if let Some(end) = rest.find('>') {
                    let target = &rest[..end];
                    if !target.contains(char::is_whitespace) {
                        links.push(CompactString::from(target));
                    }
                    i += end + 1;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    // Reference definitions: `[id]: url` at line start.
    for line in body.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix('[') {
            if let Some(close) = rest.find("]:") {
                let target = rest[close + 2..].split_whitespace().next().unwrap_or("");
                if !target.is_empty() && !target.starts_with('#') {
                    links.push(CompactString::from(target));
                }
            }
        }
    }
    links
}

/// Extract from a source-code or config file: the path's file name is the title,
/// the body provides identifiers/keywords.
pub fn extract_code(resource: &Resource, bytes: &[u8]) -> Result<Descriptor> {
    let text = String::from_utf8_lossy(bytes);
    let title = file_name(resource).map(CompactString::from);
    Ok(distill(resource, title, &[], &text, Vec::new(), None))
}

/// Extract from plain text: the first non-empty line is the title.
pub fn extract_plain(resource: &Resource, bytes: &[u8]) -> Result<Descriptor> {
    let text = String::from_utf8_lossy(bytes);
    let title = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| CompactString::from(&l[..l.len().min(120)]));
    Ok(distill(resource, title, &[], &text, Vec::new(), None))
}

fn file_name(resource: &Resource) -> Option<&str> {
    resource.url.path().rsplit('/').find(|s| !s.is_empty())
}

/// Split a leading YAML front-matter block (`---` … `---`) from the body.
fn split_front_matter(text: &str) -> (Option<&str>, &str) {
    let trimmed = text.trim_start_matches('\u{feff}');
    if !(trimmed.starts_with("---\n") || trimmed.starts_with("---\r\n")) {
        return (None, text);
    }
    let Some(nl) = trimmed.find('\n') else {
        return (None, text);
    };
    let after_open = &trimmed[nl + 1..];
    let mut idx = 0;
    for line in after_open.split_inclusive('\n') {
        let stripped = line.trim_end_matches(['\r', '\n']);
        if stripped == "---" || stripped == "..." {
            let front = &after_open[..idx];
            return (Some(front), &after_open[idx + line.len()..]);
        }
        idx += line.len();
    }
    (None, text)
}

fn front_matter_title(front: &str) -> Option<String> {
    for line in front.lines() {
        if let Some(rest) = line.trim().strip_prefix("title:") {
            let value = rest.trim().trim_matches(['"', '\'']).trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

fn front_matter_lang(front: &str) -> Option<String> {
    for line in front.lines() {
        if let Some(rest) = line.trim().strip_prefix("lang:") {
            let value = rest.trim().trim_matches(['"', '\'']).trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

fn first_atx_heading(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('#') {
            let heading = rest.trim_start_matches('#').trim();
            if !heading.is_empty() {
                return Some(heading.to_owned());
            }
        }
    }
    None
}

/// Collect up to [`MAX_HEADINGS`] ATX (`#`/`##`/`###`) heading texts with
/// their depths.
fn atx_headings(body: &str) -> (Vec<CompactString>, Vec<u8>) {
    let mut headings = Vec::new();
    let mut levels = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim_start();
        let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
        if (1..=3).contains(&hashes) {
            let heading = trimmed[hashes..].trim();
            if !heading.is_empty() {
                headings.push(CompactString::from(heading));
                levels.push(hashes as u8);
                if headings.len() >= MAX_HEADINGS {
                    break;
                }
            }
        }
    }
    (headings, levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    fn resource(url: &str) -> Resource {
        Resource::new(Url::parse(url).unwrap())
    }

    #[test]
    fn markdown_title_from_front_matter() {
        let md = "---\ntitle: \"My Guide\"\ntags: [a, b]\n---\n\n# Heading\n\nbody text here";
        let d = extract_markdown(&resource("https://x.dev/g.md"), md.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("My Guide"));
        assert!(d.bm25_terms.contains(&CompactString::from("body")));
    }

    #[test]
    fn markdown_title_from_heading_when_no_front_matter() {
        let md = "# Getting Started\n\nInstall the thing and run it.";
        let d = extract_markdown(&resource("https://x.dev/g.md"), md.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("Getting Started"));
    }

    #[test]
    fn markdown_extracts_lang_and_headings() {
        let md = "---\ntitle: Guide\nlang: fr\n---\n\n# Intro\n\ntext\n\n## Setup Steps\n\nmore";
        let d = extract_markdown(&resource("https://x.dev/g.md"), md.as_bytes()).unwrap();
        assert_eq!(d.lang.as_deref(), Some("fr"));
        assert!(d.headings.iter().any(|h| h == "Intro"));
        assert!(d.headings.iter().any(|h| h == "Setup Steps"));
        assert!(d.bm25_terms.contains(&CompactString::from("setup")));
    }

    #[test]
    fn code_title_is_file_name() {
        let code = "fn main() { let secret_token = compute(); }";
        let d = extract_code(&resource("https://x.dev/src/main.rs"), code.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("main.rs"));
        assert!(d.bm25_terms.contains(&CompactString::from("secret")));
    }

    #[test]
    fn plain_title_is_first_line() {
        let txt = "\n\nFirst real line\nsecond line";
        let d = extract_plain(&resource("https://x.dev/a.txt"), txt.as_bytes()).unwrap();
        assert_eq!(d.title.as_deref(), Some("First real line"));
    }

    #[test]
    fn handles_invalid_utf8_without_panicking() {
        let bytes = [0xff, 0xfe, b'h', b'i'];
        let d = extract_plain(&resource("https://x.dev/a.txt"), &bytes).unwrap();
        assert!(d.bm25_terms.contains(&CompactString::from("hi")));
    }

    #[test]
    fn front_matter_split_is_robust_to_missing_close() {
        // no closing delimiter → treat whole thing as body, no panic
        let md = "---\ntitle: x\nbody without close";
        let d = extract_markdown(&resource("https://x.dev/g.md"), md.as_bytes()).unwrap();
        assert!(!d.bm25_terms.is_empty());
    }
}
