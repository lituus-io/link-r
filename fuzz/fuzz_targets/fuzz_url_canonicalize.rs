// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! URL canonicalization / keying must never panic on arbitrary text.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // parse → canonicalize → key; all must be panic-free.
        let _ = link_r::UrlKey::parse(s);
    }
});
