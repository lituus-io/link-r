// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! HTML link extraction (the crawler's discovery path) must never panic on
//! malformed markup.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = link_r::extract::html::extract_links(data);
});
