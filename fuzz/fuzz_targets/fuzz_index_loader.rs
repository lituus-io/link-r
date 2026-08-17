// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! THE critical fuzz target: the index loader must never panic or trigger UB on
//! arbitrary bytes — it sits in front of the mmap + `&[f32]` reinterpret island.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Whole-file validation (header, directory, checksums, bounds, alignment).
    let _ = link_r::index::format::validate(data);
    // Section decoders on adversarial input — both the legacy and freshness layouts.
    let _ = link_r::index::meta::decode(data, false);
    let _ = link_r::index::meta::decode(data, true);
    let _ = link_r::index::sparse::Bm25::from_bytes(data);
    // The link-graph edge decoder must also reject garbage without panic/OOM.
    let _ = link_r::index::graph::decode(data);
    // The cast helper must reject mis-sized/misaligned input rather than UB.
    let _ = link_r::index::mmap::cast_f32(data);
});
