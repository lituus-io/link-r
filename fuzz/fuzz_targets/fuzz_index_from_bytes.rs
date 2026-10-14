// Author: support@lituus.io
// Copyright: lituus-io, all rights reserved
//
//! Arbitrary bytes as an in-memory index image (`Index::from_bytes`): a typed
//! error or a fully usable index — never a panic — and an accepted image
//! serializes back to exactly itself. Every buffer alignment is exercised by
//! slicing off a data-chosen prefix.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&shift, rest)) = data.split_first() else {
        return;
    };
    let shift = usize::from(shift % 8).min(rest.len());
    let mut buf = vec![0u8; shift];
    buf.extend_from_slice(rest);
    let image = bytes::Bytes::from(buf).slice(shift..);
    if let Ok(index) = link_r::Index::from_bytes(image) {
        assert_eq!(index.serialize().unwrap(), rest);
        let _ = index.dense().unwrap();
        if let Ok(builder) = index.into_builder() {
            let _ = builder.serialize();
        }
    }
});
