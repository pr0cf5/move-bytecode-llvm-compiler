// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use super::*;
use aptos_schemadb::{schema::fuzzing::assert_encode_decode, test_no_panic_decoding};

// Tests that the DB can encode / decode data
#[test]
fn test_single_entry_schema() {
    assert_encode_decode::<SingleEntrySchema>(&SingleEntryKey::LastVote, &vec![1u8, 2u8, 3u8]);
}

test_no_panic_decoding!(SingleEntrySchema);
