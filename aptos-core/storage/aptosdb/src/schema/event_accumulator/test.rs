// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use super::*;
use aptos_schemadb::{schema::fuzzing::assert_encode_decode, test_no_panic_decoding};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_encode_decode(pos in any::<u64>(), hash in any::<HashValue>()) {
        assert_encode_decode::<EventAccumulatorSchema>(
            &(pos, Position::from_inorder_index(pos)),
            &hash,
        );
    }
}

test_no_panic_decoding!(EventAccumulatorSchema);
