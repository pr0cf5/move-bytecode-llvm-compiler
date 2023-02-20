// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use super::*;
use aptos_schemadb::{schema::fuzzing::assert_encode_decode, test_no_panic_decoding};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_encode_decode(
        state_key in any::<StateKey>(),
        version in any::<Version>(),
        v in any::<Option<StateValue>>(),
    ) {
        assert_encode_decode::<StateValueSchema>(&(state_key, version), &v);
    }
}

test_no_panic_decoding!(StateValueSchema);
