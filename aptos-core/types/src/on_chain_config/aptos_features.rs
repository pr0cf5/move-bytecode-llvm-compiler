// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::on_chain_config::OnChainConfig;
use serde::{Deserialize, Serialize};

/// The feature flags define in the Move source. This must stay aligned with the constants there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum FeatureFlag {
    CODE_DEPENDENCY_CHECK = 1,
    TREAT_FRIEND_AS_PRIVATE = 2,
    SHA_512_AND_RIPEMD_160_NATIVES = 3,
    APTOS_STD_CHAIN_ID_NATIVES = 4,
    VM_BINARY_FORMAT_V6 = 5,
    //COLLECT_AND_DISTRIBUTE_GAS_FEES = 6,
    MULTI_ED25519_PK_VALIDATE_V2_NATIVES = 7,
    BLAKE2B_256_NATIVE = 8,
}

/// Representation of features on chain as a bitset.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Features {
    #[serde(with = "serde_bytes")]
    pub features: Vec<u8>,
}

impl Default for Features {
    fn default() -> Self {
        Features {
            features: vec![0b00100000],
        }
    }
}

impl OnChainConfig for Features {
    const MODULE_IDENTIFIER: &'static str = "features";
    const TYPE_IDENTIFIER: &'static str = "Features";
}

impl Features {
    pub fn is_enabled(&self, flag: FeatureFlag) -> bool {
        let val = flag as u64;
        let byte_index = (val / 8) as usize;
        let bit_mask = 1 << (val % 8);
        byte_index < self.features.len() && (self.features[byte_index] & bit_mask != 0)
    }
}

// --------------------------------------------------------------------------------------------
// Code Publishing
