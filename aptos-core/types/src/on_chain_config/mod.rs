// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    access_path::AccessPath,
    account_config::CORE_CODE_ADDRESS,
    chain_id::ChainId,
    event::{EventHandle, EventKey},
};
use anyhow::{format_err, Result};
use move_core_types::{
    ident_str,
    identifier::{IdentStr, Identifier},
    language_storage::StructTag,
    move_resource::{MoveResource, MoveStructType},
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::HashMap, fmt, sync::Arc};

mod approved_execution_hashes;
mod aptos_features;
mod aptos_version;
mod chain_id;
mod consensus_config;
mod gas_schedule;
mod validator_set;

pub use self::{
    approved_execution_hashes::ApprovedExecutionHashes,
    aptos_features::*,
    aptos_version::{
        Version, APTOS_MAX_KNOWN_VERSION, APTOS_VERSION_2, APTOS_VERSION_3, APTOS_VERSION_4,
    },
    consensus_config::{
        ConsensusConfigV1, LeaderReputationType, OnChainConsensusConfig, ProposerAndVoterConfig,
        ProposerElectionType,
    },
    gas_schedule::{GasSchedule, GasScheduleV2, StorageGasSchedule},
    validator_set::{ConsensusScheme, ValidatorSet},
};

/// To register an on-chain config in Rust:
/// 1. Implement the `OnChainConfig` trait for the Rust representation of the config
/// 2. Add the config's `ConfigID` to `ON_CHAIN_CONFIG_REGISTRY`

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct ConfigID(&'static str, &'static str, &'static str);

impl ConfigID {
    pub fn name(&self) -> String {
        self.2.to_string()
    }
}

impl fmt::Display for ConfigID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnChain config ID [address: {}, identifier: {}]",
            self.0, self.1
        )
    }
}

/// State sync will panic if the value of any config in this registry is uninitialized
pub const ON_CHAIN_CONFIG_REGISTRY: &[ConfigID] = &[
    ApprovedExecutionHashes::CONFIG_ID,
    ValidatorSet::CONFIG_ID,
    Version::CONFIG_ID,
    OnChainConsensusConfig::CONFIG_ID,
    ChainId::CONFIG_ID,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnChainConfigPayload {
    epoch: u64,
    configs: Arc<HashMap<ConfigID, Vec<u8>>>,
}

impl OnChainConfigPayload {
    pub fn new(epoch: u64, configs: Arc<HashMap<ConfigID, Vec<u8>>>) -> Self {
        Self { epoch, configs }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn get<T: OnChainConfig>(&self) -> Result<T> {
        let bytes = self
            .configs
            .get(&T::CONFIG_ID)
            .ok_or_else(|| format_err!("[on-chain cfg] config not in payload"))?;
        T::deserialize_into_config(bytes)
    }

    pub fn configs(&self) -> &HashMap<ConfigID, Vec<u8>> {
        &self.configs
    }
}

impl fmt::Display for OnChainConfigPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut config_ids = "".to_string();
        for id in self.configs.keys() {
            config_ids += &id.to_string();
        }
        write!(
            f,
            "OnChainConfigPayload [epoch: {}, configs: {}]",
            self.epoch, config_ids
        )
    }
}

/// Trait to be implemented by a storage type from which to read on-chain configs
pub trait ConfigStorage {
    fn fetch_config(&self, access_path: AccessPath) -> Option<Vec<u8>>;
}

/// Trait to be implemented by a Rust struct representation of an on-chain config
/// that is stored in storage as a serialized byte array
pub trait OnChainConfig: Send + Sync + DeserializeOwned {
    const ADDRESS: &'static str = "0x1";
    const MODULE_IDENTIFIER: &'static str;
    const TYPE_IDENTIFIER: &'static str;
    const CONFIG_ID: ConfigID = ConfigID(
        Self::ADDRESS,
        Self::MODULE_IDENTIFIER,
        Self::TYPE_IDENTIFIER,
    );

    // Single-round BCS deserialization from bytes to `Self`
    // This is the expected deserialization pattern if the Rust representation lives natively in Move.
    // but sometimes `deserialize_into_config` may need an extra customized round of deserialization
    // when the data is represented as opaque vec<u8> in Move.
    // In the override, we can reuse this default logic via this function
    // Note: we cannot directly call the default `deserialize_into_config` implementation
    // in its override - this will just refer to the override implementation itself
    fn deserialize_default_impl(bytes: &[u8]) -> Result<Self> {
        bcs::from_bytes::<Self>(bytes)
            .map_err(|e| format_err!("[on-chain config] Failed to deserialize into config: {}", e))
    }

    // Function for deserializing bytes to `Self`
    // It will by default try one round of BCS deserialization directly to `Self`
    // The implementation for the concrete type should override this function if this
    // logic needs to be customized
    fn deserialize_into_config(bytes: &[u8]) -> Result<Self> {
        Self::deserialize_default_impl(bytes)
    }

    fn fetch_config<T>(storage: &T) -> Option<Self>
    where
        T: ConfigStorage,
    {
        let access_path = Self::access_path();
        match storage.fetch_config(access_path) {
            Some(bytes) => Self::deserialize_into_config(&bytes).ok(),
            None => None,
        }
    }

    fn access_path() -> AccessPath {
        access_path_for_config(Self::CONFIG_ID)
    }

    fn struct_tag() -> StructTag {
        struct_tag_for_config(Self::CONFIG_ID)
    }
}

pub fn new_epoch_event_key() -> EventKey {
    EventKey::new(2, CORE_CODE_ADDRESS)
}

pub fn access_path_for_config(config_id: ConfigID) -> AccessPath {
    let struct_tag = struct_tag_for_config(config_id);
    AccessPath::new(CORE_CODE_ADDRESS, AccessPath::resource_path_vec(struct_tag))
}

pub fn struct_tag_for_config(config_id: ConfigID) -> StructTag {
    StructTag {
        address: CORE_CODE_ADDRESS,
        module: Identifier::new(config_id.1).expect("fail to make identifier"),
        name: Identifier::new(config_id.2).expect("fail to make identifier"),
        type_params: vec![],
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ConfigurationResource {
    epoch: u64,
    last_reconfiguration_time: u64,
    events: EventHandle,
}

impl ConfigurationResource {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn last_reconfiguration_time(&self) -> u64 {
        self.last_reconfiguration_time
    }

    pub fn events(&self) -> &EventHandle {
        &self.events
    }

    #[cfg(feature = "fuzzing")]
    pub fn bump_epoch_for_test(&self) -> Self {
        let epoch = self.epoch + 1;
        let last_reconfiguration_time = self.last_reconfiguration_time + 1;
        let mut events = self.events.clone();
        *events.count_mut() += 1;

        Self {
            epoch,
            last_reconfiguration_time,
            events,
        }
    }
}

#[cfg(feature = "fuzzing")]
impl Default for ConfigurationResource {
    fn default() -> Self {
        Self {
            epoch: 0,
            last_reconfiguration_time: 0,
            events: EventHandle::new(EventKey::new(16, CORE_CODE_ADDRESS), 0),
        }
    }
}

impl MoveStructType for ConfigurationResource {
    const MODULE_NAME: &'static IdentStr = ident_str!("reconfiguration");
    const STRUCT_NAME: &'static IdentStr = ident_str!("Configuration");
}

impl MoveResource for ConfigurationResource {}
