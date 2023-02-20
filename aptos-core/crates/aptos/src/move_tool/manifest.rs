// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::common::types::load_manifest_account_arg;
use aptos_types::account_address::AccountAddress;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// A Rust representation of the Move package manifest
///
/// Note: The original Move package manifest object used by the package system
/// can't be serialized because it uses a symbol mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovePackageManifest {
    pub package: PackageInfo,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub addresses: BTreeMap<String, ManifestNamedAddress>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, Dependency>,
}

/// Representation of an option address so we can print it as "_"
#[derive(Debug, Clone)]
pub struct ManifestNamedAddress {
    pub address: Option<AccountAddress>,
}

impl From<Option<AccountAddress>> for ManifestNamedAddress {
    fn from(opt: Option<AccountAddress>) -> Self {
        ManifestNamedAddress { address: opt }
    }
}

impl From<ManifestNamedAddress> for Option<AccountAddress> {
    fn from(addr: ManifestNamedAddress) -> Self {
        addr.address
    }
}

impl Serialize for ManifestNamedAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(address) = self.address {
            serializer.serialize_str(&address.to_hex_literal())
        } else {
            serializer.serialize_str("_")
        }
    }
}

impl<'de> Deserialize<'de> for ManifestNamedAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let str = <String>::deserialize(deserializer)?;
        Ok(ManifestNamedAddress {
            // TODO: Cleanup unwrap
            address: load_manifest_account_arg(&str).unwrap(),
        })
    }
}

/// A Rust representation of a move dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subdir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aptos: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
}

/// A Rust representation of the package info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}
