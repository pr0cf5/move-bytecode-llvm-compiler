// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::proof_of_store::ProofOfStore;
use aptos_crypto::HashValue;
use aptos_executor_types::Error;
use aptos_infallible::Mutex;
use aptos_types::{
    account_address::AccountAddress, transaction::SignedTransaction,
    validator_verifier::ValidatorVerifier,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fmt, fmt::Write, sync::Arc};
use tokio::sync::oneshot;

/// The round of a block is a consensus-internal counter, which starts with 0 and increases
/// monotonically. It is used for the protocol safety and liveness (please see the detailed
/// protocol description).
pub type Round = u64;
/// Author refers to the author's account address
pub type Author = AccountAddress;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TransactionSummary {
    pub sender: AccountAddress,
    pub sequence_number: u64,
}

impl fmt::Display for TransactionSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.sender, self.sequence_number,)
    }
}

#[derive(Clone)]
pub struct RejectedTransactionSummary {
    pub sender: AccountAddress,
    pub sequence_number: u64,
    pub hash: HashValue,
}

#[derive(Debug)]
pub enum DataStatus {
    Cached(Vec<SignedTransaction>),
    Requested(
        Vec<(
            HashValue,
            oneshot::Receiver<Result<Vec<SignedTransaction>, Error>>,
        )>,
    ),
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ProofWithData {
    pub proofs: Vec<ProofOfStore>,
    #[serde(skip)]
    pub status: Arc<Mutex<Option<DataStatus>>>,
}

impl PartialEq for ProofWithData {
    fn eq(&self, other: &Self) -> bool {
        self.proofs == other.proofs && Arc::as_ptr(&self.status) == Arc::as_ptr(&other.status)
    }
}

impl Eq for ProofWithData {}

impl ProofWithData {
    pub fn new(proofs: Vec<ProofOfStore>) -> Self {
        Self {
            proofs,
            status: Arc::new(Mutex::new(None)),
        }
    }
}

/// The payload in block.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub enum Payload {
    DirectMempool(Vec<SignedTransaction>),
    InQuorumStore(ProofWithData),
}

impl Payload {
    pub fn empty(quorum_store_enabled: bool) -> Self {
        if quorum_store_enabled {
            Payload::InQuorumStore(ProofWithData::new(Vec::new()))
        } else {
            Payload::DirectMempool(Vec::new())
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Payload::DirectMempool(txns) => txns.len(),
            Payload::InQuorumStore(proof_with_status) => proof_with_status
                .proofs
                .iter()
                .map(|proof| proof.info().num_txns as usize)
                .sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Payload::DirectMempool(txns) => txns.is_empty(),
            Payload::InQuorumStore(proof_with_status) => proof_with_status.proofs.is_empty(),
        }
    }

    pub fn is_direct(&self) -> bool {
        matches!(self, Payload::DirectMempool(_))
    }

    /// This is computationally expensive on the first call
    pub fn size(&self) -> usize {
        match self {
            Payload::DirectMempool(txns) => txns
                .par_iter()
                .with_min_len(100)
                .map(|txn| txn.raw_txn_bytes_len())
                .sum(),
            Payload::InQuorumStore(proof_with_status) => proof_with_status
                .proofs
                .iter()
                .map(|proof| proof.info().num_bytes as usize)
                .sum(),
        }
    }

    pub fn verify(
        &self,
        validator: &ValidatorVerifier,
        quorum_store_enabled: bool,
    ) -> anyhow::Result<()> {
        match (quorum_store_enabled, self) {
            (false, Payload::DirectMempool(_)) => Ok(()),
            (true, Payload::InQuorumStore(proof_with_status)) => {
                for proof in proof_with_status.proofs.iter() {
                    proof.verify(validator)?;
                }
                Ok(())
            },
            (_, _) => Err(anyhow::anyhow!(
                "Wrong payload type. Expected Payload::InQuorumStore {} got {} ",
                quorum_store_enabled,
                self
            )),
        }
    }
}

impl fmt::Display for Payload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Payload::DirectMempool(txns) => {
                write!(f, "InMemory txns: {}", txns.len())
            },
            Payload::InQuorumStore(proof_with_status) => {
                write!(f, "InMemory proofs: {}", proof_with_status.proofs.len())
            },
        }
    }
}

/// The payload to filter.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub enum PayloadFilter {
    DirectMempool(Vec<TransactionSummary>),
    InQuorumStore(HashSet<HashValue>),
    Empty,
}

impl From<&Vec<&Payload>> for PayloadFilter {
    fn from(exclude_payloads: &Vec<&Payload>) -> Self {
        if exclude_payloads.is_empty() {
            return PayloadFilter::Empty;
        }
        let direct_mode = exclude_payloads.iter().any(|payload| payload.is_direct());

        if direct_mode {
            let mut exclude_txns = Vec::new();
            for payload in exclude_payloads {
                if let Payload::DirectMempool(txns) = payload {
                    for txn in txns {
                        exclude_txns.push(TransactionSummary {
                            sender: txn.sender(),
                            sequence_number: txn.sequence_number(),
                        });
                    }
                }
            }
            PayloadFilter::DirectMempool(exclude_txns)
        } else {
            let mut exclude_proofs = HashSet::new();
            for payload in exclude_payloads {
                if let Payload::InQuorumStore(proof_with_status) = payload {
                    for proof in &proof_with_status.proofs {
                        exclude_proofs.insert(*proof.digest());
                    }
                }
            }
            PayloadFilter::InQuorumStore(exclude_proofs)
        }
    }
}

impl fmt::Display for PayloadFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadFilter::DirectMempool(excluded_txns) => {
                let mut txns_str = "".to_string();
                for tx in excluded_txns.iter() {
                    write!(txns_str, "{} ", tx)?;
                }
                write!(f, "{}", txns_str)
            },
            PayloadFilter::InQuorumStore(excluded_proofs) => {
                let mut proofs_str = "".to_string();
                for proof in excluded_proofs.iter() {
                    write!(proofs_str, "{} ", proof)?;
                }
                write!(f, "{}", proofs_str)
            },
            PayloadFilter::Empty => {
                write!(f, "Empty filter")
            },
        }
    }
}
