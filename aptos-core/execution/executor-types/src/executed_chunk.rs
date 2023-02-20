// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use crate::{StateComputeResult, TransactionData};
use anyhow::{bail, ensure, Result};
use aptos_crypto::hash::{CryptoHash, TransactionAccumulatorHasher};
use aptos_storage_interface::ExecutedTrees;
use aptos_types::{
    contract_event::ContractEvent,
    epoch_state::EpochState,
    ledger_info::LedgerInfoWithSignatures,
    proof::accumulator::InMemoryAccumulator,
    transaction::{Transaction, TransactionInfo, TransactionStatus, TransactionToCommit},
};
use std::sync::Arc;

#[derive(Default)]
pub struct ExecutedChunk {
    pub status: Vec<TransactionStatus>,
    pub to_commit: Vec<(Transaction, TransactionData)>,
    pub result_view: ExecutedTrees,
    /// If set, this is the new epoch info that should be changed to if this is committed.
    pub next_epoch_state: Option<EpochState>,
    pub ledger_info: Option<LedgerInfoWithSignatures>,
}

impl ExecutedChunk {
    pub fn new_empty(result_view: ExecutedTrees) -> Self {
        Self {
            result_view,
            ..Default::default()
        }
    }

    pub fn reconfig_suffix(&self) -> Self {
        assert!(self.next_epoch_state.is_some());
        Self {
            result_view: self.result_view.clone(),
            next_epoch_state: self.next_epoch_state.clone(),
            ..Default::default()
        }
    }

    pub fn transactions_to_commit(&self) -> Result<Vec<TransactionToCommit>> {
        self.to_commit
            .iter()
            .map(|(txn, txn_data)| {
                Ok(TransactionToCommit::new(
                    txn.clone(),
                    txn_data.txn_info.clone(),
                    txn_data.state_updates().clone(),
                    txn_data.write_set().clone(),
                    txn_data.events().to_vec(),
                    txn_data.is_reconfig(),
                ))
            })
            .collect()
    }

    pub fn transactions(&self) -> Vec<Transaction> {
        self.to_commit.iter().map(|(txn, _)| txn).cloned().collect()
    }

    pub fn events_to_commit(&self) -> Vec<ContractEvent> {
        self.to_commit
            .iter()
            .flat_map(|(_, txn_data)| txn_data.events())
            .cloned()
            .collect()
    }

    pub fn has_reconfiguration(&self) -> bool {
        self.next_epoch_state.is_some()
    }

    pub fn ensure_transaction_infos_match(
        &self,
        transaction_infos: &[TransactionInfo],
    ) -> Result<()> {
        ensure!(
            self.to_commit.len() == transaction_infos.len(),
            "Lengths don't match. {} vs {}",
            self.to_commit.len(),
            transaction_infos.len(),
        );
        let txn_info_hashes = self
            .to_commit
            .iter()
            .map(|(_, txn_data)| txn_data.txn_info_hash())
            .collect::<Vec<_>>();
        let expected_txn_info_hashes = transaction_infos
            .iter()
            .map(CryptoHash::hash)
            .collect::<Vec<_>>();

        if txn_info_hashes != expected_txn_info_hashes {
            for (idx, ((_, txn_data), expected_txn_info)) in
                itertools::zip_eq(self.to_commit.iter(), transaction_infos.iter()).enumerate()
            {
                if &txn_data.txn_info != expected_txn_info {
                    bail!(
                        "Transaction infos don't match. version: {}, txn_info:{}, expected_txn_info:{}",
                        self.result_view.txn_accumulator().version() + 1 + idx as u64
                            - self.to_commit.len() as u64,
                        &txn_data.txn_info,
                        expected_txn_info,
                    )
                }
            }
            unreachable!()
        } else {
            Ok(())
        }
    }

    /// Ensure that every block committed by consensus ends with a state checkpoint. That can be
    /// one of the two cases: 1. a reconfiguration (txns in the proposed block after the txn caused
    /// the reconfiguration will be retried) 2. a Transaction::StateCheckpoint at the end of the
    /// block.
    ///
    /// Called from `BlockExecutor`
    pub fn ensure_ends_with_state_checkpoint(&self) -> Result<()> {
        ensure!(
            self.next_epoch_state.is_some()
                || self
                    .to_commit
                    .last()
                    .map_or(true, |(t, _)| matches!(t, Transaction::StateCheckpoint(_))),
            "Chunk not ending with a state checkpoint.",
        );
        Ok(())
    }

    pub fn maybe_select_chunk_ending_ledger_info(
        &self,
        verified_target_li: &LedgerInfoWithSignatures,
        epoch_change_li: Option<&LedgerInfoWithSignatures>,
    ) -> Result<Option<LedgerInfoWithSignatures>> {
        let result_accumulator = self.result_view.txn_accumulator();

        if verified_target_li.ledger_info().version() + 1 == result_accumulator.num_leaves() {
            // If the chunk corresponds to the target LI, the target LI can be added to storage.
            ensure!(
                verified_target_li
                    .ledger_info()
                    .transaction_accumulator_hash()
                    == result_accumulator.root_hash(),
                "Root hash in target ledger info does not match local computation."
            );
            Ok(Some(verified_target_li.clone()))
        } else if let Some(epoch_change_li) = epoch_change_li {
            // If the epoch change LI is present, it must match the version of the chunk:

            // Verify that the given ledger info corresponds to the new accumulator.
            ensure!(
                epoch_change_li.ledger_info().transaction_accumulator_hash()
                    == result_accumulator.root_hash(),
                "Root hash of a given epoch LI does not match local computation."
            );
            ensure!(
                epoch_change_li.ledger_info().version() + 1 == result_accumulator.num_leaves(),
                "Version of a given epoch LI does not match local computation."
            );
            ensure!(
                epoch_change_li.ledger_info().ends_epoch(),
                "Epoch change LI does not carry validator set"
            );
            ensure!(
                epoch_change_li.ledger_info().next_epoch_state() == self.next_epoch_state.as_ref(),
                "New validator set of a given epoch LI does not match local computation"
            );
            Ok(Some(epoch_change_li.clone()))
        } else {
            ensure!(
                self.next_epoch_state.is_none(),
                "End of epoch chunk based on local computation but no EoE LedgerInfo provided."
            );
            Ok(None)
        }
    }

    pub fn combine(self, rhs: Self) -> Result<Self> {
        let mut to_commit = self.to_commit;
        to_commit.extend(rhs.to_commit.into_iter());
        let mut status = self.status;
        status.extend(rhs.status.into_iter());

        Ok(Self {
            status,
            to_commit,
            result_view: rhs.result_view,
            next_epoch_state: rhs.next_epoch_state,
            ledger_info: rhs.ledger_info,
        })
    }

    pub fn as_state_compute_result(
        &self,
        parent_accumulator: &Arc<InMemoryAccumulator<TransactionAccumulatorHasher>>,
    ) -> StateComputeResult {
        let txn_accu = self.result_view.txn_accumulator();

        let mut transaction_info_hashes = Vec::new();
        let mut reconfig_events = Vec::new();

        for (_, txn_data) in &self.to_commit {
            transaction_info_hashes.push(txn_data.txn_info_hash());
            reconfig_events.extend(txn_data.reconfig_events.iter().cloned())
        }

        StateComputeResult::new(
            txn_accu.root_hash(),
            txn_accu.frozen_subtree_roots().clone(),
            txn_accu.num_leaves(),
            parent_accumulator.frozen_subtree_roots().clone(),
            parent_accumulator.num_leaves(),
            self.next_epoch_state.clone(),
            self.status.clone(),
            transaction_info_hashes,
            reconfig_events,
        )
    }
}
