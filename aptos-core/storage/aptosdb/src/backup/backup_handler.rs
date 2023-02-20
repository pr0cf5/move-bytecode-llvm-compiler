// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    event_store::EventStore,
    ledger_store::LedgerStore,
    metrics::{
        BACKUP_EPOCH_ENDING_EPOCH, BACKUP_STATE_SNAPSHOT_LEAF_IDX, BACKUP_STATE_SNAPSHOT_VERSION,
        BACKUP_TXN_VERSION,
    },
    state_store::StateStore,
    transaction_store::TransactionStore,
};
use anyhow::{anyhow, ensure, Context, Result};
use aptos_crypto::hash::HashValue;
use aptos_types::{
    contract_event::ContractEvent,
    ledger_info::LedgerInfoWithSignatures,
    proof::{SparseMerkleRangeProof, TransactionAccumulatorRangeProof, TransactionInfoWithProof},
    state_store::{state_key::StateKey, state_value::StateValue},
    transaction::{Transaction, TransactionInfo, Version},
    write_set::WriteSet,
};
use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};

/// `BackupHandler` provides functionalities for AptosDB data backup.
#[derive(Clone)]
pub struct BackupHandler {
    ledger_store: Arc<LedgerStore>,
    transaction_store: Arc<TransactionStore>,
    state_store: Arc<StateStore>,
    event_store: Arc<EventStore>,
}

impl BackupHandler {
    pub(crate) fn new(
        ledger_store: Arc<LedgerStore>,
        transaction_store: Arc<TransactionStore>,
        state_store: Arc<StateStore>,
        event_store: Arc<EventStore>,
    ) -> Self {
        Self {
            ledger_store,
            transaction_store,
            state_store,
            event_store,
        }
    }

    /// Gets an iterator that yields a range of transactions.
    pub fn get_transaction_iter(
        &self,
        start_version: Version,
        num_transactions: usize,
    ) -> Result<
        impl Iterator<Item = Result<(Transaction, TransactionInfo, Vec<ContractEvent>, WriteSet)>> + '_,
    > {
        let txn_iter = self
            .transaction_store
            .get_transaction_iter(start_version, num_transactions)?;
        let mut txn_info_iter = self
            .ledger_store
            .get_transaction_info_iter(start_version, num_transactions)?;
        let mut event_vec_iter = self
            .event_store
            .get_events_by_version_iter(start_version, num_transactions)?;
        let mut write_set_iter = self
            .transaction_store
            .get_write_set_iter(start_version, num_transactions)?;

        let zipped = txn_iter.enumerate().map(move |(idx, txn_res)| {
            let version = start_version + idx as u64; // overflow is impossible since it's check upon txn_iter construction.

            let txn = txn_res?;
            let txn_info = txn_info_iter
                .next()
                .ok_or_else(|| anyhow!("TransactionInfo not found when Transaction exists."))
                .context(version)??;
            let event_vec = event_vec_iter
                .next()
                .ok_or_else(|| anyhow!("Events not found when Transaction exists."))
                .context(version)??;
            let write_set = write_set_iter
                .next()
                .ok_or_else(|| anyhow!("WriteSet not found when Transaction exists."))
                .context(version)??;
            BACKUP_TXN_VERSION.set(version as i64);
            Ok((txn, txn_info, event_vec, write_set))
        });
        Ok(zipped)
    }

    /// Gets the proof for a transaction chunk.
    /// N.B. the `LedgerInfo` returned will always be in the same epoch of the `last_version`.
    pub fn get_transaction_range_proof(
        &self,
        first_version: Version,
        last_version: Version,
    ) -> Result<(TransactionAccumulatorRangeProof, LedgerInfoWithSignatures)> {
        ensure!(
            last_version >= first_version,
            "Bad transaction range: [{}, {}]",
            first_version,
            last_version
        );
        let num_transactions = last_version - first_version + 1;
        let epoch = self.ledger_store.get_epoch(last_version)?;
        let ledger_info = self.ledger_store.get_latest_ledger_info_in_epoch(epoch)?;
        let accumulator_proof = self.ledger_store.get_transaction_range_proof(
            Some(first_version),
            num_transactions,
            ledger_info.ledger_info().version(),
        )?;
        Ok((accumulator_proof, ledger_info))
    }

    /// Gets an iterator which can yield all accounts in the state tree.
    pub fn get_account_iter(
        &self,
        version: Version,
    ) -> Result<Box<dyn Iterator<Item = Result<(StateKey, StateValue)>> + Send + Sync>> {
        let iterator = self
            .state_store
            .get_state_key_and_value_iter(version, HashValue::zero())?
            .enumerate()
            .map(move |(idx, res)| {
                BACKUP_STATE_SNAPSHOT_VERSION.set(version as i64);
                BACKUP_STATE_SNAPSHOT_LEAF_IDX.set(idx as i64);
                res
            });
        Ok(Box::new(iterator))
    }

    /// Gets the proof that proves a range of accounts.
    pub fn get_account_state_range_proof(
        &self,
        rightmost_key: HashValue,
        version: Version,
    ) -> Result<SparseMerkleRangeProof> {
        self.state_store
            .get_value_range_proof(rightmost_key, version)
    }

    /// Gets the epoch, committed version, and synced version of the DB.
    pub fn get_db_state(&self) -> Result<Option<DbState>> {
        Ok(self
            .ledger_store
            .get_latest_ledger_info_option()
            .map(|li| DbState {
                epoch: li.ledger_info().epoch(),
                committed_version: li.ledger_info().version(),
            }))
    }

    /// Gets the proof of the state root at specified version.
    /// N.B. the `LedgerInfo` returned will always be in the same epoch of the version.
    pub fn get_state_root_proof(
        &self,
        version: Version,
    ) -> Result<(TransactionInfoWithProof, LedgerInfoWithSignatures)> {
        let epoch = self.ledger_store.get_epoch(version)?;
        let ledger_info = self.ledger_store.get_latest_ledger_info_in_epoch(epoch)?;
        let txn_info = self
            .ledger_store
            .get_transaction_info_with_proof(version, ledger_info.ledger_info().version())?;

        Ok((txn_info, ledger_info))
    }

    pub fn get_epoch_ending_ledger_info_iter(
        &self,
        start_epoch: u64,
        end_epoch: u64,
    ) -> Result<impl Iterator<Item = Result<LedgerInfoWithSignatures>> + '_> {
        Ok(self
            .ledger_store
            .get_epoch_ending_ledger_info_iter(start_epoch, end_epoch)?
            .enumerate()
            .map(move |(idx, li)| {
                BACKUP_EPOCH_ENDING_EPOCH.set((start_epoch + idx as u64) as i64);
                li
            }))
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DbState {
    pub epoch: u64,
    pub committed_version: Version,
}

impl fmt::Display for DbState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "epoch: {}, committed_version: {}",
            self.epoch, self.committed_version,
        )
    }
}
