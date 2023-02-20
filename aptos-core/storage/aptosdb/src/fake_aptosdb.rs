// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    errors::AptosDbError,
    gauged_api,
    metrics::{LATEST_CHECKPOINT_VERSION, LEDGER_VERSION, NEXT_BLOCK_EPOCH},
    AptosDB,
};
use anyhow::{ensure, format_err, Result};
use aptos_accumulator::{HashReader, MerkleAccumulator};
use aptos_crypto::{
    hash::{CryptoHash, TransactionAccumulatorHasher, SPARSE_MERKLE_PLACEHOLDER_HASH},
    HashValue,
};
use aptos_infallible::Mutex;
use aptos_storage_interface::{
    state_delta::StateDelta, DbReader, DbWriter, ExecutedTrees, MAX_REQUEST_LIMIT,
};
use aptos_types::{
    access_path::AccessPath,
    account_address::AccountAddress,
    account_config::{AccountResource, NewBlockEvent},
    contract_event::EventWithVersion,
    epoch_state::EpochState,
    event::{EventHandle, EventKey},
    ledger_info::LedgerInfoWithSignatures,
    proof::{
        accumulator::InMemoryAccumulator, position::Position, AccumulatorConsistencyProof,
        AccumulatorRangeProof, SparseMerkleProofExt, TransactionAccumulatorProof,
        TransactionAccumulatorRangeProof, TransactionAccumulatorSummary,
        TransactionInfoListWithProof, TransactionInfoWithProof,
    },
    state_proof::StateProof,
    state_store::{
        state_key::StateKey,
        state_key_prefix::StateKeyPrefix,
        state_storage_usage::StateStorageUsage,
        state_value::{StateValue, StateValueChunkWithProof},
        table,
    },
    transaction::{
        Transaction, TransactionInfo, TransactionListWithProof, TransactionOutput,
        TransactionOutputListWithProof, TransactionToCommit, TransactionWithProof, Version,
    },
    write_set::WriteSet,
};
use arc_swap::ArcSwapOption;
use dashmap::DashMap;
use itertools::zip_eq;
use move_core_types::move_resource::MoveStructType;
use std::{collections::HashMap, mem::swap, sync::Arc};

/// Alternate implementation of [crate::state_store::buffered_state::BufferedState] for use with consensus-only-perf-test feature.
/// It stores the [StateDelta]s in memory similar to [crate::state_store::buffered_state::BufferedState] except that it does not
/// commit it to persistant storage.
#[derive(Debug)]
pub struct FakeBufferedState {
    // state until the latest checkpoint.
    state_until_checkpoint: Option<Box<StateDelta>>,
    // state after the latest checkpoint.
    state_after_checkpoint: StateDelta,
}

impl FakeBufferedState {
    pub(crate) fn new_empty() -> Self {
        let state_after_checkpoint = StateDelta::new_at_checkpoint(
            *SPARSE_MERKLE_PLACEHOLDER_HASH,
            StateStorageUsage::zero(),
            None,
        );
        let myself = Self {
            state_until_checkpoint: None,
            state_after_checkpoint,
        };
        myself.report_latest_committed_version();
        myself
    }

    pub fn current_state(&self) -> &StateDelta {
        &self.state_after_checkpoint
    }

    pub fn current_checkpoint_version(&self) -> Option<Version> {
        self.state_after_checkpoint.base_version
    }

    fn report_latest_committed_version(&self) {
        LATEST_CHECKPOINT_VERSION.set(
            self.state_after_checkpoint
                .base_version
                .map_or(-1, |v| v as i64),
        );
    }

    pub fn update(
        &mut self,
        updates_until_next_checkpoint_since_current_option: Option<
            HashMap<StateKey, Option<StateValue>>,
        >,
        mut new_state_after_checkpoint: StateDelta,
    ) -> Result<()> {
        ensure!(
            new_state_after_checkpoint.base_version >= self.state_after_checkpoint.base_version
        );
        if let Some(updates_until_next_checkpoint_since_current) =
            updates_until_next_checkpoint_since_current_option
        {
            self.state_after_checkpoint
                .updates_since_base
                .extend(updates_until_next_checkpoint_since_current);
            self.state_after_checkpoint.current = new_state_after_checkpoint.base.clone();
            self.state_after_checkpoint.current_version = new_state_after_checkpoint.base_version;
            swap(
                &mut self.state_after_checkpoint,
                &mut new_state_after_checkpoint,
            );
            if let Some(ref mut delta) = self.state_until_checkpoint {
                delta.merge(new_state_after_checkpoint);
            } else {
                self.state_until_checkpoint = Some(Box::new(new_state_after_checkpoint));
            }
        } else {
            ensure!(
                new_state_after_checkpoint.base_version == self.state_after_checkpoint.base_version
            );
            self.state_after_checkpoint = new_state_after_checkpoint;
        }
        self.report_latest_committed_version();
        Ok(())
    }
}

/// Alternate implementation of [AptosDB] for use with consensus-only-perf-test feature.
/// It stores and serves data from in-memory data structures as opposed to [AptosDB],
/// which uses RocksDB. Note that FakeAptosDB re-implements only a subset of the
/// features of [AptosDB] while passing through remaining features to the wrapped inner
/// [AptosDB].
pub struct FakeAptosDB {
    inner: AptosDB,
    // A map of transaction hash to transaction version
    txn_version_by_hash: Arc<DashMap<HashValue, Version>>,
    // A map of transaction version to Transaction
    txn_by_version: Arc<DashMap<Version, Transaction>>,
    // A map of transaction to TransactionInfo
    txn_info_by_version: Arc<DashMap<Version, TransactionInfo>>,
    // A map of Position to transaction HashValue
    txn_hash_by_position: Arc<DashMap<Position, HashValue>>,
    // Max version and transaction
    latest_txn_info: ArcSwapOption<(Version, TransactionInfo)>,
    // A map of account address to the highest executed sequence number
    account_seq_num: Arc<DashMap<AccountAddress, u64>>,
    ledger_commit_lock: std::sync::Mutex<()>,
    buffered_state: Mutex<FakeBufferedState>,
}

impl FakeAptosDB {
    pub fn new(db: AptosDB) -> Self {
        Self {
            inner: db,
            txn_by_version: Arc::new(DashMap::new()),
            txn_version_by_hash: Arc::new(DashMap::new()),
            txn_info_by_version: Arc::new(DashMap::new()),
            txn_hash_by_position: Arc::new(DashMap::new()),
            latest_txn_info: ArcSwapOption::from(None),
            account_seq_num: Arc::new(DashMap::new()),
            ledger_commit_lock: std::sync::Mutex::new(()),
            buffered_state: Mutex::new(FakeBufferedState::new_empty()),
        }
    }

    fn save_and_compute_root_hash(
        &self,
        txns_to_commit: &[TransactionToCommit],
        first_version: Version,
    ) -> Result<HashValue> {
        let txn_infos: Vec<_> = txns_to_commit
            .iter()
            .map(|t| t.transaction_info())
            .cloned()
            .collect();

        let txn_hashes: Vec<HashValue> = txn_infos.iter().map(TransactionInfo::hash).collect();
        let (root_hash, writes) =
            MerkleAccumulator::<FakeAptosDB, TransactionAccumulatorHasher>::append(
                self,
                first_version, /* num_existing_leaves */
                &txn_hashes,
            )?;
        // Store the transaction hash by position to serve [DbReader::get_latest_executed_trees] calls
        writes.iter().for_each(|(pos, hash)| {
            self.txn_hash_by_position.insert(*pos, *hash);
        });
        Ok(root_hash)
    }

    fn get_frozen_subtree_hashes(&self, num_transactions: u64) -> Result<Vec<HashValue>> {
        MerkleAccumulator::<FakeAptosDB, TransactionAccumulatorHasher>::get_frozen_subtree_hashes(
            self,
            num_transactions,
        )
    }
}

impl DbWriter for FakeAptosDB {
    fn get_state_snapshot_receiver(
        &self,
        version: Version,
        expected_root_hash: HashValue,
    ) -> Result<Box<dyn aptos_storage_interface::StateSnapshotReceiver<StateKey, StateValue>>> {
        self.inner
            .get_state_snapshot_receiver(version, expected_root_hash)
    }

    fn finalize_state_snapshot(
        &self,
        version: Version,
        output_with_proof: TransactionOutputListWithProof,
        ledger_infos: &[LedgerInfoWithSignatures],
    ) -> Result<()> {
        self.inner
            .finalize_state_snapshot(version, output_with_proof, ledger_infos)
    }

    fn save_transactions(
        &self,
        txns_to_commit: &[TransactionToCommit],
        first_version: Version,
        base_state_version: Option<Version>,
        ledger_info_with_sigs: Option<&LedgerInfoWithSignatures>,
        sync_commit: bool,
        latest_in_memory_state: StateDelta,
    ) -> Result<()> {
        gauged_api("save_transactions", || {
            // Executing and committing from more than one threads not allowed -- consensus and
            // state sync must hand over to each other after all pending execution and committing
            // complete.
            let _lock = self
                .ledger_commit_lock
                .try_lock()
                .expect("Concurrent committing detected.");

            // Persist the writeset of the genesis transaction executed on the VM. The framework
            // code in genesis is necessary for benchmark execution. Note that only the genesis
            // transaction is executed on the VM when consensus-only-perf-test feature is enabled.
            if first_version == 0 {
                self.inner.save_transactions(
                    txns_to_commit,
                    first_version,
                    base_state_version,
                    ledger_info_with_sigs,
                    sync_commit,
                    latest_in_memory_state.clone(),
                )?;
            }

            let num_txns = txns_to_commit.len() as u64;
            // ledger_info_with_sigs could be None if we are doing state synchronization. In this case
            // txns_to_commit should not be empty. Otherwise it is okay to commit empty blocks.
            ensure!(
                ledger_info_with_sigs.is_some() || num_txns > 0,
                "txns_to_commit is empty while ledger_info_with_sigs is None.",
            );

            let last_version = first_version + num_txns - 1;

            let new_root_hash = self.save_and_compute_root_hash(txns_to_commit, first_version)?;

            // If expected ledger info is provided, verify result root hash.
            if let Some(x) = ledger_info_with_sigs {
                let expected_root_hash = x.ledger_info().transaction_accumulator_hash();
                ensure!(
                    new_root_hash == expected_root_hash,
                    "Root hash calculated doesn't match expected. {:?} vs {:?}",
                    new_root_hash,
                    expected_root_hash,
                );
            }

            ensure!(Some(last_version) == latest_in_memory_state.current_version,
                "the last_version {:?} to commit doesn't match the current_version {:?} in latest_in_memory_state",
                last_version,
               latest_in_memory_state.current_version.expect("Must exist")
            );

            {
                let mut buffered_state = self.buffered_state.lock();
                ensure!(
                    base_state_version == buffered_state.state_after_checkpoint.base_version,
                    "base_state_version {:?} does not equal to the base_version {:?} in buffered state with current version {:?}",
                    base_state_version,
                    buffered_state.state_after_checkpoint.base_version,
                    buffered_state.state_after_checkpoint.current_version,
                );

                // Ensure the incoming committing requests are always consecutive and the version in
                // buffered state is consistent with that in db.
                let next_version_in_buffered_state = buffered_state
                    .state_after_checkpoint
                    .current_version
                    .map(|version| version + 1)
                    .unwrap_or(0);
                let num_transactions_in_db = self
                    .get_latest_transaction_info_option()?
                    .map(|(version, _)| version + 1)
                    .unwrap_or(0);
                ensure!(
                     num_transactions_in_db == first_version && num_transactions_in_db == next_version_in_buffered_state,
                    "The first version {} passed in, the next version in buffered state {} and the next version in db {} are inconsistent.",
                    first_version,
                    next_version_in_buffered_state,
                    num_transactions_in_db,
                );

                let updates_until_latest_checkpoint_since_current = if let Some(
                    latest_checkpoint_version,
                ) =
                    latest_in_memory_state.base_version
                {
                    if latest_checkpoint_version >= first_version {
                        let idx = (latest_checkpoint_version - first_version) as usize;
                        ensure!(
                            txns_to_commit[idx].is_state_checkpoint(),
                            "The new latest snapshot version passed in {:?} does not match with the last checkpoint version in txns_to_commit {:?}",
                            latest_checkpoint_version,
                            first_version + idx as u64
                        );
                        Some(
                            txns_to_commit[..=idx]
                                .iter()
                                .flat_map(|txn_to_commit| txn_to_commit.state_updates().clone())
                                .collect(),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };

                buffered_state.update(
                    updates_until_latest_checkpoint_since_current,
                    latest_in_memory_state,
                )?;
            }

            let last_version = first_version + txns_to_commit.len() as u64 - 1;

            // Iterate through the transactions and update the in-memory maps
            zip_eq(first_version..=last_version, txns_to_commit).try_for_each(
                |(ver, txn_to_commit)| -> Result<(), anyhow::Error> {
                    self.txn_by_version
                        .insert(ver, txn_to_commit.transaction().clone());
                    self.txn_info_by_version
                        .insert(ver, txn_to_commit.transaction_info().clone());
                    self.latest_txn_info.store(Some(Arc::new((
                        ver,
                        txn_to_commit.transaction_info().clone(),
                    ))));
                    self.txn_version_by_hash
                        .insert(txn_to_commit.transaction().hash(), ver);

                    // If it is a user transaction, also update the account sequence number
                    if let Ok(user_txn) = txn_to_commit.transaction().as_signed_user_txn() {
                        self.account_seq_num
                            .entry(user_txn.sender())
                            .and_modify(|seq_num| {
                                *seq_num = std::cmp::max(user_txn.sequence_number() + 1, *seq_num);
                            })
                            .or_insert(user_txn.sequence_number());
                    }
                    Ok::<(), anyhow::Error>(())
                },
            )?;

            // Once everything is successfully stored, update the latest in-memory ledger info.
            if let Some(x) = ledger_info_with_sigs {
                self.inner.ledger_store.set_latest_ledger_info(x.clone());

                LEDGER_VERSION.set(x.ledger_info().version() as i64);
                NEXT_BLOCK_EPOCH.set(x.ledger_info().next_block_epoch() as i64);
            }
            Ok(())
        })
    }
}

impl DbReader for FakeAptosDB {
    fn get_epoch_ending_ledger_infos(
        &self,
        start_epoch: u64,
        end_epoch: u64,
    ) -> Result<aptos_types::epoch_change::EpochChangeProof> {
        (&self.inner as &dyn DbReader).get_epoch_ending_ledger_infos(start_epoch, end_epoch)
    }

    fn get_transactions(
        &self,
        start_version: Version,
        limit: u64,
        ledger_version: Version,
        _fetch_events: bool,
    ) -> Result<TransactionListWithProof> {
        gauged_api("get_transactions", || {
            error_if_too_many_requested(limit, MAX_REQUEST_LIMIT)?;

            if start_version > ledger_version || limit == 0 {
                return Ok(TransactionListWithProof::new_empty());
            }

            let limit = std::cmp::min(limit, ledger_version - start_version + 1);

            let (txn_info_list, txn_list) = (start_version..start_version + limit)
                .map(|version| {
                    let txn_info = self
                        .txn_info_by_version
                        .get(&version)
                        .ok_or_else(|| format_err!("No transaction info at version {}", version,))?
                        .clone();

                    let txn = self
                        .txn_by_version
                        .get(&version)
                        .ok_or_else(|| format_err!("No transaction at version {}", version))?
                        .clone();

                    Ok((txn_info, txn))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();

            // None of the consumers check the proof in consensus-only-perf-test mode, so it is fine to
            // return an empty proof.
            Ok(TransactionListWithProof::new(
                txn_list,
                None,
                Some(start_version),
                TransactionInfoListWithProof::new(
                    AccumulatorRangeProof::new_empty(),
                    txn_info_list,
                ),
            ))
        })
    }

    fn get_gas_prices(
        &self,
        start_version: Version,
        limit: u64,
        ledger_version: Version,
    ) -> Result<Vec<u64>> {
        self.inner
            .get_gas_prices(start_version, limit, ledger_version)
    }

    fn get_transaction_by_hash(
        &self,
        hash: HashValue,
        ledger_version: Version,
        fetch_events: bool,
    ) -> Result<Option<TransactionWithProof>> {
        gauged_api("get_transaction_by_hash", || {
            self.txn_version_by_hash
                .get(&hash)
                .as_deref()
                .map(|version| {
                    self.get_transaction_by_version(*version, ledger_version, fetch_events)
                })
                .transpose()
        })
    }

    fn get_transaction_by_version(
        &self,
        version: Version,
        _ledger_version: Version,
        _fetch_events: bool,
    ) -> Result<TransactionWithProof> {
        gauged_api("get_transaction_by_version", || {
            let txn_info = self
                .txn_info_by_version
                .get(&version)
                .ok_or_else(|| format_err!("No transaction info at version {}", version,))?
                .clone();

            let txn = self
                .txn_by_version
                .get(&version)
                .ok_or_else(|| format_err!("No transaction at version {}", version))?
                .clone();

            let txn_info_with_proof =
                TransactionInfoWithProof::new(TransactionAccumulatorProof::new(vec![]), txn_info);

            Ok(TransactionWithProof::new(
                version,
                txn,
                None,
                txn_info_with_proof,
            ))
        })
    }

    fn get_first_txn_version(&self) -> Result<Option<Version>> {
        self.inner.get_first_txn_version()
    }

    fn get_first_viable_txn_version(&self) -> Result<Version> {
        self.inner.get_first_viable_txn_version()
    }

    fn get_first_write_set_version(&self) -> Result<Option<Version>> {
        self.inner.get_first_write_set_version()
    }

    fn get_transaction_outputs(
        &self,
        start_version: Version,
        limit: u64,
        ledger_version: Version,
    ) -> Result<TransactionOutputListWithProof> {
        gauged_api("get_transactions_outputs", || {
            error_if_too_many_requested(limit, MAX_REQUEST_LIMIT)?;

            if start_version > ledger_version || limit == 0 {
                return Ok(TransactionOutputListWithProof::new_empty());
            }

            let limit = std::cmp::min(limit, ledger_version - start_version + 1);

            let (txn_infos, txns_and_outputs) = (start_version..start_version + limit)
                .map(|version| {
                    let txn_info = self
                        .txn_info_by_version
                        .get(&version)
                        .ok_or_else(|| format_err!("No transaction info at version {}", version,))?
                        .clone();
                    let events = vec![];
                    let write_set = WriteSet::default();
                    let txn = self
                        .txn_by_version
                        .get(&version)
                        .ok_or_else(|| format_err!("No transaction at version {}", version,))?
                        .clone();
                    let txn_output = TransactionOutput::new(
                        write_set,
                        events,
                        txn_info.gas_used(),
                        txn_info.status().clone().into(),
                    );
                    Ok((txn_info, (txn, txn_output)))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();
            let proof = TransactionInfoListWithProof::new(
                TransactionAccumulatorRangeProof::new_empty(),
                txn_infos,
            );

            Ok(TransactionOutputListWithProof::new(
                txns_and_outputs,
                Some(start_version),
                proof,
            ))
        })
    }

    fn get_events(
        &self,
        event_key: &aptos_types::event::EventKey,
        start: u64,
        order: aptos_storage_interface::Order,
        limit: u64,
        ledger_version: Version,
    ) -> Result<Vec<EventWithVersion>> {
        self.inner
            .get_events(event_key, start, order, limit, ledger_version)
    }

    fn get_block_timestamp(&self, version: Version) -> Result<u64> {
        self.inner.get_block_timestamp(version)
    }

    fn get_next_block_event(&self, version: Version) -> Result<(Version, NewBlockEvent)> {
        self.inner.get_next_block_event(version)
    }

    fn get_block_info_by_version(
        &self,
        version: Version,
    ) -> Result<(Version, Version, NewBlockEvent)> {
        self.inner.get_block_info_by_version(version)
    }

    fn get_block_info_by_height(&self, height: u64) -> Result<(Version, Version, NewBlockEvent)> {
        self.inner.get_block_info_by_height(height)
    }

    fn get_last_version_before_timestamp(
        &self,
        timestamp: u64,
        ledger_version: Version,
    ) -> Result<Version> {
        self.inner
            .get_last_version_before_timestamp(timestamp, ledger_version)
    }

    fn get_latest_epoch_state(&self) -> Result<EpochState> {
        self.inner.get_latest_epoch_state()
    }

    fn get_prefixed_state_value_iterator(
        &self,
        key_prefix: &StateKeyPrefix,
        cursor: Option<&StateKey>,
        version: Version,
    ) -> Result<Box<dyn Iterator<Item = anyhow::Result<(StateKey, StateValue)>> + '_>> {
        self.inner
            .get_prefixed_state_value_iterator(key_prefix, cursor, version)
    }

    fn get_latest_ledger_info_option(&self) -> Result<Option<LedgerInfoWithSignatures>> {
        self.inner.get_latest_ledger_info_option()
    }

    fn get_latest_state_checkpoint_version(&self) -> Result<Option<Version>> {
        gauged_api("get_latest_state_checkpoint_version", || {
            Ok(self
                .buffered_state
                .lock()
                .state_after_checkpoint
                .current_version)
        })
    }

    fn get_state_snapshot_before(
        &self,
        next_version: Version,
    ) -> Result<Option<(Version, HashValue)>> {
        self.inner.get_state_snapshot_before(next_version)
    }

    fn get_account_transaction(
        &self,
        address: aptos_types::PeerId,
        seq_num: u64,
        include_events: bool,
        ledger_version: Version,
    ) -> Result<Option<TransactionWithProof>> {
        self.inner
            .get_account_transaction(address, seq_num, include_events, ledger_version)
    }

    fn get_account_transactions(
        &self,
        address: aptos_types::PeerId,
        seq_num: u64,
        limit: u64,
        include_events: bool,
        ledger_version: Version,
    ) -> Result<aptos_types::transaction::AccountTransactionsWithProof> {
        self.inner
            .get_account_transactions(address, seq_num, limit, include_events, ledger_version)
    }

    fn get_state_proof_with_ledger_info(
        &self,
        known_version: u64,
        ledger_info: LedgerInfoWithSignatures,
    ) -> Result<StateProof> {
        self.inner
            .get_state_proof_with_ledger_info(known_version, ledger_info)
    }

    fn get_state_proof(&self, known_version: u64) -> Result<StateProof> {
        self.inner.get_state_proof(known_version)
    }

    fn get_state_value_by_version(
        &self,
        state_key: &StateKey,
        version: Version,
    ) -> Result<Option<StateValue>> {
        let access_path = AccessPath::try_from(state_key.clone())?;
        let account_address = access_path.address;
        let struct_tag = access_path.get_struct_tag();

        // Since the genesis write set is persisted with AptosDB, we call
        // it to serve state values targetting the framework account
        // (to access AptosCoin, for example).
        // The in-memory data structures only handles "normal user" accounts.
        if account_address != AccountAddress::ONE
            && struct_tag.is_some()
            && struct_tag.unwrap() == AccountResource::struct_tag()
        {
            let seq_num = match self.account_seq_num.get(&account_address).as_deref() {
                Some(seq_num) => *seq_num,
                None => {
                    let initial_seq_num = 0;
                    self.account_seq_num
                        .insert(account_address, initial_seq_num);
                    initial_seq_num
                },
            };
            let account = AccountResource::new(
                seq_num,
                vec![],
                EventHandle::new(EventKey::new(0, account_address), 0),
                EventHandle::new(EventKey::new(1, account_address), 0),
            );
            let bytes = bcs::to_bytes(&account)?;
            Ok(Some(StateValue::new(bytes)))
        } else {
            self.inner.get_state_value_by_version(state_key, version)
        }
    }

    fn get_state_proof_by_version_ext(
        &self,
        state_key: &StateKey,
        version: Version,
    ) -> Result<SparseMerkleProofExt> {
        self.inner
            .get_state_proof_by_version_ext(state_key, version)
    }

    fn get_state_value_with_proof_by_version_ext(
        &self,
        state_key: &StateKey,
        version: Version,
    ) -> Result<(Option<StateValue>, SparseMerkleProofExt)> {
        self.inner
            .get_state_value_with_proof_by_version_ext(state_key, version)
    }

    fn get_latest_executed_trees(&self) -> Result<ExecutedTrees> {
        // If the genesis is not executed yet, we need to get the executed trees from the inner AptosDB
        // This is because when we call save_transactions for the genesis block, we call [AptosDB::save_transactions]
        // where there is an expectation that the root of the SMTs are the same pointers. Here,
        // we get from the inner AptosDB which ensures that the pointers match when save_transactions is called.
        if self.get_latest_version().unwrap_or_default() == 0 {
            return self.inner.get_latest_executed_trees();
        }

        gauged_api("get_latest_executed_trees", || {
            let buffered_state = self.buffered_state.lock();
            let num_txns = buffered_state
                .current_state()
                .current_version
                .map_or(0, |v| v + 1);

            let frozen_subtrees = self.get_frozen_subtree_hashes(num_txns)?;
            let transaction_accumulator =
                Arc::new(InMemoryAccumulator::new(frozen_subtrees, num_txns)?);
            let executed_trees = ExecutedTrees::new(
                buffered_state.current_state().clone(),
                transaction_accumulator,
            );
            Ok(executed_trees)
        })
    }

    fn get_epoch_ending_ledger_info(&self, known_version: u64) -> Result<LedgerInfoWithSignatures> {
        self.inner.get_epoch_ending_ledger_info(known_version)
    }

    fn get_latest_transaction_info_option(
        &self,
    ) -> Result<Option<(Version, aptos_types::transaction::TransactionInfo)>> {
        Ok(self
            .latest_txn_info
            .load_full()
            .map(|txn| txn.as_ref().clone()))
    }

    fn get_accumulator_root_hash(&self, _version: Version) -> Result<HashValue> {
        Ok(HashValue::zero())
    }

    fn get_accumulator_consistency_proof(
        &self,
        client_known_version: Option<Version>,
        ledger_version: Version,
    ) -> Result<AccumulatorConsistencyProof> {
        self.inner
            .get_accumulator_consistency_proof(client_known_version, ledger_version)
    }

    fn get_accumulator_summary(
        &self,
        ledger_version: Version,
    ) -> Result<TransactionAccumulatorSummary> {
        let num_txns = ledger_version + 1;
        let frozen_subtrees = self.get_frozen_subtree_hashes(num_txns)?;
        TransactionAccumulatorSummary::new(InMemoryAccumulator::new(frozen_subtrees, num_txns)?)
    }

    fn get_state_leaf_count(&self, version: Version) -> Result<usize> {
        self.inner.get_state_leaf_count(version)
    }

    fn get_state_value_chunk_with_proof(
        &self,
        version: Version,
        start_idx: usize,
        chunk_size: usize,
    ) -> Result<StateValueChunkWithProof> {
        self.inner
            .get_state_value_chunk_with_proof(version, start_idx, chunk_size)
    }

    fn is_state_pruner_enabled(&self) -> Result<bool> {
        self.inner.is_state_pruner_enabled()
    }

    fn get_epoch_snapshot_prune_window(&self) -> Result<usize> {
        self.inner.get_epoch_snapshot_prune_window()
    }

    fn is_ledger_pruner_enabled(&self) -> Result<bool> {
        self.inner.is_ledger_pruner_enabled()
    }

    fn get_ledger_prune_window(&self) -> Result<usize> {
        self.inner.get_ledger_prune_window()
    }

    fn get_table_info(&self, handle: table::TableHandle) -> Result<table::TableInfo> {
        self.inner.get_table_info(handle)
    }

    fn indexer_enabled(&self) -> bool {
        self.inner.indexer_enabled()
    }

    fn get_state_storage_usage(&self, version: Option<Version>) -> Result<StateStorageUsage> {
        self.inner.get_state_storage_usage(version)
    }
}

/// This is necessary for constructing the [ExecutedTrees] to serve [DbReader::get_latest_executed_trees]
/// requests.
impl HashReader for FakeAptosDB {
    fn get(&self, position: Position) -> Result<HashValue> {
        self.txn_hash_by_position
            .get(&position)
            .as_deref()
            .cloned()
            .ok_or_else(|| format_err!("Position not found: {}", position))
    }
}

fn error_if_too_many_requested(num_requested: u64, max_allowed: u64) -> Result<()> {
    if num_requested > max_allowed {
        Err(AptosDbError::TooManyRequested(num_requested, max_allowed).into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        fake_aptosdb::FakeAptosDB,
        test_helper::{arb_blocks_to_commit, update_in_memory_state},
        AptosDB,
    };
    use anyhow::{ensure, Result};
    use aptos_crypto::{hash::CryptoHash, HashValue};
    use aptos_storage_interface::{DbReader, DbWriter};
    use aptos_temppath::TempPath;
    use aptos_types::{
        account_address::AccountAddress,
        ledger_info::LedgerInfoWithSignatures,
        transaction::{
            TransactionListWithProof, TransactionOutputListWithProof, TransactionStatus,
            TransactionToCommit, TransactionWithProof, Version,
        },
    };
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_save_transactions(input in arb_blocks_to_commit()) {
            let tmp_dir = TempPath::new();
            let db = FakeAptosDB::new(AptosDB::new_for_test(&tmp_dir));

            let mut in_memory_state = db
                .inner
                .buffered_state()
                .lock()
                .current_state()
                .clone();

            let mut cur_ver: Version = 0;
            for (txns_to_commit, ledger_info_with_sigs) in input.iter() {
                update_in_memory_state(&mut in_memory_state, txns_to_commit.as_slice());
                db.save_transactions(
                    txns_to_commit,
                    cur_ver,                /* first_version */
                    cur_ver.checked_sub(1), /* base_state_version */
                    Some(ledger_info_with_sigs),
                    false, /* sync_commit */
                    in_memory_state.clone(),
                )
                .unwrap();

                assert_eq!(
                    db.get_latest_ledger_info().unwrap(),
                    *ledger_info_with_sigs
                );
                verify_committed_transactions(
                    &db,
                    txns_to_commit,
                    cur_ver,
                    ledger_info_with_sigs,
                );

                cur_ver += txns_to_commit.len() as u64;
            }
        }
    }

    fn verify_committed_transactions(
        db: &FakeAptosDB,
        txns_to_commit: &[TransactionToCommit],
        first_version: Version,
        ledger_info_with_sigs: &LedgerInfoWithSignatures,
    ) {
        let ledger_info = ledger_info_with_sigs.ledger_info();
        let ledger_version = ledger_info.version();
        assert_eq!(
            db.get_accumulator_root_hash(ledger_version).unwrap(),
            HashValue::zero(),
        );

        let mut cur_ver = first_version;
        for txn_to_commit in txns_to_commit {
            let txn_info = &*db.txn_info_by_version.get(&cur_ver).unwrap();

            // Verify transaction hash.
            assert_eq!(
                txn_info.transaction_hash(),
                txn_to_commit.transaction().hash()
            );

            if !txn_to_commit.is_state_checkpoint() {
                // Fetch and verify transaction itself.
                let txn = txn_to_commit.transaction().as_signed_user_txn().unwrap();
                let txn_with_proof = db
                    .get_transaction_by_hash(
                        txn_to_commit.transaction().hash(),
                        ledger_version,
                        true,
                    )
                    .unwrap()
                    .unwrap();

                assert_eq!(
                    txn_with_proof.transaction.hash(),
                    txn_to_commit.transaction().hash()
                );

                verify_user_txn(
                    &txn_with_proof,
                    cur_ver,
                    txn.sender(),
                    txn.sequence_number(),
                )
                .unwrap();

                let txn_with_proof = db
                    .get_transaction_by_version(cur_ver, ledger_version, true)
                    .unwrap();
                verify_user_txn(
                    &txn_with_proof,
                    cur_ver,
                    txn.sender(),
                    txn.sequence_number(),
                )
                .unwrap();

                let txn_list_with_proof = db
                    .get_transactions(cur_ver, 1, ledger_version, true /* fetch_events */)
                    .unwrap();
                verify_txn_list(&txn_list_with_proof, Some(cur_ver)).unwrap();
                assert_eq!(txn_list_with_proof.transactions.len(), 1);

                let txn_output_list_with_proof = db
                    .get_transaction_outputs(cur_ver, 1, ledger_version)
                    .unwrap();
                verify_txn_outputs(&txn_output_list_with_proof, Some(cur_ver)).unwrap();
                assert_eq!(txn_output_list_with_proof.transactions_and_outputs.len(), 1);
            }
            cur_ver += 1;
        }
    }

    fn verify_user_txn(
        transaction_with_proof: &TransactionWithProof,
        version: Version,
        sender: AccountAddress,
        sequence_number: u64,
    ) -> Result<()> {
        let signed_transaction = transaction_with_proof.transaction.as_signed_user_txn()?;

        ensure!(
            transaction_with_proof.version == version,
            "Version ({}) is not expected ({}).",
            transaction_with_proof.version,
            version,
        );
        ensure!(
            signed_transaction.sender() == sender,
            "Sender ({}) not expected ({}).",
            signed_transaction.sender(),
            sender,
        );
        ensure!(
            signed_transaction.sequence_number() == sequence_number,
            "Sequence number ({}) not expected ({}).",
            signed_transaction.sequence_number(),
            sequence_number,
        );

        let txn_hash = transaction_with_proof.transaction.hash();
        ensure!(
            txn_hash
                == transaction_with_proof
                    .proof
                    .transaction_info()
                    .transaction_hash(),
            "Transaction hash ({}) not expected ({}).",
            txn_hash,
            transaction_with_proof
                .proof
                .transaction_info()
                .transaction_hash(),
        );

        Ok(())
    }

    fn verify_txn_outputs(
        txn_outputs_with_proof: &TransactionOutputListWithProof,
        first_transaction_output_version: Option<Version>,
    ) -> Result<()> {
        // Verify the first transaction/output versions match
        ensure!(
            txn_outputs_with_proof.first_transaction_output_version
                == first_transaction_output_version,
            "First transaction and output version ({:?}) doesn't match given version ({:?}).",
            txn_outputs_with_proof.first_transaction_output_version,
            first_transaction_output_version,
        );

        // Verify the lengths of the transaction(output)s and transaction infos match
        ensure!(
            txn_outputs_with_proof.proof.transaction_infos.len()
                == txn_outputs_with_proof.transactions_and_outputs.len(),
            "The number of TransactionInfo objects ({}) does not match the number of \
             transactions and outputs ({}).",
            txn_outputs_with_proof.proof.transaction_infos.len(),
            txn_outputs_with_proof.transactions_and_outputs.len(),
        );

        // Verify the events, status, gas used and transaction hashes.
        itertools::zip_eq(
            &txn_outputs_with_proof.transactions_and_outputs,
            &txn_outputs_with_proof.proof.transaction_infos,
        )
        .map(|((txn, txn_output), txn_info)| {
            // Verify the gas matches for both the transaction info and output
            ensure!(
                txn_output.gas_used() == txn_info.gas_used(),
                "The gas used in transaction output does not match the transaction info \
                     in proof. Gas used in transaction output: {}. Gas used in txn_info: {}.",
                txn_output.gas_used(),
                txn_info.gas_used(),
            );

            // Verify the execution status matches for both the transaction info and output.
            ensure!(
                *txn_output.status() == TransactionStatus::Keep(txn_info.status().clone()),
                "The execution status of transaction output does not match the transaction \
                     info in proof. Status in transaction output: {:?}. Status in txn_info: {:?}.",
                txn_output.status(),
                txn_info.status(),
            );

            // Verify the transaction hashes match those of the transaction infos
            let txn_hash = txn.hash();
            ensure!(
                txn_hash == txn_info.transaction_hash(),
                "The transaction hash does not match the hash in transaction info. \
                     Transaction hash: {:x}. Transaction hash in txn_info: {:x}.",
                txn_hash,
                txn_info.transaction_hash(),
            );
            Ok(())
        })
        .collect::<Result<Vec<_>>>()?;

        Ok(())
    }

    fn verify_txn_list(
        txn_list: &TransactionListWithProof,
        first_transaction_version: Option<Version>,
    ) -> Result<()> {
        // Verify the first transaction versions match
        ensure!(
            txn_list.first_transaction_version == first_transaction_version,
            "First transaction version ({:?}) doesn't match given version ({:?}).",
            txn_list.first_transaction_version,
            first_transaction_version,
        );

        // Verify the lengths of the transactions and transaction infos match
        ensure!(
            txn_list.proof.transaction_infos.len() == txn_list.transactions.len(),
            "The number of TransactionInfo objects ({}) does not match the number of \
             transactions ({}).",
            txn_list.proof.transaction_infos.len(),
            txn_list.transactions.len(),
        );

        // Verify the transaction hashes match those of the transaction infos
        let transaction_hashes: Vec<_> =
            txn_list.transactions.iter().map(CryptoHash::hash).collect();
        itertools::zip_eq(transaction_hashes, &txn_list.proof.transaction_infos)
            .map(|(txn_hash, txn_info)| {
                ensure!(
                    txn_hash == txn_info.transaction_hash(),
                    "The hash of transaction does not match the transaction info in proof. \
                     Transaction hash: {:x}. Transaction hash in txn_info: {:x}.",
                    txn_hash,
                    txn_info.transaction_hash(),
                );
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(())
    }
}
