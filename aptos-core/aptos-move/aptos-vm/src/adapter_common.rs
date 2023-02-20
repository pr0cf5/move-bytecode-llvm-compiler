// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    counters::*,
    data_cache::AsMoveResolver,
    logging::AdapterLogSchema,
    move_vm_ext::{MoveResolverExt, SessionExt, SessionId},
};
use anyhow::Result;
use aptos_aggregator::transaction::TransactionOutputExt;
use aptos_state_view::StateView;
use aptos_types::{
    block_metadata::BlockMetadata,
    transaction::{
        SignatureCheckedTransaction, SignedTransaction, Transaction, TransactionOutput,
        TransactionStatus, VMValidatorResult, WriteSetPayload,
    },
    vm_status::{StatusCode, VMStatus},
    write_set::WriteSet,
};

/// This trait describes the VM adapter's interface.
/// TODO: bring more of the execution logic in aptos_vm into this file.
pub trait VMAdapter {
    /// Creates a new Session backed by the given storage.
    /// TODO: this doesn't belong in this trait. We should be able to remove
    /// this after redesigning cache ownership model.
    fn new_session<'r, R: MoveResolverExt>(
        &self,
        remote: &'r R,
        session_id: SessionId,
    ) -> SessionExt<'r, '_, R>;

    /// Checks the signature of the given signed transaction and returns
    /// `Ok(SignatureCheckedTransaction)` if the signature is valid.
    fn check_signature(txn: SignedTransaction) -> Result<SignatureCheckedTransaction>;

    /// Check if the transaction format is supported.
    fn check_transaction_format(&self, txn: &SignedTransaction) -> Result<(), VMStatus>;

    /// Runs the prologue for the given transaction.
    fn run_prologue<S: MoveResolverExt>(
        &self,
        session: &mut SessionExt<S>,
        storage: &S,
        transaction: &SignatureCheckedTransaction,
        log_context: &AdapterLogSchema,
    ) -> Result<(), VMStatus>;

    /// TODO: maybe remove this after more refactoring of execution logic.
    fn should_restart_execution(output: &TransactionOutput) -> bool;

    /// Execute a single transaction.
    fn execute_single_transaction<S: MoveResolverExt + StateView>(
        &self,
        txn: &PreprocessedTransaction,
        data_cache: &S,
        log_context: &AdapterLogSchema,
    ) -> Result<(VMStatus, TransactionOutputExt, Option<String>), VMStatus>;
}

/// Validate a signed transaction by performing the following:
/// 1. Check the signature(s) included in the signed transaction
/// 2. Check that the transaction is allowed in the context provided by the `adapter`
/// 3. Run the prologue to perform additional on-chain checks
/// The returned `VMValidatorResult` will have status `None` and if all checks succeeded
/// and `Some(DiscardedVMStatus)` otherwise.
pub fn validate_signed_transaction<A: VMAdapter>(
    adapter: &A,
    transaction: SignedTransaction,
    state_view: &impl StateView,
) -> VMValidatorResult {
    let _timer = TXN_VALIDATION_SECONDS.start_timer();
    let log_context = AdapterLogSchema::new(state_view.id(), 0);
    let txn = match A::check_signature(transaction) {
        Ok(t) => t,
        _ => {
            return VMValidatorResult::error(StatusCode::INVALID_SIGNATURE);
        },
    };

    let resolver = state_view.as_move_resolver();
    let mut session = adapter.new_session(&resolver, SessionId::txn(&txn));

    let validation_result = validate_signature_checked_transaction(
        adapter,
        &mut session,
        &resolver,
        &txn,
        true,
        &log_context,
    );

    // Increment the counter for transactions verified.
    let (counter_label, result) = match validation_result {
        Ok(_) => (
            "success",
            VMValidatorResult::new(None, txn.gas_unit_price()),
        ),
        Err(err) => (
            "failure",
            VMValidatorResult::new(Some(err.status_code()), 0),
        ),
    };
    TRANSACTIONS_VALIDATED
        .with_label_values(&[counter_label])
        .inc();

    result
}

pub(crate) fn validate_signature_checked_transaction<S: MoveResolverExt, A: VMAdapter>(
    adapter: &A,
    session: &mut SessionExt<S>,
    storage: &S,
    transaction: &SignatureCheckedTransaction,
    allow_too_new: bool,
    log_context: &AdapterLogSchema,
) -> Result<(), VMStatus> {
    adapter.check_transaction_format(transaction)?;

    let prologue_status = adapter.run_prologue(session, storage, transaction, log_context);
    match prologue_status {
        Err(err) if !allow_too_new || err.status_code() != StatusCode::SEQUENCE_NUMBER_TOO_NEW => {
            Err(err)
        },
        _ => Ok(()),
    }
}

/// Transactions after signature checking:
/// Waypoints and BlockPrologues are not signed and are unaffected by signature checking,
/// but a user transaction or writeset transaction is transformed to a SignatureCheckedTransaction.
#[derive(Debug)]
pub enum PreprocessedTransaction {
    UserTransaction(Box<SignatureCheckedTransaction>),
    WaypointWriteSet(WriteSetPayload),
    BlockMetadata(BlockMetadata),
    InvalidSignature,
    StateCheckpoint,
}

/// Check the signature (if any) of a transaction. If the signature is OK, the result
/// is a PreprocessedTransaction, where a user transaction is translated to a
/// SignatureCheckedTransaction and also categorized into either a UserTransaction
/// or a WriteSet transaction.
pub(crate) fn preprocess_transaction<A: VMAdapter>(txn: Transaction) -> PreprocessedTransaction {
    match txn {
        Transaction::BlockMetadata(b) => PreprocessedTransaction::BlockMetadata(b),
        Transaction::GenesisTransaction(ws) => PreprocessedTransaction::WaypointWriteSet(ws),
        Transaction::UserTransaction(txn) => {
            let checked_txn = match A::check_signature(txn) {
                Ok(checked_txn) => checked_txn,
                _ => {
                    return PreprocessedTransaction::InvalidSignature;
                },
            };
            PreprocessedTransaction::UserTransaction(Box::new(checked_txn))
        },
        Transaction::StateCheckpoint(_) => PreprocessedTransaction::StateCheckpoint,
    }
}

pub(crate) fn discard_error_vm_status(err: VMStatus) -> (VMStatus, TransactionOutputExt) {
    let vm_status = err.clone();
    let error_code = match err.keep_or_discard() {
        Ok(_) => {
            debug_assert!(false, "discarding non-discardable error: {:?}", vm_status);
            vm_status.status_code()
        },
        Err(code) => code,
    };
    (vm_status, discard_error_output(error_code))
}

pub(crate) fn discard_error_output(err: StatusCode) -> TransactionOutputExt {
    // Since this transaction will be discarded, no writeset will be included.
    TransactionOutputExt::from(TransactionOutput::new(
        WriteSet::default(),
        vec![],
        0,
        TransactionStatus::Discard(err),
    ))
}
