// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    account_address::AccountAddress,
    block_metadata::BlockMetadata,
    chain_id::ChainId,
    contract_event::ContractEvent,
    ledger_info::LedgerInfo,
    proof::{
        accumulator::InMemoryAccumulator, TransactionInfoListWithProof, TransactionInfoWithProof,
    },
    transaction::authenticator::{AccountAuthenticator, TransactionAuthenticator},
    vm_status::{DiscardedVMStatus, KeptVMStatus, StatusCode, StatusType, VMStatus},
    write_set::WriteSet,
};
use anyhow::{ensure, format_err, Error, Result};
use aptos_crypto::{
    ed25519::*,
    hash::{CryptoHash, EventAccumulatorHasher},
    multi_ed25519::{MultiEd25519PublicKey, MultiEd25519Signature},
    traits::{signing_message, SigningKey},
    CryptoMaterialError, HashValue,
};
use aptos_crypto_derive::{BCSCryptoHash, CryptoHasher};
use move_core_types::transaction_argument::convert_txn_args;
#[cfg(any(test, feature = "fuzzing"))]
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    convert::TryFrom,
    fmt,
    fmt::{Debug, Display, Formatter},
};

pub mod authenticator;
mod change_set;
mod module;
mod script;
mod transaction_argument;

use crate::state_store::{state_key::StateKey, state_value::StateValue};
#[cfg(any(test, feature = "fuzzing"))]
pub use change_set::NoOpChangeSetChecker;
pub use change_set::{ChangeSet, CheckChangeSet};
pub use module::{Module, ModuleBundle};
use move_core_types::vm_status::AbortLocation;
use once_cell::sync::OnceCell;
pub use script::{
    ArgumentABI, EntryABI, EntryFunction, EntryFunctionABI, Script, TransactionScriptABI,
    TypeArgumentABI,
};
use std::{collections::BTreeSet, hash::Hash, ops::Deref, sync::atomic::AtomicU64};
pub use transaction_argument::{parse_transaction_argument, TransactionArgument};

pub type Version = u64; // Height - also used for MVCC in StateDB
pub type AtomicVersion = AtomicU64;

/// RawTransaction is the portion of a transaction that a client signs.
#[derive(
    Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, CryptoHasher, BCSCryptoHash,
)]
pub struct RawTransaction {
    /// Sender's address.
    sender: AccountAddress,

    /// Sequence number of this transaction. This must match the sequence number
    /// stored in the sender's account at the time the transaction executes.
    sequence_number: u64,

    /// The transaction payload, e.g., a script to execute.
    payload: TransactionPayload,

    /// Maximal total gas to spend for this transaction.
    max_gas_amount: u64,

    /// Price to be paid per gas unit.
    gas_unit_price: u64,

    /// Expiration timestamp for this transaction, represented
    /// as seconds from the Unix Epoch. If the current blockchain timestamp
    /// is greater than or equal to this time, then the transaction has
    /// expired and will be discarded. This can be set to a large value far
    /// in the future to indicate that a transaction does not expire.
    expiration_timestamp_secs: u64,

    /// Chain ID of the Aptos network this transaction is intended for.
    chain_id: ChainId,
}

impl RawTransaction {
    /// Create a new `RawTransaction` with a payload.
    ///
    /// It can be either to publish a module, to execute a script, or to issue a writeset
    /// transaction.
    pub fn new(
        sender: AccountAddress,
        sequence_number: u64,
        payload: TransactionPayload,
        max_gas_amount: u64,
        gas_unit_price: u64,
        expiration_timestamp_secs: u64,
        chain_id: ChainId,
    ) -> Self {
        RawTransaction {
            sender,
            sequence_number,
            payload,
            max_gas_amount,
            gas_unit_price,
            expiration_timestamp_secs,
            chain_id,
        }
    }

    /// Create a new `RawTransaction` with a script.
    ///
    /// A script transaction contains only code to execute. No publishing is allowed in scripts.
    pub fn new_script(
        sender: AccountAddress,
        sequence_number: u64,
        script: Script,
        max_gas_amount: u64,
        gas_unit_price: u64,
        expiration_timestamp_secs: u64,
        chain_id: ChainId,
    ) -> Self {
        RawTransaction {
            sender,
            sequence_number,
            payload: TransactionPayload::Script(script),
            max_gas_amount,
            gas_unit_price,
            expiration_timestamp_secs,
            chain_id,
        }
    }

    /// Create a new `RawTransaction` with a entry function.
    ///
    /// A script transaction contains only code to execute. No publishing is allowed in scripts.
    pub fn new_entry_function(
        sender: AccountAddress,
        sequence_number: u64,
        entry_function: EntryFunction,
        max_gas_amount: u64,
        gas_unit_price: u64,
        expiration_timestamp_secs: u64,
        chain_id: ChainId,
    ) -> Self {
        RawTransaction {
            sender,
            sequence_number,
            payload: TransactionPayload::EntryFunction(entry_function),
            max_gas_amount,
            gas_unit_price,
            expiration_timestamp_secs,
            chain_id,
        }
    }

    /// Create a new `RawTransaction` with a module to publish.
    ///
    /// A module transaction is the only way to publish code. Only one module per transaction
    /// can be published.
    pub fn new_module(
        sender: AccountAddress,
        sequence_number: u64,
        module: Module,
        max_gas_amount: u64,
        gas_unit_price: u64,
        expiration_timestamp_secs: u64,
        chain_id: ChainId,
    ) -> Self {
        RawTransaction {
            sender,
            sequence_number,
            payload: TransactionPayload::ModuleBundle(ModuleBundle::from(module)),
            max_gas_amount,
            gas_unit_price,
            expiration_timestamp_secs,
            chain_id,
        }
    }

    /// Create a new `RawTransaction` with a list of modules to publish.
    ///
    /// A module transaction is the only way to publish code. Multiple modules per transaction
    /// can be published.
    pub fn new_module_bundle(
        sender: AccountAddress,
        sequence_number: u64,
        modules: ModuleBundle,
        max_gas_amount: u64,
        gas_unit_price: u64,
        expiration_timestamp_secs: u64,
        chain_id: ChainId,
    ) -> Self {
        RawTransaction {
            sender,
            sequence_number,
            payload: TransactionPayload::ModuleBundle(modules),
            max_gas_amount,
            gas_unit_price,
            expiration_timestamp_secs,
            chain_id,
        }
    }

    /// Signs the given `RawTransaction`. Note that this consumes the `RawTransaction` and turns it
    /// into a `SignatureCheckedTransaction`.
    ///
    /// For a transaction that has just been signed, its signature is expected to be valid.
    pub fn sign(
        self,
        private_key: &Ed25519PrivateKey,
        public_key: Ed25519PublicKey,
    ) -> Result<SignatureCheckedTransaction> {
        let signature = private_key.sign(&self)?;
        Ok(SignatureCheckedTransaction(SignedTransaction::new(
            self, public_key, signature,
        )))
    }

    /// Signs the given multi-agent `RawTransaction`, which is a transaction with secondary
    /// signers in addition to a sender. The private keys of the sender and the
    /// secondary signers are used to sign the transaction.
    ///
    /// The order and length of the secondary keys provided here have to match the order and
    /// length of the `secondary_signers`.
    pub fn sign_multi_agent(
        self,
        sender_private_key: &Ed25519PrivateKey,
        secondary_signers: Vec<AccountAddress>,
        secondary_private_keys: Vec<&Ed25519PrivateKey>,
    ) -> Result<SignatureCheckedTransaction> {
        let message =
            RawTransactionWithData::new_multi_agent(self.clone(), secondary_signers.clone());
        let sender_signature = sender_private_key.sign(&message)?;
        let sender_authenticator = AccountAuthenticator::ed25519(
            Ed25519PublicKey::from(sender_private_key),
            sender_signature,
        );

        if secondary_private_keys.len() != secondary_signers.len() {
            return Err(format_err!(
                "number of secondary private keys and number of secondary signers don't match"
            ));
        }
        let mut secondary_authenticators = vec![];
        for priv_key in secondary_private_keys {
            let signature = priv_key.sign(&message)?;
            secondary_authenticators.push(AccountAuthenticator::ed25519(
                Ed25519PublicKey::from(priv_key),
                signature,
            ));
        }

        Ok(SignatureCheckedTransaction(
            SignedTransaction::new_multi_agent(
                self,
                sender_authenticator,
                secondary_signers,
                secondary_authenticators,
            ),
        ))
    }

    #[cfg(any(test, feature = "fuzzing"))]
    pub fn multi_sign_for_testing(
        self,
        private_key: &Ed25519PrivateKey,
        public_key: Ed25519PublicKey,
    ) -> Result<SignatureCheckedTransaction> {
        let signature = private_key.sign(&self)?;
        Ok(SignatureCheckedTransaction(
            SignedTransaction::new_multisig(self, public_key.into(), signature.into()),
        ))
    }

    pub fn into_payload(self) -> TransactionPayload {
        self.payload
    }

    pub fn format_for_client(&self, get_transaction_name: impl Fn(&[u8]) -> String) -> String {
        let (code, args) = match &self.payload {
            TransactionPayload::Script(script) => (
                get_transaction_name(script.code()),
                convert_txn_args(script.args()),
            ),
            TransactionPayload::EntryFunction(script_fn) => (
                format!("{}::{}", script_fn.module(), script_fn.function()),
                script_fn.args().to_vec(),
            ),
            TransactionPayload::ModuleBundle(_) => ("module publishing".to_string(), vec![]),
        };
        let mut f_args: String = "".to_string();
        for arg in args {
            f_args = format!("{}\n\t\t\t{:02X?},", f_args, arg);
        }
        format!(
            "RawTransaction {{ \n\
             \tsender: {}, \n\
             \tsequence_number: {}, \n\
             \tpayload: {{, \n\
             \t\ttransaction: {}, \n\
             \t\targs: [ {} \n\
             \t\t]\n\
             \t}}, \n\
             \tmax_gas_amount: {}, \n\
             \tgas_unit_price: {}, \n\
             \texpiration_timestamp_secs: {:#?}, \n\
             \tchain_id: {},
             }}",
            self.sender,
            self.sequence_number,
            code,
            f_args,
            self.max_gas_amount,
            self.gas_unit_price,
            self.expiration_timestamp_secs,
            self.chain_id,
        )
    }

    /// Return the sender of this transaction.
    pub fn sender(&self) -> AccountAddress {
        self.sender
    }

    /// Return the signing message for creating transaction signature.
    pub fn signing_message(&self) -> Result<Vec<u8>, CryptoMaterialError> {
        signing_message(self)
    }
}

#[derive(
    Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize, CryptoHasher, BCSCryptoHash,
)]
pub enum RawTransactionWithData {
    MultiAgent {
        raw_txn: RawTransaction,
        secondary_signer_addresses: Vec<AccountAddress>,
    },
}

impl RawTransactionWithData {
    pub fn new_multi_agent(
        raw_txn: RawTransaction,
        secondary_signer_addresses: Vec<AccountAddress>,
    ) -> Self {
        Self::MultiAgent {
            raw_txn,
            secondary_signer_addresses,
        }
    }
}

/// Different kinds of transactions.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TransactionPayload {
    /// A transaction that executes code.
    Script(Script),
    /// A transaction that publishes multiple modules at the same time.
    ModuleBundle(ModuleBundle),
    /// A transaction that executes an existing entry function published on-chain.
    EntryFunction(EntryFunction),
}

impl TransactionPayload {
    pub fn into_entry_function(self) -> EntryFunction {
        match self {
            Self::EntryFunction(f) => f,
            payload => panic!("Expected EntryFunction(_) payload, found: {:#?}", payload),
        }
    }
}

/// Two different kinds of WriteSet transactions.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum WriteSetPayload {
    /// Directly passing in the WriteSet.
    Direct(ChangeSet),
    /// Generate the WriteSet by running a script.
    Script {
        /// Execute the script as the designated signer.
        execute_as: AccountAddress,
        /// Script body that gets executed.
        script: Script,
    },
}

impl WriteSetPayload {
    pub fn should_trigger_reconfiguration_by_default(&self) -> bool {
        match self {
            Self::Direct(_) => true,
            Self::Script { .. } => false,
        }
    }
}

/// A transaction that has been signed.
///
/// A `SignedTransaction` is a single transaction that can be atomically executed. Clients submit
/// these to validator nodes, and the validator and executor submits these to the VM.
///
/// **IMPORTANT:** The signature of a `SignedTransaction` is not guaranteed to be verified. For a
/// transaction whose signature is statically guaranteed to be verified, see
/// [`SignatureCheckedTransaction`].
#[derive(Clone, Eq, Serialize, Deserialize)]
pub struct SignedTransaction {
    /// The raw transaction
    raw_txn: RawTransaction,

    /// Public key and signature to authenticate
    authenticator: TransactionAuthenticator,

    /// A cached size of the raw transaction bytes.
    /// Prevents serializing the same transaction multiple times to determine size.
    #[serde(skip)]
    size: OnceCell<usize>,
}

/// PartialEq ignores the "bytes" field as this is a OnceCell that may or
/// may not be initialized during runtime comparison.
impl PartialEq for SignedTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.raw_txn == other.raw_txn && self.authenticator == other.authenticator
    }
}

/// A transaction for which the signature has been verified. Created by
/// [`SignedTransaction::check_signature`] and [`RawTransaction::sign`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignatureCheckedTransaction(SignedTransaction);

impl SignatureCheckedTransaction {
    /// Returns the `SignedTransaction` within.
    pub fn into_inner(self) -> SignedTransaction {
        self.0
    }

    /// Returns the `RawTransaction` within.
    pub fn into_raw_transaction(self) -> RawTransaction {
        self.0.into_raw_transaction()
    }
}

impl Deref for SignatureCheckedTransaction {
    type Target = SignedTransaction;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for SignedTransaction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SignedTransaction {{ \n \
             {{ raw_txn: {:#?}, \n \
             authenticator: {:#?}, \n \
             }} \n \
             }}",
            self.raw_txn, self.authenticator
        )
    }
}

impl SignedTransaction {
    pub fn new(
        raw_txn: RawTransaction,
        public_key: Ed25519PublicKey,
        signature: Ed25519Signature,
    ) -> SignedTransaction {
        let authenticator = TransactionAuthenticator::ed25519(public_key, signature);
        SignedTransaction {
            raw_txn,
            authenticator,
            size: OnceCell::new(),
        }
    }

    pub fn new_multisig(
        raw_txn: RawTransaction,
        public_key: MultiEd25519PublicKey,
        signature: MultiEd25519Signature,
    ) -> SignedTransaction {
        let authenticator = TransactionAuthenticator::multi_ed25519(public_key, signature);
        SignedTransaction {
            raw_txn,
            authenticator,
            size: OnceCell::new(),
        }
    }

    pub fn new_multi_agent(
        raw_txn: RawTransaction,
        sender: AccountAuthenticator,
        secondary_signer_addresses: Vec<AccountAddress>,
        secondary_signers: Vec<AccountAuthenticator>,
    ) -> Self {
        SignedTransaction {
            raw_txn,
            authenticator: TransactionAuthenticator::multi_agent(
                sender,
                secondary_signer_addresses,
                secondary_signers,
            ),
            size: OnceCell::new(),
        }
    }

    pub fn new_with_authenticator(
        raw_txn: RawTransaction,
        authenticator: TransactionAuthenticator,
    ) -> Self {
        Self {
            raw_txn,
            authenticator,
            size: OnceCell::new(),
        }
    }

    pub fn authenticator(&self) -> TransactionAuthenticator {
        self.authenticator.clone()
    }

    pub fn sender(&self) -> AccountAddress {
        self.raw_txn.sender
    }

    pub fn into_raw_transaction(self) -> RawTransaction {
        self.raw_txn
    }

    pub fn sequence_number(&self) -> u64 {
        self.raw_txn.sequence_number
    }

    pub fn chain_id(&self) -> ChainId {
        self.raw_txn.chain_id
    }

    pub fn payload(&self) -> &TransactionPayload {
        &self.raw_txn.payload
    }

    pub fn max_gas_amount(&self) -> u64 {
        self.raw_txn.max_gas_amount
    }

    pub fn gas_unit_price(&self) -> u64 {
        self.raw_txn.gas_unit_price
    }

    pub fn expiration_timestamp_secs(&self) -> u64 {
        self.raw_txn.expiration_timestamp_secs
    }

    pub fn raw_txn_bytes_len(&self) -> usize {
        *self.size.get_or_init(|| {
            bcs::to_bytes(&self.raw_txn)
                .expect("Unable to serialize RawTransaction")
                .len()
        })
    }

    /// Checks that the signature of given transaction. Returns `Ok(SignatureCheckedTransaction)` if
    /// the signature is valid.
    pub fn check_signature(self) -> Result<SignatureCheckedTransaction> {
        self.authenticator.verify(&self.raw_txn)?;
        Ok(SignatureCheckedTransaction(self))
    }

    /// Checks that the signature of given transaction inplace. Returns `Ok(())` if
    /// the signature is valid.
    pub fn signature_is_valid(&self) -> bool {
        self.authenticator.verify(&self.raw_txn).is_ok()
    }

    pub fn contains_duplicate_signers(&self) -> bool {
        let mut all_signer_addresses = self.authenticator.secondary_signer_addreses();
        all_signer_addresses.push(self.sender());
        let mut s = BTreeSet::new();
        all_signer_addresses.iter().any(|a| !s.insert(*a))
    }

    pub fn format_for_client(&self, get_transaction_name: impl Fn(&[u8]) -> String) -> String {
        format!(
            "SignedTransaction {{ \n \
             raw_txn: {}, \n \
             authenticator: {:#?}, \n \
             }}",
            self.raw_txn.format_for_client(get_transaction_name),
            self.authenticator
        )
    }

    pub fn is_multi_agent(&self) -> bool {
        matches!(
            self.authenticator,
            TransactionAuthenticator::MultiAgent { .. }
        )
    }

    /// Returns the hash when the transaction is commited onchain.
    pub fn committed_hash(self) -> HashValue {
        Transaction::UserTransaction(self).hash()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
pub struct TransactionWithProof {
    pub version: Version,
    pub transaction: Transaction,
    pub events: Option<Vec<ContractEvent>>,
    pub proof: TransactionInfoWithProof,
}

impl TransactionWithProof {
    pub fn new(
        version: Version,
        transaction: Transaction,
        events: Option<Vec<ContractEvent>>,
        proof: TransactionInfoWithProof,
    ) -> Self {
        Self {
            version,
            transaction,
            events,
            proof,
        }
    }

    /// Verifies the transaction with the proof, both carried by `self`.
    ///
    /// A few things are ensured if no error is raised:
    ///   1. This transaction exists in the ledger represented by `ledger_info`.
    ///   2. This transaction is a `UserTransaction`.
    ///   3. And this user transaction has the same `version`, `sender`, and `sequence_number` as
    ///      indicated by the parameter list. If any of these parameter is unknown to the call site
    ///      that is supposed to be informed via this struct, get it from the struct itself, such
    ///      as version and sender.
    pub fn verify_user_txn(
        &self,
        ledger_info: &LedgerInfo,
        version: Version,
        sender: AccountAddress,
        sequence_number: u64,
    ) -> Result<()> {
        let signed_transaction = self.transaction.as_signed_user_txn()?;

        ensure!(
            self.version == version,
            "Version ({}) is not expected ({}).",
            self.version,
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

        let txn_hash = self.transaction.hash();
        ensure!(
            txn_hash == self.proof.transaction_info().transaction_hash(),
            "Transaction hash ({}) not expected ({}).",
            txn_hash,
            self.proof.transaction_info().transaction_hash(),
        );

        if let Some(events) = &self.events {
            let event_hashes: Vec<_> = events.iter().map(CryptoHash::hash).collect();
            let event_root_hash =
                InMemoryAccumulator::<EventAccumulatorHasher>::from_leaves(&event_hashes[..])
                    .root_hash();
            ensure!(
                event_root_hash == self.proof.transaction_info().event_root_hash(),
                "Event root hash ({}) not expected ({}).",
                event_root_hash,
                self.proof.transaction_info().event_root_hash(),
            );
        }

        self.proof.verify(ledger_info, version)
    }
}

/// The status of VM execution, which contains more detailed failure info
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
#[cfg_attr(any(test, feature = "fuzzing"), proptest(no_params))]
pub enum ExecutionStatus {
    Success,
    OutOfGas,
    MoveAbort {
        location: AbortLocation,
        code: u64,
        info: Option<AbortInfo>,
    },
    ExecutionFailure {
        location: AbortLocation,
        function: u16,
        code_offset: u16,
    },
    MiscellaneousError(Option<StatusCode>),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
#[cfg_attr(any(test, feature = "fuzzing"), proptest(no_params))]
pub struct AbortInfo {
    pub reason_name: String,
    pub description: String,
}

impl From<KeptVMStatus> for ExecutionStatus {
    fn from(kept_status: KeptVMStatus) -> Self {
        match kept_status {
            KeptVMStatus::Executed => ExecutionStatus::Success,
            KeptVMStatus::OutOfGas => ExecutionStatus::OutOfGas,
            KeptVMStatus::MoveAbort(location, code) => ExecutionStatus::MoveAbort {
                location,
                code,
                info: None,
            },
            KeptVMStatus::ExecutionFailure {
                location: loc,
                function: func,
                code_offset: offset,
            } => ExecutionStatus::ExecutionFailure {
                location: loc,
                function: func,
                code_offset: offset,
            },
            KeptVMStatus::MiscellaneousError => ExecutionStatus::MiscellaneousError(None),
        }
    }
}

impl ExecutionStatus {
    pub fn is_success(&self) -> bool {
        matches!(self, ExecutionStatus::Success)
    }
}

/// The status of executing a transaction. The VM decides whether or not we should `Keep` the
/// transaction output or `Discard` it based upon the execution of the transaction. We wrap these
/// decisions around a `VMStatus` that provides more detail on the final execution state of the VM.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TransactionStatus {
    /// Discard the transaction output
    Discard(DiscardedVMStatus),

    /// Keep the transaction output
    Keep(ExecutionStatus),

    /// Retry the transaction, e.g., after a reconfiguration
    Retry,
}

impl TransactionStatus {
    pub fn status(&self) -> Result<ExecutionStatus, StatusCode> {
        match self {
            TransactionStatus::Keep(status) => Ok(status.clone()),
            TransactionStatus::Discard(code) => Err(*code),
            TransactionStatus::Retry => Err(StatusCode::UNKNOWN_VALIDATION_STATUS),
        }
    }

    pub fn is_discarded(&self) -> bool {
        match self {
            TransactionStatus::Discard(_) => true,
            TransactionStatus::Keep(_) => false,
            TransactionStatus::Retry => true,
        }
    }

    pub fn as_kept_status(&self) -> Result<ExecutionStatus> {
        match self {
            TransactionStatus::Keep(s) => Ok(s.clone()),
            _ => Err(format_err!("Not Keep.")),
        }
    }
}

impl From<VMStatus> for TransactionStatus {
    fn from(vm_status: VMStatus) -> Self {
        let status_code = vm_status.status_code();
        match vm_status.keep_or_discard() {
            Ok(recorded) => match recorded {
                KeptVMStatus::MiscellaneousError => {
                    TransactionStatus::Keep(ExecutionStatus::MiscellaneousError(Some(status_code)))
                },
                _ => TransactionStatus::Keep(recorded.into()),
            },
            Err(code) => TransactionStatus::Discard(code),
        }
    }
}

impl From<ExecutionStatus> for TransactionStatus {
    fn from(txn_execution_status: ExecutionStatus) -> Self {
        TransactionStatus::Keep(txn_execution_status)
    }
}

/// The result of running the transaction through the VM validator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VMValidatorResult {
    /// Result of the validation: `None` if the transaction was successfully validated
    /// or `Some(DiscardedVMStatus)` if the transaction should be discarded.
    status: Option<DiscardedVMStatus>,

    /// Score for ranking the transaction priority (e.g., based on the gas price).
    /// Only used when the status is `None`. Higher values indicate a higher priority.
    score: u64,
}

impl VMValidatorResult {
    pub fn new(vm_status: Option<DiscardedVMStatus>, score: u64) -> Self {
        debug_assert!(
            match vm_status {
                None => true,
                Some(status) =>
                    status.status_type() == StatusType::Unknown
                        || status.status_type() == StatusType::Validation
                        || status.status_type() == StatusType::InvariantViolation,
            },
            "Unexpected discarded status: {:?}",
            vm_status
        );
        Self {
            status: vm_status,
            score,
        }
    }

    pub fn error(vm_status: DiscardedVMStatus) -> Self {
        Self {
            status: Some(vm_status),
            score: 0,
        }
    }

    pub fn status(&self) -> Option<DiscardedVMStatus> {
        self.status
    }

    pub fn score(&self) -> u64 {
        self.score
    }
}

/// The output of executing a transaction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransactionOutput {
    /// The list of writes this transaction intends to do.
    write_set: WriteSet,

    /// The list of events emitted during this transaction.
    events: Vec<ContractEvent>,

    /// The amount of gas used during execution.
    gas_used: u64,

    /// The execution status.
    status: TransactionStatus,
}

impl TransactionOutput {
    pub fn new(
        write_set: WriteSet,
        events: Vec<ContractEvent>,
        gas_used: u64,
        status: TransactionStatus,
    ) -> Self {
        TransactionOutput {
            write_set,
            events,
            gas_used,
            status,
        }
    }

    pub fn into(self) -> (WriteSet, Vec<ContractEvent>) {
        (self.write_set, self.events)
    }

    pub fn write_set(&self) -> &WriteSet {
        &self.write_set
    }

    pub fn events(&self) -> &[ContractEvent] {
        &self.events
    }

    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }

    pub fn status(&self) -> &TransactionStatus {
        &self.status
    }

    pub fn unpack(self) -> (WriteSet, Vec<ContractEvent>, u64, TransactionStatus) {
        let Self {
            write_set,
            events,
            gas_used,
            status,
        } = self;
        (write_set, events, gas_used, status)
    }
}

/// `TransactionInfo` is the object we store in the transaction accumulator. It consists of the
/// transaction as well as the execution result of this transaction.
#[derive(Clone, CryptoHasher, BCSCryptoHash, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
pub enum TransactionInfo {
    V0(TransactionInfoV0),
}

impl TransactionInfo {
    pub fn new(
        transaction_hash: HashValue,
        state_change_hash: HashValue,
        event_root_hash: HashValue,
        state_checkpoint_hash: Option<HashValue>,
        gas_used: u64,
        status: ExecutionStatus,
    ) -> Self {
        Self::V0(TransactionInfoV0::new(
            transaction_hash,
            state_change_hash,
            event_root_hash,
            state_checkpoint_hash,
            gas_used,
            status,
        ))
    }

    #[cfg(any(test, feature = "fuzzing"))]
    pub fn new_placeholder(
        gas_used: u64,
        state_checkpoint_hash: Option<HashValue>,
        status: ExecutionStatus,
    ) -> Self {
        Self::new(
            HashValue::default(),
            HashValue::default(),
            HashValue::default(),
            state_checkpoint_hash,
            gas_used,
            status,
        )
    }
}

impl Deref for TransactionInfo {
    type Target = TransactionInfoV0;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::V0(txn_info) => txn_info,
        }
    }
}

#[derive(Clone, CryptoHasher, BCSCryptoHash, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
pub struct TransactionInfoV0 {
    /// The amount of gas used.
    gas_used: u64,

    /// The vm status. If it is not `Executed`, this will provide the general error class. Execution
    /// failures and Move abort's receive more detailed information. But other errors are generally
    /// categorized with no status code or other information
    status: ExecutionStatus,

    /// The hash of this transaction.
    transaction_hash: HashValue,

    /// The root hash of Merkle Accumulator storing all events emitted during this transaction.
    event_root_hash: HashValue,

    /// The hash value summarizing all changes caused to the world state by this transaction.
    /// i.e. hash of the output write set.
    state_change_hash: HashValue,

    /// The root hash of the Sparse Merkle Tree describing the world state at the end of this
    /// transaction. Depending on the protocol configuration, this can be generated periodical
    /// only, like per block.
    state_checkpoint_hash: Option<HashValue>,

    /// Potentially summarizes all evicted items from state. Always `None` for now.
    state_cemetery_hash: Option<HashValue>,
}

impl TransactionInfoV0 {
    pub fn new(
        transaction_hash: HashValue,
        state_change_hash: HashValue,
        event_root_hash: HashValue,
        state_checkpoint_hash: Option<HashValue>,
        gas_used: u64,
        status: ExecutionStatus,
    ) -> Self {
        Self {
            gas_used,
            status,
            transaction_hash,
            event_root_hash,
            state_change_hash,
            state_checkpoint_hash,
            state_cemetery_hash: None,
        }
    }

    pub fn transaction_hash(&self) -> HashValue {
        self.transaction_hash
    }

    pub fn state_change_hash(&self) -> HashValue {
        self.state_change_hash
    }

    pub fn is_state_checkpoint(&self) -> bool {
        self.state_checkpoint_hash().is_some()
    }

    pub fn state_checkpoint_hash(&self) -> Option<HashValue> {
        self.state_checkpoint_hash
    }

    pub fn ensure_state_checkpoint_hash(&self) -> Result<HashValue> {
        self.state_checkpoint_hash
            .ok_or_else(|| format_err!("State checkpoint hash not present in TransactionInfo"))
    }

    pub fn event_root_hash(&self) -> HashValue {
        self.event_root_hash
    }

    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }

    pub fn status(&self) -> &ExecutionStatus {
        &self.status
    }
}

impl Display for TransactionInfo {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "TransactionInfo: [txn_hash: {}, state_change_hash: {}, event_root_hash: {}, state_checkpoint_hash: {:?}, gas_used: {}, recorded_status: {:?}]",
            self.transaction_hash(), self.state_change_hash(), self.event_root_hash(), self.state_checkpoint_hash(), self.gas_used(), self.status(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TransactionToCommit {
    transaction: Transaction,
    transaction_info: TransactionInfo,
    state_updates: HashMap<StateKey, Option<StateValue>>,
    write_set: WriteSet,
    events: Vec<ContractEvent>,
    is_reconfig: bool,
}

impl TransactionToCommit {
    pub fn new(
        transaction: Transaction,
        transaction_info: TransactionInfo,
        state_updates: HashMap<StateKey, Option<StateValue>>,
        write_set: WriteSet,
        events: Vec<ContractEvent>,
        is_reconfig: bool,
    ) -> Self {
        TransactionToCommit {
            transaction,
            transaction_info,
            state_updates,
            write_set,
            events,
            is_reconfig,
        }
    }

    pub fn transaction(&self) -> &Transaction {
        &self.transaction
    }

    pub fn transaction_info(&self) -> &TransactionInfo {
        &self.transaction_info
    }

    pub fn is_state_checkpoint(&self) -> bool {
        self.transaction_info().is_state_checkpoint()
    }

    #[cfg(any(test, feature = "fuzzing"))]
    pub fn set_transaction_info(&mut self, txn_info: TransactionInfo) {
        self.transaction_info = txn_info
    }

    pub fn state_updates(&self) -> &HashMap<StateKey, Option<StateValue>> {
        &self.state_updates
    }

    pub fn write_set(&self) -> &WriteSet {
        &self.write_set
    }

    pub fn events(&self) -> &[ContractEvent] {
        &self.events
    }

    pub fn gas_used(&self) -> u64 {
        self.transaction_info.gas_used
    }

    pub fn status(&self) -> &ExecutionStatus {
        &self.transaction_info.status
    }

    pub fn is_reconfig(&self) -> bool {
        self.is_reconfig
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TransactionListWithProof {
    pub transactions: Vec<Transaction>,
    pub events: Option<Vec<Vec<ContractEvent>>>,
    pub first_transaction_version: Option<Version>,
    pub proof: TransactionInfoListWithProof,
}

impl TransactionListWithProof {
    /// Constructor.
    pub fn new(
        transactions: Vec<Transaction>,
        events: Option<Vec<Vec<ContractEvent>>>,
        first_transaction_version: Option<Version>,
        proof: TransactionInfoListWithProof,
    ) -> Self {
        Self {
            transactions,
            events,
            first_transaction_version,
            proof,
        }
    }

    /// A convenience function to create an empty proof. Mostly used for tests.
    pub fn new_empty() -> Self {
        Self::new(
            vec![],
            None,
            None,
            TransactionInfoListWithProof::new_empty(),
        )
    }

    /// Verifies the transaction list with proof using the given `ledger_info`.
    /// This method will ensure:
    /// 1. All transactions exist on the given `ledger_info`.
    /// 2. All transactions in the list have consecutive versions.
    /// 3. If `first_transaction_version` is None, the transaction list is empty.
    ///    Otherwise, the transaction list starts at `first_transaction_version`.
    /// 4. If events exist, they match the expected event root hashes in the proof.
    pub fn verify(
        &self,
        ledger_info: &LedgerInfo,
        first_transaction_version: Option<Version>,
    ) -> Result<()> {
        // Verify the first transaction versions match
        ensure!(
            self.first_transaction_version == first_transaction_version,
            "First transaction version ({:?}) doesn't match given version ({:?}).",
            self.first_transaction_version,
            first_transaction_version,
        );

        // Verify the lengths of the transactions and transaction infos match
        ensure!(
            self.proof.transaction_infos.len() == self.transactions.len(),
            "The number of TransactionInfo objects ({}) does not match the number of \
             transactions ({}).",
            self.proof.transaction_infos.len(),
            self.transactions.len(),
        );

        // Verify the transaction hashes match those of the transaction infos
        let transaction_hashes: Vec<_> = self.transactions.iter().map(CryptoHash::hash).collect();
        itertools::zip_eq(transaction_hashes, &self.proof.transaction_infos)
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

        // Verify the transaction infos are proven by the ledger info.
        self.proof
            .verify(ledger_info, self.first_transaction_version)?;

        // Verify the events if they exist.
        if let Some(event_lists) = &self.events {
            ensure!(
                event_lists.len() == self.transactions.len(),
                "The length of event_lists ({}) does not match the number of transactions ({}).",
                event_lists.len(),
                self.transactions.len(),
            );
            itertools::zip_eq(event_lists, &self.proof.transaction_infos)
                .map(|(events, txn_info)| verify_events_against_root_hash(events, txn_info))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(())
    }
}

/// This differs from TransactionListWithProof in that TransactionOutputs are
/// stored (no transactions). Events are stored inside each TransactionOutput.
///
/// Note: the proof cannot verify the TransactionOutputs themselves. This
/// requires speculative execution of each TransactionOutput to verify that the
/// resulting state matches the expected state in the proof (for each version).
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct TransactionOutputListWithProof {
    pub transactions_and_outputs: Vec<(Transaction, TransactionOutput)>,
    pub first_transaction_output_version: Option<Version>,
    pub proof: TransactionInfoListWithProof,
}

impl TransactionOutputListWithProof {
    pub fn new(
        transactions_and_outputs: Vec<(Transaction, TransactionOutput)>,
        first_transaction_output_version: Option<Version>,
        proof: TransactionInfoListWithProof,
    ) -> Self {
        Self {
            transactions_and_outputs,
            first_transaction_output_version,
            proof,
        }
    }

    /// A convenience function to create an empty proof. Mostly used for tests.
    pub fn new_empty() -> Self {
        Self::new(vec![], None, TransactionInfoListWithProof::new_empty())
    }

    /// Verifies the transaction output list with proof using the given `ledger_info`.
    /// This method will ensure:
    /// 1. All transaction infos exist on the given `ledger_info`.
    /// 2. If `first_transaction_output_version` is None, the transaction output list is empty.
    ///    Otherwise, the list starts at `first_transaction_output_version`.
    /// 3. Events, gas, status in each transaction output match the expected event root hashes,
    ///    the gas used and the transaction execution status in the proof, respectively.
    /// 4. The transaction hashes match those of the transaction infos.
    ///
    /// Note: the proof cannot verify the TransactionOutputs themselves. This
    /// requires speculative execution of each TransactionOutput to verify that the
    /// resulting state matches the expected state in the proof (for each version).
    pub fn verify(
        &self,
        ledger_info: &LedgerInfo,
        first_transaction_output_version: Option<Version>,
    ) -> Result<()> {
        // Verify the first transaction/output versions match
        ensure!(
            self.first_transaction_output_version == first_transaction_output_version,
            "First transaction and output version ({:?}) doesn't match given version ({:?}).",
            self.first_transaction_output_version,
            first_transaction_output_version,
        );

        // Verify the lengths of the transaction(output)s and transaction infos match
        ensure!(
            self.proof.transaction_infos.len() == self.transactions_and_outputs.len(),
            "The number of TransactionInfo objects ({}) does not match the number of \
             transactions and outputs ({}).",
            self.proof.transaction_infos.len(),
            self.transactions_and_outputs.len(),
        );

        // Verify the events, status, gas used and transaction hashes.
        itertools::zip_eq(
            &self.transactions_and_outputs,
            &self.proof.transaction_infos,
        )
        .map(|((txn, txn_output), txn_info)| {
            // Check the events against the expected events root hash
            verify_events_against_root_hash(&txn_output.events, txn_info)?;

            // Verify the write set matches for both the transaction info and output
            let write_set_hash = CryptoHash::hash(&txn_output.write_set);
            ensure!(
                txn_info.state_change_hash == write_set_hash,
                "The write set in transaction output does not match the transaction info \
                     in proof. Hash of write set in transaction output: {}. Write set hash in txn_info: {}.",
                write_set_hash,
                txn_info.state_change_hash,
            );

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

        // Verify the transaction infos are proven by the ledger info.
        self.proof
            .verify(ledger_info, self.first_transaction_output_version)?;

        Ok(())
    }
}

/// Verifies a list of events against an expected event root hash. This is done
/// by calculating the hash of the events using an event accumulator hasher.
fn verify_events_against_root_hash(
    events: &[ContractEvent],
    transaction_info: &TransactionInfo,
) -> Result<()> {
    let event_hashes: Vec<_> = events.iter().map(CryptoHash::hash).collect();
    let event_root_hash =
        InMemoryAccumulator::<EventAccumulatorHasher>::from_leaves(&event_hashes).root_hash();
    ensure!(
        event_root_hash == transaction_info.event_root_hash(),
        "The event root hash calculated doesn't match that carried on the \
                         transaction info! Calculated hash {:?}, transaction info hash {:?}",
        event_root_hash,
        transaction_info.event_root_hash()
    );
    Ok(())
}

/// A list of transactions under an account that are contiguous by sequence number
/// and include proofs.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
pub struct AccountTransactionsWithProof(pub Vec<TransactionWithProof>);

impl AccountTransactionsWithProof {
    pub fn new(txns_with_proofs: Vec<TransactionWithProof>) -> Self {
        Self(txns_with_proofs)
    }

    pub fn new_empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn inner(&self) -> &[TransactionWithProof] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<TransactionWithProof> {
        self.0
    }

    /// 1. Verify all transactions are consistent with the given ledger info.
    /// 2. All transactions were sent by `account`.
    /// 3. The transactions are contiguous by sequence number, starting at `start_seq_num`.
    /// 4. No more transactions than limit.
    /// 5. Events are present when requested (and not present when not requested).
    /// 6. Transactions are not newer than requested ledger version.
    pub fn verify(
        &self,
        ledger_info: &LedgerInfo,
        account: AccountAddress,
        start_seq_num: u64,
        limit: u64,
        include_events: bool,
        ledger_version: Version,
    ) -> Result<()> {
        ensure!(
            self.len() as u64 <= limit,
            "number of account transactions ({}) exceeded limit ({})",
            self.len(),
            limit,
        );

        self.0
            .iter()
            .enumerate()
            .try_for_each(|(seq_num_offset, txn_with_proof)| {
                let expected_seq_num = start_seq_num.saturating_add(seq_num_offset as u64);
                let txn_version = txn_with_proof.version;

                ensure!(
                    include_events == txn_with_proof.events.is_some(),
                    "unexpected events or missing events"
                );
                ensure!(
                    txn_version <= ledger_version,
                    "transaction with version ({}) greater than requested ledger version ({})",
                    txn_version,
                    ledger_version,
                );

                txn_with_proof.verify_user_txn(ledger_info, txn_version, account, expected_seq_num)
            })
    }
}

/// `Transaction` will be the transaction type used internally in the aptos node to represent the
/// transaction to be processed and persisted.
///
/// We suppress the clippy warning here as we would expect most of the transaction to be user
/// transaction.
#[allow(clippy::large_enum_variant)]
#[cfg_attr(any(test, feature = "fuzzing"), derive(Arbitrary))]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, CryptoHasher, BCSCryptoHash)]
pub enum Transaction {
    /// Transaction submitted by the user. e.g: P2P payment transaction, publishing module
    /// transaction, etc.
    /// TODO: We need to rename SignedTransaction to SignedUserTransaction, as well as all the other
    ///       transaction types we had in our codebase.
    UserTransaction(SignedTransaction),

    /// Transaction that applies a WriteSet to the current storage, it's applied manually via aptos-db-bootstrapper.
    GenesisTransaction(WriteSetPayload),

    /// Transaction to update the block metadata resource at the beginning of a block.
    BlockMetadata(BlockMetadata),

    /// Transaction to let the executor update the global state tree and record the root hash
    /// in the TransactionInfo
    /// The hash value inside is unique block id which can generate unique hash of state checkpoint transaction
    StateCheckpoint(HashValue),
}

impl Transaction {
    pub fn as_signed_user_txn(&self) -> Result<&SignedTransaction> {
        match self {
            Transaction::UserTransaction(txn) => Ok(txn),
            _ => Err(format_err!("Not a user transaction.")),
        }
    }

    pub fn format_for_client(&self, get_transaction_name: impl Fn(&[u8]) -> String) -> String {
        match self {
            Transaction::UserTransaction(user_txn) => {
                user_txn.format_for_client(get_transaction_name)
            },
            // TODO: display proper information for client
            Transaction::GenesisTransaction(_write_set) => String::from("genesis"),
            // TODO: display proper information for client
            Transaction::BlockMetadata(_block_metadata) => String::from("block_metadata"),
            // TODO: display proper information for client
            Transaction::StateCheckpoint(_) => String::from("state_checkpoint"),
        }
    }
}

impl TryFrom<Transaction> for SignedTransaction {
    type Error = Error;

    fn try_from(txn: Transaction) -> Result<Self> {
        match txn {
            Transaction::UserTransaction(txn) => Ok(txn),
            _ => Err(format_err!("Not a user transaction.")),
        }
    }
}
