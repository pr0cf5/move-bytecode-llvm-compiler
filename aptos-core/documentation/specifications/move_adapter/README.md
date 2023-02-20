# Move Adapter: Transaction Validation and Execution

## Introduction

The **Move Adapter** is responsible for the validation and execution of
transactions. **Validation** and **Execution** are different steps, exposed
through different entry points and using different instances of the adapter.

_[Validation](#Validation)_ performs various checks on a transaction to
determine if it is well formed and to evaluate its priority. When entering
the system, each user transaction is validated relative to a starting state
and either discarded if it is malformed or added to a ranked pool of pending
transactions otherwise. A **discarded** transaction is removed from the
system without being recorded on the blockchain. A **validated** transaction
can proceed but may still be discarded during the execution step.

_[Execution](#Execution)_ receives a block of transactions and a starting
state and evaluates those transactions in order. Each transaction is first
re-validated. The system state may have changed since the validation step, so
that a transaction has become invalid and needs to be discarded. Otherwise,
the execution process computes a set of side effects for each transaction.
Those side effects are used in two ways: they are included in the final
per-transaction results of the execution step, and they are applied locally
within the execution process so that they are visible to subsequent
transactions within the same block.

The following diagram shows how the Move Adapter fits into the system.

![abstract](../images/move_adapter.png)

The **Move Adapter** uses a **[Move Virtual Machine](#Move-VM)** (VM) to
execute code in the context of the Aptos ecosystem. The Move VM is not aware
of the structure and semantics of Aptos transactions, nor is it aware of the
Aptos storage layer; it only knows how to execute Move functions. It is the
adapter's job to use the VM in a way that honors the Aptos protocol.

We will describe the Move Adapter and the Move VM architecture in Aptos and how
they share responsibilities.

## Validation

Validation operates on a transaction (`SignedTransaction`) with a given state
(`StateView`). The state is read-only: validation does not execute the
transaction and so it does not compute side effects. If a transaction is
malformed the adapter returns a result (`VMValidatorResult`) indicating that
the transaction must be discarded. If the transaction is successfully
validated, the result also includes information to rank the transaction
priority.

```rust
/// The entry point to validate a transaction
pub trait VMValidator {
    fn validate_transaction(
        &self,
        transaction: SignedTransaction,
        state_view: &impl StateView,
    ) -> VMValidatorResult;
}

/// `StateView` is a read-only snapshot of the global state.
pub trait StateView {
    fn id(&self) -> StateViewId;
    fn get(&self, access_path: &AccessPath) -> Result<Option<Vec<u8>>>;
    fn is_genesis(&self) -> bool;
}

/// The result of running the transaction through the VM validator.
pub struct VMValidatorResult {
    /// Result of the validation: `None` if the transaction was successfully validated
    /// or `Some(DiscardedVMStatus)` if the transaction should be discarded.
    status: Option<DiscardedVMStatus>,

    /// Score for ranking the transaction priority (e.g., based on the gas price).
    /// Only used when the status is `None`. Higher values indicate a higher priority.
    score: u64,
}
```

Validation examines the content of a transaction to determine if it is well
formed. The transaction content is organized into three layers:
`SignedTransaction`, `RawTransaction`, and the transaction payload
(`TransactionPayload` and `WriteSetPayload`):

```rust
/// A transaction that has been signed.
pub struct SignedTransaction {
    /// The raw transaction
    raw_txn: RawTransaction,

    /// Public key and signature to authenticate
    authenticator: TransactionAuthenticator,
}

/// RawTransaction is the portion of a transaction that a client signs.
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
    /// is greater than or equal to this time, then the transaction is
    /// expired and will be discarded. This can be set to a large value far
    /// in the future to indicate that a transaction does not expire.
    expiration_timestamp_secs: u64,

    /// Chain ID of the Aptos network this transaction is intended for.
    chain_id: ChainId,
}

/// Different kinds of transactions.
pub enum TransactionPayload {
    /// A system maintenance transaction.
    WriteSet(WriteSetPayload),
    /// A transaction that executes code.
    Script(Script),
    /// A transaction that publishes code.
    Module(Module),
    /// A transaction that executes an existing entry function published on-chain.
    EntryFunction(EntryFunction),
}

/// Two different kinds of WriteSet transactions.
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

/// The set of changes and events produced by a transaction.
pub struct ChangeSet {
    write_set: WriteSet,
    events: Vec<ContractEvent>,
}

/// A transaction script runs code with a specified set of arguments.
pub struct Script {
    code: Vec<u8>,
    ty_args: Vec<TypeTag>,
    args: Vec<TransactionArgument>,
}

/// Call a Move entry function.
pub struct EntryFunction {
    module: ModuleId,
    function: Identifier,
    ty_args: Vec<TypeTag>,
    args: Vec<Vec<u8>>,
}

/// A module publishing transaction contains the code to be published.
pub struct Module {
    code: Vec<u8>,
}
```

There are several different kinds of transactions that can be stored in the
transaction payload: executing a script or entry function,
publishing a module, and applying a
WriteSet for system maintenance or updates. The payload is stored inside a
`RawTransaction` structure that includes the various fields that are common to
all of these transactions, and the `RawTransaction` is signed and wrapped
inside a `SignedTransaction` structure that includes the signature and public
key.

The adapter performs a sequence of checks to validate a transaction. Some of
these checks are implemented directly in the adapter and others are specified
in Move code and evaluated via the [Move VM](#Move-VM). Some of the checks
apply to all transactions and others are specific to the type of payload.
To ensure consistent error handling, the checks should be performed in the
order specified here.

### Signature Verification

The adapter performs signature verification for any transaction, regardless of
the payload. The adapter checks if the sender and secondary signers' signatures in
the `SignedTransaction` are consistent with their public keys and the content of the
transaction. If not, this check fails with an `INVALID_SIGNATURE` status code.

```rust
pub enum TransactionAuthenticator {
    /// Single signature
    Ed25519 {
        public_key: Ed25519PublicKey,
        signature: Ed25519Signature,
    },
    /// K-of-N multisignature
    MultiEd25519 {
        public_key: MultiEd25519PublicKey,
        signature: MultiEd25519Signature,
    },
    /// Multi-agent transaction.
    MultiAgent {
        sender: AccountAuthenticator,
        secondary_signer_addresses: Vec<AccountAddress>,
        secondary_signers: Vec<AccountAuthenticator>,
    },
}
```

The `TransactionAuthenticator` stored in the `SignedTransaction` is responsible for
performing this check:
* If the transaction is single-agent, then only the sender's signature is included
and checked against the `RawTransaction` content.
* If the transaction is multi-agent, then both the primary signer (sender) and the
secondary signers' signatures are included and checked against a struct containing both
the `RawTransaction` and a vector of secondary signers' addresses. The addresses have
to be in the same order as
the signatures.

Note that comparing the transaction's public keys against the sender and secondary
signer accounts' authorization keys is done separately in [Move code](#Prologue-Checks).


### General Checks

Besides signature verification, the adapter performs the following steps for any transaction, regardless of
the payload:

* If secondary signers exist, check that all signers including the sender and secondary
signers have distinct account addresses. If not, validation fails with a
`SIGNERS_CONTAIN_DUPLICATES` status code.

* Load the `RoleId` resource from the sender's account. If the validation is
successful, this value is returned as the `governance_role` field of the
`VMValidatorResult` so that the client can choose to prioritize governance
transactions.

* If the transaction payload is a `EntryFunction`, check if the on-chain
Aptos Version number is 2 or later. For version 1, validation will fail with
a `FEATURE_UNDER_GATING` status code.

### Gas and Size Checks

Next, there are a series of checks related to the transaction size and gas
parameters. These checks are performed for `Script`, `EntryFunction`,
and `Module` payloads, but
not for `WriteSet` transactions. The constraints for these checks are defined
by the `GasConstants` structure in the `VMConfig` module.

* Check if the transaction size exceeds the limit specified by the
`max_transaction_size_in_bytes` field of `GasConstants`. If the transaction is
too big, validation fails with an `EXCEEDED_MAX_TRANSACTION_SIZE` status code.

* If the `max_gas_amount` field in the `RawTransaction` is larger than the
`maximum_number_of_gas_units` field of `GasConstants`, then validation fails
with a status code of `MAX_GAS_UNITS_EXCEEDS_MAX_GAS_UNITS_BOUND`.

* There is also a minimum gas amount based on the transaction size. The minimum
charge is calculated in terms of internal gas units that are scaled up by the
`gas_unit_scaling_factor` field of `GasConstants` to allow more fine grained
accounting. First, the `GasConstants` structure specifies a
`min_transaction_gas_units` value that is charged for all transactions
regardless of their size. Next, if the transaction size in bytes is larger
than the `large_transaction_cutoff` value, then the minimum gas amount is
increased by `intrinsic_gas_per_byte` for every byte in excess of
`large_transaction_cutoff`. The resulting value is divided by the
`gas_unit_scaling_factor` to obtain the minimum gas amount. If the
`max_gas_amount` for the transaction is less than this minimum requirement,
validation fails with a status code of
`MAX_GAS_UNITS_BELOW_MIN_TRANSACTION_GAS_UNITS`.

* The `gas_unit_price` from the `RawTransaction` must be within the range
specified by the `GasConstants`. If the price is less than
`min_price_per_gas_unit`, validation fails with a status code of
`GAS_UNIT_PRICE_BELOW_MIN_BOUND`. If the price is more than
`max_price_per_gas_unit`, validation fails with a status code of
`GAS_UNIT_PRICE_ABOVE_MAX_BOUND`.

### Prologue Checks

The rest of the validation is performed in Move code, which is run using the
Move VM with gas metering disabled. Each kind of transaction payload has a
corresponding prologue function that is used for validation. These prologue
functions are defined in the `AptosAccount` module of the Aptos Framework:

* Single-agent `EntryFunction` and `Script`: The prologue function is `script_prologue`.
In addition to the common checks listed below, it also calls the `is_script_allowed`
function in the `TransactionPublishingOption` module to determine if the script
should be allowed. A script sent by an account with `has_aptos_root_role` is always
allowed. Otherwise, a `Script` payload is allowed if the hash of the
script bytecode is on the list of allowed scripts published at
`0x1::TransactionPublishingOption::TransactionPublishingOption.script_allowlist`.
`EntryFunction` payloads, for which the adapter uses an empty vector in place
of the script hash, are always allowed. If the script is not allowed, validation
fails with an `UNKNOWN_SCRIPT` status code.

* Multi-agent `EntryFunction` and `Script`: The prologue function is `multi_agent_script_prologue`.
In addition to the common checks listed below, it also performs the following checks:
    * Check that the number of secondary signer addresses provided is the same
      as the number of secondary signer public key hashes provided. If not,
      validation fails with a `SECONDARY_KEYS_ADDRESSES_COUNT_MISMATCH` status code.
    * For each secondary signer, check if the secondary signer has an account,
      and if not, validation fails with a `SENDING_ACCOUNT_DOES_NOT_EXIST` status code.
    * For each secondary signer, check that the hash of the secondary signer's
      public key (from the `authenticator` in the `SignedTransaction`) matches
      the authentication key in the secondary signer's account. If not, validation
      fails with an `INVALID_AUTH_KEY` status code.


* `Module`: The prologue function is `module_prologue`. In addition to the
common checks listed below, it also calls the `is_module_allowed` function in
the `TransactionPublishingOption` module to see if publishing is allowed
for the transaction sender. If not, validation fails with a
`INVALID_MODULE_PUBLISHER` status code.

The following checks are performed by all the prologue functions:

* If the transaction's `chain_id` value does not match the expected value for
the blockchain, validation fails with a `BAD_CHAIN_ID` status code.

* Check if the transaction sender has an account, and if not, fail with a
`SENDING_ACCOUNT_DOES_NOT_EXIST` status code.

* Call the `AccountFreezing::account_is_frozen` function to check if the
transaction sender's account is frozen. If so, the status code is set to
`SENDING_ACCOUNT_FROZEN`.

* Check that the hash of the transaction's public key (from the `authenticator`
in the `SignedTransaction`) matches the authentication key in the sender's
account. If not, validation fails with an `INVALID_AUTH_KEY` status code.

* The transaction sender must be able to pay the maximum transaction fee. The
maximum fee is the product of the transaction's `max_gas_amount` and
`gas_unit_price` fields. If the sender's account balance
for the Aptos coin is less than the maximum fee, validation fails with an
`INSUFFICIENT_BALANCE_FOR_TRANSACTION_FEE` status code. For `WriteSet`
transactions, the maximum fee is treated as zero, regardless of the gas
parameters specified in the transaction.

* Check if the transaction is expired. If the transaction's
`expiration_timestamp_secs` field is greater than or equal to the current
blockchain timestamp, fail with a `TRANSACTION_EXPIRED` status code.

* Check if the transaction's `sequence_number` is already the maximum value,
such that it would overflow if the transaction was processed. If so,
validation fails with a `SEQUENCE_NUMBER_TOO_BIG` status code.

* The transaction's `sequence_number` must match the current sequence number in
the sender's account. If the transaction sequence number is too low,
validation fails with a `SEQUENCE_NUMBER_TOO_OLD` status code. If the number
is too high, the behavior depends on whether it is the initial validation or
the re-validation done as part of the execution phase. Multiple transactions
with consecutive sequence numbers from the same account can be in flight at
the same time, but they must be executed strictly in order. For that reason, a
transaction sequence number higher than expected for the sender's account is
accepted during the initial validation, but rejected with a
`SEQUENCE_NUMBER_TOO_NEW` status code during the execution phase. Note that
this check for "too new" sequence numbers must be the last check in the
prologue function so that a transaction cannot get through the initial
validation when it has some other fatal error.

If the prologue function fails for any other reason, which would indicate some
kind of unexpected problem, validation fails with a status code of
`UNEXPECTED_ERROR_FROM_KNOWN_MOVE_FUNCTION`.

### Configuration

Validation instantiates a Move Adapter and the adapter loads configuration
data from the view provided. Configuration data resides in the blockchain
storage, and the adapter queries the view for that data. The adapter also
instantiates a Move VM at creation time.

Invocation into the Move VM will load all code related to the transaction
prologue functions on first reference. As described later in the [Move VM
section](#Code-Cache), that code is never released and lives in the code cache
of the VM instance for as long as the instance is alive.

The Move code executed during validation involves exclusively the call graph
rooted at the transaction prologues, all within code published at genesis. The
transaction prologues do not make any updates to on-chain data; they must
always be pure.

Transactions may alter the configuration of the system or force-upgrade code
that was published at genesis. In those cases the system publishes specific
events that require clients to restart the adapter. It is a client's
responsibility to watch for those events and restart an adapter, since the
adapter has no knowledge of execution or the client architecture.

In conclusion, any changes in either configuration data or transaction
prologue code require a restart of the adapter. Failure to do so can lead to
incorrect validation.

### Tests

We have both positive and negative end-to-end tests. All tests that
successfully execute transactions will naturally exercise the path through the
validation logic. Each of the conditions checked in validation has a
corresponding test that fails with the expected status code.

### Threats

Validation has an important role in the system. A bad policy can stall the
system either by depriving it of transactions to execute or by submitting
transactions that will be discarded.

* Stale view: The state view provided by the client to the adapter may be out
of date. For example, after a key rotation transaction executes, subsequent
transactions from that account cannot be validated until the state view is
updated with the new key. The delay in accepting any transaction for that
account is a function of the validator latency in refreshing the view. If a
validator can be "stalled" into a view, a user may be locked out of the system
and unable to submit transactions. What is a tolerable delay? Can
administrative accounts (e.g., AptosRoot) be locked out of the system?

* Configuration updates: The validator must be restarted after a configuration
update, and failure to notice an update may result in inconsistent behavior of
the entire system.

* Panics: What happens if the validator panics? How will a panic be noticed?

### Monitoring and Logging

* Monitoring:

    - `aptos_vm_transactions_validated`: Number of transactions processed by
      the validator, with either "success" or "failure" labels
    - `aptos_vm_txn_validation_seconds`: Histogram of validation time (in
      seconds) per transaction
    - `aptos_vm_critical_errors`: Counter for critical internal errors;
      intended to trigger alerts, not for display on a dashboard

* Logging on catastrophic events must report enough info to debug.

### Runbook

* Related to monitoring and logging obviously.
* Configuration and setup for the adapter? where?
* Common/known problems

## Execution

Execution takes a vector of transactions (`Transaction`) and an initial state
(`StateView`) and produces a corresponding vector of side effects
(`TransactionOutput`) for each transaction. The transactions are executed in
the order of their position in the input vector. The side effects for each
transaction are computed, cached, and accounted for in subsequent transaction
execution. Each `TransactionOutput` entry in the output vector contains the
results of the transaction at the corresponding position in the input
vector. Clients are responsible to apply the side effects in the transaction
output.

```rust
pub trait VMExecutor: Send {
    /// Executes a block of transactions and returns the output for each one of them.
    fn execute_block(
        transactions: Vec<Transaction>,
        state_view: &impl StateView,
    ) -> Result<Vec<TransactionOutput>, VMStatus>;
}

pub enum Transaction {
    /// Transaction submitted by a user.
    UserTransaction(SignedTransaction),

    /// Transaction that applies a WriteSet to the current storage.
    GenesisTransaction(WriteSetPayload),

    /// Transaction to update the block metadata resource at the beginning of a block.
    BlockMetadata(BlockMetadata),
}

pub struct BlockMetadata {
    id: HashValue,
    epoch: u64,
    round: u64,
    proposer: AccountAddress,
    previous_block_votes: Vec<bool>,
    failed_proposer_indices: Vec<u32>,
    timestamp_usecs: u64,
}

/// The output of executing a transaction.
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

/// `WriteSet` contains all access paths that one transaction modifies.
/// Each of them is a `WriteOp` where `Value(val)` means that serialized
/// representation should be updated to `val`, and `Deletion` means that
/// we are going to delete this access path.
pub struct WriteSet {
    write_set: Vec<(AccessPath, WriteOp)>,
}

pub enum WriteOp {
    Deletion,
    Value(Vec<u8>),
}

/// The status of executing a transaction. The wrapped `VMStatus` value
/// provides more detail on the final execution state of the VM.
pub enum TransactionStatus {
    /// Discard the transaction output
    Discard(DiscardedVMStatus),

    /// Keep the transaction output
    Keep(KeptVMStatus),

    /// Retry the transaction, e.g., after a reconfiguration
    Retry,
}
```

The `Transaction` type has three variants:

* `UserTransaction`: This variant is for all transactions that are signed by a
user (possibly a system administrator) and submitted to the system. This uses
the `SignedTransaction` type described in the [Validation
section](#Validation).

* `GenesisTransaction`: This is for a transaction that resets the blockchain to
the original genesis state or to a fixed waypoint. `WriteSetPayload` is the
same type used for WriteSet transactions submitted by a user, but this variant
is needed to allow resetting the system when the consensus system is not
operational.

* `BlockMetadata`: Each block on the blockchain begins with a `BlockMetadata`
transaction that records things like a timestamp and round number. This kind
of transaction may not always be at the beginning of the vector of
transactions given to `execute_block` because the input vector does not
necessarily correspond to a block on the blockchain (e.g., when replaying
transactions).

Because user transactions originate from outside the system, they must go
through validation before they are executed. The other transaction variants do
not go through validation.

The `TransactionOutput` has four components:

* `write_set`: This `WriteSet` value contains a vector of the write operations
performed by the transaction. Each write operation is associated with a
resource at a particular access path in the blockchain state. The operation
may either delete that resource or set it to a new value, specified as a
vector of bytes. No access path should be included more than once in a single
`WriteSet`, and a non-existent access path should never be deleted.

* `events`: This is a vector of `ContractEvent` values describing the events
emitted while executing the transaction. Events in Aptos are stored separately
from the blockchain state, so these events are not part of the transaction
`WriteSet`.

* `gas_used`: This is the number of gas units consumed while executing the
transaction.

* `status`: The status of the transaction execution typically falls into either
the `Discard` or `Keep` categories, indicating whether the transaction should
be discarded or recorded on the blockchain. In both cases, there is an
associated value that records a more detailed status code from the Move
VM. There is also a third category of `Retry` that indicates that the client
should resubmit the transaction following a configuration change. Note that a
transaction with a `Keep` status may not have completed successfully, but it
still needs to be written to the blockchain, for example, to record the
transaction fee.

Execution is a static entry point in the adapter. The adapter creates its
implementation on the stack and its lifetime is limited to the `execute_block`
call. The Move Adapter considers any transaction that may produce a
configuration change as the end of a block. All subsequent transactions in the
input vector are marked `Retry` so that the client will add them to a later
block after the configuration change. Thus, beyond checking for transactions
to retry, the execution client need not do anything special to reconfigure the
adapter.

## Implementation

The adapter uses a data cache to keep track of the side effects computed, so
subsequent transactions can execute on a consistent state. (The `StateView`
input is a read-only view of the blockchain state.) The state changes in the
`WriteSet` for each transaction are recorded in this cache. When reading a
value from the state, the adapter checks first in the cache and uses any value
there before falling back to retrieve the original value from the input state.

The `execute_block` function processes each transaction in its input vector
according to the kind of transaction:

* `GenesisTransaction`: The `WriteSetPayload` from the transaction may contain
either a `Direct` value or a `Script` value.

  - A `Direct` value specifies a `ChangeSet` holding both the `WriteSet` of
    state changes and a vector of events. These are simply copied to the
    transaction output.
  - For a `Script` value, the Move transaction script is run via the Move
    VM. The script runs with the `execute_as` address passed into the vector
    of senders given to the VM session's `execute_script` function. Gas
    metering is disabled while running the script. A failure from executing
    the script is reported with the `INVALID_WRITE_SET` status code.

For both kinds of payload, each access path from the output `WriteSet` is
read from the state. This is done to to maintain a read-before-write
invariant for states. If one of those reads fails, the transaction will fail
with a `STORAGE_ERROR` status code.

* `BlockMetadata`: This transaction is handled by running the `block_prologue`
function from the `Block` module in the Aptos Framework to record the
metadata at the start of a new block. The sender of the transaction is set to
the reserved VM address (zero), and the `round`, `timestamp_usecs`,
`previous_block_votes` and `proposer` fields are extracted from the
transaction and passed as arguments to the function. Gas metering is disabled
when running this in the Move VM, and any error is reported with the
`UNEXPECTED_ERROR_FROM_KNOWN_MOVE_FUNCTION` status code.

* `UserTransaction`: The first step here is to repeat the transaction validation
process in case things have changed since the initial validation. See the
[Validation section](#Validation) for details. Invalid transactions are
discarded. The signature verification step of validation is computationally
expensive, so for performance reasons, it is a good idea for the adapter
implementation to do that checking in parallel for all the input transactions,
instead of waiting to do it as the transactions are executed
sequentially. After successful validation, the `TransactionPayload` is
processed as described in the following sections for different kinds of
payloads.

Whenever the Move VM is used in this transaction processing, the adapter must
translate the results from the VM's `Session` into both a
`WriteSet` and a set of events. That translation can fail with either an
`DATA_FORMAT_ERROR` or an `EVENT_KEY_MISMATCH` status code.

Regardless of the kind of transaction, the `WriteSet` changes that it produces
are stored in the adapter's data cache, so they will be seen when processing
subsequent transactions.

The adapter needs to check if a transaction reconfigures the blockchain. It
does that by checking the generated events to see if a `NewEpochEvent` was
emitted. If so, the `execute_block` function returns after marking the
`TransactionStatus` of all subsequent transactions as `Retry`.

Errors for user transactions, either running scripts or publishing modules, do
not stop the execution. A failed user transaction may be kept on-chain to
charge for gas, or it may be discarded, but it will not cause the
`execute_block` function to return an error. However, if an error occurs for a
`WriteSet` transaction or other system transaction, the error is immediately
propagated to the result of the `execute_block` function, so that none of the
input transactions are executed.

### Script and Module Transactions

After re-validating a user transaction, the adapter processes the payload,
depending on its contents.

Aptos version 1 had a fixed set of script transactions for general user
transactions, with the script hash values stored in the on-chain allowlist,
that are now implemented as entry functions in Aptos version 2 and later.
If the on-chain Aptos Version number is 2 or later, the adapter first checks
if the script is one of those special scripts, and if so, remaps it to the
corresponding entry function. Because this remapping is fixed to Aptos
version 1 and is never expected to change, the remapping to entry functions
is hardcoded in the adapter.

In the common case, the payload is either a script
function, script, or a module:

* `EntryFunction`: The Move VM is used to [execute](#Script-Function-Execution)
the entry function with the types and arguments specified in the transaction.

* `Script`: The Move VM is used to [execute](#Script-Execution) the script with
the types and arguments specified in the transaction.

* `Module`: The Move VM is used to [publish](#Publishing) the code module from
the transaction.

The Move VM operations used here consume gas according to the gas schedule
that is stored in an on-chain configuration. The adapter loads that gas
schedule before invoking the VM.

If the script or module payload is processed successfully, the Move VM is next
used to [execute](#Function-Execution) the `epilogue` function from the
`AptosAccount` module. The epilogue increments the sender's `sequence_number`
and deducts the transaction fee based on the gas price and the amount of gas
consumed. This function execution is done using the same VM `Session` that was
used when processing the payload, so that all the side effects are
combined. The epilogue function is run with gas metering disabled.

If an error occurs when processing the payload or when running the epilogue,
the adapter will discard all the side effects from the transaction, but it
still needs to charge the transaction fee for gas consumption to the sender's
account. It does that by creating a new VM `Session` to run the epilogue
function. Note that the epilogue function may be run twice. For example, the
transaction may make a payment that drops the account balance so that the
first attempt to run the epilogue fails because of insufficient funds to pay
the transaction fee. After dropping the side effects of the payment, however,
the second attempt to run the epilogue should always succeed, because the
validation process ensures that the account balance can cover the maximum
transaction fee. If the second "failure" epilogue execution somehow fails (due
to an internal inconsistency in the system), execution fails with an
`UNEXPECTED_ERROR_FROM_KNOWN_MOVE_FUNCTION` status, and the transaction is
discarded.

### WriteSet Transactions

The third possible kind of user transaction is a `WriteSet`, and there are
some differences in handling those compared to other user transactions. For
the most part, the `WriteSetPayload` is processed just like a
`GenesisTransaction`, but there are two differences. First, for a `Script`
payload, the vector of senders passed to the VM has the transaction sender as
the first value followed by the `execute_as` value from the
transaction. Second, if there is an error reading the states touched by the
`WriteSet`, the error status code is reported as `INVALID_WRITE_SET` instead
of `STORAGE_ERROR`. There are no gas charges for `WriteSet` transactions.

Instead of the standard epilogue function, for `WriteSet` transactions the
adapter executes the special `writeset_epilogue` function from the
`AptosAccount` module. The `writeset_epilogue` calls the standard epilogue to
increment the `sequence_number`, emits an `AdminTransactionEvent`, and if the
`WriteSetPayload` is a `Direct` value, it also emits a `NewEpochEvent` to
trigger reconfiguration. For a `Script` value in the `WriteSetPayload`, it is
the responsibility of the code in the script to determine whether a
reconfiguration is necessary, and if so, to emit the appropriate
`NewEpochEvent`. If the epilogue does not execute successfully, the status
code is set to `UNEXPECTED_ERROR_FROM_KNOWN_MOVE_FUNCTION`.

The epilogue is run with a new VM `Session`. (If the `WriteSetPayload` was
`Direct` then there is no other `Session` to use.) Because of that, the side
effects of the epilogue function, both the state changes and the events, need
to be merged with those from the payload. If those changes conflict, i.e., if
they modify the same access paths or events, the transaction fails with an
`INVALID_WRITE_SET` status.

If errors occur while processing either the payload or the epilogue, the
transaction fails and the `execute_block` function returns an error.

## Tests

End to end, mostly/exclusively positive tests: smoke tests

Versioning: ...

Unit test: e2e tests

## Threats

* Configuration updates: configuration updates may (and conservatively should)
require a refresh of the adapter. Both configuration data and the VM should
be reloaded with the updated view. What are the guarantees of the adapter?

* Panics

* Transaction executed, transaction executed by the VM, success vs fail,
certain error type.

* Monitoring:

    - `aptos_vm_user_transactions_executed`: Number of user transactions executed,
      with either "success" or "failure" labels
    - `aptos_vm_system_transactions_executed`: Number of system transactions
      executed, with either "success" or "failure" labels
    - `aptos_vm_txn_total_seconds`: Histogram of execution time (in seconds) per
      user transaction
    - `aptos_vm_num_txns_per_block`: Histogram of number of transactions per block
    - `aptos_vm_critical_errors`: Counter for critical internal errors; intended
      to trigger alerts, not for display on a dashboard

* Logging on catastrophic events must report enough info to debug.

## Runbook

* Related to monitoring and logging obviously.
* Configuration and setup for the adapter? where?
* Common/known problems

## Move VM

Instantiation of a Move VM just initializes an instance of a `Loader`, that
is, a small set of empty tables (few instances of `HashMap` and `Vec` behind
`Mutex`). Initialization of a VM is reasonably inexpensive. The `Loader` is
effectively the code cache. The code cache has the lifetime of the VM. Code
is loaded at runtime when functions and scripts are executed. Once loaded,
modules and scripts are reused in their loaded form and ready to be executed
immediately. Loading code is expensive and the VM performs eager
loading. When execution starts, no more loading takes place, all code through
any possible control flow is ready to be executed and cached at load time.
Maybe, more importantly, the eager model guarantees that no runtime errors can
come from linking at runtime, and that a given invocation will not fail
loading/linking because of different code paths. The consistency of the
invocation is guaranteed before execution starts. Obviously runtime errors are
still possible and "expected".

This model fits well Aptos requirements:

* Validation uses only few functions published at genesis. Once loaded, code is
always fetched from the cache and immediately available.

* Execution is in the context of a given data view, a stable and immutable
view. As such code is stable too, and it is important to optimize the process
of loading. Also, transactions are reasonably homogeneous and reuse of code
leads to significant improvements in performance and stability.

The VM in its current form is optimized for Aptos, and it offers an API that is
targeted for that environment. In particular the VM has an internal
implementation for a data cache that relieves the Aptos client from an
important responsibility (data cache consistency). That abstraction is behind
a `Session` which is the only way to talk to the runtime.

The objective of a `Session` is to create and manage the data cache for a set
of invocations into the VM. It is also intended to return side effects in a
format that is suitable to the adapter, and in line with Aptos and the
generation of a `WriteSet`.
A `Session` forwards calls to the `Runtime` which is where the logic and
implementation of the VM lives and starts.

### Code Cache

When loading a Module for the first time, the VM queries the data store for
the Module. That binary is deserialized, verified, loaded and cached by the
loader. Once loaded, a Module is never requested again for the lifetime of
that VM instance. Code is an immutable resource in the system.

The process of loading can be summarized through the following steps:

1. a binary—Module in a serialized form, `Vec<u8>`—is fetched from the data store.
This may require a network access
2. the binary is deserialized and verified
3. dependencies of the module are loaded (repeat 1.–4. for each dependency)
4. the module is linked to its dependencies (transformed in a representation
suitable for runtime) and cached by the loader.

So a reference to a loaded module does not perform any fetching from the
network, or verification, or transformations into runtime structures
(e.g. linking).

In Aptos, consistency of the code cache can be broken by a system transaction
that performs a hard upgrade, requiring the adapter to stop processing
transactions until a restart takes place. Other clients may have different
"code models" (e.g. some form of versioning).

Overall, a client holding an instance of a Move VM has to be aware of the
behavior of the code cache and provide data views (`DataStore`) that are
compatible with the loaded code. Moreover, a client is responsible to release
and instantiate a new VM when specific conditions may alter the consistency of
the code cache.

### Publishing

Clients may publish modules in the system by calling:

```rust
pub fn publish_module(
    &mut self,
    module: Vec<u8>,
    sender: AccountAddress,
    gas_status: &mut GasStatus,
) -> VMResult<()>;
```

The `module` is in a [serialized form](#Binary-Format) and the VM performs the
following steps:

* Deserialize the module: If the module does not deserialize, an error is
returned with a proper `StatusCode`.

* Check that the module address and the `sender` address are the same: This
check verifies that the publisher is the account that will eventually [hold
the module](#References-to-Data-and-Code). If the two addresses do not match, an
error with `StatusCode::MODULE_ADDRESS_DOES_NOT_MATCH_SENDER` is returned.

* Check that the module is not already published: Code is immutable in
Move. An attempt to overwrite an exiting module results in an error with
`StatusCode::DUPLICATE_MODULE_NAME`.

* Verify loading: The VM performs [verification](#Verification) of the
module to prove correctness. However, neither the module nor any of its
dependencies are actually saved in the cache. The VM ensures that the module
will be loadable when a reference will be found. If a module would fail to
load an error with proper `StatusCode` is returned.

* Publish: The VM writes the serialized bytes of the module
with the [proper key](#References-to-Data-and-Code) to the storage.
After this step any reference to the
module is valid.

## Script Execution

The VM allows the execution of [scripts](#Binary-Format). A script is a
Move function declared in a `script` block that performs
calls into the Aptos Framework to accomplish a
logical transaction. A script is not saved in storage and
it cannot be invoked by other scripts or modules.

```rust
pub fn execute_script(
    &mut self,
    script: Vec<u8>,
    ty_args: Vec<TypeTag>,
    args: Vec<Vec<u8>>,
    senders: Vec<AccountAddress>,
    gas_status: &mut GasStatus,
) -> VMResult<()>;
```

The `script` is specified in a [serialized form](#Binary-Format).
If the script is generic, the `ty_args` vector contains the `TypeTag`
values for the type arguments. The `signer` account addresses for the
script are specified in the `senders` vector. Any additional arguments
are provided in the `args` vector, where each argument is a BCS-serialized
vector of bytes. The VM
performs the following steps:

* Load the Script and the main function:

    - The `sha3_256` hash value of the `script` binary is computed.
    - The hash is used to access the script cache to see if the script was
      loaded. The hash is used for script identity.
    - If not in the cache the script is [loaded](#Loading). If loading fails,
      execution stops and an error with a proper `StatusCode` is returned.
    - The script main function is [checked against the
      type argument instantiation](#Verification) and if there are
      errors, execution stops and the error returned.

* Build the argument list: The first arguments are `Signer` values created by
the VM for the account addresses in the `senders` vector. Any other arguments
from the `args` vector are then checked against a whitelisted set of permitted
types and added to the arguments for the script.
The VM returns an error with `StatusCode::TYPE_MISMATCH` if
any of the types is not permitted.

* Execute the script: The VM invokes the interpreter to [execute the
script](#Interpreter). Any error during execution is returned, and the
transaction aborted. The VM returns whether execution succeeded or
failed.

## Entry Function Execution

Entry functions (in version 2 and later of the Move VM) are similar to scripts
except that the Move bytecode comes from a Move function with `script` visibility
in an on-chain module. The entry function is specified by the module and function
name:

```rust
pub fn execute_entry_function(
    &mut self,
    module: &ModuleId,
    function_name: &IdentStr,
    ty_args: Vec<TypeTag>,
    args: Vec<Vec<u8>>,
    senders: Vec<AccountAddress>,
    gas_status: &mut GasStatus,
) -> VMResult<()>;
```

Execution of entry functions is similar to scripts. Instead of using the Move bytecodes
from a script, the entry function is loaded from the on-chain module, and the Move VM
checks that it has `script` visibility. The rest of the entry function execution is
the same as for scripts. If the function does not exist, execution fails with a
`FUNCTION_RESOLUTION_FAILURE` status code. If the function does not have `script` visibility,
it will fail with the `EXECUTE_SCRIPT_FUNCTION_CALLED_ON_NON_SCRIPT_VISIBLE` status code.

## Function Execution

The VM allows the execution of [any function in a module](#Binary-Format)
through a `ModuleId` and a function name. Function names are unique within a
module (no overloading), so the signature of the function is not
required. Argument checking is done by the [interpreter](#Interpreter).

The adapter uses this entry point to run specific system functions as
described in [validation](#Validation) and [execution](#Execution). This is a
very powerful entry point into the system given there are no visibility
checks. Clients would likely use this entry point internally (e.g., for
constructing a genesis state), or wrap and expose it with restrictions.

```rust
pub fn execute_function(
    &mut self,
    module: &ModuleId,
    function_name: &IdentStr,
    ty_args: Vec<TypeTag>,
    args: Vec<Vec<u8>>,
    gas_status: &mut GasStatus,
) -> VMResult<()>;
```

The VM performs the following steps:

* Load the function:

    - The specified `module` is first [loaded](#Loading).
      An error in loading halts execution and returns the error with a proper
      `StatusCode`.
    - The VM looks up the function in the module. Failure to resolve the
      function returns an error with a proper `StatusCode`.
    - Every type in the `ty_args` vector is [loaded](#Loading). An error
      in loading halts execution and returns the error with a proper `StatusCode`.
      Type arguments are checked against type parameters and an error returned
      if there is a mismatch (i.e., argument inconsistent with generic declaration).

* Build the argument list: Arguments are checked against a whitelisted set
of permitted types (_specify which types_). The VM returns an error with
`StatusCode::TYPE_MISMATCH` if any of the types is not permitted.

* Execute the function: The VM invokes the interpreter to [execute the
function](#Interpreter). Any error during execution aborts the interpreter
and returns the error. The VM returns whether execution succeeded or
failed.

## Binary Format

Modules and Scripts can only enter the VM in binary form, and Modules are
saved on chain in binary form. A Module is logically a collection of
functions and data structures. A Script is just an entry point, a single
function with arguments and no return value.

Modules can be thought as library or shared code, whereas Scripts can only
come in input with the Transaction.

Binaries are composed of headers and a set of tables. Some of
those tables are common to both Modules and Scripts, others specific to one or
the other. There is also data specific only to Modules or Scripts.

The binary format makes a heavy use of
[ULEB128](https://en.wikipedia.org/wiki/LEB128) to compress integers. Most of
the data in a binary is in the form of indices, and as such compression offers
an important saving. Integers, when used with no compression are in
[little-endian](https://en.wikipedia.org/wiki/Endianness) form.

Vectors are serialized with the size first, in ULEB128 form, followed by the
elements contiguously.

### Binary Header

Every binary starts with a header that has the following format:

* `Magic`: 4 bytes 0xA1, 0x1C, 0xEB, 0x0B (aka "A11CEB0B" or "AliceBob")
* `Version`: 4 byte little-endian unsigned integer
* `Table count`: number of tables in ULEB128 form. The current maximum number
of tables is contained in 1 byte, so this is effectively the count of tables in
one byte. Not all tables need to be present. Each kind of table can only be
present once; table repetitions are not allowed. Tables can be serialized in any
order.

### Table Headers

Following the binary header are the table headers. There are as many tables as
defined in "table count". Each table header
has the following format:

* `Table Kind`: 1 byte for the [kind of table](#Tables) that is serialized at
the location defined by the next 2 entries
* `Table Offset`: ULEB128 offset from the end of the table headers where the
table content starts
* `Table Length`: ULEB128 byte count of the table content

Tables must be contiguous to each other, starting from the end of the table
headers. There must not be any gap between the content of the tables. Table
content must not overlap.

### Tables

A `Table Kind` is 1 byte, and it is one of:

* `0x1`: `MODULE_HANDLES` - for both Modules and Scripts
* `0x2`: `STRUCT_HANDLES` - for both Modules and Scripts
* `0x3`: `FUNCTION_HANDLES` - for both Modules and Scriptss
* `0x4`: `FUNCTION_INSTANTIATIONS` - for both Modules and Scripts
* `0x5`: `SIGNATURES` - for both Modules and Scripts
* `0x6`: `CONSTANT_POOL` - for both Modules and Scripts
* `0x7`: `IDENTIFIERS` - for both Modules and Scripts
* `0x8`: `ADDRESS_IDENTIFIERS` - for both Modules and Scripts
* `0xA`: `STRUCT_DEFINITIONS` - only for Modules
* `0xB`: `STRUCT_DEF_INSTANTIATIONS` - only for Modules
* `0xC`: `FUNCTION_DEFINITIONS` - only for Modules
* `0xD`: `FIELD_HANDLES` - only for Modules
* `0xE`: `FIELD_INSTANTIATIONS` - only for Modules
* `0xF`: `FRIEND_DECLS` - only for Modules, version 2 and later

The formats of the tables are:

* `MODULE_HANDLES`: A `Module Handle` is a pair of indices that identify
the location of a module:

    * `address`: ULEB128 index into the `ADDRESS_IDENTIFIERS` table of
    the account under which the module is published
    * `name`: ULEB128 index into the `IDENTIFIERS` table of the name of the module

* `STRUCT_HANDLES`: A `Struct Handle` contains all the information to
uniquely identify a user type:

    * `module`: ULEB128 index in the `MODULE_HANDLES` table of the module
    where the struct is defined
    * `name`: ULEB128 index into the `IDENTIFIERS` table of the name of the struct
    * `nominal resource`: U8 bool defining whether the
    struct is a resource (true/1) or not (false/0)
    * `type parameters`: vector of [type parameter kinds](#Kinds) if the
    struct is generic, an empty vector otherwise:
        * `length`: ULEB128 length of the vector, effectively the number of type
        parameters for the generic struct
        * `kinds`: array of `length` U8 kind values; not present if length is 0

* `FUNCTION_HANDLES`: A `Function Handle` contains all the information to uniquely
identify a function:

    * `module`: ULEB128 index in the `MODULE_HANDLES` table of the module where
    the function is defined
    * `name`: ULEB128 index into the `IDENTIFIERS` table of the name of the function
    * `parameters`: ULEB128 index into the `SIGNATURES` table for the argument types
    of the function
    * `return`: ULEB128 index into the `SIGNATURES` table for the return types of the function
    * `type parameters`: vector of [type parameter kinds](#Kinds) if the function
    is generic, an empty vector otherwise:
        * `length`: ULEB128 length of the vector, effectively the number of type
        parameters for the generic function
        * `kinds`: array of `length` U8 kind values; not present if length is 0

* `FUNCTION_INSTANTIATIONS`: A `Function Instantiation` describes the
instantation of a generic function. Function Instantiation can be full or
partial. E.g., given a generic function `f<K, V>()` a full instantiation would
be `f<U8, Bool>()` whereas a partial instantiation would be `f<U8, Z>()` where
`Z` is a type parameter in a given context (typically another function
`g<Z>()`).

    * `function handle`: ULEB128 index into the `FUNCTION_HANDLES` table of the
    generic function for this instantiation (e.g., `f<K, W>()`)
    * `instantiation`: ULEB128 index into the `SIGNATURES` table for the
    instantiation of the function

* `SIGNATURES`: The set of signatures in this binary. A signature is a
vector of [Signature Tokens](#SignatureTokens), so every signature will carry
the length (in ULEB128 form) followed by the Signature Tokens.

* `CONSTANT_POOL`: The set of constants in the binary. A constant is a
copyable primitive value or a vector of vectors of primitives. Constants
cannot be user types. Constants are serialized according to the rule defined
in [Move Values](#Move-Values) and stored in the table in serialized form. A
constant in the constant pool has the following entries:

    * `type`: the [Signature Token](#SignatureTokens) (type) of the value that follows
    * `length`: the length of the serialized value in bytes
    * `value`: the serialized value

* `IDENTIFIERS`: The set of identifiers in this binary. Identifiers are
vectors of chars. Their format is the length of the vector in ULEB128 form
followed by the chars. An identifier can only have characters in the ASCII set
and specifically: must start with a letter or '\_', followed by a letter, '\_'
or digit

* `ADDRESS_IDENTIFIERS`: The set of addresses used in ModuleHandles.
Addresses are fixed size so they are stored contiguously in this table.

* `STRUCT_DEFINITIONS`: The structs or user types defined in the binary. A
struct definition contains the following fields:

    * `struct_handle`: ULEB128 index in the `STRUCT_HANDLES` table for the
    handle of this definition
    * `field_information`: Field Information provides information about the
    fields of the struct or whether the struct is native

        * `tag`: 1 byte, either `0x1` if the struct is native, or `0x2` if the struct
        contains fields, in which case it is followed by:
        * `field count`: ULEB128 number of fields for this struct
        * `fields`: a field count of

            * `name`: ULEB128 index in the `IDENTIFIERS` table containing the
            name of the field
            * `field type`: [SignatureToken](#SignatureTokens) - the type of
            the field

* `STRUCT_DEF_INSTANTIATIONS`: the set of instantiation for any given
generic struct. It contains the following fields:

    * `struct handle`: ULEB128 index into the `STRUCT_HANDLES` table of the
    generic struct for this instantiation (e.g., `struct X<T>`)
    * `instantiation`: ULEB128 index into the `SIGNATURES` table for the
    instantiation of the struct. The instantiation can be either partial or complete
    (e.g., `X<U64>` or `X<Z>` when inside another generic function or generic struct
    with type parameter `Z`)

* `FUNCTION_DEFINITIONS`: the set of functions defined in this binary. A
function definition contains the following fields:

    * `function_handle`: ULEB128 index in the `FUNCTION_HANDLES` table for
    the handle of this definition
    * `visibility`: 1 byte for the function visibility (only used in version 2 and later)

        * `0x0` if the function is private to the Module
        * `0x1` if the function is public and thus visible outside this module
        * `0x2` for a `script` function
        * `0x3` if the function is private but also visible to `friend` modules

    * `flags`: 1 byte:

        * `0x0` if the function is private to the Module (version 1 only)
        * `0x1` if the function is public and thus visible outside this module (version 1 only)
        * `0x2` if the function is native, not implemented in Move

    * `acquires_global_resources`: resources accessed by this function

        * `length`: ULEB128 length of the vector, number of resources
        acquired by this function
        * `resources`: array of `length` ULEB128 indices into the `STRUCT_DEFS` table,
        for the resources acquired by this function

    * `code_unit`: if the function is not native, the code unit follows:

        * `locals`: ULEB128 index into the `SIGNATURES` table for the types
        of the locals of the function
        * `code`: vector of [Bytecodes](#Bytecodes), the body of this function

            * `length`: the count of bytecodes the follows
            * `bytecodes`: Bytecodes, they are variable size

* `FIELD_HANDLES`: the set of fields accessed in code. A field handle is
composed by the following fields:

    * `owner`: ULEB128 index into the `STRUCT_DEFS` table of the type that owns the field
    * `index`: ULEB128 position of the field in the vector of fields of the `owner`

* `FIELD_INSTANTIATIONS`: the set of generic fields accessed in code. A
field instantiation is a pair of indices:

    * `field_handle`: ULEB128 index into the `FIELD_HANDLES` table for the generic field
    * `instantiation`: ULEB128 index into the `SIGNATURES` table for the instantiation of
    the type that owns the field

* `FRIEND_DECLS`: the set of declared friend modules with the following for each one:

    * `address`: ULEB128 index into the `ADDRESS_IDENTIFIERS` table of
    the account under which the module is published
    * `name`: ULEB128 index into the `IDENTIFIERS` table of the name of the module

### Kinds

A `Type Parameter Kind` is 1 byte, and it is one of:

* `0x1`: `ALL` - the type parameter can be substituted by either a resource, or a copyable type
* `0x2`: `COPYABLE` - the type parameter must be substituted by a copyable type
* `0x3`: `RESOURCE` - the type parameter must be substituted by a resource type

### SignatureTokens

A `SignatureToken` is 1 byte, and it is one of:

* `0x1`: `BOOL` - a boolean
* `0x2`: `U8` - a U8 (byte)
* `0x3`: `U64` - a 64-bit unsigned integer
* `0x4`: `U128` - a 128-bit unsigned integer
* `0x5`: `ADDRESS` - an `AccountAddress` in Aptos, which is a 128-bit unsigned integer
* `0x6`: `REFERENCE` - a reference; must be followed by another SignatureToken
representing the type referenced
* `0x7`: `MUTABLE_REFERENCE` - a mutable reference; must be followed by another
SignatureToken representing the type referenced
* `0x8`: `STRUCT` - a structure; must be followed by the index into the
`STRUCT_HANDLES` table describing the type. That index is in ULEB128 form
* `0x9`: `TYPE_PARAMETER` - a type parameter of a generic struct or a generic
function; must be followed by the index into the type parameters vector of its container.
The index is in ULEB128 form
* `0xA`: `VECTOR` - a vector - must be followed by another SignatureToken
representing the type of the vector
* `0xB`: `STRUCT_INST` - a struct instantiation; must be followed by an index
into the `STRUCT_HANDLES` table for the generic type of the instantiation, and a
vector describing the substitution types, that is, a vector of SignatureTokens
* `0xC`: `SIGNER` - a signer type, which is a special type for the VM
representing the "entity" that signed the transaction. Signer is a resource type

Signature tokens examples:

* `u8, u128` -> `0x2 0x2 0x4` - size(`0x2`), U8(`0x2`), u128(`0x4`)
* `u8, u128, A` where A is a struct -> `0x3 0x2 0x4 0x8 0x10` - size(`0x3`),
U8(`0x2`), u128(`0x4`), Struct::A
(`0x8 0x10` assuming the struct is in the `STRUCT_HANDLES` table at position `0x10`)
* `vector<address>, &A` where A is a struct -> `0x2 0xA 0x5 0x8 0x10` - size(`0x2`),
vector<address>(`0xA 0x5`), &Struct::A
(`0x6 0x8 0x10` assuming the struct is in the `STRUCT_HANDLES` table at position `0x10`)
* `vector<A>, &A<B>` where A and B are a struct ->
`0x2 0xA 0x8 0x10 0x6 0xB 0x10 0x1 0x8 0x11` -
size(`0x2`), vector\<A\>(`0xA 0x8 0x10`),
&Struct::A\<Struct::B\> (`0x6` &, `0xB 0x10` A<\_>, `0x1 0x8 0x11` B type
instantiation; assuming the struct are in the `STRUCT_HANDLES` table at position
`0x10` and `0x11` respectively)

### Bytecodes

Bytecodes are variable size instructions for the Move VM. Bytecodes are
composed by opcodes (1 byte) followed by a possible payload which depends on
the specific opcode and specified in "()" below:

* `0x01`: `POP`
* `0x02`: `RET`
* `0x03`: `BR_TRUE(offset)` - offset is in ULEB128 form, and it is the target
offset in the code stream from the beginning of the code stream
* `0x04`: `BR_FALSE(offset)` - offset is in ULEB128 form, and it is the
target offset in the code stream from the beginning of the code stream
* `0x05`: `BRANCH(offset)` - offset is in ULEB128 form, and it is the target
offset in the code stream from the beginning of the code stream
* `0x06`: `LD_U64(value)` - value is a U64 in little-endian form
* `0x07`: `LD_CONST(index)` - index is in ULEB128 form, and it is an index
in the `CONSTANT_POOL` table
* `0x08`: `LD_TRUE`
* `0x09`: `LD_FALSE`
* `0x0A`: `COPY_LOC(index)` - index is in ULEB128 form, and it is an index
referring to either an argument or a local of the function. From a bytecode
perspective arguments and locals lengths are added and the index must be in that
range. If index is less than the length of arguments it refers to one of the
arguments otherwise it refers to one of the locals
* `0x0B`: `MOVE_LOC(index)` - index is in ULEB128 form, and it is an index
referring to either an argument or a local of the function. From a bytecode
perspective arguments and locals lengths are added and the index must be in that
range. If index is less than the length of arguments it refers to one of the
arguments otherwise it refers to one of the locals
* `0x0C`: `ST_LOC(index)` - index is in ULEB128 form, and it is an index
referring to either an argument or a local of the function. From a bytecode
perspective arguments and locals lengths are added and the index must be in that
range. If index is less than the length of arguments it refers to one of the
arguments otherwise it refers to one of the locals
* `0x0D`: `MUT_BORROW_LOC(index)` - index is in ULEB128 form, and it is an
index referring to either an argument or a local of the function. From a
bytecode perspective arguments and locals lengths are added and the index must
be in that range. If index is less than the length of arguments it refers to one
of the arguments otherwise it refers to one of the locals
* `0x0E`: `IMM_BORROW_LOC(index)` - index is in ULEB128 form, and it is an
index referring to either an argument or a local of the function. From a
bytecode perspective arguments and locals lengths are added and the index must
be in that range. If index is less than the length of arguments it refers to one
of the arguments otherwise it refers to one of the locals
* `0x0F`: `MUT_BORROW_FIELD(index)` - index is in ULEB128 form, and it is an
index in the `FIELD_HANDLES` table
* `0x10`: `IMM_BORROW_FIELD(index)` - index is in ULEB128 form, and it is an
index in the `FIELD_HANDLES` table
* `0x11`: `CALL(index)` - index is in ULEB128 form, and it is an index in the
`FUNCTION_HANDLES` table
* `0x12`: `PACK(index)` - index is in ULEB128 form, and it is an index in the
`STRUCT_DEFINITIONS` table
* `0x13`: `UNPACK(index)` - index is in ULEB128 form, and it is an index in
the `STRUCT_DEFINITIONS` table
* `0x14`: `READ_REF`
* `0x15`: `WRITE_REF`
* `0x16`: `ADD`
* `0x17`: `SUB`
* `0x18`: `MUL`
* `0x19`: `MOD`
* `0x1A`: `DIV`
* `0x1B`: `BIT_OR`
* `0x1C`: `BIT_AND`
* `0x1D`: `XOR`
* `0x1E`: `OR`
* `0x1F`: `AND`
* `0x20`: `NOT`
* `0x21`: `EQ`
* `0x22`: `NEQ`
* `0x23`: `LT`
* `0x24`: `GT`
* `0x25`: `LE`
* `0x26`: `GE`
* `0x27`: `ABORT`
* `0x28`: `NOP`
* `0x29`: `EXISTS(index)` - index is in ULEB128 form, and it is an index in
the `STRUCT_DEFINITIONS` table
* `0x2A`: `MUT_BORROW_GLOBAL(index)` - index is in ULEB128 form, and it is
an index in the `STRUCT_DEFINITIONS` table
* `0x2B`: `IMM_BORROW_GLOBAL(index)` - index is in ULEB128 form, and it is
an index in the `STRUCT_DEFINITIONS` table
* `0x2C`: `MOVE_FROM(index)` - index is in ULEB128 form, and it is an index
in the `STRUCT_DEFINITIONS` table
* `0x2D`: `MOVE_TO(index)` - index is in ULEB128 form, and it is an index
in the `STRUCT_DEFINITIONS` table
* `0x2E`: `FREEZE_REF`
* `0x2F`: `SHL`
* `0x30`: `SHR`
* `0x31`: `LD_U8(value)` - value is a U8
* `0x32`: `LD_U128(value)` - value is a U128 in little-endian form
* `0x33`: `CAST_U8`
* `0x34`: `CAST_U64`
* `0x35`: `CAST_U128`
* `0x36`: `MUT_BORROW_FIELD_GENERIC(index)` - index is in ULEB128 form,
and it is an index in the `FIELD_INSTANTIATIONS` table
* `0x37`: `IMM_BORROW_FIELD_GENERIC(index)` - index is in ULEB128 form,
and it is an index in the `FIELD_INSTANTIATIONS` table
* `0x38`: `CALL_GENERIC(index)` - index is in ULEB128 form, and it is an
index in the `FUNCTION_INSTANTIATIONS` table
* `0x39`: `PACK_GENERIC(index)` - index is in ULEB128 form, and it is an
index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3A`: `UNPACK_GENERIC(index)` - index is in ULEB128 form, and it is an
index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3B`: `EXISTS_GENERIC(index)` - index is in ULEB128 form, and it is an
index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3C`: `MUT_BORROW_GLOBAL_GENERIC(index)` - index is in ULEB128 form,
and it is an index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3D`: `IMM_BORROW_GLOBAL_GENERIC(index)` - index is in ULEB128 form,
and it is an index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3E`: `MOVE_FROM_GENERIC(index)` - index is in ULEB128 form, and it
is an index in the `STRUCT_DEF_INSTANTIATIONS` table
* `0x3F`: `MOVE_TO_GENERIC(index)` - index is in ULEB128 form, and it is
an index in the `STRUCT_DEF_INSTANTIATIONS` table

### Module Specific Data

A binary for a Module contains an index in ULEB128 form as its last
entry. That is after all tables. That index points to the ModuleHandle table
and it is the self module. It is where the module is stored, and a
specification of which one of the Modules in the `MODULE_HANDLES` tables is the
self one.

### Script Specific Data

A Script does not have a `FUNCTION_DEFINITIONS` table, and the entry point is
explicitly described in the following entries, at the end of a Script
Binary, in the order below:

* `type parameters`: if the script entry point is generic, the number and
kind of the type parameters is in this vector.

    * `length`: ULEB128 length of the vector, effectively the number of
    type parameters for the generic entry point. 0 if the script is not generic
    * `kinds`: array of `length` U8 [kind](#Kinds) values, not present
    if length is 0

* `parameters`: ULEB128 index into the `SIGNATURES` table for the argument
types of the entry point

* `code`: vector of [Bytecodes](#Bytecodes), the body of this function
    * `length`: the count of bytecodes
    * `bytecodes`: Bytecodes contiguously serialized, they are variable size
