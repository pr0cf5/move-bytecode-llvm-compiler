// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::quorum_store::types::SerializedTransaction;
use aptos_crypto::{
    ed25519::{Ed25519PrivateKey, Ed25519Signature},
    PrivateKey, Uniform,
};
use aptos_types::{
    account_address::AccountAddress,
    chain_id::ChainId,
    transaction::{RawTransaction, Script, SignedTransaction, Transaction, TransactionPayload},
};

// Creates a single test transaction
fn create_transaction() -> Transaction {
    let private_key = Ed25519PrivateKey::generate_for_testing();
    let public_key = private_key.public_key();

    let transaction_payload = TransactionPayload::Script(Script::new(vec![], vec![], vec![]));
    let raw_transaction = RawTransaction::new(
        AccountAddress::random(),
        0,
        transaction_payload,
        0,
        0,
        0,
        ChainId::new(10),
    );
    let signed_transaction = SignedTransaction::new(
        raw_transaction,
        public_key,
        Ed25519Signature::dummy_signature(),
    );

    Transaction::UserTransaction(signed_transaction)
}

fn create_vec_signed_transactions(size: u64) -> Vec<SignedTransaction> {
    (0..size)
        .map(|_| match create_transaction() {
            Transaction::UserTransaction(inner) => inner,
            _ => panic!("Not a user transaction."),
        })
        .collect()
}

pub(crate) fn create_vec_serialized_transactions(size: u64) -> Vec<SerializedTransaction> {
    create_vec_signed_transactions(size)
        .iter()
        .map(SerializedTransaction::from_signed_txn)
        .collect()
}
