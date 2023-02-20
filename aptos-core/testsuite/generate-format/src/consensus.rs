// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use aptos_crypto::{
    bls12381,
    ed25519::Ed25519PrivateKey,
    multi_ed25519::{MultiEd25519PublicKey, MultiEd25519Signature},
    traits::{SigningKey, Uniform},
    PrivateKey,
};
use aptos_crypto_derive::{BCSCryptoHash, CryptoHasher};
use aptos_types::{
    contract_event, event, state_store::state_key::StateKey, transaction, write_set,
};
use move_core_types::language_storage;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_reflection::{Registry, Result, Samples, Tracer, TracerConfig};

/// Return a relative path to start tracking changes in commits.
pub fn output_file() -> Option<&'static str> {
    Some("tests/staged/consensus.yaml")
}

/// This aims at signing canonically serializable BCS data
#[derive(CryptoHasher, BCSCryptoHash, Serialize, Deserialize)]
struct TestAptosCrypto(String);

/// Record sample values for crypto types used by consensus.
fn trace_crypto_values(tracer: &mut Tracer, samples: &mut Samples) -> Result<()> {
    let message = TestAptosCrypto("Hello, World".to_string());

    let mut rng: StdRng = SeedableRng::from_seed([0; 32]);

    let private_key = Ed25519PrivateKey::generate(&mut rng);
    let public_key = private_key.public_key();
    let signature = private_key.sign(&message).unwrap();

    let bls_private_key = bls12381::PrivateKey::generate(&mut rng);
    let bls_public_key = bls_private_key.public_key();
    let bls_signature = bls_private_key.sign(&message).unwrap();

    tracer.trace_value(samples, &public_key)?;
    tracer.trace_value(samples, &signature)?;
    tracer.trace_value(samples, &bls_public_key)?;
    tracer.trace_value(samples, &bls_signature)?;
    tracer.trace_value::<MultiEd25519PublicKey>(samples, &public_key.into())?;
    tracer.trace_value::<MultiEd25519Signature>(samples, &signature.into())?;
    Ok(())
}

/// Create a registry for consensus types.
pub fn get_registry() -> Result<Registry> {
    let mut tracer =
        Tracer::new(TracerConfig::default().is_human_readable(bcs::is_human_readable()));
    let mut samples = Samples::new();
    // 1. Record samples for types with custom deserializers.
    trace_crypto_values(&mut tracer, &mut samples)?;
    tracer.trace_value(
        &mut samples,
        &aptos_consensus_types::block::Block::make_genesis_block(),
    )?;
    tracer.trace_value(&mut samples, &event::EventKey::random())?;

    // 2. Trace the main entry point(s) + every enum separately.
    tracer.trace_type::<contract_event::ContractEvent>(&samples)?;
    tracer.trace_type::<language_storage::TypeTag>(&samples)?;
    tracer.trace_type::<transaction::Transaction>(&samples)?;
    tracer.trace_type::<transaction::TransactionArgument>(&samples)?;
    tracer.trace_type::<transaction::TransactionPayload>(&samples)?;
    tracer.trace_type::<transaction::WriteSetPayload>(&samples)?;
    tracer.trace_type::<transaction::authenticator::AccountAuthenticator>(&samples)?;
    tracer.trace_type::<transaction::authenticator::TransactionAuthenticator>(&samples)?;
    tracer.trace_type::<write_set::WriteOp>(&samples)?;

    tracer.trace_type::<StateKey>(&samples)?;
    tracer.trace_type::<aptos_consensus::network_interface::ConsensusMsg>(&samples)?;
    tracer.trace_type::<aptos_consensus_types::block_data::BlockType>(&samples)?;
    tracer.trace_type::<aptos_consensus_types::block_retrieval::BlockRetrievalStatus>(&samples)?;
    tracer.trace_type::<aptos_consensus_types::common::Payload>(&samples)?;

    tracer.registry()
}
