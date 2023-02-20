// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "testing")]
use crate::natives::util::make_test_only_native_from_func;
use crate::natives::{cryptography::ed25519::GasParameters, util::make_native_from_func};
#[cfg(feature = "testing")]
use aptos_crypto::ed25519::{Ed25519PrivateKey, Ed25519PublicKey};
#[cfg(feature = "testing")]
use aptos_crypto::test_utils::KeyPair;
use aptos_crypto::{
    ed25519::{ED25519_PUBLIC_KEY_LENGTH, ED25519_SIGNATURE_LENGTH},
    multi_ed25519,
    traits::*,
};
use curve25519_dalek::edwards::CompressedEdwardsY;
use move_binary_format::errors::PartialVMResult;
#[cfg(feature = "testing")]
use move_core_types::gas_algebra::InternalGas;
use move_core_types::gas_algebra::{InternalGasPerArg, NumArgs, NumBytes};
use move_vm_runtime::native_functions::{NativeContext, NativeFunction};
use move_vm_types::{
    loaded_data::runtime_types::Type, natives::function::NativeResult, pop_arg, values::Value,
};
#[cfg(feature = "testing")]
use rand_core::OsRng;
use smallvec::smallvec;
use std::{collections::VecDeque, convert::TryFrom};

/// DEPRECATED: See `public_key_validate_internal` comments in `multi_ed25519.move`.
///
/// Simply put, this function should have checked that `num_sub_pks > 0` and should abort PK
/// validation as soon as an invalid sub-PK is found, charging gas accordingly, rather than charge
/// gas for validating all `num_sub_pks` sub-PKs.
fn native_public_key_validate(
    gas_params: &GasParameters,
    _context: &mut NativeContext,
    _ty_args: Vec<Type>,
    mut arguments: VecDeque<Value>,
) -> PartialVMResult<NativeResult> {
    debug_assert!(_ty_args.is_empty());
    debug_assert!(arguments.len() == 1);

    let pks_bytes = pop_arg!(arguments, Vec<u8>);

    let num_sub_pks = pks_bytes.len() / ED25519_PUBLIC_KEY_LENGTH;

    let mut cost = gas_params.base;

    if num_sub_pks > multi_ed25519::MAX_NUM_OF_KEYS {
        return Ok(NativeResult::ok(cost, smallvec![Value::bool(false)]));
    };

    let num_valid = pks_bytes
        .chunks_exact(ED25519_PUBLIC_KEY_LENGTH)
        .filter(|&pk_bytes| {
            <[u8; ED25519_PUBLIC_KEY_LENGTH]>::try_from(pk_bytes)
                .ok()
                .and_then(|slice| CompressedEdwardsY(slice).decompress())
                .map_or(false, |point| !point.is_small_order())
        })
        .count();

    let all_valid = num_valid == num_sub_pks;
    let mut num_checked = num_valid;
    if !all_valid {
        num_checked += 1;
    }

    let num_checked = NumArgs::new(num_checked as u64);
    cost += gas_params.per_pubkey_deserialize * num_checked
        + gas_params.per_pubkey_small_order_check * num_checked;

    Ok(NativeResult::ok(cost, smallvec![Value::bool(all_valid)]))
}

/// See `public_key_validate_v2_internal` comments in `multi_ed25519.move`.
fn native_public_key_validate_v2(
    gas_params: &GasParameters,
    _context: &mut NativeContext,
    _ty_args: Vec<Type>,
    mut arguments: VecDeque<Value>,
) -> PartialVMResult<NativeResult> {
    debug_assert!(_ty_args.is_empty());
    debug_assert!(arguments.len() == 1);

    let pks_bytes = pop_arg!(arguments, Vec<u8>);

    let mut cost = gas_params.base;

    let (all_valid, num_deserializations, num_small_order_checks) =
        multi_ed25519::MultiEd25519PublicKey::validate_bytes_and_count_checks(pks_bytes.as_slice());

    cost += gas_params.per_pubkey_deserialize * NumArgs::new(num_deserializations as u64)
        + gas_params.per_pubkey_small_order_check * NumArgs::new(num_small_order_checks as u64);

    Ok(NativeResult::ok(cost, smallvec![Value::bool(all_valid)]))
}

fn native_signature_verify_strict(
    gas_params: &GasParameters,
    _context: &mut NativeContext,
    _ty_args: Vec<Type>,
    mut arguments: VecDeque<Value>,
) -> PartialVMResult<NativeResult> {
    debug_assert!(_ty_args.is_empty());
    debug_assert!(arguments.len() == 3);

    let msg = pop_arg!(arguments, Vec<u8>);
    let pubkey = pop_arg!(arguments, Vec<u8>);
    let signature = pop_arg!(arguments, Vec<u8>);

    let mut cost = gas_params.base;

    let num_sub_pks = NumArgs::new((pubkey.len() / ED25519_PUBLIC_KEY_LENGTH) as u64);
    let num_sub_sigs = NumArgs::new((signature.len() / ED25519_SIGNATURE_LENGTH) as u64);

    cost += gas_params.per_pubkey_deserialize * num_sub_pks;
    let pk = match multi_ed25519::MultiEd25519PublicKey::try_from(pubkey.as_slice()) {
        Ok(pk) => pk,
        Err(_) => {
            return Ok(NativeResult::ok(cost, smallvec![Value::bool(false)]));
        },
    };

    cost += gas_params.per_sig_deserialize * num_sub_sigs;
    let sig = match multi_ed25519::MultiEd25519Signature::try_from(signature.as_slice()) {
        Ok(sig) => sig,
        Err(_) => {
            return Ok(NativeResult::ok(cost, smallvec![Value::bool(false)]));
        },
    };

    // TODO(Gas): Have Victor improve type safety here
    cost += gas_params.per_sig_strict_verify * num_sub_sigs
        + gas_params.per_msg_hashing_base * num_sub_sigs
        + InternalGasPerArg::from(u64::from(
            gas_params.per_msg_byte_hashing * NumBytes::new(msg.len() as u64),
        )) * num_sub_sigs;

    let verify_result = sig.verify_arbitrary_msg(msg.as_slice(), &pk).is_ok();
    Ok(NativeResult::ok(cost, smallvec![Value::bool(
        verify_result
    )]))
}

#[cfg(feature = "testing")]
fn native_generate_keys(
    _context: &mut NativeContext,
    _ty_args: Vec<Type>,
    mut arguments: VecDeque<Value>,
) -> PartialVMResult<NativeResult> {
    let n = pop_arg!(arguments, u8);
    let threshold = pop_arg!(arguments, u8);
    let key_pairs: Vec<KeyPair<Ed25519PrivateKey, Ed25519PublicKey>> = (0..n)
        .map(|_i| KeyPair::<Ed25519PrivateKey, Ed25519PublicKey>::generate(&mut OsRng))
        .collect();
    let private_keys = key_pairs
        .iter()
        .map(|pair| pair.private_key.clone())
        .collect();
    let public_keys = key_pairs
        .iter()
        .map(|pair| pair.public_key.clone())
        .collect();
    let group_sk = multi_ed25519::MultiEd25519PrivateKey::new(private_keys, threshold).unwrap();
    let group_pk = multi_ed25519::MultiEd25519PublicKey::new(public_keys, threshold).unwrap();
    Ok(NativeResult::ok(InternalGas::zero(), smallvec![
        Value::vector_u8(group_sk.to_bytes()),
        Value::vector_u8(group_pk.to_bytes()),
    ]))
}

#[cfg(feature = "testing")]
fn native_sign(
    _context: &mut NativeContext,
    _ty_args: Vec<Type>,
    mut arguments: VecDeque<Value>,
) -> PartialVMResult<NativeResult> {
    let message = pop_arg!(arguments, Vec<u8>);
    let sk_bytes = pop_arg!(arguments, Vec<u8>);
    let group_sk = multi_ed25519::MultiEd25519PrivateKey::try_from(sk_bytes.as_slice()).unwrap();
    let sig = group_sk.sign_arbitrary_message(message.as_slice());
    Ok(NativeResult::ok(InternalGas::zero(), smallvec![
        Value::vector_u8(sig.to_bytes()),
    ]))
}
/***************************************************************************************************
 * module
 *
 **************************************************************************************************/

pub fn make_all(gas_params: GasParameters) -> impl Iterator<Item = (String, NativeFunction)> {
    let mut natives = vec![];
    natives.append(&mut vec![
        // MultiEd25519
        (
            "public_key_validate_internal",
            make_native_from_func(gas_params.clone(), native_public_key_validate),
        ),
        (
            "public_key_validate_v2_internal",
            make_native_from_func(gas_params.clone(), native_public_key_validate_v2),
        ),
        (
            "signature_verify_strict_internal",
            make_native_from_func(gas_params, native_signature_verify_strict),
        ),
    ]);
    #[cfg(feature = "testing")]
    natives.append(&mut vec![
        (
            "generate_keys_internal",
            make_test_only_native_from_func(native_generate_keys),
        ),
        (
            "sign_internal",
            make_test_only_native_from_func(native_sign),
        ),
    ]);
    crate::natives::helpers::make_module_natives(natives)
}
