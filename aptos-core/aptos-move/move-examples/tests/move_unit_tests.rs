// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use aptos_gas::{AbstractValueSizeGasParameters, NativeGasParameters, LATEST_GAS_FEATURE_VERSION};
use aptos_types::account_address::{create_resource_address, AccountAddress};
use aptos_vm::natives;
use move_cli::base::test::{run_move_unit_tests, UnitTestResult};
use move_unit_test::UnitTestingConfig;
use move_vm_runtime::native_functions::NativeFunctionTable;
use std::{collections::BTreeMap, path::PathBuf};
use tempfile::tempdir;

pub fn path_in_crate<S>(relative: S) -> PathBuf
where
    S: Into<String>,
{
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(relative.into());
    path
}

pub fn run_tests_for_pkg(
    path_to_pkg: impl Into<String>,
    named_addr: BTreeMap<String, AccountAddress>,
) {
    let pkg_path = path_in_crate(path_to_pkg);
    let ok = run_move_unit_tests(
        &pkg_path,
        move_package::BuildConfig {
            test_mode: true,
            install_dir: Some(tempdir().unwrap().path().to_path_buf()),
            additional_named_addresses: named_addr,
            ..Default::default()
        },
        UnitTestingConfig::default_with_bound(Some(100_000)),
        // TODO(Gas): we may want to switch to non-zero costs in the future
        aptos_test_natives(),
        /* cost_table */ None,
        /* compute_coverage */ false,
        &mut std::io::stdout(),
    )
    .unwrap();
    if ok != UnitTestResult::Success {
        panic!("move unit tests failed")
    }
}

pub fn aptos_test_natives() -> NativeFunctionTable {
    natives::configure_for_unit_test();
    natives::aptos_natives(
        NativeGasParameters::zeros(),
        AbstractValueSizeGasParameters::zeros(),
        LATEST_GAS_FEATURE_VERSION,
    )
}

fn test_common(pkg: &str) {
    let named_address = BTreeMap::from([(
        String::from(pkg),
        AccountAddress::from_hex_literal("0xf00d").unwrap(),
    )]);
    run_tests_for_pkg(pkg, named_address);
}

fn test_resource_account_common(pkg: &str) {
    let named_address = BTreeMap::from([(
        String::from(pkg),
        create_resource_address(AccountAddress::from_hex_literal("0xcafe").unwrap(), &[]),
    )]);
    run_tests_for_pkg(pkg, named_address);
}

#[test]
fn test_data_structures() {
    test_common("data_structures");
}

#[test]
fn test_defi() {
    test_common("defi");
}

#[test]
fn test_hello_blockchain() {
    test_common("hello_blockchain");
}

#[test]
fn test_marketplace() {
    test_common("marketplace")
}

#[test]
fn test_message_board() {
    test_common("message_board");
}

#[test]
fn test_mint_nft() {
    let addr = AccountAddress::from_hex_literal("0xcafe").unwrap();
    let named_address = BTreeMap::from([
        (String::from("mint_nft"), create_resource_address(addr, &[])),
        (String::from("source_addr"), addr),
    ]);
    run_tests_for_pkg("mint_nft/4-Getting-Production-Ready", named_address);
}

#[test]
fn test_minter() {
    run_tests_for_pkg("scripts/minter", BTreeMap::new());
}

#[test]
fn test_resource_account() {
    test_resource_account_common("resource_account");
}

#[test]
fn test_shared_account() {
    test_common("shared_account");
}

#[test]
fn test_two_by_two_transfer() {
    run_tests_for_pkg("scripts/two_by_two_transfer", BTreeMap::new());
}

#[test]
fn test_post_mint_reveal_nft() {
    let addr = AccountAddress::from_hex_literal("0xcafe").unwrap();
    let named_address = BTreeMap::from([
        (
            String::from("post_mint_reveal_nft"),
            create_resource_address(addr, &[]),
        ),
        (String::from("source_addr"), addr),
    ]);
    run_tests_for_pkg("post_mint_reveal_nft", named_address);
}
