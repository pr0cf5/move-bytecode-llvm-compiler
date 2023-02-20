// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use aptos_forge::{forge_main, ForgeConfig, InitialVersion, LocalFactory, Options, Result};
use aptos_testcases::compatibility_test::SimpleValidatorUpgrade;
use std::num::NonZeroUsize;

fn main() -> Result<()> {
    ::aptos_logger::Logger::init_for_testing();

    let tests = ForgeConfig::default()
        .with_initial_validator_count(NonZeroUsize::new(4).unwrap())
        .with_initial_version(InitialVersion::Oldest)
        .with_network_tests(vec![&SimpleValidatorUpgrade]);

    let options = Options::from_args();
    forge_main(
        tests,
        LocalFactory::with_upstream_merge_base_and_workspace()?,
        &options,
    )
}
