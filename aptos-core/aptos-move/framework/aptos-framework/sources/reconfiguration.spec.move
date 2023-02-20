spec aptos_framework::reconfiguration {
    spec module {
        pragma verify = true;
        pragma aborts_if_is_strict;

        // After genesis, `Configuration` exists.
        invariant [suspendable] chain_status::is_operating() ==> exists<Configuration>(@aptos_framework);
        invariant [suspendable] chain_status::is_operating() ==>
            (timestamp::spec_now_microseconds() >= last_reconfiguration_time());
    }

    /// Make sure the signer address is @aptos_framework.
    spec schema AbortsIfNotAptosFramework {
        aptos_framework: &signer;

        let addr = signer::address_of(aptos_framework);
        aborts_if !system_addresses::is_aptos_framework_address(addr);
    }

    /// Address @aptos_framework must exist resource Account and Configuration.
    /// Already exists in framework account.
    /// Guid_creation_num should be 2 according to logic.
    spec initialize(aptos_framework: &signer) {
        use std::signer;
        use aptos_framework::account::{Account};

        include AbortsIfNotAptosFramework;
        let addr = signer::address_of(aptos_framework);
        requires exists<Account>(addr);
        aborts_if !(global<Account>(addr).guid_creation_num == 2);
        aborts_if exists<Configuration>(@aptos_framework);
    }

    spec current_epoch(): u64 {
        aborts_if !exists<Configuration>(@aptos_framework);
    }

    spec disable_reconfiguration(aptos_framework: &signer) {
        include AbortsIfNotAptosFramework;
        aborts_if exists<DisableReconfiguration>(@aptos_framework);
    }

    /// Make sure the caller is admin and check the resource DisableReconfiguration.
    spec enable_reconfiguration(aptos_framework: &signer) {
        use aptos_framework::reconfiguration::{DisableReconfiguration};
        include AbortsIfNotAptosFramework;
        aborts_if !exists<DisableReconfiguration>(@aptos_framework);
    }

    /// When genesis_event emit the epoch and the `last_reconfiguration_time` .
    /// Should equal to 0
    spec emit_genesis_reconfiguration_event {
        use aptos_framework::reconfiguration::{Configuration};

        aborts_if !exists<Configuration>(@aptos_framework);
        let config_ref = global<Configuration>(@aptos_framework);
        aborts_if !(config_ref.epoch == 0 && config_ref.last_reconfiguration_time == 0);
    }

    spec last_reconfiguration_time {
        aborts_if !exists<Configuration>(@aptos_framework);
    }

    spec reconfigure {
        use aptos_framework::coin::CoinInfo;
        use aptos_framework::aptos_coin::AptosCoin;

        requires exists<stake::ValidatorFees>(@aptos_framework);
        requires exists<CoinInfo<AptosCoin>>(@aptos_framework);

        aborts_if false;
    }
}
