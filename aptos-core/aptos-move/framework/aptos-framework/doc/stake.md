
<a name="0x1_stake"></a>

# Module `0x1::stake`


* Validator lifecycle:
* 1. Prepare a validator node set up and call stake::initialize_validator
* 2. Once ready to deposit stake (or have funds assigned by a staking service in exchange for ownership capability),
* call stake::add_stake (or *_with_cap versions if called from the staking service)
* 3. Call stake::join_validator_set (or _with_cap version) to join the active validator set. Changes are effective in
* the next epoch.
* 4. Validate and gain rewards. The stake will automatically be locked up for a fixed duration (set by governance) and
* automatically renewed at expiration.
* 5. At any point, if the validator operator wants to update the consensus key or network/fullnode addresses, they can
* call stake::rotate_consensus_key and stake::update_network_and_fullnode_addresses. Similar to changes to stake, the
* changes to consensus key/network/fullnode addresses are only effective in the next epoch.
* 6. Validator can request to unlock their stake at any time. However, their stake will only become withdrawable when
* their current lockup expires. This can be at most as long as the fixed lockup duration.
* 7. After exiting, the validator can either explicitly leave the validator set by calling stake::leave_validator_set
* or if their stake drops below the min required, they would get removed at the end of the epoch.
* 8. Validator can always rejoin the validator set by going through steps 2-3 again.
* 9. An owner can always switch operators by calling stake::set_operator.
* 10. An owner can always switch designated voter by calling stake::set_designated_voter.


-  [Resource `OwnerCapability`](#0x1_stake_OwnerCapability)
-  [Resource `StakePool`](#0x1_stake_StakePool)
-  [Resource `ValidatorConfig`](#0x1_stake_ValidatorConfig)
-  [Struct `ValidatorInfo`](#0x1_stake_ValidatorInfo)
-  [Resource `ValidatorSet`](#0x1_stake_ValidatorSet)
-  [Resource `AptosCoinCapabilities`](#0x1_stake_AptosCoinCapabilities)
-  [Struct `IndividualValidatorPerformance`](#0x1_stake_IndividualValidatorPerformance)
-  [Resource `ValidatorPerformance`](#0x1_stake_ValidatorPerformance)
-  [Struct `RegisterValidatorCandidateEvent`](#0x1_stake_RegisterValidatorCandidateEvent)
-  [Struct `SetOperatorEvent`](#0x1_stake_SetOperatorEvent)
-  [Struct `AddStakeEvent`](#0x1_stake_AddStakeEvent)
-  [Struct `ReactivateStakeEvent`](#0x1_stake_ReactivateStakeEvent)
-  [Struct `RotateConsensusKeyEvent`](#0x1_stake_RotateConsensusKeyEvent)
-  [Struct `UpdateNetworkAndFullnodeAddressesEvent`](#0x1_stake_UpdateNetworkAndFullnodeAddressesEvent)
-  [Struct `IncreaseLockupEvent`](#0x1_stake_IncreaseLockupEvent)
-  [Struct `JoinValidatorSetEvent`](#0x1_stake_JoinValidatorSetEvent)
-  [Struct `DistributeRewardsEvent`](#0x1_stake_DistributeRewardsEvent)
-  [Struct `UnlockStakeEvent`](#0x1_stake_UnlockStakeEvent)
-  [Struct `WithdrawStakeEvent`](#0x1_stake_WithdrawStakeEvent)
-  [Struct `LeaveValidatorSetEvent`](#0x1_stake_LeaveValidatorSetEvent)
-  [Resource `ValidatorFees`](#0x1_stake_ValidatorFees)
-  [Resource `AllowedValidators`](#0x1_stake_AllowedValidators)
-  [Constants](#@Constants_0)
-  [Function `initialize_validator_fees`](#0x1_stake_initialize_validator_fees)
-  [Function `add_transaction_fee`](#0x1_stake_add_transaction_fee)
-  [Function `get_lockup_secs`](#0x1_stake_get_lockup_secs)
-  [Function `get_remaining_lockup_secs`](#0x1_stake_get_remaining_lockup_secs)
-  [Function `get_stake`](#0x1_stake_get_stake)
-  [Function `get_validator_state`](#0x1_stake_get_validator_state)
-  [Function `get_current_epoch_voting_power`](#0x1_stake_get_current_epoch_voting_power)
-  [Function `get_delegated_voter`](#0x1_stake_get_delegated_voter)
-  [Function `get_operator`](#0x1_stake_get_operator)
-  [Function `get_owned_pool_address`](#0x1_stake_get_owned_pool_address)
-  [Function `get_validator_index`](#0x1_stake_get_validator_index)
-  [Function `get_current_epoch_proposal_counts`](#0x1_stake_get_current_epoch_proposal_counts)
-  [Function `get_validator_config`](#0x1_stake_get_validator_config)
-  [Function `stake_pool_exists`](#0x1_stake_stake_pool_exists)
-  [Function `initialize`](#0x1_stake_initialize)
-  [Function `store_aptos_coin_mint_cap`](#0x1_stake_store_aptos_coin_mint_cap)
-  [Function `remove_validators`](#0x1_stake_remove_validators)
-  [Function `initialize_stake_owner`](#0x1_stake_initialize_stake_owner)
-  [Function `initialize_validator`](#0x1_stake_initialize_validator)
-  [Function `initialize_owner`](#0x1_stake_initialize_owner)
-  [Function `extract_owner_cap`](#0x1_stake_extract_owner_cap)
-  [Function `deposit_owner_cap`](#0x1_stake_deposit_owner_cap)
-  [Function `destroy_owner_cap`](#0x1_stake_destroy_owner_cap)
-  [Function `set_operator`](#0x1_stake_set_operator)
-  [Function `set_operator_with_cap`](#0x1_stake_set_operator_with_cap)
-  [Function `set_delegated_voter`](#0x1_stake_set_delegated_voter)
-  [Function `set_delegated_voter_with_cap`](#0x1_stake_set_delegated_voter_with_cap)
-  [Function `add_stake`](#0x1_stake_add_stake)
-  [Function `add_stake_with_cap`](#0x1_stake_add_stake_with_cap)
-  [Function `reactivate_stake`](#0x1_stake_reactivate_stake)
-  [Function `reactivate_stake_with_cap`](#0x1_stake_reactivate_stake_with_cap)
-  [Function `rotate_consensus_key`](#0x1_stake_rotate_consensus_key)
-  [Function `update_network_and_fullnode_addresses`](#0x1_stake_update_network_and_fullnode_addresses)
-  [Function `increase_lockup`](#0x1_stake_increase_lockup)
-  [Function `increase_lockup_with_cap`](#0x1_stake_increase_lockup_with_cap)
-  [Function `join_validator_set`](#0x1_stake_join_validator_set)
-  [Function `join_validator_set_internal`](#0x1_stake_join_validator_set_internal)
-  [Function `unlock`](#0x1_stake_unlock)
-  [Function `unlock_with_cap`](#0x1_stake_unlock_with_cap)
-  [Function `withdraw`](#0x1_stake_withdraw)
-  [Function `withdraw_with_cap`](#0x1_stake_withdraw_with_cap)
-  [Function `leave_validator_set`](#0x1_stake_leave_validator_set)
-  [Function `is_current_epoch_validator`](#0x1_stake_is_current_epoch_validator)
-  [Function `update_performance_statistics`](#0x1_stake_update_performance_statistics)
-  [Function `on_new_epoch`](#0x1_stake_on_new_epoch)
-  [Function `update_stake_pool`](#0x1_stake_update_stake_pool)
-  [Function `calculate_rewards_amount`](#0x1_stake_calculate_rewards_amount)
-  [Function `distribute_rewards`](#0x1_stake_distribute_rewards)
-  [Function `append`](#0x1_stake_append)
-  [Function `find_validator`](#0x1_stake_find_validator)
-  [Function `generate_validator_info`](#0x1_stake_generate_validator_info)
-  [Function `get_next_epoch_voting_power`](#0x1_stake_get_next_epoch_voting_power)
-  [Function `update_voting_power_increase`](#0x1_stake_update_voting_power_increase)
-  [Function `assert_stake_pool_exists`](#0x1_stake_assert_stake_pool_exists)
-  [Function `configure_allowed_validators`](#0x1_stake_configure_allowed_validators)
-  [Function `is_allowed`](#0x1_stake_is_allowed)
-  [Function `assert_owner_cap_exists`](#0x1_stake_assert_owner_cap_exists)
-  [Specification](#@Specification_1)
    -  [Function `get_validator_state`](#@Specification_1_get_validator_state)
    -  [Function `initialize`](#@Specification_1_initialize)
    -  [Function `remove_validators`](#@Specification_1_remove_validators)
    -  [Function `initialize_stake_owner`](#@Specification_1_initialize_stake_owner)
    -  [Function `extract_owner_cap`](#@Specification_1_extract_owner_cap)
    -  [Function `deposit_owner_cap`](#@Specification_1_deposit_owner_cap)
    -  [Function `set_operator_with_cap`](#@Specification_1_set_operator_with_cap)
    -  [Function `set_delegated_voter_with_cap`](#@Specification_1_set_delegated_voter_with_cap)
    -  [Function `add_stake`](#@Specification_1_add_stake)
    -  [Function `add_stake_with_cap`](#@Specification_1_add_stake_with_cap)
    -  [Function `reactivate_stake_with_cap`](#@Specification_1_reactivate_stake_with_cap)
    -  [Function `rotate_consensus_key`](#@Specification_1_rotate_consensus_key)
    -  [Function `update_network_and_fullnode_addresses`](#@Specification_1_update_network_and_fullnode_addresses)
    -  [Function `increase_lockup_with_cap`](#@Specification_1_increase_lockup_with_cap)
    -  [Function `unlock_with_cap`](#@Specification_1_unlock_with_cap)
    -  [Function `is_current_epoch_validator`](#@Specification_1_is_current_epoch_validator)
    -  [Function `update_performance_statistics`](#@Specification_1_update_performance_statistics)
    -  [Function `on_new_epoch`](#@Specification_1_on_new_epoch)
    -  [Function `update_stake_pool`](#@Specification_1_update_stake_pool)
    -  [Function `calculate_rewards_amount`](#@Specification_1_calculate_rewards_amount)
    -  [Function `distribute_rewards`](#@Specification_1_distribute_rewards)
    -  [Function `append`](#@Specification_1_append)
    -  [Function `find_validator`](#@Specification_1_find_validator)
    -  [Function `update_voting_power_increase`](#@Specification_1_update_voting_power_increase)
    -  [Function `assert_stake_pool_exists`](#@Specification_1_assert_stake_pool_exists)
    -  [Function `configure_allowed_validators`](#@Specification_1_configure_allowed_validators)
    -  [Function `assert_owner_cap_exists`](#@Specification_1_assert_owner_cap_exists)


<pre><code><b>use</b> <a href="account.md#0x1_account">0x1::account</a>;
<b>use</b> <a href="aptos_coin.md#0x1_aptos_coin">0x1::aptos_coin</a>;
<b>use</b> <a href="../../aptos-stdlib/doc/bls12381.md#0x1_bls12381">0x1::bls12381</a>;
<b>use</b> <a href="coin.md#0x1_coin">0x1::coin</a>;
<b>use</b> <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error">0x1::error</a>;
<b>use</b> <a href="event.md#0x1_event">0x1::event</a>;
<b>use</b> <a href="../../aptos-stdlib/../move-stdlib/doc/features.md#0x1_features">0x1::features</a>;
<b>use</b> <a href="../../aptos-stdlib/doc/math64.md#0x1_math64">0x1::math64</a>;
<b>use</b> <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option">0x1::option</a>;
<b>use</b> <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">0x1::signer</a>;
<b>use</b> <a href="staking_config.md#0x1_staking_config">0x1::staking_config</a>;
<b>use</b> <a href="system_addresses.md#0x1_system_addresses">0x1::system_addresses</a>;
<b>use</b> <a href="../../aptos-stdlib/doc/table.md#0x1_table">0x1::table</a>;
<b>use</b> <a href="timestamp.md#0x1_timestamp">0x1::timestamp</a>;
<b>use</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">0x1::vector</a>;
</code></pre>



<a name="0x1_stake_OwnerCapability"></a>

## Resource `OwnerCapability`

Capability that represents ownership and can be used to control the validator and the associated stake pool.
Having this be separate from the signer for the account that the validator resources are hosted at allows
modules to have control over a validator.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a> <b>has</b> store, key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_StakePool"></a>

## Resource `StakePool`

Each validator has a separate StakePool resource and can provide a stake.
Changes in stake for an active validator:
1. If a validator calls add_stake, the newly added stake is moved to pending_active.
2. If validator calls unlock, their stake is moved to pending_inactive.
2. When the next epoch starts, any pending_inactive stake is moved to inactive and can be withdrawn.
Any pending_active stake is moved to active and adds to the validator's voting power.

Changes in stake for an inactive validator:
1. If a validator calls add_stake, the newly added stake is moved directly to active.
2. If validator calls unlock, their stake is moved directly to inactive.
3. When the next epoch starts, the validator can be activated if their active stake is more than the minimum.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>active: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>inactive: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>pending_active: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>pending_inactive: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>locked_until_secs: u64</code>
</dt>
<dd>

</dd>
<dt>
<code>operator_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>delegated_voter: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>initialize_validator_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_RegisterValidatorCandidateEvent">stake::RegisterValidatorCandidateEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>set_operator_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_SetOperatorEvent">stake::SetOperatorEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>add_stake_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_AddStakeEvent">stake::AddStakeEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>reactivate_stake_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_ReactivateStakeEvent">stake::ReactivateStakeEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>rotate_consensus_key_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_RotateConsensusKeyEvent">stake::RotateConsensusKeyEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>update_network_and_fullnode_addresses_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_UpdateNetworkAndFullnodeAddressesEvent">stake::UpdateNetworkAndFullnodeAddressesEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>increase_lockup_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_IncreaseLockupEvent">stake::IncreaseLockupEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>join_validator_set_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_JoinValidatorSetEvent">stake::JoinValidatorSetEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>distribute_rewards_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_DistributeRewardsEvent">stake::DistributeRewardsEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>unlock_stake_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_UnlockStakeEvent">stake::UnlockStakeEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>withdraw_stake_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_WithdrawStakeEvent">stake::WithdrawStakeEvent</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>leave_validator_set_events: <a href="event.md#0x1_event_EventHandle">event::EventHandle</a>&lt;<a href="stake.md#0x1_stake_LeaveValidatorSetEvent">stake::LeaveValidatorSetEvent</a>&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ValidatorConfig"></a>

## Resource `ValidatorConfig`

Validator info stored in validator address.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> <b>has</b> <b>copy</b>, drop, store, key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>validator_index: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ValidatorInfo"></a>

## Struct `ValidatorInfo`

Consensus information per validator, stored in ValidatorSet.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a> <b>has</b> <b>copy</b>, drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>addr: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>voting_power: u64</code>
</dt>
<dd>

</dd>
<dt>
<code>config: <a href="stake.md#0x1_stake_ValidatorConfig">stake::ValidatorConfig</a></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ValidatorSet"></a>

## Resource `ValidatorSet`

Full ValidatorSet, stored in @aptos_framework.
1. join_validator_set adds to pending_active queue.
2. leave_valdiator_set moves from active to pending_inactive queue.
3. on_new_epoch processes two pending queues and refresh ValidatorInfo from the owner's address.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>consensus_scheme: u8</code>
</dt>
<dd>

</dd>
<dt>
<code>active_validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>pending_inactive: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>pending_active: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>total_voting_power: u128</code>
</dt>
<dd>

</dd>
<dt>
<code>total_joining_power: u128</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_AptosCoinCapabilities"></a>

## Resource `AptosCoinCapabilities`

AptosCoin capabilities, set during genesis and stored in @CoreResource account.
This allows the Stake module to mint rewards to stakers.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>mint_cap: <a href="coin.md#0x1_coin_MintCapability">coin::MintCapability</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_IndividualValidatorPerformance"></a>

## Struct `IndividualValidatorPerformance`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_IndividualValidatorPerformance">IndividualValidatorPerformance</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>successful_proposals: u64</code>
</dt>
<dd>

</dd>
<dt>
<code>failed_proposals: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ValidatorPerformance"></a>

## Resource `ValidatorPerformance`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_IndividualValidatorPerformance">stake::IndividualValidatorPerformance</a>&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_RegisterValidatorCandidateEvent"></a>

## Struct `RegisterValidatorCandidateEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_RegisterValidatorCandidateEvent">RegisterValidatorCandidateEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_SetOperatorEvent"></a>

## Struct `SetOperatorEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_SetOperatorEvent">SetOperatorEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>old_operator: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>new_operator: <b>address</b></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_AddStakeEvent"></a>

## Struct `AddStakeEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_AddStakeEvent">AddStakeEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>amount_added: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ReactivateStakeEvent"></a>

## Struct `ReactivateStakeEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ReactivateStakeEvent">ReactivateStakeEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>amount: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_RotateConsensusKeyEvent"></a>

## Struct `RotateConsensusKeyEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_RotateConsensusKeyEvent">RotateConsensusKeyEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>old_consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>new_consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_UpdateNetworkAndFullnodeAddressesEvent"></a>

## Struct `UpdateNetworkAndFullnodeAddressesEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_UpdateNetworkAndFullnodeAddressesEvent">UpdateNetworkAndFullnodeAddressesEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>old_network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>new_network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>old_fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
<dt>
<code>new_fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_IncreaseLockupEvent"></a>

## Struct `IncreaseLockupEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_IncreaseLockupEvent">IncreaseLockupEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>old_locked_until_secs: u64</code>
</dt>
<dd>

</dd>
<dt>
<code>new_locked_until_secs: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_JoinValidatorSetEvent"></a>

## Struct `JoinValidatorSetEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_JoinValidatorSetEvent">JoinValidatorSetEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_DistributeRewardsEvent"></a>

## Struct `DistributeRewardsEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_DistributeRewardsEvent">DistributeRewardsEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>rewards_amount: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_UnlockStakeEvent"></a>

## Struct `UnlockStakeEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_UnlockStakeEvent">UnlockStakeEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>amount_unlocked: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_WithdrawStakeEvent"></a>

## Struct `WithdrawStakeEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_WithdrawStakeEvent">WithdrawStakeEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
<dt>
<code>amount_withdrawn: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_LeaveValidatorSetEvent"></a>

## Struct `LeaveValidatorSetEvent`



<pre><code><b>struct</b> <a href="stake.md#0x1_stake_LeaveValidatorSetEvent">LeaveValidatorSetEvent</a> <b>has</b> drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>pool_address: <b>address</b></code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_ValidatorFees"></a>

## Resource `ValidatorFees`

Stores transaction fees assigned to validators. All fees are distributed to validators
at the end of the epoch.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>fees_table: <a href="../../aptos-stdlib/doc/table.md#0x1_table_Table">table::Table</a>&lt;<b>address</b>, <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_stake_AllowedValidators"></a>

## Resource `AllowedValidators`

This provides an ACL for Testnet purposes. In testnet, everyone is a whale, a whale can be a validator.
This allows a testnet to bring additional entities into the validator set without compromising the
security of the testnet. This will NOT be enabled in Mainnet.


<pre><code><b>struct</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> <b>has</b> key
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>accounts: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="@Constants_0"></a>

## Constants


<a name="0x1_stake_MAX_REWARDS_RATE"></a>

Limit the maximum value of <code>rewards_rate</code> in order to avoid any arithmetic overflow.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_MAX_REWARDS_RATE">MAX_REWARDS_RATE</a>: u64 = 1000000;
</code></pre>



<a name="0x1_stake_EALREADY_ACTIVE_VALIDATOR"></a>

Account is already a validator or pending validator.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EALREADY_ACTIVE_VALIDATOR">EALREADY_ACTIVE_VALIDATOR</a>: u64 = 4;
</code></pre>



<a name="0x1_stake_EALREADY_REGISTERED"></a>

Account is already registered as a validator candidate.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EALREADY_REGISTERED">EALREADY_REGISTERED</a>: u64 = 8;
</code></pre>



<a name="0x1_stake_EFEES_TABLE_ALREADY_EXISTS"></a>

Table to store collected transaction fees for each validator already exists.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EFEES_TABLE_ALREADY_EXISTS">EFEES_TABLE_ALREADY_EXISTS</a>: u64 = 19;
</code></pre>



<a name="0x1_stake_EINELIGIBLE_VALIDATOR"></a>

Validator is not defined in the ACL of entities allowed to be validators


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EINELIGIBLE_VALIDATOR">EINELIGIBLE_VALIDATOR</a>: u64 = 17;
</code></pre>



<a name="0x1_stake_EINVALID_LOCKUP"></a>

Cannot update stake pool's lockup to earlier than current lockup.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EINVALID_LOCKUP">EINVALID_LOCKUP</a>: u64 = 18;
</code></pre>



<a name="0x1_stake_EINVALID_PUBLIC_KEY"></a>

Invalid consensus public key


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EINVALID_PUBLIC_KEY">EINVALID_PUBLIC_KEY</a>: u64 = 11;
</code></pre>



<a name="0x1_stake_ELAST_VALIDATOR"></a>

Can't remove last validator.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ELAST_VALIDATOR">ELAST_VALIDATOR</a>: u64 = 6;
</code></pre>



<a name="0x1_stake_ENOT_OPERATOR"></a>

Account does not have the right operator capability.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ENOT_OPERATOR">ENOT_OPERATOR</a>: u64 = 9;
</code></pre>



<a name="0x1_stake_ENOT_VALIDATOR"></a>

Account is not a validator.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ENOT_VALIDATOR">ENOT_VALIDATOR</a>: u64 = 5;
</code></pre>



<a name="0x1_stake_ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED"></a>

Validators cannot join or leave post genesis on this test network.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED">ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED</a>: u64 = 10;
</code></pre>



<a name="0x1_stake_EOWNER_CAP_ALREADY_EXISTS"></a>

An account cannot own more than one owner capability.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EOWNER_CAP_ALREADY_EXISTS">EOWNER_CAP_ALREADY_EXISTS</a>: u64 = 16;
</code></pre>



<a name="0x1_stake_EOWNER_CAP_NOT_FOUND"></a>

Owner capability does not exist at the provided account.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EOWNER_CAP_NOT_FOUND">EOWNER_CAP_NOT_FOUND</a>: u64 = 15;
</code></pre>



<a name="0x1_stake_ESTAKE_EXCEEDS_MAX"></a>

Total stake exceeds maximum allowed.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ESTAKE_EXCEEDS_MAX">ESTAKE_EXCEEDS_MAX</a>: u64 = 7;
</code></pre>



<a name="0x1_stake_ESTAKE_POOL_DOES_NOT_EXIST"></a>

Stake pool does not exist at the provided pool address.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ESTAKE_POOL_DOES_NOT_EXIST">ESTAKE_POOL_DOES_NOT_EXIST</a>: u64 = 14;
</code></pre>



<a name="0x1_stake_ESTAKE_TOO_HIGH"></a>

Too much stake to join validator set.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ESTAKE_TOO_HIGH">ESTAKE_TOO_HIGH</a>: u64 = 3;
</code></pre>



<a name="0x1_stake_ESTAKE_TOO_LOW"></a>

Not enough stake to join validator set.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_ESTAKE_TOO_LOW">ESTAKE_TOO_LOW</a>: u64 = 2;
</code></pre>



<a name="0x1_stake_EVALIDATOR_CONFIG"></a>

Validator Config not published.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EVALIDATOR_CONFIG">EVALIDATOR_CONFIG</a>: u64 = 1;
</code></pre>



<a name="0x1_stake_EVALIDATOR_SET_TOO_LARGE"></a>

Validator set exceeds the limit


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EVALIDATOR_SET_TOO_LARGE">EVALIDATOR_SET_TOO_LARGE</a>: u64 = 12;
</code></pre>



<a name="0x1_stake_EVOTING_POWER_INCREASE_EXCEEDS_LIMIT"></a>

Voting power increase has exceeded the limit for this current epoch.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_EVOTING_POWER_INCREASE_EXCEEDS_LIMIT">EVOTING_POWER_INCREASE_EXCEEDS_LIMIT</a>: u64 = 13;
</code></pre>



<a name="0x1_stake_MAX_VALIDATOR_SET_SIZE"></a>

Limit the maximum size to u16::max, it's the current limit of the bitvec
https://github.com/aptos-labs/aptos-core/blob/main/crates/aptos-bitvec/src/lib.rs#L20


<pre><code><b>const</b> <a href="stake.md#0x1_stake_MAX_VALIDATOR_SET_SIZE">MAX_VALIDATOR_SET_SIZE</a>: u64 = 65536;
</code></pre>



<a name="0x1_stake_VALIDATOR_STATUS_ACTIVE"></a>



<pre><code><b>const</b> <a href="stake.md#0x1_stake_VALIDATOR_STATUS_ACTIVE">VALIDATOR_STATUS_ACTIVE</a>: u64 = 2;
</code></pre>



<a name="0x1_stake_VALIDATOR_STATUS_INACTIVE"></a>



<pre><code><b>const</b> <a href="stake.md#0x1_stake_VALIDATOR_STATUS_INACTIVE">VALIDATOR_STATUS_INACTIVE</a>: u64 = 4;
</code></pre>



<a name="0x1_stake_VALIDATOR_STATUS_PENDING_ACTIVE"></a>

Validator status enum. We can switch to proper enum later once Move supports it.


<pre><code><b>const</b> <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_ACTIVE">VALIDATOR_STATUS_PENDING_ACTIVE</a>: u64 = 1;
</code></pre>



<a name="0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE"></a>



<pre><code><b>const</b> <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE">VALIDATOR_STATUS_PENDING_INACTIVE</a>: u64 = 3;
</code></pre>



<a name="0x1_stake_initialize_validator_fees"></a>

## Function `initialize_validator_fees`

Initializes the resource storing information about collected transaction fees per validator.
Used by <code><a href="transaction_fee.md#0x1_transaction_fee">transaction_fee</a>.<b>move</b></code> to initialize fee collection and distribution.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_initialize_validator_fees">initialize_validator_fees</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_initialize_validator_fees">initialize_validator_fees</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>) {
    <a href="system_addresses.md#0x1_system_addresses_assert_aptos_framework">system_addresses::assert_aptos_framework</a>(aptos_framework);
    <b>assert</b>!(
        !<b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a>&gt;(@aptos_framework),
        <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_already_exists">error::already_exists</a>(<a href="stake.md#0x1_stake_EFEES_TABLE_ALREADY_EXISTS">EFEES_TABLE_ALREADY_EXISTS</a>)
    );
    <b>move_to</b>(aptos_framework, <a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a> { fees_table: <a href="../../aptos-stdlib/doc/table.md#0x1_table_new">table::new</a>() });
}
</code></pre>



</details>

<a name="0x1_stake_add_transaction_fee"></a>

## Function `add_transaction_fee`

Stores the transaction fee collected to the specified validator address.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_add_transaction_fee">add_transaction_fee</a>(validator_addr: <b>address</b>, fee: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_add_transaction_fee">add_transaction_fee</a>(validator_addr: <b>address</b>, fee: Coin&lt;AptosCoin&gt;) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a> {
    <b>let</b> fees_table = &<b>mut</b> <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a>&gt;(@aptos_framework).fees_table;
    <b>if</b> (<a href="../../aptos-stdlib/doc/table.md#0x1_table_contains">table::contains</a>(fees_table, validator_addr)) {
        <b>let</b> collected_fee = <a href="../../aptos-stdlib/doc/table.md#0x1_table_borrow_mut">table::borrow_mut</a>(fees_table, validator_addr);
        <a href="coin.md#0x1_coin_merge">coin::merge</a>(collected_fee, fee);
    } <b>else</b> {
        <a href="../../aptos-stdlib/doc/table.md#0x1_table_add">table::add</a>(fees_table, validator_addr, fee);
    }
}
</code></pre>



</details>

<a name="0x1_stake_get_lockup_secs"></a>

## Function `get_lockup_secs`

Return the lockup expiration of the stake pool at <code>pool_address</code>.
This will throw an error if there's no stake pool at <code>pool_address</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_lockup_secs">get_lockup_secs</a>(pool_address: <b>address</b>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_lockup_secs">get_lockup_secs</a>(pool_address: <b>address</b>): u64 <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).locked_until_secs
}
</code></pre>



</details>

<a name="0x1_stake_get_remaining_lockup_secs"></a>

## Function `get_remaining_lockup_secs`

Return the remaining lockup of the stake pool at <code>pool_address</code>.
This will throw an error if there's no stake pool at <code>pool_address</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_remaining_lockup_secs">get_remaining_lockup_secs</a>(pool_address: <b>address</b>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_remaining_lockup_secs">get_remaining_lockup_secs</a>(pool_address: <b>address</b>): u64 <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> lockup_time = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).locked_until_secs;
    <b>if</b> (lockup_time &lt;= <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>()) {
        0
    } <b>else</b> {
        lockup_time - <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>()
    }
}
</code></pre>



</details>

<a name="0x1_stake_get_stake"></a>

## Function `get_stake`

Return the different stake amounts for <code>pool_address</code> (whether the validator is active or not).
The returned amounts are for (active, inactive, pending_active, pending_inactive) stake respectively.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_stake">get_stake</a>(pool_address: <b>address</b>): (u64, u64, u64, u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_stake">get_stake</a>(pool_address: <b>address</b>): (u64, u64, u64, u64) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    (
        <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.active),
        <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.inactive),
        <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.pending_active),
        <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.pending_inactive),
    )
}
</code></pre>



</details>

<a name="0x1_stake_get_validator_state"></a>

## Function `get_validator_state`

Returns the validator's state.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address: <b>address</b>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address: <b>address</b>): u64 <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> validator_set = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&<a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.pending_active, pool_address))) {
        <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_ACTIVE">VALIDATOR_STATUS_PENDING_ACTIVE</a>
    } <b>else</b> <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&<a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.active_validators, pool_address))) {
        <a href="stake.md#0x1_stake_VALIDATOR_STATUS_ACTIVE">VALIDATOR_STATUS_ACTIVE</a>
    } <b>else</b> <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&<a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.pending_inactive, pool_address))) {
        <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE">VALIDATOR_STATUS_PENDING_INACTIVE</a>
    } <b>else</b> {
        <a href="stake.md#0x1_stake_VALIDATOR_STATUS_INACTIVE">VALIDATOR_STATUS_INACTIVE</a>
    }
}
</code></pre>



</details>

<a name="0x1_stake_get_current_epoch_voting_power"></a>

## Function `get_current_epoch_voting_power`

Return the voting power of the validator in the current epoch.
This is the same as the validator's total active and pending_inactive stake.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_current_epoch_voting_power">get_current_epoch_voting_power</a>(pool_address: <b>address</b>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_current_epoch_voting_power">get_current_epoch_voting_power</a>(pool_address: <b>address</b>): u64 <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> validator_state = <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address);
    // Both active and pending inactive validators can still vote in the current epoch.
    <b>if</b> (validator_state == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_ACTIVE">VALIDATOR_STATUS_ACTIVE</a> || validator_state == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE">VALIDATOR_STATUS_PENDING_INACTIVE</a>) {
        <b>let</b> active_stake = <a href="coin.md#0x1_coin_value">coin::value</a>(&<b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).active);
        <b>let</b> pending_inactive_stake = <a href="coin.md#0x1_coin_value">coin::value</a>(&<b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).pending_inactive);
        active_stake + pending_inactive_stake
    } <b>else</b> {
        0
    }
}
</code></pre>



</details>

<a name="0x1_stake_get_delegated_voter"></a>

## Function `get_delegated_voter`

Return the delegated voter of the validator at <code>pool_address</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_delegated_voter">get_delegated_voter</a>(pool_address: <b>address</b>): <b>address</b>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_delegated_voter">get_delegated_voter</a>(pool_address: <b>address</b>): <b>address</b> <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).delegated_voter
}
</code></pre>



</details>

<a name="0x1_stake_get_operator"></a>

## Function `get_operator`

Return the operator of the validator at <code>pool_address</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_operator">get_operator</a>(pool_address: <b>address</b>): <b>address</b>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_operator">get_operator</a>(pool_address: <b>address</b>): <b>address</b> <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address).operator_address
}
</code></pre>



</details>

<a name="0x1_stake_get_owned_pool_address"></a>

## Function `get_owned_pool_address`

Return the pool address in <code>owner_cap</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_owned_pool_address">get_owned_pool_address</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>): <b>address</b>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_owned_pool_address">get_owned_pool_address</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>): <b>address</b> {
    owner_cap.pool_address
}
</code></pre>



</details>

<a name="0x1_stake_get_validator_index"></a>

## Function `get_validator_index`

Return the validator index for <code>pool_address</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_index">get_validator_index</a>(pool_address: <b>address</b>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_index">get_validator_index</a>(pool_address: <b>address</b>): u64 <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address).validator_index
}
</code></pre>



</details>

<a name="0x1_stake_get_current_epoch_proposal_counts"></a>

## Function `get_current_epoch_proposal_counts`

Return the number of successful and failed proposals for the proposal at the given validator index.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_current_epoch_proposal_counts">get_current_epoch_proposal_counts</a>(validator_index: u64): (u64, u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_current_epoch_proposal_counts">get_current_epoch_proposal_counts</a>(validator_index: u64): (u64, u64) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a> {
    <b>let</b> validator_performances = &<b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(@aptos_framework).validators;
    <b>let</b> validator_performance = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(validator_performances, validator_index);
    (validator_performance.successful_proposals, validator_performance.failed_proposals)
}
</code></pre>



</details>

<a name="0x1_stake_get_validator_config"></a>

## Function `get_validator_config`

Return the validator's config.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_config">get_validator_config</a>(pool_address: <b>address</b>): (<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_config">get_validator_config</a>(pool_address: <b>address</b>): (<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> validator_config = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
    (validator_config.consensus_pubkey, validator_config.network_addresses, validator_config.fullnode_addresses)
}
</code></pre>



</details>

<a name="0x1_stake_stake_pool_exists"></a>

## Function `stake_pool_exists`



<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(addr: <b>address</b>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(addr: <b>address</b>): bool {
    <b>exists</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(addr)
}
</code></pre>



</details>

<a name="0x1_stake_initialize"></a>

## Function `initialize`

Initialize validator set to the core resource account.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_initialize">initialize</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_initialize">initialize</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>) {
    <a href="system_addresses.md#0x1_system_addresses_assert_aptos_framework">system_addresses::assert_aptos_framework</a>(aptos_framework);

    <b>move_to</b>(aptos_framework, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
        consensus_scheme: 0,
        active_validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        pending_active: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        pending_inactive: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        total_voting_power: 0,
        total_joining_power: 0,
    });

    <b>move_to</b>(aptos_framework, <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a> {
        validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
    });
}
</code></pre>



</details>

<a name="0x1_stake_store_aptos_coin_mint_cap"></a>

## Function `store_aptos_coin_mint_cap`

This is only called during Genesis, which is where MintCapability<AptosCoin> can be created.
Beyond genesis, no one can create AptosCoin mint/burn capabilities.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_store_aptos_coin_mint_cap">store_aptos_coin_mint_cap</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, mint_cap: <a href="coin.md#0x1_coin_MintCapability">coin::MintCapability</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_store_aptos_coin_mint_cap">store_aptos_coin_mint_cap</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, mint_cap: MintCapability&lt;AptosCoin&gt;) {
    <a href="system_addresses.md#0x1_system_addresses_assert_aptos_framework">system_addresses::assert_aptos_framework</a>(aptos_framework);
    <b>move_to</b>(aptos_framework, <a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a> { mint_cap })
}
</code></pre>



</details>

<a name="0x1_stake_remove_validators"></a>

## Function `remove_validators`

Allow on chain governance to remove validators from the validator set.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_remove_validators">remove_validators</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, validators: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_remove_validators">remove_validators</a>(
    aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    validators: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;,
) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <a href="system_addresses.md#0x1_system_addresses_assert_aptos_framework">system_addresses::assert_aptos_framework</a>(aptos_framework);

    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    <b>let</b> active_validators = &<b>mut</b> validator_set.active_validators;
    <b>let</b> pending_inactive = &<b>mut</b> validator_set.pending_inactive;
    <b>let</b> len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(validators);
    <b>let</b> i = 0;
    // Remove each validator from the validator set.
    <b>while</b> ({
        <b>spec</b> {
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(active_validators);
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validator_indices_are_valid">spec_validator_indices_are_valid</a>(active_validators);
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(pending_inactive);
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validator_indices_are_valid">spec_validator_indices_are_valid</a>(pending_inactive);
        };
        i &lt; len
    }) {
        <b>let</b> validator = *<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(validators, i);
        <b>let</b> validator_index = <a href="stake.md#0x1_stake_find_validator">find_validator</a>(active_validators, validator);
        <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&validator_index)) {
            <b>let</b> validator_info = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_swap_remove">vector::swap_remove</a>(active_validators, *<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_borrow">option::borrow</a>(&validator_index));
            <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(pending_inactive, validator_info);
        };
        i = i + 1;
    };
}
</code></pre>



</details>

<a name="0x1_stake_initialize_stake_owner"></a>

## Function `initialize_stake_owner`

Initialize the validator account and give ownership to the signing account
except it leaves the ValidatorConfig to be set by another entity.
Note: this triggers setting the operator and owner, set it to the account's address
to set later.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_initialize_stake_owner">initialize_stake_owner</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, initial_stake_amount: u64, operator: <b>address</b>, voter: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_initialize_stake_owner">initialize_stake_owner</a>(
    owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    initial_stake_amount: u64,
    operator: <b>address</b>,
    voter: <b>address</b>,
) <b>acquires</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>, <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <a href="stake.md#0x1_stake_initialize_owner">initialize_owner</a>(owner);
    <b>move_to</b>(owner, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
        consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>(),
        validator_index: 0,
    });

    <b>if</b> (initial_stake_amount &gt; 0) {
        <a href="stake.md#0x1_stake_add_stake">add_stake</a>(owner, initial_stake_amount);
    };

    <b>let</b> account_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <b>if</b> (account_address != operator) {
        <a href="stake.md#0x1_stake_set_operator">set_operator</a>(owner, operator)
    };
    <b>if</b> (account_address != voter) {
        <a href="stake.md#0x1_stake_set_delegated_voter">set_delegated_voter</a>(owner, voter)
    };
}
</code></pre>



</details>

<a name="0x1_stake_initialize_validator"></a>

## Function `initialize_validator`

Initialize the validator account and give ownership to the signing account.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_initialize_validator">initialize_validator</a>(<a href="account.md#0x1_account">account</a>: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, proof_of_possession: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_initialize_validator">initialize_validator</a>(
    <a href="account.md#0x1_account">account</a>: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
    proof_of_possession: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
    network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
    fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
) <b>acquires</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> {
    // Checks the <b>public</b> key <b>has</b> a valid proof-of-possession <b>to</b> prevent rogue-key attacks.
    <b>let</b> pubkey_from_pop = &<b>mut</b> <a href="../../aptos-stdlib/doc/bls12381.md#0x1_bls12381_public_key_from_bytes_with_pop">bls12381::public_key_from_bytes_with_pop</a>(
        consensus_pubkey,
        &proof_of_possession_from_bytes(proof_of_possession)
    );
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(pubkey_from_pop), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EINVALID_PUBLIC_KEY">EINVALID_PUBLIC_KEY</a>));

    <a href="stake.md#0x1_stake_initialize_owner">initialize_owner</a>(<a href="account.md#0x1_account">account</a>);
    <b>move_to</b>(<a href="account.md#0x1_account">account</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
        consensus_pubkey,
        network_addresses,
        fullnode_addresses,
        validator_index: 0,
    });
}
</code></pre>



</details>

<a name="0x1_stake_initialize_owner"></a>

## Function `initialize_owner`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_initialize_owner">initialize_owner</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_initialize_owner">initialize_owner</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>) <b>acquires</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <b>assert</b>!(<a href="stake.md#0x1_stake_is_allowed">is_allowed</a>(owner_address), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_not_found">error::not_found</a>(<a href="stake.md#0x1_stake_EINELIGIBLE_VALIDATOR">EINELIGIBLE_VALIDATOR</a>));
    <b>assert</b>!(!<a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(owner_address), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_already_exists">error::already_exists</a>(<a href="stake.md#0x1_stake_EALREADY_REGISTERED">EALREADY_REGISTERED</a>));

    <b>move_to</b>(owner, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
        active: <a href="coin.md#0x1_coin_zero">coin::zero</a>&lt;AptosCoin&gt;(),
        pending_active: <a href="coin.md#0x1_coin_zero">coin::zero</a>&lt;AptosCoin&gt;(),
        pending_inactive: <a href="coin.md#0x1_coin_zero">coin::zero</a>&lt;AptosCoin&gt;(),
        inactive: <a href="coin.md#0x1_coin_zero">coin::zero</a>&lt;AptosCoin&gt;(),
        locked_until_secs: 0,
        operator_address: owner_address,
        delegated_voter: owner_address,
        // Events.
        initialize_validator_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_RegisterValidatorCandidateEvent">RegisterValidatorCandidateEvent</a>&gt;(owner),
        set_operator_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_SetOperatorEvent">SetOperatorEvent</a>&gt;(owner),
        add_stake_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_AddStakeEvent">AddStakeEvent</a>&gt;(owner),
        reactivate_stake_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_ReactivateStakeEvent">ReactivateStakeEvent</a>&gt;(owner),
        rotate_consensus_key_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_RotateConsensusKeyEvent">RotateConsensusKeyEvent</a>&gt;(owner),
        update_network_and_fullnode_addresses_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_UpdateNetworkAndFullnodeAddressesEvent">UpdateNetworkAndFullnodeAddressesEvent</a>&gt;(owner),
        increase_lockup_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_IncreaseLockupEvent">IncreaseLockupEvent</a>&gt;(owner),
        join_validator_set_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_JoinValidatorSetEvent">JoinValidatorSetEvent</a>&gt;(owner),
        distribute_rewards_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_DistributeRewardsEvent">DistributeRewardsEvent</a>&gt;(owner),
        unlock_stake_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_UnlockStakeEvent">UnlockStakeEvent</a>&gt;(owner),
        withdraw_stake_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_WithdrawStakeEvent">WithdrawStakeEvent</a>&gt;(owner),
        leave_validator_set_events: <a href="account.md#0x1_account_new_event_handle">account::new_event_handle</a>&lt;<a href="stake.md#0x1_stake_LeaveValidatorSetEvent">LeaveValidatorSetEvent</a>&gt;(owner),
    });

    <b>move_to</b>(owner, <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a> { pool_address: owner_address });
}
</code></pre>



</details>

<a name="0x1_stake_extract_owner_cap"></a>

## Function `extract_owner_cap`

Extract and return owner capability from the signing account.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_extract_owner_cap">extract_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>): <a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_extract_owner_cap">extract_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>): <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a> <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>move_from</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address)
}
</code></pre>



</details>

<a name="0x1_stake_deposit_owner_cap"></a>

## Function `deposit_owner_cap`

Deposit <code>owner_cap</code> into <code><a href="account.md#0x1_account">account</a></code>. This requires <code><a href="account.md#0x1_account">account</a></code> to not already have owernship of another
staking pool.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_deposit_owner_cap">deposit_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, owner_cap: <a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_deposit_owner_cap">deposit_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, owner_cap: <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>) {
    <b>assert</b>!(!<b>exists</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner)), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_not_found">error::not_found</a>(<a href="stake.md#0x1_stake_EOWNER_CAP_ALREADY_EXISTS">EOWNER_CAP_ALREADY_EXISTS</a>));
    <b>move_to</b>(owner, owner_cap);
}
</code></pre>



</details>

<a name="0x1_stake_destroy_owner_cap"></a>

## Function `destroy_owner_cap`

Destroy <code>owner_cap</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_destroy_owner_cap">destroy_owner_cap</a>(owner_cap: <a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_destroy_owner_cap">destroy_owner_cap</a>(owner_cap: <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>) {
    <b>let</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a> { pool_address: _ } = owner_cap;
}
</code></pre>



</details>

<a name="0x1_stake_set_operator"></a>

## Function `set_operator`

Allows an owner to change the operator of the stake pool.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_set_operator">set_operator</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, new_operator: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_set_operator">set_operator</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, new_operator: <b>address</b>) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_set_operator_with_cap">set_operator_with_cap</a>(ownership_cap, new_operator);
}
</code></pre>



</details>

<a name="0x1_stake_set_operator_with_cap"></a>

## Function `set_operator_with_cap`

Allows an account with ownership capability to change the operator of the stake pool.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_operator_with_cap">set_operator_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, new_operator: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_operator_with_cap">set_operator_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, new_operator: <b>address</b>) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>let</b> old_operator = stake_pool.operator_address;
    stake_pool.operator_address = new_operator;

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.set_operator_events,
        <a href="stake.md#0x1_stake_SetOperatorEvent">SetOperatorEvent</a> {
            pool_address,
            old_operator,
            new_operator,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_set_delegated_voter"></a>

## Function `set_delegated_voter`

Allows an owner to change the delegated voter of the stake pool.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_set_delegated_voter">set_delegated_voter</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, new_voter: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_set_delegated_voter">set_delegated_voter</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, new_voter: <b>address</b>) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_set_delegated_voter_with_cap">set_delegated_voter_with_cap</a>(ownership_cap, new_voter);
}
</code></pre>



</details>

<a name="0x1_stake_set_delegated_voter_with_cap"></a>

## Function `set_delegated_voter_with_cap`

Allows an owner to change the delegated voter of the stake pool.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_delegated_voter_with_cap">set_delegated_voter_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, new_voter: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_delegated_voter_with_cap">set_delegated_voter_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, new_voter: <b>address</b>) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    stake_pool.delegated_voter = new_voter;
}
</code></pre>



</details>

<a name="0x1_stake_add_stake"></a>

## Function `add_stake`

Add <code>amount</code> of coins from the <code><a href="account.md#0x1_account">account</a></code> owning the StakePool.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_add_stake">add_stake</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_add_stake">add_stake</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_add_stake_with_cap">add_stake_with_cap</a>(ownership_cap, <a href="coin.md#0x1_coin_withdraw">coin::withdraw</a>&lt;AptosCoin&gt;(owner, amount));
}
</code></pre>



</details>

<a name="0x1_stake_add_stake_with_cap"></a>

## Function `add_stake_with_cap`

Add <code>coins</code> into <code>pool_address</code>. this requires the corresponding <code>owner_cap</code> to be passed in.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_add_stake_with_cap">add_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, coins: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_add_stake_with_cap">add_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, coins: Coin&lt;AptosCoin&gt;) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);

    <b>let</b> amount = <a href="coin.md#0x1_coin_value">coin::value</a>(&coins);
    <b>if</b> (amount == 0) {
        <a href="coin.md#0x1_coin_destroy_zero">coin::destroy_zero</a>(coins);
        <b>return</b>
    };

    // Only track and validate <a href="voting.md#0x1_voting">voting</a> power increase for active and pending_active validator.
    // Pending_inactive validator will be removed from the validator set in the next epoch.
    // Inactive validator's total <a href="stake.md#0x1_stake">stake</a> will be tracked when they join the validator set.
    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    // Search directly rather using get_validator_state <b>to</b> save on unnecessary loops.
    <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&<a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.active_validators, pool_address)) ||
        <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&<a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.pending_active, pool_address))) {
        <a href="stake.md#0x1_stake_update_voting_power_increase">update_voting_power_increase</a>(amount);
    };

    // Add <b>to</b> pending_active <b>if</b> it's a current validator because the <a href="stake.md#0x1_stake">stake</a> is not counted until the next epoch.
    // Otherwise, the delegation can be added <b>to</b> active directly <b>as</b> the validator is also activated in the epoch.
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>if</b> (<a href="stake.md#0x1_stake_is_current_epoch_validator">is_current_epoch_validator</a>(pool_address)) {
        <a href="coin.md#0x1_coin_merge">coin::merge</a>&lt;AptosCoin&gt;(&<b>mut</b> stake_pool.pending_active, coins);
    } <b>else</b> {
        <a href="coin.md#0x1_coin_merge">coin::merge</a>&lt;AptosCoin&gt;(&<b>mut</b> stake_pool.active, coins);
    };

    <b>let</b> (_, maximum_stake) = <a href="staking_config.md#0x1_staking_config_get_required_stake">staking_config::get_required_stake</a>(&<a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>());
    <b>let</b> voting_power = <a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool);
    <b>assert</b>!(voting_power &lt;= maximum_stake, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ESTAKE_EXCEEDS_MAX">ESTAKE_EXCEEDS_MAX</a>));

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.add_stake_events,
        <a href="stake.md#0x1_stake_AddStakeEvent">AddStakeEvent</a> {
            pool_address,
            amount_added: amount,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_reactivate_stake"></a>

## Function `reactivate_stake`

Move <code>amount</code> of coins from pending_inactive to active.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_reactivate_stake">reactivate_stake</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_reactivate_stake">reactivate_stake</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_reactivate_stake_with_cap">reactivate_stake_with_cap</a>(ownership_cap, amount);
}
</code></pre>



</details>

<a name="0x1_stake_reactivate_stake_with_cap"></a>

## Function `reactivate_stake_with_cap`



<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_reactivate_stake_with_cap">reactivate_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_reactivate_stake_with_cap">reactivate_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, amount: u64) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);

    // Cap the amount <b>to</b> reactivate by the amount in pending_inactive.
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>let</b> total_pending_inactive = <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.pending_inactive);
    amount = <b>min</b>(amount, total_pending_inactive);

    // Since this does not count <b>as</b> a <a href="voting.md#0x1_voting">voting</a> power change (pending inactive still counts <b>as</b> <a href="voting.md#0x1_voting">voting</a> power in the
    // current epoch), <a href="stake.md#0x1_stake">stake</a> can be immediately moved from pending inactive <b>to</b> active.
    // We also don't need <b>to</b> check <a href="voting.md#0x1_voting">voting</a> power increase <b>as</b> there's none.
    <b>let</b> reactivated_coins = <a href="coin.md#0x1_coin_extract">coin::extract</a>(&<b>mut</b> stake_pool.pending_inactive, amount);
    <a href="coin.md#0x1_coin_merge">coin::merge</a>(&<b>mut</b> stake_pool.active, reactivated_coins);

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.reactivate_stake_events,
        <a href="stake.md#0x1_stake_ReactivateStakeEvent">ReactivateStakeEvent</a> {
            pool_address,
            amount,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_rotate_consensus_key"></a>

## Function `rotate_consensus_key`

Rotate the consensus key of the validator, it'll take effect in next epoch.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_rotate_consensus_key">rotate_consensus_key</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>, new_consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, proof_of_possession: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_rotate_consensus_key">rotate_consensus_key</a>(
    operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    pool_address: <b>address</b>,
    new_consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
    proof_of_possession: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(operator) == stake_pool.operator_address, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_unauthenticated">error::unauthenticated</a>(<a href="stake.md#0x1_stake_ENOT_OPERATOR">ENOT_OPERATOR</a>));

    <b>assert</b>!(<b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_not_found">error::not_found</a>(<a href="stake.md#0x1_stake_EVALIDATOR_CONFIG">EVALIDATOR_CONFIG</a>));
    <b>let</b> validator_info = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
    <b>let</b> old_consensus_pubkey = validator_info.consensus_pubkey;
    // Checks the <b>public</b> key <b>has</b> a valid proof-of-possession <b>to</b> prevent rogue-key attacks.
    <b>let</b> pubkey_from_pop = &<b>mut</b> <a href="../../aptos-stdlib/doc/bls12381.md#0x1_bls12381_public_key_from_bytes_with_pop">bls12381::public_key_from_bytes_with_pop</a>(
        new_consensus_pubkey,
        &proof_of_possession_from_bytes(proof_of_possession)
    );
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(pubkey_from_pop), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EINVALID_PUBLIC_KEY">EINVALID_PUBLIC_KEY</a>));
    validator_info.consensus_pubkey = new_consensus_pubkey;

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.rotate_consensus_key_events,
        <a href="stake.md#0x1_stake_RotateConsensusKeyEvent">RotateConsensusKeyEvent</a> {
            pool_address,
            old_consensus_pubkey,
            new_consensus_pubkey,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_update_network_and_fullnode_addresses"></a>

## Function `update_network_and_fullnode_addresses`

Update the network and full node addresses of the validator. This only takes effect in the next epoch.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_update_network_and_fullnode_addresses">update_network_and_fullnode_addresses</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>, new_network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, new_fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_update_network_and_fullnode_addresses">update_network_and_fullnode_addresses</a>(
    operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    pool_address: <b>address</b>,
    new_network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
    new_fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;,
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(operator) == stake_pool.operator_address, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_unauthenticated">error::unauthenticated</a>(<a href="stake.md#0x1_stake_ENOT_OPERATOR">ENOT_OPERATOR</a>));

    <b>assert</b>!(<b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_not_found">error::not_found</a>(<a href="stake.md#0x1_stake_EVALIDATOR_CONFIG">EVALIDATOR_CONFIG</a>));
    <b>let</b> validator_info = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
    <b>let</b> old_network_addresses = validator_info.network_addresses;
    validator_info.network_addresses = new_network_addresses;
    <b>let</b> old_fullnode_addresses = validator_info.fullnode_addresses;
    validator_info.fullnode_addresses = new_fullnode_addresses;

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.update_network_and_fullnode_addresses_events,
        <a href="stake.md#0x1_stake_UpdateNetworkAndFullnodeAddressesEvent">UpdateNetworkAndFullnodeAddressesEvent</a> {
            pool_address,
            old_network_addresses,
            new_network_addresses,
            old_fullnode_addresses,
            new_fullnode_addresses,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_increase_lockup"></a>

## Function `increase_lockup`

Similar to increase_lockup_with_cap but will use ownership capability from the signing account.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_increase_lockup">increase_lockup</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_increase_lockup">increase_lockup</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_increase_lockup_with_cap">increase_lockup_with_cap</a>(ownership_cap);
}
</code></pre>



</details>

<a name="0x1_stake_increase_lockup_with_cap"></a>

## Function `increase_lockup_with_cap`

Unlock from active delegation, it's moved to pending_inactive if locked_until_secs < current_time or
directly inactive if it's not from an active validator.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_increase_lockup_with_cap">increase_lockup_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_increase_lockup_with_cap">increase_lockup_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> config = <a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>();

    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>let</b> old_locked_until_secs = stake_pool.locked_until_secs;
    <b>let</b> new_locked_until_secs = <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>() + <a href="staking_config.md#0x1_staking_config_get_recurring_lockup_duration">staking_config::get_recurring_lockup_duration</a>(&config);
    <b>assert</b>!(old_locked_until_secs &lt; new_locked_until_secs, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EINVALID_LOCKUP">EINVALID_LOCKUP</a>));
    stake_pool.locked_until_secs = new_locked_until_secs;

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.increase_lockup_events,
        <a href="stake.md#0x1_stake_IncreaseLockupEvent">IncreaseLockupEvent</a> {
            pool_address,
            old_locked_until_secs,
            new_locked_until_secs,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_join_validator_set"></a>

## Function `join_validator_set`

This can only called by the operator of the validator/staking pool.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_join_validator_set">join_validator_set</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_join_validator_set">join_validator_set</a>(
    operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    pool_address: <b>address</b>
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>assert</b>!(
        <a href="staking_config.md#0x1_staking_config_get_allow_validator_set_change">staking_config::get_allow_validator_set_change</a>(&<a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>()),
        <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED">ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED</a>),
    );

    <a href="stake.md#0x1_stake_join_validator_set_internal">join_validator_set_internal</a>(operator, pool_address);
}
</code></pre>



</details>

<a name="0x1_stake_join_validator_set_internal"></a>

## Function `join_validator_set_internal`

Request to have <code>pool_address</code> join the validator set. Can only be called after calling <code>initialize_validator</code>.
If the validator has the required stake (more than minimum and less than maximum allowed), they will be
added to the pending_active queue. All validators in this queue will be added to the active set when the next
epoch starts (eligibility will be rechecked).

This internal version can only be called by the Genesis module during Genesis.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_join_validator_set_internal">join_validator_set_internal</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_join_validator_set_internal">join_validator_set_internal</a>(
    operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    pool_address: <b>address</b>
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(operator) == stake_pool.operator_address, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_unauthenticated">error::unauthenticated</a>(<a href="stake.md#0x1_stake_ENOT_OPERATOR">ENOT_OPERATOR</a>));
    <b>assert</b>!(
        <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address) == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_INACTIVE">VALIDATOR_STATUS_INACTIVE</a>,
        <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_state">error::invalid_state</a>(<a href="stake.md#0x1_stake_EALREADY_ACTIVE_VALIDATOR">EALREADY_ACTIVE_VALIDATOR</a>),
    );

    <b>let</b> config = <a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>();
    <b>let</b> (minimum_stake, maximum_stake) = <a href="staking_config.md#0x1_staking_config_get_required_stake">staking_config::get_required_stake</a>(&config);
    <b>let</b> voting_power = <a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool);
    <b>assert</b>!(voting_power &gt;= minimum_stake, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ESTAKE_TOO_LOW">ESTAKE_TOO_LOW</a>));
    <b>assert</b>!(voting_power &lt;= maximum_stake, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ESTAKE_TOO_HIGH">ESTAKE_TOO_HIGH</a>));

    // Track and validate <a href="voting.md#0x1_voting">voting</a> power increase.
    <a href="stake.md#0x1_stake_update_voting_power_increase">update_voting_power_increase</a>(voting_power);

    // Add validator <b>to</b> pending_active, <b>to</b> be activated in the next epoch.
    <b>let</b> validator_config = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
    <b>assert</b>!(!<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_is_empty">vector::is_empty</a>(&validator_config.consensus_pubkey), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EINVALID_PUBLIC_KEY">EINVALID_PUBLIC_KEY</a>));

    // Validate the current validator set size <b>has</b> not exceeded the limit.
    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(&<b>mut</b> validator_set.pending_active, <a href="stake.md#0x1_stake_generate_validator_info">generate_validator_info</a>(pool_address, stake_pool, *validator_config));
    <b>let</b> validator_set_size = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.active_validators) + <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.pending_active);
    <b>assert</b>!(validator_set_size &lt;= <a href="stake.md#0x1_stake_MAX_VALIDATOR_SET_SIZE">MAX_VALIDATOR_SET_SIZE</a>, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EVALIDATOR_SET_TOO_LARGE">EVALIDATOR_SET_TOO_LARGE</a>));

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.join_validator_set_events,
        <a href="stake.md#0x1_stake_JoinValidatorSetEvent">JoinValidatorSetEvent</a> { pool_address },
    );
}
</code></pre>



</details>

<a name="0x1_stake_unlock"></a>

## Function `unlock`

Similar to unlock_with_cap but will use ownership capability from the signing account.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_unlock">unlock</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_unlock">unlock</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <a href="stake.md#0x1_stake_unlock_with_cap">unlock_with_cap</a>(amount, ownership_cap);
}
</code></pre>



</details>

<a name="0x1_stake_unlock_with_cap"></a>

## Function `unlock_with_cap`

Unlock <code>amount</code> from the active stake. Only possible if the lockup has expired.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_unlock_with_cap">unlock_with_cap</a>(amount: u64, owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_unlock_with_cap">unlock_with_cap</a>(amount: u64, owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a> {
    // Short-circuit <b>if</b> amount <b>to</b> unlock is 0 so we don't emit events.
    <b>if</b> (amount == 0) {
        <b>return</b>
    };

    // Unlocked coins are moved <b>to</b> pending_inactive. When the current lockup cycle expires, they will be moved into
    // inactive in the earliest possible epoch transition.
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    // Cap amount <b>to</b> unlock by maximum active <a href="stake.md#0x1_stake">stake</a>.
    <b>let</b> amount = <b>min</b>(amount, <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.active));
    <b>let</b> unlocked_stake = <a href="coin.md#0x1_coin_extract">coin::extract</a>(&<b>mut</b> stake_pool.active, amount);
    <a href="coin.md#0x1_coin_merge">coin::merge</a>&lt;AptosCoin&gt;(&<b>mut</b> stake_pool.pending_inactive, unlocked_stake);

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.unlock_stake_events,
        <a href="stake.md#0x1_stake_UnlockStakeEvent">UnlockStakeEvent</a> {
            pool_address,
            amount_unlocked: amount,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_withdraw"></a>

## Function `withdraw`

Withdraw from <code><a href="account.md#0x1_account">account</a></code>'s inactive stake.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_withdraw">withdraw</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, withdraw_amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_withdraw">withdraw</a>(
    owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    withdraw_amount: u64
) <b>acquires</b> <a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>, <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
    <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner_address);
    <b>let</b> ownership_cap = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
    <b>let</b> coins = <a href="stake.md#0x1_stake_withdraw_with_cap">withdraw_with_cap</a>(ownership_cap, withdraw_amount);
    <a href="coin.md#0x1_coin_deposit">coin::deposit</a>&lt;AptosCoin&gt;(owner_address, coins);
}
</code></pre>



</details>

<a name="0x1_stake_withdraw_with_cap"></a>

## Function `withdraw_with_cap`

Withdraw from <code>pool_address</code>'s inactive stake with the corresponding <code>owner_cap</code>.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_withdraw_with_cap">withdraw_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, withdraw_amount: u64): <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_withdraw_with_cap">withdraw_with_cap</a>(
    owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>,
    withdraw_amount: u64
): Coin&lt;AptosCoin&gt; <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> pool_address = owner_cap.pool_address;
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    // There's an edge case <b>where</b> a validator unlocks their <a href="stake.md#0x1_stake">stake</a> and leaves the validator set before
    // the <a href="stake.md#0x1_stake">stake</a> is fully unlocked (the current lockup cycle <b>has</b> not expired yet).
    // This can leave their <a href="stake.md#0x1_stake">stake</a> stuck in pending_inactive even after the current lockup cycle expires.
    <b>if</b> (<a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address) == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_INACTIVE">VALIDATOR_STATUS_INACTIVE</a> &&
        <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>() &gt;= stake_pool.locked_until_secs) {
        <b>let</b> pending_inactive_stake = <a href="coin.md#0x1_coin_extract_all">coin::extract_all</a>(&<b>mut</b> stake_pool.pending_inactive);
        <a href="coin.md#0x1_coin_merge">coin::merge</a>(&<b>mut</b> stake_pool.inactive, pending_inactive_stake);
    };

    // Cap withdraw amount by total ianctive coins.
    withdraw_amount = <b>min</b>(withdraw_amount, <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.inactive));
    <b>if</b> (withdraw_amount == 0) <b>return</b> <a href="coin.md#0x1_coin_zero">coin::zero</a>&lt;AptosCoin&gt;();

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.withdraw_stake_events,
        <a href="stake.md#0x1_stake_WithdrawStakeEvent">WithdrawStakeEvent</a> {
            pool_address,
            amount_withdrawn: withdraw_amount,
        },
    );

    <a href="coin.md#0x1_coin_extract">coin::extract</a>(&<b>mut</b> stake_pool.inactive, withdraw_amount)
}
</code></pre>



</details>

<a name="0x1_stake_leave_validator_set"></a>

## Function `leave_validator_set`

Request to have <code>pool_address</code> leave the validator set. The validator is only actually removed from the set when
the next epoch starts.
The last validator in the set cannot leave. This is an edge case that should never happen as long as the network
is still operational.

Can only be called by the operator of the validator/staking pool.


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_leave_validator_set">leave_validator_set</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_leave_validator_set">leave_validator_set</a>(
    operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>,
    pool_address: <b>address</b>
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> config = <a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>();
    <b>assert</b>!(
        <a href="staking_config.md#0x1_staking_config_get_allow_validator_set_change">staking_config::get_allow_validator_set_change</a>(&config),
        <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED">ENO_POST_GENESIS_VALIDATOR_SET_CHANGE_ALLOWED</a>),
    );

    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
    // Account <b>has</b> <b>to</b> be the operator.
    <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(operator) == stake_pool.operator_address, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_unauthenticated">error::unauthenticated</a>(<a href="stake.md#0x1_stake_ENOT_OPERATOR">ENOT_OPERATOR</a>));

    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    // If the validator is still pending_active, directly kick the validator out.
    <b>let</b> maybe_pending_active_index = <a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.pending_active, pool_address);
    <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&maybe_pending_active_index)) {
        <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_swap_remove">vector::swap_remove</a>(
            &<b>mut</b> validator_set.pending_active, <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_extract">option::extract</a>(&<b>mut</b> maybe_pending_active_index));

        // Decrease the <a href="voting.md#0x1_voting">voting</a> power increase <b>as</b> the pending validator's <a href="voting.md#0x1_voting">voting</a> power was added when they requested
        // <b>to</b> join. Now that they changed their mind, their <a href="voting.md#0x1_voting">voting</a> power should not affect the joining limit of this
        // epoch.
        <b>let</b> validator_stake = (<a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool) <b>as</b> u128);
        // total_joining_power should be larger than validator_stake but just in case there <b>has</b> been a small
        // rounding <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error">error</a> somewhere that can lead <b>to</b> an underflow, we still want <b>to</b> allow this transaction <b>to</b>
        // succeed.
        <b>if</b> (validator_set.total_joining_power &gt; validator_stake) {
            validator_set.total_joining_power = validator_set.total_joining_power - validator_stake;
        } <b>else</b> {
            validator_set.total_joining_power = 0;
        };
    } <b>else</b> {
        // Validate that the validator is already part of the validator set.
        <b>let</b> maybe_active_index = <a href="stake.md#0x1_stake_find_validator">find_validator</a>(&validator_set.active_validators, pool_address);
        <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&maybe_active_index), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_state">error::invalid_state</a>(<a href="stake.md#0x1_stake_ENOT_VALIDATOR">ENOT_VALIDATOR</a>));
        <b>let</b> validator_info = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_swap_remove">vector::swap_remove</a>(
            &<b>mut</b> validator_set.active_validators, <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_extract">option::extract</a>(&<b>mut</b> maybe_active_index));
        <b>assert</b>!(<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.active_validators) &gt; 0, <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_state">error::invalid_state</a>(<a href="stake.md#0x1_stake_ELAST_VALIDATOR">ELAST_VALIDATOR</a>));
        <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(&<b>mut</b> validator_set.pending_inactive, validator_info);

        <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
            &<b>mut</b> stake_pool.leave_validator_set_events,
            <a href="stake.md#0x1_stake_LeaveValidatorSetEvent">LeaveValidatorSetEvent</a> {
                pool_address,
            },
        );
    };
}
</code></pre>



</details>

<a name="0x1_stake_is_current_epoch_validator"></a>

## Function `is_current_epoch_validator`

Returns true if the current validator can still vote in the current epoch.
This includes validators that requested to leave but are still in the pending_inactive queue and will be removed
when the epoch starts.


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_is_current_epoch_validator">is_current_epoch_validator</a>(pool_address: <b>address</b>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_is_current_epoch_validator">is_current_epoch_validator</a>(pool_address: <b>address</b>): bool <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address);
    <b>let</b> validator_state = <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address);
    validator_state == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_ACTIVE">VALIDATOR_STATUS_ACTIVE</a> || validator_state == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE">VALIDATOR_STATUS_PENDING_INACTIVE</a>
}
</code></pre>



</details>

<a name="0x1_stake_update_performance_statistics"></a>

## Function `update_performance_statistics`

Update the validator performance (proposal statistics). This is only called by block::prologue().
This function cannot abort.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_update_performance_statistics">update_performance_statistics</a>(proposer_index: <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;u64&gt;, failed_proposer_indices: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u64&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_update_performance_statistics">update_performance_statistics</a>(proposer_index: Option&lt;u64&gt;, failed_proposer_indices: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u64&gt;) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a> {
    // Validator set cannot change until the end of the epoch, so the validator index in arguments should
    // match <b>with</b> those of the validators in <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a> resource.
    <b>let</b> validator_perf = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(@aptos_framework);
    <b>let</b> validator_len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_perf.validators);

    // proposer_index is an <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option">option</a> because it can be missing (for NilBlocks)
    <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(&proposer_index)) {
        <b>let</b> cur_proposer_index = <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_extract">option::extract</a>(&<b>mut</b> proposer_index);
        // Here, and in all other <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>, skip <a href="../../aptos-stdlib/doc/any.md#0x1_any">any</a> validator indices that are out of bounds,
        // this <b>ensures</b> that this function doesn't <b>abort</b> <b>if</b> there are out of bounds errors.
        <b>if</b> (cur_proposer_index &lt; validator_len) {
            <b>let</b> validator = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow_mut">vector::borrow_mut</a>(&<b>mut</b> validator_perf.validators, cur_proposer_index);
            <b>spec</b> {
                <b>assume</b> validator.successful_proposals + 1 &lt;= MAX_U64;
            };
            validator.successful_proposals = validator.successful_proposals + 1;
        };
    };

    <b>let</b> f = 0;
    <b>let</b> f_len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&failed_proposer_indices);
    <b>while</b> ({
        <b>spec</b> {
            <b>invariant</b> len(validator_perf.validators) == validator_len;
        };
        f &lt; f_len
    }) {
        <b>let</b> validator_index = *<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(&failed_proposer_indices, f);
        <b>if</b> (validator_index &lt; validator_len) {
            <b>let</b> validator = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow_mut">vector::borrow_mut</a>(&<b>mut</b> validator_perf.validators, validator_index);
            <b>spec</b> {
                <b>assume</b> validator.failed_proposals + 1 &lt;= MAX_U64;
            };
            validator.failed_proposals = validator.failed_proposals + 1;
        };
        f = f + 1;
    };
}
</code></pre>



</details>

<a name="0x1_stake_on_new_epoch"></a>

## Function `on_new_epoch`

Triggers at epoch boundary. This function shouldn't abort.

1. Distribute transaction fees and rewards to stake pools of active and pending inactive validators (requested
to leave but not yet removed).
2. Officially move pending active stake to active and move pending inactive stake to inactive.
The staking pool's voting power in this new epoch will be updated to the total active stake.
3. Add pending active validators to the active set if they satisfy requirements so they can vote and remove
pending inactive validators so they no longer can vote.
4. The validator's voting power in the validator set is updated to be the corresponding staking pool's voting
power.


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_on_new_epoch">on_new_epoch</a>()
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_on_new_epoch">on_new_epoch</a>() <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>, <a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>, <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>, <a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a> {
    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    <b>let</b> config = <a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>();
    <b>let</b> validator_perf = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(@aptos_framework);

    // Process pending <a href="stake.md#0x1_stake">stake</a> and distribute transaction fees and rewards for each currently active validator.
    <b>let</b> i = 0;
    <b>let</b> len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.active_validators);
    <b>while</b> (i &lt; len) {
        <b>let</b> validator = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(&validator_set.active_validators, i);
        <a href="stake.md#0x1_stake_update_stake_pool">update_stake_pool</a>(validator_perf, validator.addr, &config);
        i = i + 1;
    };

    // Process pending <a href="stake.md#0x1_stake">stake</a> and distribute transaction fees and rewards for each currently pending_inactive validator
    // (requested <b>to</b> leave but not removed yet).
    <b>let</b> i = 0;
    <b>let</b> len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.pending_inactive);
    <b>while</b> (i &lt; len) {
        <b>let</b> validator = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(&validator_set.pending_inactive, i);
        <a href="stake.md#0x1_stake_update_stake_pool">update_stake_pool</a>(validator_perf, validator.addr, &config);
        i = i + 1;
    };

    // Activate currently pending_active validators.
    <a href="stake.md#0x1_stake_append">append</a>(&<b>mut</b> validator_set.active_validators, &<b>mut</b> validator_set.pending_active);

    // Officially deactivate all pending_inactive validators. They will now no longer receive rewards.
    validator_set.pending_inactive = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>();

    // Update active validator set so that network <b>address</b>/<b>public</b> key change takes effect.
    // Moreover, recalculate the total <a href="voting.md#0x1_voting">voting</a> power, and deactivate the validator whose
    // <a href="voting.md#0x1_voting">voting</a> power is less than the minimum required <a href="stake.md#0x1_stake">stake</a>.
    <b>let</b> next_epoch_validators = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>();
    <b>let</b> (minimum_stake, _) = <a href="staking_config.md#0x1_staking_config_get_required_stake">staking_config::get_required_stake</a>(&config);
    <b>let</b> vlen = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.active_validators);
    <b>let</b> total_voting_power = 0;
    <b>let</b> i = 0;
    <b>while</b> ({
        <b>spec</b> {
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(next_epoch_validators);
        };
        i &lt; vlen
    }) {
        <b>let</b> old_validator_info = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow_mut">vector::borrow_mut</a>(&<b>mut</b> validator_set.active_validators, i);
        <b>let</b> pool_address = old_validator_info.addr;
        <b>let</b> validator_config = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
        <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
        <b>let</b> new_validator_info = <a href="stake.md#0x1_stake_generate_validator_info">generate_validator_info</a>(pool_address, stake_pool, *validator_config);

        // A validator needs at least the <b>min</b> <a href="stake.md#0x1_stake">stake</a> required <b>to</b> join the validator set.
        <b>if</b> (new_validator_info.voting_power &gt;= minimum_stake) {
            <b>spec</b> {
                <b>assume</b> total_voting_power + new_validator_info.voting_power &lt;= MAX_U128;
            };
            total_voting_power = total_voting_power + (new_validator_info.voting_power <b>as</b> u128);
            <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(&<b>mut</b> next_epoch_validators, new_validator_info);
        };
        i = i + 1;
    };

    validator_set.active_validators = next_epoch_validators;
    validator_set.total_voting_power = total_voting_power;
    validator_set.total_joining_power = 0;

    // Update validator indices, reset performance scores, and renew lockups.
    validator_perf.validators = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_empty">vector::empty</a>();
    <b>let</b> recurring_lockup_duration_secs = <a href="staking_config.md#0x1_staking_config_get_recurring_lockup_duration">staking_config::get_recurring_lockup_duration</a>(&config);
    <b>let</b> vlen = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(&validator_set.active_validators);
    <b>let</b> validator_index = 0;
    <b>while</b> ({
        <b>spec</b> {
            <b>invariant</b> <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(validator_set.active_validators);
            <b>invariant</b> len(validator_set.pending_active) == 0;
            <b>invariant</b> len(validator_set.pending_inactive) == 0;
            <b>invariant</b> 0 &lt;= validator_index && validator_index &lt;= vlen;
            <b>invariant</b> vlen == len(validator_set.active_validators);
            <b>invariant</b> <b>forall</b> i in 0..validator_index:
                <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(validator_set.active_validators[i].addr).validator_index &lt; validator_index;
            <b>invariant</b> len(validator_perf.validators) == validator_index;
        };
        validator_index &lt; vlen
    }) {
        <b>let</b> validator_info = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow_mut">vector::borrow_mut</a>(&<b>mut</b> validator_set.active_validators, validator_index);
        validator_info.config.validator_index = validator_index;
        <b>let</b> validator_config = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(validator_info.addr);
        validator_config.validator_index = validator_index;

        <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(&<b>mut</b> validator_perf.validators, <a href="stake.md#0x1_stake_IndividualValidatorPerformance">IndividualValidatorPerformance</a> {
            successful_proposals: 0,
            failed_proposals: 0,
        });

        // Automatically renew a validator's lockup for validators that will still be in the validator set in the
        // next epoch.
        <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(validator_info.addr);
        <b>if</b> (stake_pool.locked_until_secs &lt;= <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>()) {
            <b>spec</b> {
                <b>assume</b> <a href="timestamp.md#0x1_timestamp_spec_now_seconds">timestamp::spec_now_seconds</a>() + recurring_lockup_duration_secs &lt;= MAX_U64;
            };
            stake_pool.locked_until_secs =
                <a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>() + recurring_lockup_duration_secs;
        };

        validator_index = validator_index + 1;
    };
}
</code></pre>



</details>

<a name="0x1_stake_update_stake_pool"></a>

## Function `update_stake_pool`

Update individual validator's stake pool
1. distribute transaction fees to active/pending_inactive delegations
2. distribute rewards to active/pending_inactive delegations
3. process pending_active, pending_inactive correspondingly
This function shouldn't abort.


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_stake_pool">update_stake_pool</a>(validator_perf: &<a href="stake.md#0x1_stake_ValidatorPerformance">stake::ValidatorPerformance</a>, pool_address: <b>address</b>, <a href="staking_config.md#0x1_staking_config">staking_config</a>: &<a href="staking_config.md#0x1_staking_config_StakingConfig">staking_config::StakingConfig</a>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_stake_pool">update_stake_pool</a>(
    validator_perf: &<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>,
    pool_address: <b>address</b>,
    <a href="staking_config.md#0x1_staking_config">staking_config</a>: &StakingConfig,
) <b>acquires</b> <a href="stake.md#0x1_stake_StakePool">StakePool</a>, <a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a>, <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>, <a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a> {
    <b>let</b> stake_pool = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);

    // First, distribute transaction fees.
    <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/features.md#0x1_features_collect_and_distribute_gas_fees">features::collect_and_distribute_gas_fees</a>()) {
        <b>let</b> fees_table = &<b>mut</b> <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorFees">ValidatorFees</a>&gt;(@aptos_framework).fees_table;
        <b>if</b> (<a href="../../aptos-stdlib/doc/table.md#0x1_table_contains">table::contains</a>(fees_table, pool_address)) {
            <b>let</b> <a href="coin.md#0x1_coin">coin</a> = <a href="../../aptos-stdlib/doc/table.md#0x1_table_remove">table::remove</a>(fees_table, pool_address);
            <a href="coin.md#0x1_coin_merge">coin::merge</a>(&<b>mut</b> stake_pool.active, <a href="coin.md#0x1_coin">coin</a>);
        };
    };

    <b>let</b> validator_config = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
    <b>let</b> cur_validator_perf = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(&validator_perf.validators, validator_config.validator_index);
    <b>let</b> num_successful_proposals = cur_validator_perf.successful_proposals;
    <b>spec</b> {
        // The following addition should not overflow because `num_total_proposals` cannot be larger than 86400,
        // the maximum number of proposals in a day (1 proposal per second).
        <b>assume</b> cur_validator_perf.successful_proposals + cur_validator_perf.failed_proposals &lt;= MAX_U64;
    };
    <b>let</b> num_total_proposals = cur_validator_perf.successful_proposals + cur_validator_perf.failed_proposals;

    <b>let</b> (rewards_rate, rewards_rate_denominator) = <a href="staking_config.md#0x1_staking_config_get_reward_rate">staking_config::get_reward_rate</a>(<a href="staking_config.md#0x1_staking_config">staking_config</a>);
    <b>let</b> rewards_active = <a href="stake.md#0x1_stake_distribute_rewards">distribute_rewards</a>(
        &<b>mut</b> stake_pool.active,
        num_successful_proposals,
        num_total_proposals,
        rewards_rate,
        rewards_rate_denominator
    );
    <b>let</b> rewards_pending_inactive = <a href="stake.md#0x1_stake_distribute_rewards">distribute_rewards</a>(
        &<b>mut</b> stake_pool.pending_inactive,
        num_successful_proposals,
        num_total_proposals,
        rewards_rate,
        rewards_rate_denominator
    );
    <b>spec</b> {
        <b>assume</b> rewards_active + rewards_pending_inactive &lt;= MAX_U64;
    };
    <b>let</b> rewards_amount = rewards_active + rewards_pending_inactive;
    // Pending active <a href="stake.md#0x1_stake">stake</a> can now be active.
    <a href="coin.md#0x1_coin_merge">coin::merge</a>(&<b>mut</b> stake_pool.active, <a href="coin.md#0x1_coin_extract_all">coin::extract_all</a>(&<b>mut</b> stake_pool.pending_active));

    // Pending inactive <a href="stake.md#0x1_stake">stake</a> is only fully unlocked and moved into inactive <b>if</b> the current lockup cycle <b>has</b> expired
    <b>let</b> current_lockup_expiration = stake_pool.locked_until_secs;
    <b>if</b> (<a href="timestamp.md#0x1_timestamp_now_seconds">timestamp::now_seconds</a>() &gt;= current_lockup_expiration) {
        <a href="coin.md#0x1_coin_merge">coin::merge</a>(
            &<b>mut</b> stake_pool.inactive,
            <a href="coin.md#0x1_coin_extract_all">coin::extract_all</a>(&<b>mut</b> stake_pool.pending_inactive),
        );
    };

    <a href="event.md#0x1_event_emit_event">event::emit_event</a>(
        &<b>mut</b> stake_pool.distribute_rewards_events,
        <a href="stake.md#0x1_stake_DistributeRewardsEvent">DistributeRewardsEvent</a> {
            pool_address,
            rewards_amount,
        },
    );
}
</code></pre>



</details>

<a name="0x1_stake_calculate_rewards_amount"></a>

## Function `calculate_rewards_amount`

Calculate the rewards amount.


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_calculate_rewards_amount">calculate_rewards_amount</a>(stake_amount: u64, num_successful_proposals: u64, num_total_proposals: u64, rewards_rate: u64, rewards_rate_denominator: u64): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_calculate_rewards_amount">calculate_rewards_amount</a>(
    stake_amount: u64,
    num_successful_proposals: u64,
    num_total_proposals: u64,
    rewards_rate: u64,
    rewards_rate_denominator: u64,
): u64 {
    <b>spec</b> {
        // The following condition must hold because
        // (1) num_successful_proposals &lt;= num_total_proposals, and
        // (2) `num_total_proposals` cannot be larger than 86400, the maximum number of proposals
        //     in a day (1 proposal per second), and `num_total_proposals` is reset <b>to</b> 0 every epoch.
        <b>assume</b> num_successful_proposals * <a href="stake.md#0x1_stake_MAX_REWARDS_RATE">MAX_REWARDS_RATE</a> &lt;= MAX_U64;
    };
    // The rewards amount is equal <b>to</b> (<a href="stake.md#0x1_stake">stake</a> amount * rewards rate * performance multiplier).
    // We do multiplication in u128 before division <b>to</b> avoid the overflow and minimize the rounding <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error">error</a>.
    <b>let</b> rewards_numerator = (stake_amount <b>as</b> u128) * (rewards_rate <b>as</b> u128) * (num_successful_proposals <b>as</b> u128);
    <b>let</b> rewards_denominator = (rewards_rate_denominator <b>as</b> u128) * (num_total_proposals <b>as</b> u128);
    <b>if</b> (rewards_denominator &gt; 0) {
        ((rewards_numerator / rewards_denominator) <b>as</b> u64)
    } <b>else</b> {
        0
    }
}
</code></pre>



</details>

<a name="0x1_stake_distribute_rewards"></a>

## Function `distribute_rewards`

Mint rewards corresponding to current epoch's <code><a href="stake.md#0x1_stake">stake</a></code> and <code>num_successful_votes</code>.


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_distribute_rewards">distribute_rewards</a>(<a href="stake.md#0x1_stake">stake</a>: &<b>mut</b> <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;, num_successful_proposals: u64, num_total_proposals: u64, rewards_rate: u64, rewards_rate_denominator: u64): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_distribute_rewards">distribute_rewards</a>(
    <a href="stake.md#0x1_stake">stake</a>: &<b>mut</b> Coin&lt;AptosCoin&gt;,
    num_successful_proposals: u64,
    num_total_proposals: u64,
    rewards_rate: u64,
    rewards_rate_denominator: u64,
): u64 <b>acquires</b> <a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a> {
    <b>let</b> stake_amount = <a href="coin.md#0x1_coin_value">coin::value</a>(<a href="stake.md#0x1_stake">stake</a>);
    <b>let</b> rewards_amount = <b>if</b> (stake_amount &gt; 0) {
        <a href="stake.md#0x1_stake_calculate_rewards_amount">calculate_rewards_amount</a>(stake_amount, num_successful_proposals, num_total_proposals, rewards_rate, rewards_rate_denominator)
    } <b>else</b> {
        0
    };
    <b>if</b> (rewards_amount &gt; 0) {
        <b>let</b> mint_cap = &<b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a>&gt;(@aptos_framework).mint_cap;
        <b>let</b> rewards = <a href="coin.md#0x1_coin_mint">coin::mint</a>(rewards_amount, mint_cap);
        <a href="coin.md#0x1_coin_merge">coin::merge</a>(<a href="stake.md#0x1_stake">stake</a>, rewards);
    };
    rewards_amount
}
</code></pre>



</details>

<a name="0x1_stake_append"></a>

## Function `append`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_append">append</a>&lt;T&gt;(v1: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;, v2: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_append">append</a>&lt;T&gt;(v1: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;, v2: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;) {
    <b>while</b> (!<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_is_empty">vector::is_empty</a>(v2)) {
        <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_push_back">vector::push_back</a>(v1, <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_pop_back">vector::pop_back</a>(v2));
    }
}
</code></pre>



</details>

<a name="0x1_stake_find_validator"></a>

## Function `find_validator`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_find_validator">find_validator</a>(v: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>&gt;, addr: <b>address</b>): <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;u64&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_find_validator">find_validator</a>(v: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a>&gt;, addr: <b>address</b>): Option&lt;u64&gt; {
    <b>let</b> i = 0;
    <b>let</b> len = <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_length">vector::length</a>(v);
    <b>while</b> ({
        <b>spec</b> {
            <b>invariant</b> !(<b>exists</b> j in 0..i: v[j].addr == addr);
        };
        i &lt; len
    }) {
        <b>if</b> (<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_borrow">vector::borrow</a>(v, i).addr == addr) {
            <b>return</b> <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_some">option::some</a>(i)
        };
        i = i + 1;
    };
    <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_none">option::none</a>()
}
</code></pre>



</details>

<a name="0x1_stake_generate_validator_info"></a>

## Function `generate_validator_info`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_generate_validator_info">generate_validator_info</a>(addr: <b>address</b>, stake_pool: &<a href="stake.md#0x1_stake_StakePool">stake::StakePool</a>, config: <a href="stake.md#0x1_stake_ValidatorConfig">stake::ValidatorConfig</a>): <a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_generate_validator_info">generate_validator_info</a>(addr: <b>address</b>, stake_pool: &<a href="stake.md#0x1_stake_StakePool">StakePool</a>, config: <a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>): <a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a> {
    <b>let</b> voting_power = <a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool);
    <a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a> {
        addr,
        voting_power,
        config,
    }
}
</code></pre>



</details>

<a name="0x1_stake_get_next_epoch_voting_power"></a>

## Function `get_next_epoch_voting_power`

Returns validator's next epoch voting power, including pending_active, active, and pending_inactive stake.


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool: &<a href="stake.md#0x1_stake_StakePool">stake::StakePool</a>): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_get_next_epoch_voting_power">get_next_epoch_voting_power</a>(stake_pool: &<a href="stake.md#0x1_stake_StakePool">StakePool</a>): u64 {
    <b>let</b> value_pending_active = <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.pending_active);
    <b>let</b> value_active = <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.active);
    <b>let</b> value_pending_inactive = <a href="coin.md#0x1_coin_value">coin::value</a>(&stake_pool.pending_inactive);
    <b>spec</b> {
        <b>assume</b> value_pending_active + value_active + value_pending_inactive &lt;= MAX_U64;
    };
    value_pending_active + value_active + value_pending_inactive
}
</code></pre>



</details>

<a name="0x1_stake_update_voting_power_increase"></a>

## Function `update_voting_power_increase`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_voting_power_increase">update_voting_power_increase</a>(increase_amount: u64)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_voting_power_increase">update_voting_power_increase</a>(increase_amount: u64) <b>acquires</b> <a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a> {
    <b>let</b> validator_set = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
    <b>let</b> voting_power_increase_limit =
        (<a href="staking_config.md#0x1_staking_config_get_voting_power_increase_limit">staking_config::get_voting_power_increase_limit</a>(&<a href="staking_config.md#0x1_staking_config_get">staking_config::get</a>()) <b>as</b> u128);
    validator_set.total_joining_power = validator_set.total_joining_power + (increase_amount <b>as</b> u128);

    // Only validator <a href="voting.md#0x1_voting">voting</a> power increase <b>if</b> the current validator set's <a href="voting.md#0x1_voting">voting</a> power &gt; 0.
    <b>if</b> (validator_set.total_voting_power &gt; 0) {
        <b>assert</b>!(
            validator_set.total_joining_power &lt;= validator_set.total_voting_power * voting_power_increase_limit / 100,
            <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_EVOTING_POWER_INCREASE_EXCEEDS_LIMIT">EVOTING_POWER_INCREASE_EXCEEDS_LIMIT</a>),
        );
    }
}
</code></pre>



</details>

<a name="0x1_stake_assert_stake_pool_exists"></a>

## Function `assert_stake_pool_exists`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address: <b>address</b>) {
    <b>assert</b>!(<a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(pool_address), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_invalid_argument">error::invalid_argument</a>(<a href="stake.md#0x1_stake_ESTAKE_POOL_DOES_NOT_EXIST">ESTAKE_POOL_DOES_NOT_EXIST</a>));
}
</code></pre>



</details>

<a name="0x1_stake_configure_allowed_validators"></a>

## Function `configure_allowed_validators`



<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_configure_allowed_validators">configure_allowed_validators</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, accounts: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_configure_allowed_validators">configure_allowed_validators</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, accounts: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;) <b>acquires</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> {
    <b>let</b> aptos_framework_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(aptos_framework);
    <a href="system_addresses.md#0x1_system_addresses_assert_aptos_framework">system_addresses::assert_aptos_framework</a>(aptos_framework);
    <b>if</b> (!<b>exists</b>&lt;<a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>&gt;(aptos_framework_address)) {
        <b>move_to</b>(aptos_framework, <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> { accounts });
    } <b>else</b> {
        <b>let</b> allowed = <b>borrow_global_mut</b>&lt;<a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>&gt;(aptos_framework_address);
        allowed.accounts = accounts;
    }
}
</code></pre>



</details>

<a name="0x1_stake_is_allowed"></a>

## Function `is_allowed`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_is_allowed">is_allowed</a>(<a href="account.md#0x1_account">account</a>: <b>address</b>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_is_allowed">is_allowed</a>(<a href="account.md#0x1_account">account</a>: <b>address</b>): bool <b>acquires</b> <a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a> {
    <b>if</b> (!<b>exists</b>&lt;<a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>&gt;(@aptos_framework)) {
        <b>true</b>
    } <b>else</b> {
        <b>let</b> allowed = <b>borrow_global</b>&lt;<a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>&gt;(@aptos_framework);
        <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector_contains">vector::contains</a>(&allowed.accounts, &<a href="account.md#0x1_account">account</a>)
    }
}
</code></pre>



</details>

<a name="0x1_stake_assert_owner_cap_exists"></a>

## Function `assert_owner_cap_exists`



<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner: <b>address</b>)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner: <b>address</b>) {
    <b>assert</b>!(<b>exists</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner), <a href="../../aptos-stdlib/../move-stdlib/doc/error.md#0x1_error_not_found">error::not_found</a>(<a href="stake.md#0x1_stake_EOWNER_CAP_NOT_FOUND">EOWNER_CAP_NOT_FOUND</a>));
}
</code></pre>



</details>

<a name="@Specification_1"></a>

## Specification



<pre><code><b>invariant</b> [suspendable] <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework) ==&gt; <a href="stake.md#0x1_stake_validator_set_is_valid">validator_set_is_valid</a>();
<b>invariant</b> [suspendable] <a href="chain_status.md#0x1_chain_status_is_operating">chain_status::is_operating</a>() ==&gt; <b>exists</b>&lt;<a href="stake.md#0x1_stake_AptosCoinCapabilities">AptosCoinCapabilities</a>&gt;(@aptos_framework);
<b>invariant</b> [suspendable] <a href="chain_status.md#0x1_chain_status_is_operating">chain_status::is_operating</a>() ==&gt; <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(@aptos_framework);
<b>invariant</b> [suspendable] <a href="chain_status.md#0x1_chain_status_is_operating">chain_status::is_operating</a>() ==&gt; <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
</code></pre>




<a name="0x1_stake_validator_set_is_valid"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_validator_set_is_valid">validator_set_is_valid</a>(): bool {
   <b>let</b> validator_set = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
   <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(validator_set.active_validators) &&
       <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(validator_set.pending_inactive) &&
       <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(validator_set.pending_active) &&
       <a href="stake.md#0x1_stake_spec_validator_indices_are_valid">spec_validator_indices_are_valid</a>(validator_set.active_validators) &&
       <a href="stake.md#0x1_stake_spec_validator_indices_are_valid">spec_validator_indices_are_valid</a>(validator_set.pending_inactive)
}
</code></pre>



<a name="@Specification_1_get_validator_state"></a>

### Function `get_validator_state`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_get_validator_state">get_validator_state</a>(pool_address: <b>address</b>): u64
</code></pre>




<pre><code><b>let</b> validator_set = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
<b>ensures</b> result == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_ACTIVE">VALIDATOR_STATUS_PENDING_ACTIVE</a> ==&gt; <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_active, pool_address);
<b>ensures</b> result == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_ACTIVE">VALIDATOR_STATUS_ACTIVE</a> ==&gt; <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.active_validators, pool_address);
<b>ensures</b> result == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_PENDING_INACTIVE">VALIDATOR_STATUS_PENDING_INACTIVE</a> ==&gt; <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_inactive, pool_address);
<b>ensures</b> result == <a href="stake.md#0x1_stake_VALIDATOR_STATUS_INACTIVE">VALIDATOR_STATUS_INACTIVE</a> ==&gt; (
    !<a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_active, pool_address)
        && !<a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.active_validators, pool_address)
        && !<a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_inactive, pool_address)
);
</code></pre>



<a name="@Specification_1_initialize"></a>

### Function `initialize`


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_initialize">initialize</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>)
</code></pre>




<pre><code><b>let</b> aptos_addr = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(aptos_framework);
<b>aborts_if</b> !<a href="system_addresses.md#0x1_system_addresses_is_aptos_framework_address">system_addresses::is_aptos_framework_address</a>(aptos_addr);
<b>aborts_if</b> <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(aptos_addr);
<b>aborts_if</b> <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(aptos_addr);
</code></pre>



<a name="@Specification_1_remove_validators"></a>

### Function `remove_validators`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_remove_validators">remove_validators</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, validators: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;)
</code></pre>




<pre><code><b>requires</b> <a href="chain_status.md#0x1_chain_status_is_operating">chain_status::is_operating</a>();
</code></pre>



<a name="@Specification_1_initialize_stake_owner"></a>

### Function `initialize_stake_owner`


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_initialize_stake_owner">initialize_stake_owner</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, initial_stake_amount: u64, operator: <b>address</b>, voter: <b>address</b>)
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
</code></pre>



<a name="@Specification_1_extract_owner_cap"></a>

### Function `extract_owner_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_extract_owner_cap">extract_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>): <a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>
</code></pre>




<pre><code><b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
</code></pre>



<a name="@Specification_1_deposit_owner_cap"></a>

### Function `deposit_owner_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_deposit_owner_cap">deposit_owner_cap</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, owner_cap: <a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>




<pre><code><b>let</b> owner_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(owner);
<b>aborts_if</b> <b>exists</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner_address);
</code></pre>



<a name="@Specification_1_set_operator_with_cap"></a>

### Function `set_operator_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_operator_with_cap">set_operator_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, new_operator: <b>address</b>)
</code></pre>




<pre><code><b>let</b> pool_address = owner_cap.pool_address;
<b>let</b> <b>post</b> stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>ensures</b> stake_pool.operator_address == new_operator;
</code></pre>



<a name="@Specification_1_set_delegated_voter_with_cap"></a>

### Function `set_delegated_voter_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_set_delegated_voter_with_cap">set_delegated_voter_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, new_voter: <b>address</b>)
</code></pre>




<pre><code><b>let</b> pool_address = owner_cap.pool_address;
<b>let</b> <b>post</b> stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>ensures</b> stake_pool.delegated_voter == new_voter;
</code></pre>



<a name="@Specification_1_add_stake"></a>

### Function `add_stake`


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_add_stake">add_stake</a>(owner: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, amount: u64)
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
</code></pre>



<a name="@Specification_1_add_stake_with_cap"></a>

### Function `add_stake_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_add_stake_with_cap">add_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, coins: <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;)
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
</code></pre>



<a name="@Specification_1_reactivate_stake_with_cap"></a>

### Function `reactivate_stake_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_reactivate_stake_with_cap">reactivate_stake_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>, amount: u64)
</code></pre>




<pre><code><b>let</b> pool_address = owner_cap.pool_address;
<b>aborts_if</b> !<a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(pool_address);
<b>let</b> pre_stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> <b>post</b> stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> min_amount = aptos_std::math64::min(amount,pre_stake_pool.pending_inactive.value);
<b>ensures</b> stake_pool.active.value == pre_stake_pool.active.value + min_amount;
</code></pre>



<a name="@Specification_1_rotate_consensus_key"></a>

### Function `rotate_consensus_key`


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_rotate_consensus_key">rotate_consensus_key</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>, new_consensus_pubkey: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, proof_of_possession: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>




<pre><code><b>let</b> pre_stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> <b>post</b> validator_info = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>ensures</b> validator_info.consensus_pubkey == new_consensus_pubkey;
</code></pre>



<a name="@Specification_1_update_network_and_fullnode_addresses"></a>

### Function `update_network_and_fullnode_addresses`


<pre><code><b>public</b> entry <b>fun</b> <a href="stake.md#0x1_stake_update_network_and_fullnode_addresses">update_network_and_fullnode_addresses</a>(operator: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, pool_address: <b>address</b>, new_network_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, new_fullnode_addresses: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;)
</code></pre>




<pre><code><b>let</b> pre_stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> <b>post</b> validator_info = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>aborts_if</b> <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(operator) != pre_stake_pool.operator_address;
<b>ensures</b> validator_info.network_addresses == new_network_addresses;
<b>ensures</b> validator_info.fullnode_addresses == new_fullnode_addresses;
</code></pre>



<a name="@Specification_1_increase_lockup_with_cap"></a>

### Function `increase_lockup_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_increase_lockup_with_cap">increase_lockup_with_cap</a>(owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>




<pre><code><b>let</b> config = <b>global</b>&lt;<a href="staking_config.md#0x1_staking_config_StakingConfig">staking_config::StakingConfig</a>&gt;(@aptos_framework);
<b>let</b> pool_address = owner_cap.pool_address;
<b>let</b> pre_stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> <b>post</b> stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> now_seconds = <a href="timestamp.md#0x1_timestamp_spec_now_seconds">timestamp::spec_now_seconds</a>();
<b>let</b> lockup = config.recurring_lockup_duration_secs;
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>aborts_if</b> pre_stake_pool.locked_until_secs &gt;= lockup + now_seconds;
<b>aborts_if</b> lockup + now_seconds &gt; MAX_U64;
<b>aborts_if</b> !<b>exists</b>&lt;<a href="timestamp.md#0x1_timestamp_CurrentTimeMicroseconds">timestamp::CurrentTimeMicroseconds</a>&gt;(@aptos_framework);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="staking_config.md#0x1_staking_config_StakingConfig">staking_config::StakingConfig</a>&gt;(@aptos_framework);
<b>ensures</b> stake_pool.locked_until_secs == lockup + now_seconds;
</code></pre>



<a name="@Specification_1_unlock_with_cap"></a>

### Function `unlock_with_cap`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_unlock_with_cap">unlock_with_cap</a>(amount: u64, owner_cap: &<a href="stake.md#0x1_stake_OwnerCapability">stake::OwnerCapability</a>)
</code></pre>




<pre><code><b>let</b> pool_address = owner_cap.pool_address;
<b>let</b> pre_stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> <b>post</b> stake_pool = <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>modifies</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>let</b> min_amount = aptos_std::math64::min(amount,pre_stake_pool.active.value);
<b>ensures</b> stake_pool.pending_inactive.value == pre_stake_pool.pending_inactive.value + min_amount;
</code></pre>



<a name="@Specification_1_is_current_epoch_validator"></a>

### Function `is_current_epoch_validator`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_is_current_epoch_validator">is_current_epoch_validator</a>(pool_address: <b>address</b>): bool
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
<b>aborts_if</b> !<a href="stake.md#0x1_stake_spec_has_stake_pool">spec_has_stake_pool</a>(pool_address);
<b>ensures</b> result == <a href="stake.md#0x1_stake_spec_is_current_epoch_validator">spec_is_current_epoch_validator</a>(pool_address);
</code></pre>



<a name="@Specification_1_update_performance_statistics"></a>

### Function `update_performance_statistics`


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_update_performance_statistics">update_performance_statistics</a>(proposer_index: <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;u64&gt;, failed_proposer_indices: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u64&gt;)
</code></pre>




<pre><code><b>requires</b> <a href="chain_status.md#0x1_chain_status_is_operating">chain_status::is_operating</a>();
<b>aborts_if</b> <b>false</b>;
</code></pre>



<a name="@Specification_1_on_new_epoch"></a>

### Function `on_new_epoch`


<pre><code><b>public</b>(<b>friend</b>) <b>fun</b> <a href="stake.md#0x1_stake_on_new_epoch">on_new_epoch</a>()
</code></pre>




<pre><code><b>pragma</b> disable_invariants_in_body;
<b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
<b>aborts_if</b> <b>false</b>;
</code></pre>



<a name="@Specification_1_update_stake_pool"></a>

### Function `update_stake_pool`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_stake_pool">update_stake_pool</a>(validator_perf: &<a href="stake.md#0x1_stake_ValidatorPerformance">stake::ValidatorPerformance</a>, pool_address: <b>address</b>, <a href="staking_config.md#0x1_staking_config">staking_config</a>: &<a href="staking_config.md#0x1_staking_config_StakingConfig">staking_config::StakingConfig</a>)
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(pool_address);
<b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address);
<b>aborts_if</b> <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(pool_address).validator_index &gt;= len(validator_perf.validators);
</code></pre>



<a name="@Specification_1_calculate_rewards_amount"></a>

### Function `calculate_rewards_amount`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_calculate_rewards_amount">calculate_rewards_amount</a>(stake_amount: u64, num_successful_proposals: u64, num_total_proposals: u64, rewards_rate: u64, rewards_rate_denominator: u64): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
<b>requires</b> rewards_rate &lt;= <a href="stake.md#0x1_stake_MAX_REWARDS_RATE">MAX_REWARDS_RATE</a>;
<b>requires</b> rewards_rate_denominator &gt; 0;
<b>requires</b> rewards_rate &lt;= rewards_rate_denominator;
<b>requires</b> num_successful_proposals &lt;= num_total_proposals;
<b>ensures</b> [concrete] (rewards_rate_denominator * num_total_proposals == 0) ==&gt; result == 0;
<b>ensures</b> [concrete] (rewards_rate_denominator * num_total_proposals &gt; 0) ==&gt; {
    <b>let</b> amount = ((stake_amount * rewards_rate * num_successful_proposals) /
        (rewards_rate_denominator * num_total_proposals));
    result == amount
};
<b>aborts_if</b> <b>false</b>;
<b>ensures</b> [abstract] result == <a href="stake.md#0x1_stake_spec_rewards_amount">spec_rewards_amount</a>(
    stake_amount,
    num_successful_proposals,
    num_total_proposals,
    rewards_rate,
    rewards_rate_denominator);
</code></pre>



<a name="@Specification_1_distribute_rewards"></a>

### Function `distribute_rewards`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_distribute_rewards">distribute_rewards</a>(<a href="stake.md#0x1_stake">stake</a>: &<b>mut</b> <a href="coin.md#0x1_coin_Coin">coin::Coin</a>&lt;<a href="aptos_coin.md#0x1_aptos_coin_AptosCoin">aptos_coin::AptosCoin</a>&gt;, num_successful_proposals: u64, num_total_proposals: u64, rewards_rate: u64, rewards_rate_denominator: u64): u64
</code></pre>




<pre><code><b>include</b> <a href="stake.md#0x1_stake_ResourceRequirement">ResourceRequirement</a>;
<b>requires</b> rewards_rate &lt;= <a href="stake.md#0x1_stake_MAX_REWARDS_RATE">MAX_REWARDS_RATE</a>;
<b>requires</b> rewards_rate_denominator &gt; 0;
<b>requires</b> rewards_rate &lt;= rewards_rate_denominator;
<b>requires</b> num_successful_proposals &lt;= num_total_proposals;
<b>aborts_if</b> <b>false</b>;
<b>ensures</b> <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value) &gt; 0 ==&gt;
    result == <a href="stake.md#0x1_stake_spec_rewards_amount">spec_rewards_amount</a>(
        <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value),
        num_successful_proposals,
        num_total_proposals,
        rewards_rate,
        rewards_rate_denominator);
<b>ensures</b> <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value) &gt; 0 ==&gt;
    <a href="stake.md#0x1_stake">stake</a>.value == <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value) + <a href="stake.md#0x1_stake_spec_rewards_amount">spec_rewards_amount</a>(
        <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value),
        num_successful_proposals,
        num_total_proposals,
        rewards_rate,
        rewards_rate_denominator);
<b>ensures</b> <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value) == 0 ==&gt; result == 0;
<b>ensures</b> <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value) == 0 ==&gt; <a href="stake.md#0x1_stake">stake</a>.value == <b>old</b>(<a href="stake.md#0x1_stake">stake</a>.value);
</code></pre>



<a name="@Specification_1_append"></a>

### Function `append`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_append">append</a>&lt;T&gt;(v1: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;, v2: &<b>mut</b> <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;T&gt;)
</code></pre>




<pre><code><b>pragma</b> opaque, verify = <b>false</b>;
<b>aborts_if</b> <b>false</b>;
<b>ensures</b> len(v1) == <b>old</b>(len(v1) + len(v2));
<b>ensures</b> len(v2) == 0;
<b>ensures</b> (<b>forall</b> i in 0..<b>old</b>(len(v1)): v1[i] == <b>old</b>(v1[i]));
<b>ensures</b> (<b>forall</b> i in <b>old</b>(len(v1))..len(v1): v1[i] == <b>old</b>(v2[len(v2) - (i - len(v1)) - 1]));
</code></pre>



<a name="@Specification_1_find_validator"></a>

### Function `find_validator`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_find_validator">find_validator</a>(v: &<a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">stake::ValidatorInfo</a>&gt;, addr: <b>address</b>): <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;u64&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
<b>aborts_if</b> <b>false</b>;
<b>ensures</b> <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_none">option::is_none</a>(result) ==&gt; (<b>forall</b> i in 0..len(v): v[i].addr != addr);
<b>ensures</b> <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(result) ==&gt; v[<a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_borrow">option::borrow</a>(result)].addr == addr;
<b>ensures</b> <a href="../../aptos-stdlib/../move-stdlib/doc/option.md#0x1_option_is_some">option::is_some</a>(result) ==&gt; <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(v, addr);
</code></pre>



<a name="@Specification_1_update_voting_power_increase"></a>

### Function `update_voting_power_increase`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_update_voting_power_increase">update_voting_power_increase</a>(increase_amount: u64)
</code></pre>




<pre><code><b>let</b> aptos = @aptos_framework;
<b>let</b> pre_validator_set = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(aptos);
<b>let</b> <b>post</b> validator_set = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(aptos);
<b>let</b> <a href="staking_config.md#0x1_staking_config">staking_config</a> = <b>global</b>&lt;<a href="staking_config.md#0x1_staking_config_StakingConfig">staking_config::StakingConfig</a>&gt;(aptos);
<b>let</b> voting_power_increase_limit = <a href="staking_config.md#0x1_staking_config">staking_config</a>.voting_power_increase_limit;
<b>ensures</b> validator_set.total_voting_power &gt; 0 ==&gt; validator_set.total_joining_power &lt;= validator_set.total_voting_power * voting_power_increase_limit / 100;
<b>ensures</b> validator_set.total_joining_power == pre_validator_set.total_joining_power + increase_amount;
</code></pre>



<a name="@Specification_1_assert_stake_pool_exists"></a>

### Function `assert_stake_pool_exists`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_stake_pool_exists">assert_stake_pool_exists</a>(pool_address: <b>address</b>)
</code></pre>




<pre><code><b>aborts_if</b> !<a href="stake.md#0x1_stake_stake_pool_exists">stake_pool_exists</a>(pool_address);
</code></pre>



<a name="@Specification_1_configure_allowed_validators"></a>

### Function `configure_allowed_validators`


<pre><code><b>public</b> <b>fun</b> <a href="stake.md#0x1_stake_configure_allowed_validators">configure_allowed_validators</a>(aptos_framework: &<a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer">signer</a>, accounts: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<b>address</b>&gt;)
</code></pre>




<pre><code><b>let</b> aptos_framework_address = <a href="../../aptos-stdlib/../move-stdlib/doc/signer.md#0x1_signer_address_of">signer::address_of</a>(aptos_framework);
<b>aborts_if</b> !<a href="system_addresses.md#0x1_system_addresses_is_aptos_framework_address">system_addresses::is_aptos_framework_address</a>(aptos_framework_address);
<b>let</b> <b>post</b> allowed = <b>global</b>&lt;<a href="stake.md#0x1_stake_AllowedValidators">AllowedValidators</a>&gt;(aptos_framework_address);
<b>ensures</b> allowed.accounts == accounts;
</code></pre>



<a name="@Specification_1_assert_owner_cap_exists"></a>

### Function `assert_owner_cap_exists`


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_assert_owner_cap_exists">assert_owner_cap_exists</a>(owner: <b>address</b>)
</code></pre>




<pre><code><b>aborts_if</b> !<b>exists</b>&lt;<a href="stake.md#0x1_stake_OwnerCapability">OwnerCapability</a>&gt;(owner);
</code></pre>




<a name="0x1_stake_spec_validators_are_initialized"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_validators_are_initialized">spec_validators_are_initialized</a>(validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a>&gt;): bool {
   <b>forall</b> i in 0..len(validators):
       <a href="stake.md#0x1_stake_spec_has_stake_pool">spec_has_stake_pool</a>(validators[i].addr) &&
           <a href="stake.md#0x1_stake_spec_has_validator_config">spec_has_validator_config</a>(validators[i].addr)
}
</code></pre>




<a name="0x1_stake_spec_validator_indices_are_valid"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_validator_indices_are_valid">spec_validator_indices_are_valid</a>(validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a>&gt;): bool {
   <b>forall</b> i in 0..len(validators):
       <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(validators[i].addr).validator_index &lt; <a href="stake.md#0x1_stake_spec_validator_index_upper_bound">spec_validator_index_upper_bound</a>()
}
</code></pre>




<a name="0x1_stake_spec_validator_index_upper_bound"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_validator_index_upper_bound">spec_validator_index_upper_bound</a>(): u64 {
   len(<b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorPerformance">ValidatorPerformance</a>&gt;(@aptos_framework).validators)
}
</code></pre>




<a name="0x1_stake_spec_has_stake_pool"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_has_stake_pool">spec_has_stake_pool</a>(a: <b>address</b>): bool {
   <b>exists</b>&lt;<a href="stake.md#0x1_stake_StakePool">StakePool</a>&gt;(a)
}
</code></pre>




<a name="0x1_stake_spec_has_validator_config"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_has_validator_config">spec_has_validator_config</a>(a: <b>address</b>): bool {
   <b>exists</b>&lt;<a href="stake.md#0x1_stake_ValidatorConfig">ValidatorConfig</a>&gt;(a)
}
</code></pre>




<a name="0x1_stake_spec_rewards_amount"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_rewards_amount">spec_rewards_amount</a>(
   stake_amount: u64,
   num_successful_proposals: u64,
   num_total_proposals: u64,
   rewards_rate: u64,
   rewards_rate_denominator: u64,
): u64;
</code></pre>




<a name="0x1_stake_spec_contains"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validators: <a href="../../aptos-stdlib/../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="stake.md#0x1_stake_ValidatorInfo">ValidatorInfo</a>&gt;, addr: <b>address</b>): bool {
   <b>exists</b> i in 0..len(validators): validators[i].addr == addr
}
</code></pre>




<a name="0x1_stake_spec_is_current_epoch_validator"></a>


<pre><code><b>fun</b> <a href="stake.md#0x1_stake_spec_is_current_epoch_validator">spec_is_current_epoch_validator</a>(pool_address: <b>address</b>): bool {
   <b>let</b> validator_set = <b>global</b>&lt;<a href="stake.md#0x1_stake_ValidatorSet">ValidatorSet</a>&gt;(@aptos_framework);
   !<a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_active, pool_address)
       && (<a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.active_validators, pool_address)
       || <a href="stake.md#0x1_stake_spec_contains">spec_contains</a>(validator_set.pending_inactive, pool_address))
}
</code></pre>


[move-book]: https://move-language.github.io/move/introduction.html
