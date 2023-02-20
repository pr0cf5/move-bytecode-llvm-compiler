// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    emitter::{wait_for_single_account_sequence, RETRY_POLICY, SEND_AMOUNT},
    query_sequence_number, EmitJobRequest, EmitModeParams,
};
use anyhow::{anyhow, format_err, Context, Result};
use aptos::common::{types::EncodingType, utils::prompt_yes};
use aptos_crypto::ed25519::{Ed25519PrivateKey, Ed25519PublicKey};
use aptos_infallible::Mutex;
use aptos_logger::{debug, info, sample, sample::SampleRate, warn};
use aptos_rest_client::{aptos_api_types::AptosError, Client as RestClient};
use aptos_sdk::{
    transaction_builder::{aptos_stdlib, TransactionFactory},
    types::{
        transaction::{
            authenticator::{AuthenticationKey, AuthenticationKeyPreimage},
            SignedTransaction,
        },
        AccountKey, LocalAccount,
    },
};
use core::{
    cmp::min,
    result::Result::{Err, Ok},
};
use futures::StreamExt;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use std::{
    collections::HashMap,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

#[derive(Debug)]
pub struct AccountMinter<'t> {
    txn_factory: TransactionFactory,
    rng: StdRng,
    source_account: &'t mut LocalAccount,
}

impl<'t> AccountMinter<'t> {
    pub fn new(
        source_account: &'t mut LocalAccount,
        txn_factory: TransactionFactory,
        rng: StdRng,
    ) -> Self {
        Self {
            source_account,
            txn_factory,
            rng,
        }
    }

    /// workflow of create accounts:
    /// 1. Use given source_account as the money source
    /// 1a. Optionally, and if it is root account, mint balance to that account
    /// 2. load tc account to create seed accounts, one seed account for each endpoint
    /// 3. mint coins from faucet to new created seed accounts
    /// 4. split number of requested accounts into equally size of groups
    /// 5. each seed account take responsibility to create one size of group requested accounts and mint coins to them
    /// example:
    /// requested totally 100 new accounts with 10 endpoints
    /// will create 10 seed accounts, each seed account create 10 new accounts
    pub async fn create_accounts(
        &mut self,
        req: &EmitJobRequest,
        mode_params: &EmitModeParams,
        total_requested_accounts: usize,
    ) -> Result<Vec<LocalAccount>> {
        let mut accounts = vec![];
        let expected_num_seed_accounts =
            (total_requested_accounts / 50).clamp(1, CREATION_PARALLELISM);
        let num_accounts = total_requested_accounts - accounts.len(); // Only minting extra accounts
        let coins_per_account = (req.expected_max_txns / total_requested_accounts as u64)
            .checked_mul(SEND_AMOUNT + req.expected_gas_per_txn)
            .unwrap()
            .checked_add(aptos_global_constants::MAX_GAS_AMOUNT * req.gas_price)
            .unwrap(); // extra coins for secure to pay none zero gas price
        let txn_factory = self.txn_factory.clone();
        let expected_children_per_seed_account =
            (num_accounts + expected_num_seed_accounts - 1) / expected_num_seed_accounts;
        let coins_per_seed_account = (expected_children_per_seed_account as u64)
            .checked_mul(coins_per_account + req.expected_gas_per_txn)
            .unwrap()
            .checked_add(aptos_global_constants::MAX_GAS_AMOUNT * req.gas_price)
            .unwrap();
        let coins_for_source = coins_per_seed_account
            .checked_mul(expected_num_seed_accounts as u64)
            .unwrap()
            .checked_add(aptos_global_constants::MAX_GAS_AMOUNT * req.gas_price)
            .unwrap();
        info!(
            "Account creation plan created for {} accounts with {} balance each.",
            num_accounts, coins_per_account
        );
        info!(
            "    through {} seed accounts with {} each",
            expected_num_seed_accounts, coins_per_seed_account
        );
        info!(
            "    because of expecting {} txns and {} gas fee for each ",
            req.expected_max_txns, req.expected_gas_per_txn
        );

        if req.mint_to_root {
            self.mint_to_root(&req.rest_clients, coins_for_source)
                .await?;
        } else {
            let balance = &self
                .pick_mint_client(&req.rest_clients)
                .get_account_balance(self.source_account.address())
                .await?
                .into_inner();
            info!(
                "Source account {} current balance is {}, needed {} coins",
                self.source_account.address(),
                balance.get(),
                coins_for_source
            );

            if req.prompt_before_spending {
                if !prompt_yes(&format!(
                    "plan will consume in total {} balance, are you sure you want to proceed",
                    coins_for_source
                )) {
                    panic!("Aborting");
                }
            } else {
                let max_allowed = 2 * req
                    .expected_max_txns
                    .checked_mul(req.expected_gas_per_txn)
                    .unwrap();
                assert!(coins_for_source <= max_allowed,
                    "Estimated total coins needed for load test ({}) are larger than expected_max_txns * expected_gas_per_txn, multiplied by 2 to account for rounding up ({})",
                    coins_for_source,
                    max_allowed,
                );
            }

            if balance.get() < coins_for_source {
                return Err(anyhow!(
                    "Source ({}) doesn't have enough coins, balance {} < needed {}",
                    self.source_account.address(),
                    balance.get(),
                    coins_for_source
                ));
            }
        }

        let failed_requests = AtomicUsize::new(0);
        // Create seed accounts with which we can create actual accounts concurrently. Adding
        // additional fund for paying gas fees later.
        let seed_accounts = self
            .create_and_fund_seed_accounts(
                &req.rest_clients,
                expected_num_seed_accounts,
                coins_per_seed_account,
                mode_params.max_submit_batch_size,
                &failed_requests,
            )
            .await?;
        let actual_num_seed_accounts = seed_accounts.len();
        let num_new_child_accounts =
            (num_accounts + actual_num_seed_accounts - 1) / actual_num_seed_accounts;
        info!(
            "Completed creating {} seed accounts, each with {} coins, had to retry {} transactions",
            seed_accounts.len(),
            coins_per_seed_account,
            failed_requests.into_inner(),
        );
        info!(
            "Minting additional {} accounts with {} coins each",
            num_accounts, coins_per_account
        );

        let seed_rngs = gen_rng_for_reusable_account(actual_num_seed_accounts);
        let failed_requests = AtomicUsize::new(0);
        // For each seed account, create a future and transfer coins from that seed account to new accounts
        let account_futures = seed_accounts
            .into_iter()
            .enumerate()
            .map(|(i, seed_account)| {
                // Spawn new threads
                let index = i % req.rest_clients.len();
                let cur_client = req.rest_clients[index].clone();
                create_and_fund_new_accounts(
                    seed_account,
                    num_new_child_accounts,
                    coins_per_account,
                    mode_params.max_submit_batch_size,
                    cur_client,
                    &txn_factory,
                    req.reuse_accounts,
                    if req.reuse_accounts {
                        seed_rngs[i].clone()
                    } else {
                        StdRng::from_rng(self.rng()).unwrap()
                    },
                    &failed_requests,
                )
            });

        // Each future creates 10 accounts, limit concurrency to 100.
        let stream = futures::stream::iter(account_futures).buffer_unordered(CREATION_PARALLELISM);
        // wait for all futures to complete
        let mut minted_accounts = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .map_err(|e| format_err!("Failed to mint accounts: {:?}", e))?
            .into_iter()
            .flatten()
            .collect();

        accounts.append(&mut minted_accounts);
        assert!(
            accounts.len() >= num_accounts,
            "Something wrong in mint_account, wanted to mint {}, only have {}",
            total_requested_accounts,
            accounts.len()
        );
        info!(
            "Successfully completed creating accounts, had to retry {} transactions",
            failed_requests.into_inner()
        );
        Ok(accounts)
    }

    pub async fn mint_to_root(&mut self, rest_clients: &[RestClient], amount: u64) -> Result<()> {
        info!("Minting new coins to root");

        let txn = self
            .source_account
            .sign_with_transaction_builder(self.txn_factory.payload(
                aptos_stdlib::aptos_coin_mint(self.source_account.address(), amount),
            ));
        execute_and_wait_transactions(
            &self.pick_mint_client(rest_clients).clone(),
            self.source_account,
            vec![txn],
            &AtomicUsize::new(0),
        )
        .await?;

        Ok(())
    }

    pub async fn create_and_fund_seed_accounts(
        &mut self,
        rest_clients: &[RestClient],
        seed_account_num: usize,
        coins_per_seed_account: u64,
        max_submit_batch_size: usize,
        failed_requests: &AtomicUsize,
    ) -> Result<Vec<LocalAccount>> {
        info!("Creating and minting seeds accounts");
        let mut i = 0;
        let mut seed_accounts = vec![];
        while i < seed_account_num {
            let client = self.pick_mint_client(rest_clients).clone();
            let batch_size = min(max_submit_batch_size, seed_account_num - i);
            let mut rng = StdRng::from_rng(self.rng()).unwrap();
            let mut batch = gen_random_accounts(batch_size, &mut rng);
            let source_account = &mut self.source_account;
            let txn_factory = &self.txn_factory;
            let create_requests = batch
                .iter()
                .map(|account| {
                    create_and_fund_account_request(
                        source_account,
                        coins_per_seed_account,
                        account.public_key(),
                        txn_factory,
                    )
                })
                .collect();
            execute_and_wait_transactions(
                &client,
                source_account,
                create_requests,
                failed_requests,
            )
            .await?;
            i += batch_size;
            seed_accounts.append(&mut batch);
        }

        Ok(seed_accounts)
    }

    pub async fn load_vasp_account(
        &self,
        client: &RestClient,
        index: usize,
    ) -> Result<LocalAccount> {
        let file = "vasp".to_owned() + index.to_string().as_str() + ".key";
        let mint_key: Ed25519PrivateKey = EncodingType::BCS
            .load_key("vasp private key", Path::new(&file))
            .unwrap();
        let account_key = AccountKey::from_private_key(mint_key);
        let address = account_key.authentication_key().derived_address();
        let sequence_number = query_sequence_number(client, address).await.map_err(|e| {
            format_err!(
                "query_sequence_number on {:?} for dd account {} failed: {:?}",
                client.path_prefix_string(),
                index,
                e
            )
        })?;
        Ok(LocalAccount::new(address, account_key, sequence_number))
    }

    fn pick_mint_client<'a>(&mut self, clients: &'a [RestClient]) -> &'a RestClient {
        clients
            .choose(self.rng())
            .expect("json-rpc clients can not be empty")
    }

    pub fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

fn gen_rng_for_reusable_account(count: usize) -> Vec<StdRng> {
    // use same seed for reuse account creation and reuse
    // TODO: Investigate why we use the same seed and then consider changing
    // this so that we don't do this, since it causes conflicts between
    // runs of the emitter.
    let mut seed = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let mut rngs = vec![];
    for i in 0..count {
        seed[31] = i as u8;
        rngs.push(StdRng::from_seed(seed));
    }
    rngs
}

/// Create `num_new_accounts` by transferring coins from `source_account`. Return Vec of created
/// accounts
async fn create_and_fund_new_accounts<R>(
    mut source_account: LocalAccount,
    num_new_accounts: usize,
    coins_per_new_account: u64,
    max_num_accounts_per_batch: usize,
    client: RestClient,
    txn_factory: &TransactionFactory,
    reuse_account: bool,
    mut rng: R,
    failed_requests: &AtomicUsize,
) -> Result<Vec<LocalAccount>>
where
    R: ::rand_core::RngCore + ::rand_core::CryptoRng,
{
    let mut i = 0;
    let mut accounts = vec![];

    // Wait for source account to exist, this can happen because the corresponding REST endpoint might
    // not be up to date with the latest ledger state and requires some time for syncing.
    wait_for_single_account_sequence(&client, &source_account, Duration::from_secs(60)).await?;
    while i < num_new_accounts {
        let batch_size = min(max_num_accounts_per_batch, num_new_accounts - i);
        let mut batch = if reuse_account {
            info!("Loading {} accounts if they exist", batch_size);
            gen_reusable_accounts(&client, batch_size, &mut rng).await?
        } else {
            let batch = gen_random_accounts(batch_size, &mut rng);
            let creation_requests = batch
                .as_slice()
                .iter()
                .map(|account| {
                    create_and_fund_account_request(
                        &mut source_account,
                        coins_per_new_account,
                        account.public_key(),
                        txn_factory,
                    )
                })
                .collect();
            execute_and_wait_transactions(
                &client,
                &mut source_account,
                creation_requests,
                failed_requests,
            )
            .await
            .with_context(|| format!("Account {} couldn't mint", source_account.address()))?;
            batch
        };

        i += batch.len();
        accounts.append(&mut batch);
    }
    Ok(accounts)
}

async fn gen_reusable_accounts<R>(
    client: &RestClient,
    num_accounts: usize,
    rng: &mut R,
) -> Result<Vec<LocalAccount>>
where
    R: rand_core::RngCore + ::rand_core::CryptoRng,
{
    let mut vasp_accounts = vec![];
    let mut i = 0;
    while i < num_accounts {
        vasp_accounts.push(gen_reusable_account(client, rng).await?);
        i += 1;
    }
    Ok(vasp_accounts)
}

async fn gen_reusable_account<R>(client: &RestClient, rng: &mut R) -> Result<LocalAccount>
where
    R: ::rand_core::RngCore + ::rand_core::CryptoRng,
{
    let account_key = AccountKey::generate(rng);
    let address = account_key.authentication_key().derived_address();
    let sequence_number = query_sequence_number(client, address).await.unwrap_or(0);
    Ok(LocalAccount::new(address, account_key, sequence_number))
}

fn gen_random_accounts<R>(num_accounts: usize, rng: &mut R) -> Vec<LocalAccount>
where
    R: ::rand_core::RngCore + ::rand_core::CryptoRng,
{
    (0..num_accounts)
        .map(|_| LocalAccount::generate(rng))
        .collect()
}

pub fn create_and_fund_account_request(
    creation_account: &mut LocalAccount,
    amount: u64,
    pubkey: &Ed25519PublicKey,
    txn_factory: &TransactionFactory,
) -> SignedTransaction {
    let preimage = AuthenticationKeyPreimage::ed25519(pubkey);
    let auth_key = AuthenticationKey::from_preimage(&preimage);
    creation_account.sign_with_transaction_builder(txn_factory.payload(
        aptos_stdlib::aptos_account_transfer(auth_key.derived_address(), amount),
    ))
}

struct SubmittingTxnState {
    pub txns: Vec<SignedTransaction>,
    pub failures: Vec<Option<AptosError>>,
}

pub async fn execute_and_wait_transactions(
    client: &RestClient,
    account: &mut LocalAccount,
    txns: Vec<SignedTransaction>,
    failure_counter: &AtomicUsize,
) -> Result<()> {
    let start_seq_num = account.sequence_number();
    debug!(
        "[{:?}] Submitting transactions {} - {} for {}",
        client.path_prefix_string(),
        start_seq_num - txns.len() as u64,
        start_seq_num,
        account.address()
    );

    async fn submit_batch(
        client: &RestClient,
        state_mutex: &Mutex<SubmittingTxnState>,
        local_failure_counter: &AtomicUsize,
    ) -> Result<()> {
        let (indices, txns) = {
            let state = state_mutex.lock();
            let mut indices = Vec::new();
            let mut txns = Vec::new();
            for (i, txn) in state.txns.iter().enumerate() {
                if state.failures.get(i).map(|r| r.is_some()).unwrap_or(true) {
                    indices.push(i);
                    txns.push(txn.clone());
                }
            }
            (indices, txns)
        };

        let response = client.submit_batch_bcs(&txns).await;
        let results = match response {
            Err(e) => {
                warn!(
                    "[{:?}] Submitting transactions connection refused: {:?}, num txns: {}, first txn: {:?}",
                    client.path_prefix_string(), e, txns.len(), txns.first()
                );
                return Err(format_err!("{:?}", e));
            },
            Ok(result) => result.into_inner(),
        };
        let mut failures = results
            .transaction_failures
            .into_iter()
            .map(|f| (f.transaction_index, f.error))
            .collect::<HashMap<_, _>>();
        if !failures.is_empty() {
            local_failure_counter.fetch_add(failures.len(), Ordering::Relaxed);
            sample!(
                SampleRate::Duration(Duration::from_secs(120)),
                warn!(
                    "[{:?}] Submitting transactions failed for: {:?}",
                    client.path_prefix_string(),
                    failures,
                )
            );
        }
        let mut state = state_mutex.lock();
        for (request_idx, input_idx) in indices.iter().enumerate() {
            let value = failures.remove(&request_idx);
            if state.failures.len() == *input_idx {
                // We need to fill it up on the first call:
                state.failures.push(value);
            } else {
                state.failures[*input_idx] = value;
            }
        }

        if state.failures.iter().any(|r| r.is_some()) {
            Err(format_err!(""))
        } else {
            Ok(())
        }
    }

    let state = Mutex::new(SubmittingTxnState {
        txns,
        failures: vec![],
    });
    let state_ref = &state;
    let local_failure_counter = AtomicUsize::new(0);
    let local_failure_counter_ref = &local_failure_counter;
    let result = RETRY_POLICY
        .retry(move || submit_batch(client, state_ref, local_failure_counter_ref))
        .await
        .map_err(|e| {
            format_err!(
                "Failed to submit transactions: {:?}, {:?}",
                state
                    .lock()
                    .failures
                    .iter()
                    .enumerate()
                    .filter(|(_idx, r)| r.is_some())
                    .collect::<Vec<_>>(),
                e
            )
        });

    let local_failures = local_failure_counter.into_inner();
    if local_failures > 0 {
        sample!(
            SampleRate::Duration(Duration::from_secs(120)),
            warn!(
                "[{:?}] {} failures occured during submission.",
                client.path_prefix_string(),
                local_failures
            )
        );
        failure_counter.fetch_add(local_failures, Ordering::Relaxed);
    }
    // Log error, but not return, because timeout or other errors can commit the transaction in the background,
    // and the wait for transaction below will fail if transaction is not there.
    if let Err(e) = result {
        sample!(
            SampleRate::Duration(Duration::from_secs(120)),
            warn!(
                "[{:?}] Appears that we couldn't submit all transactions, rechecking. Details: {:?}",
                client.path_prefix_string(),
                e
            )
        );
    }

    let state = state.into_inner();

    for txn in state.txns.iter() {
        client
            .wait_for_transaction_by_hash_bcs(
                txn.clone().committed_hash(),
                txn.expiration_timestamp_secs(),
                Some(Duration::from_secs(120)),
                None,
            )
            .await
            .map_err(|e| {
                warn!(
                    "Failed to wait for transactions: {:?}, txn: {:?}. [{:?}] We were submitting transactions for account {}: from {} - {}, now at {}",
                    e,
                    txn,
                    client.path_prefix_string(),
                    account.address(),
                    start_seq_num - state.txns.len() as u64,
                    start_seq_num,
                    account.sequence_number()
                );

                // We shouldn't be able to reach this point.
                // This failure is unrecoverable, we end the test at this point.
                // It it sporadically happens in forge, and we need to debug why.
                // By default, we end the test and stop the nodes, before the next
                // counters poll happens after this.
                // Wait for 30s here, to make sure Grafana counters for expired transactions, etc,
                // get pulled from all the nodes, so we can investigate.
                std::thread::sleep(Duration::from_secs(30));
                format_err!(
                    "Failed to wait for transactions: {:?}",
                    e,
                )
            })?;
    }

    debug!(
        "[{:?}] Account {} is at sequence number {} now",
        client.path_prefix_string(),
        account.address(),
        account.sequence_number()
    );
    Ok(())
}

const CREATION_PARALLELISM: usize = 100;
