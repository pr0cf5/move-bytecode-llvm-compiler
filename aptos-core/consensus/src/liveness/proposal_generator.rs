// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use super::{
    proposer_election::ProposerElection, unequivocal_proposer_election::UnequivocalProposerElection,
};
use crate::{
    block_storage::BlockReader, counters::CHAIN_HEALTH_BACKOFF_TRIGGERED,
    state_replication::PayloadClient, util::time_service::TimeService,
};
use anyhow::{bail, ensure, format_err, Context};
use aptos_config::config::ChainHealthBackoffValues;
use aptos_consensus_types::{
    block::Block,
    block_data::BlockData,
    common::{Author, Payload, PayloadFilter, Round},
    quorum_cert::QuorumCert,
};
use aptos_logger::{error, sample, sample::SampleRate, warn};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[cfg(test)]
#[path = "proposal_generator_test.rs"]
mod proposal_generator_test;

#[derive(Clone)]
pub struct ChainHealthBackoffConfig {
    backoffs: BTreeMap<usize, ChainHealthBackoffValues>,
}

impl ChainHealthBackoffConfig {
    pub fn new(backoffs: Vec<ChainHealthBackoffValues>) -> Self {
        let original_len = backoffs.len();
        let backoffs = backoffs
            .into_iter()
            .map(|v| (v.backoff_if_below_participating_voting_power_percentage, v))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(original_len, backoffs.len());
        Self { backoffs }
    }

    #[allow(dead_code)]
    pub fn new_no_backoff() -> Self {
        Self {
            backoffs: BTreeMap::new(),
        }
    }

    pub fn get_backoff(
        &self,
        round: Round,
        proposer_election: &dyn ProposerElection,
    ) -> Option<&ChainHealthBackoffValues> {
        if self.backoffs.is_empty() {
            return None;
        }

        let voting_power_ratio = proposer_election.get_voting_power_participation_ratio(round);
        if voting_power_ratio < 2.0 / 3.0 {
            error!("Voting power ratio {} is below 2f + 1", voting_power_ratio);
        }
        let voting_power_percentage = (voting_power_ratio * 100.0).floor() as usize;
        if voting_power_percentage > 100 {
            error!(
                "Voting power participation percentatge {} is > 100, before rounding {}",
                voting_power_percentage, voting_power_ratio
            );
        }
        self.backoffs
            .range(voting_power_percentage..)
            .next()
            .map(|(_, v)| {
                sample!(
                    SampleRate::Duration(Duration::from_secs(10)),
                    warn!(
                        "Using chain health backoff config for {} voting power ratio: {:?}",
                        voting_power_percentage, v
                    )
                );
                v
            })
    }
}

/// ProposalGenerator is responsible for generating the proposed block on demand: it's typically
/// used by a validator that believes it's a valid candidate for serving as a proposer at a given
/// round.
/// ProposalGenerator is the one choosing the branch to extend:
/// - round is given by the caller (typically determined by RoundState).
/// The transactions for the proposed block are delivered by TxnManager.
///
/// TxnManager should be aware of the pending transactions in the branch that it is extending,
/// such that it will filter them out to avoid transaction duplication.
pub struct ProposalGenerator {
    // The account address of this validator
    author: Author,
    // Block store is queried both for finding the branch to extend and for generating the
    // proposed block.
    block_store: Arc<dyn BlockReader + Send + Sync>,
    // ProofOfStore manager is delivering the ProofOfStores.
    payload_client: Arc<dyn PayloadClient>,
    // Transaction manager is delivering the transactions.
    // Time service to generate block timestamps
    time_service: Arc<dyn TimeService>,
    // Max number of transactions to be added to a proposed block.
    max_block_txns: u64,
    // Max number of bytes to be added to a proposed block.
    max_block_bytes: u64,
    // Max number of failed authors to be added to a proposed block.
    max_failed_authors_to_store: usize,

    chain_health_backoff_config: ChainHealthBackoffConfig,

    // Last round that a proposal was generated
    last_round_generated: Round,
    quorum_store_enabled: bool,
}

impl ProposalGenerator {
    pub fn new(
        author: Author,
        block_store: Arc<dyn BlockReader + Send + Sync>,
        payload_client: Arc<dyn PayloadClient>,
        time_service: Arc<dyn TimeService>,
        max_block_txns: u64,
        max_block_bytes: u64,
        max_failed_authors_to_store: usize,
        chain_health_backoff_config: ChainHealthBackoffConfig,
        quorum_store_enabled: bool,
    ) -> Self {
        Self {
            author,
            block_store,
            payload_client,
            time_service,
            max_block_txns,
            max_block_bytes,
            max_failed_authors_to_store,
            chain_health_backoff_config,
            last_round_generated: 0,
            quorum_store_enabled,
        }
    }

    pub fn author(&self) -> Author {
        self.author
    }

    /// Creates a NIL block proposal extending the highest certified block from the block store.
    pub fn generate_nil_block(
        &self,
        round: Round,
        proposer_election: &mut UnequivocalProposerElection,
    ) -> anyhow::Result<Block> {
        let hqc = self.ensure_highest_quorum_cert(round)?;
        let quorum_cert = hqc.as_ref().clone();
        let failed_authors = self.compute_failed_authors(
            round, // to include current round, as that is what failed
            quorum_cert.certified_block().round(),
            true,
            proposer_election,
        );
        Ok(Block::new_nil(round, quorum_cert, failed_authors))
    }

    /// The function generates a new proposal block: the returned future is fulfilled when the
    /// payload is delivered by the TxnManager implementation.  At most one proposal can be
    /// generated per round (no proposal equivocation allowed).
    /// Errors returned by the TxnManager implementation are propagated to the caller.
    /// The logic for choosing the branch to extend is as follows:
    /// 1. The function gets the highest head of a one-chain from block tree.
    /// The new proposal must extend hqc to ensure optimistic responsiveness.
    /// 2. The round is provided by the caller.
    /// 3. In case a given round is not greater than the calculated parent, return an OldRound
    /// error.
    pub async fn generate_proposal(
        &mut self,
        round: Round,
        proposer_election: &mut UnequivocalProposerElection,
        wait_callback: BoxFuture<'static, ()>,
    ) -> anyhow::Result<BlockData> {
        if self.last_round_generated < round {
            self.last_round_generated = round;
        } else {
            bail!("Already proposed in the round {}", round);
        }

        let chain_health_backoff = self
            .chain_health_backoff_config
            .get_backoff(round, proposer_election);

        let hqc = self.ensure_highest_quorum_cert(round)?;

        let (payload, timestamp) = if hqc.certified_block().has_reconfiguration() {
            // Reconfiguration rule - we propose empty blocks with parents' timestamp
            // after reconfiguration until it's committed
            (
                Payload::empty(self.quorum_store_enabled),
                hqc.certified_block().timestamp_usecs(),
            )
        } else {
            // One needs to hold the blocks with the references to the payloads while get_block is
            // being executed: pending blocks vector keeps all the pending ancestors of the extended branch.
            let mut pending_blocks = self
                .block_store
                .path_from_commit_root(hqc.certified_block().id())
                .ok_or_else(|| format_err!("HQC {} already pruned", hqc.certified_block().id()))?;
            // Avoid txn manager long poll if the root block has txns, so that the leader can
            // deliver the commit proof to others without delay.
            pending_blocks.push(self.block_store.commit_root());

            // Exclude all the pending transactions: these are all the ancestors of
            // parent (including) up to the root (including).
            let exclude_payload: Vec<_> = pending_blocks
                .iter()
                .flat_map(|block| block.payload())
                .collect();
            let payload_filter = PayloadFilter::from(&exclude_payload);

            let pending_ordering = self
                .block_store
                .path_from_ordered_root(hqc.certified_block().id())
                .ok_or_else(|| format_err!("HQC {} already pruned", hqc.certified_block().id()))?
                .iter()
                .any(|block| !block.payload().map_or(true, |txns| txns.is_empty()));

            // All proposed blocks in a branch are guaranteed to have increasing timestamps
            // since their predecessor block will not be added to the BlockStore until
            // the local time exceeds it.
            let timestamp = self.time_service.get_current_timestamp();

            let (max_block_txns, max_block_bytes) = if let Some(value) = chain_health_backoff {
                let max_block_txns = self
                    .max_block_txns
                    .min(value.max_sending_block_txns_override);
                let max_block_bytes = self
                    .max_block_bytes
                    .min(value.max_sending_block_bytes_override);

                CHAIN_HEALTH_BACKOFF_TRIGGERED.inc();
                warn!(
                    "Generating proposal reducing limits to {} txns and {} bytes, due to chain health backoff",
                    max_block_txns,
                    max_block_bytes,
                );
                (max_block_txns, max_block_bytes)
            } else {
                (self.max_block_txns, self.max_block_bytes)
            };

            let payload = self
                .payload_client
                .pull_payload(
                    round,
                    max_block_txns,
                    max_block_bytes,
                    payload_filter,
                    wait_callback,
                    pending_ordering,
                )
                .await
                .context("Fail to retrieve payload")?;

            (payload, timestamp.as_micros() as u64)
        };

        let quorum_cert = hqc.as_ref().clone();
        let failed_authors = self.compute_failed_authors(
            round,
            quorum_cert.certified_block().round(),
            false,
            proposer_election,
        );
        // create block proposal
        Ok(BlockData::new_proposal(
            payload,
            self.author,
            failed_authors,
            round,
            timestamp,
            quorum_cert,
        ))
    }

    fn ensure_highest_quorum_cert(&self, round: Round) -> anyhow::Result<Arc<QuorumCert>> {
        let hqc = self.block_store.highest_quorum_cert();
        ensure!(
            hqc.certified_block().round() < round,
            "Given round {} is lower than hqc round {}",
            round,
            hqc.certified_block().round()
        );
        ensure!(
            !hqc.ends_epoch(),
            "The epoch has already ended,a proposal is not allowed to generated"
        );

        Ok(hqc)
    }

    /// Compute the list of consecutive proposers from the
    /// immediately preceeding rounds that didn't produce a successful block
    pub fn compute_failed_authors(
        &self,
        round: Round,
        previous_round: Round,
        include_cur_round: bool,
        proposer_election: &mut UnequivocalProposerElection,
    ) -> Vec<(Round, Author)> {
        let end_round = round + u64::from(include_cur_round);
        let mut failed_authors = Vec::new();
        let start = std::cmp::max(
            previous_round + 1,
            end_round.saturating_sub(self.max_failed_authors_to_store as u64),
        );
        for i in start..end_round {
            failed_authors.push((i, proposer_election.get_valid_proposer(i)));
        }

        failed_authors
    }
}
