// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

//! Processes that are directly spawned by shared mempool runtime initialization
use super::types::MempoolClientRequest;
use crate::{
    core_mempool::{CoreMempool, TimelineState},
    counters,
    logging::{LogEntry, LogEvent, LogSchema},
    network::MempoolSyncMsg,
    shared_mempool::{
        tasks,
        tasks::process_committed_transactions,
        types::{notify_subscribers, ScheduledBroadcast, SharedMempool, SharedMempoolNotification},
    },
    MempoolEventsReceiver, QuorumStoreRequest,
};
use aptos_bounded_executor::BoundedExecutor;
use aptos_config::network_id::{NetworkId, PeerNetworkId};
use aptos_consensus_types::common::TransactionSummary;
use aptos_event_notifications::ReconfigNotificationListener;
use aptos_infallible::Mutex;
use aptos_logger::prelude::*;
use aptos_mempool_notifications::{MempoolCommitNotification, MempoolNotificationListener};
use aptos_network::{
    application::interface::{NetworkClientInterface, NetworkServiceEvents},
    protocols::network::Event,
};
use aptos_types::on_chain_config::OnChainConfigPayload;
use aptos_vm_validator::vm_validator::TransactionValidation;
use futures::{
    channel::mpsc,
    stream::{select_all, FuturesUnordered},
    StreamExt,
};
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::{runtime::Handle, time::interval};
use tokio_stream::wrappers::IntervalStream;

/// Coordinator that handles inbound network events and outbound txn broadcasts.
pub(crate) async fn coordinator<NetworkClient, TransactionValidator>(
    mut smp: SharedMempool<NetworkClient, TransactionValidator>,
    executor: Handle,
    network_service_events: NetworkServiceEvents<MempoolSyncMsg>,
    mut client_events: MempoolEventsReceiver,
    mut quorum_store_requests: mpsc::Receiver<QuorumStoreRequest>,
    mut mempool_listener: MempoolNotificationListener,
    mut mempool_reconfig_events: ReconfigNotificationListener,
) where
    NetworkClient: NetworkClientInterface<MempoolSyncMsg> + 'static,
    TransactionValidator: TransactionValidation + 'static,
{
    info!(LogSchema::event_log(
        LogEntry::CoordinatorRuntime,
        LogEvent::Start
    ));

    // Transform events to also include the network id
    let network_events: Vec<_> = network_service_events
        .into_network_and_events()
        .into_iter()
        .map(|(network_id, events)| events.map(move |event| (network_id, event)))
        .collect();
    let mut events = select_all(network_events).fuse();
    let mut scheduled_broadcasts = FuturesUnordered::new();

    // Use a BoundedExecutor to restrict only `workers_available` concurrent
    // worker tasks that can process incoming transactions.
    let workers_available = smp.config.shared_mempool_max_concurrent_inbound_syncs;
    let bounded_executor = BoundedExecutor::new(workers_available, executor.clone());

    let initial_reconfig = mempool_reconfig_events
        .next()
        .await
        .expect("Reconfig sender dropped, unable to start mempool");
    handle_mempool_reconfig_event(
        &mut smp,
        &bounded_executor,
        initial_reconfig.on_chain_configs,
    )
    .await;

    loop {
        let _timer = counters::MAIN_LOOP.start_timer();
        ::futures::select! {
            msg = client_events.select_next_some() => {
                handle_client_request(&mut smp, &bounded_executor, msg).await;
            },
            msg = quorum_store_requests.select_next_some() => {
                tasks::process_quorum_store_request(&smp, msg);
            },
            msg = mempool_listener.select_next_some() => {
                handle_commit_notification(&mut smp, msg, &mut mempool_listener);
            },
            reconfig_notification = mempool_reconfig_events.select_next_some() => {
                handle_mempool_reconfig_event(&mut smp, &bounded_executor, reconfig_notification.on_chain_configs).await;
            },
            (peer, backoff) = scheduled_broadcasts.select_next_some() => {
                tasks::execute_broadcast(peer, backoff, &mut smp, &mut scheduled_broadcasts, executor.clone()).await;
            },
            (network_id, event) = events.select_next_some() => {
                handle_network_event(&executor, &bounded_executor, &mut scheduled_broadcasts, &mut smp, network_id, event).await;
            },
            complete => break,
        }
    }
    error!(LogSchema::event_log(
        LogEntry::CoordinatorRuntime,
        LogEvent::Terminated
    ));
}

/// Spawn a task for processing `MempoolClientRequest`s from a client such as API service
async fn handle_client_request<NetworkClient, TransactionValidator>(
    smp: &mut SharedMempool<NetworkClient, TransactionValidator>,
    bounded_executor: &BoundedExecutor,
    request: MempoolClientRequest,
) where
    NetworkClient: NetworkClientInterface<MempoolSyncMsg> + 'static,
    TransactionValidator: TransactionValidation + 'static,
{
    match request {
        MempoolClientRequest::SubmitTransaction(txn, callback) => {
            // This timer measures how long it took for the bounded executor to *schedule* the
            // task.
            let _timer = counters::task_spawn_latency_timer(
                counters::CLIENT_EVENT_LABEL,
                counters::SPAWN_LABEL,
            );
            // This timer measures how long it took for the task to go from scheduled to started.
            let task_start_timer = counters::task_spawn_latency_timer(
                counters::CLIENT_EVENT_LABEL,
                counters::START_LABEL,
            );
            bounded_executor
                .spawn(tasks::process_client_transaction_submission(
                    smp.clone(),
                    txn,
                    callback,
                    task_start_timer,
                ))
                .await;
        },
        MempoolClientRequest::GetTransactionByHash(hash, callback) => {
            // This timer measures how long it took for the bounded executor to *schedule* the
            // task.
            let _timer = counters::task_spawn_latency_timer(
                counters::CLIENT_EVENT_GET_TXN_LABEL,
                counters::SPAWN_LABEL,
            );
            // This timer measures how long it took for the task to go from scheduled to started.
            let task_start_timer = counters::task_spawn_latency_timer(
                counters::CLIENT_EVENT_GET_TXN_LABEL,
                counters::START_LABEL,
            );
            bounded_executor
                .spawn(tasks::process_client_get_transaction(
                    smp.clone(),
                    hash,
                    callback,
                    task_start_timer,
                ))
                .await;
        },
    }
}

/// Handle removing committed transactions from local mempool immediately.  This should be done
/// immediately to ensure broadcasts of committed transactions stop as soon as possible.
fn handle_commit_notification<NetworkClient, TransactionValidator>(
    smp: &mut SharedMempool<NetworkClient, TransactionValidator>,
    msg: MempoolCommitNotification,
    mempool_listener: &mut MempoolNotificationListener,
) where
    NetworkClient: NetworkClientInterface<MempoolSyncMsg>,
    TransactionValidator: TransactionValidation,
{
    debug!(
        block_timestamp_usecs = msg.block_timestamp_usecs,
        num_committed_txns = msg.transactions.len(),
        LogSchema::event_log(LogEntry::StateSyncCommit, LogEvent::Received),
    );

    // Process and time committed user transactions.
    let start_time = Instant::now();
    counters::mempool_service_transactions(
        counters::COMMIT_STATE_SYNC_LABEL,
        msg.transactions.len(),
    );
    process_committed_transactions(
        &smp.mempool,
        msg.transactions
            .iter()
            .map(|txn| TransactionSummary {
                sender: txn.sender,
                sequence_number: txn.sequence_number,
            })
            .collect(),
        msg.block_timestamp_usecs,
    );
    smp.validator.write().notify_commit();
    let counter_result = if mempool_listener.ack_commit_notification(msg).is_err() {
        error!(LogSchema::event_log(
            LogEntry::StateSyncCommit,
            LogEvent::CallbackFail
        ));
        counters::REQUEST_FAIL_LABEL
    } else {
        counters::REQUEST_SUCCESS_LABEL
    };
    let latency = start_time.elapsed();
    counters::mempool_service_latency(counters::COMMIT_STATE_SYNC_LABEL, counter_result, latency);
}

/// Spawn a task to restart the transaction validator with the new reconfig data.
async fn handle_mempool_reconfig_event<NetworkClient, TransactionValidator>(
    smp: &mut SharedMempool<NetworkClient, TransactionValidator>,
    bounded_executor: &BoundedExecutor,
    config_update: OnChainConfigPayload,
) where
    NetworkClient: NetworkClientInterface<MempoolSyncMsg> + 'static,
    TransactionValidator: TransactionValidation + 'static,
{
    info!(LogSchema::event_log(
        LogEntry::ReconfigUpdate,
        LogEvent::Received
    ));
    let _timer =
        counters::task_spawn_latency_timer(counters::RECONFIG_EVENT_LABEL, counters::SPAWN_LABEL);

    bounded_executor
        .spawn(tasks::process_config_update(
            config_update,
            smp.validator.clone(),
            smp.broadcast_within_validator_network.clone(),
        ))
        .await;
}

/// Handles all NewPeer, LostPeer, and network messages.
/// - NewPeer events start new automatic broadcasts if the peer is upstream. If the peer is not upstream, we ignore it.
/// - LostPeer events disable the upstream peer, which will cancel ongoing broadcasts.
/// - Network messages follow a simple Request/Response framework to accept new transactions
/// TODO: Move to RPC off of DirectSend
async fn handle_network_event<NetworkClient, TransactionValidator>(
    executor: &Handle,
    bounded_executor: &BoundedExecutor,
    scheduled_broadcasts: &mut FuturesUnordered<ScheduledBroadcast>,
    smp: &mut SharedMempool<NetworkClient, TransactionValidator>,
    network_id: NetworkId,
    event: Event<MempoolSyncMsg>,
) where
    NetworkClient: NetworkClientInterface<MempoolSyncMsg> + 'static,
    TransactionValidator: TransactionValidation + 'static,
{
    match event {
        Event::NewPeer(metadata) => {
            counters::shared_mempool_event_inc("new_peer");
            let peer = PeerNetworkId::new(network_id, metadata.remote_peer_id);
            let is_new_peer = smp.network_interface.add_peer(peer, metadata.clone());
            let is_upstream_peer = smp
                .network_interface
                .is_upstream_peer(&peer, Some(&metadata));
            debug!(LogSchema::new(LogEntry::NewPeer)
                .peer(&peer)
                .is_upstream_peer(is_upstream_peer));
            notify_subscribers(SharedMempoolNotification::PeerStateChange, &smp.subscribers);
            if is_new_peer && is_upstream_peer {
                tasks::execute_broadcast(peer, false, smp, scheduled_broadcasts, executor.clone())
                    .await;
            }
        },
        Event::LostPeer(metadata) => {
            counters::shared_mempool_event_inc("lost_peer");
            let peer = PeerNetworkId::new(network_id, metadata.remote_peer_id);
            debug!(LogSchema::new(LogEntry::LostPeer)
                .peer(&peer)
                .is_upstream_peer(
                    smp.network_interface
                        .is_upstream_peer(&peer, Some(&metadata))
                ));
            smp.network_interface.disable_peer(peer);
            notify_subscribers(SharedMempoolNotification::PeerStateChange, &smp.subscribers);
        },
        Event::Message(peer_id, msg) => {
            counters::shared_mempool_event_inc("message");
            match msg {
                MempoolSyncMsg::BroadcastTransactionsRequest {
                    request_id,
                    transactions,
                } => {
                    let smp_clone = smp.clone();
                    let peer = PeerNetworkId::new(network_id, peer_id);
                    let ineligible_for_broadcast = (smp.network_interface.is_validator()
                        && !smp.broadcast_within_validator_network())
                        || smp.network_interface.is_upstream_peer(&peer, None);
                    let timeline_state = if ineligible_for_broadcast {
                        TimelineState::NonQualified
                    } else {
                        TimelineState::NotReady
                    };
                    // This timer measures how long it took for the bounded executor to
                    // *schedule* the task.
                    let _timer = counters::task_spawn_latency_timer(
                        counters::PEER_BROADCAST_EVENT_LABEL,
                        counters::SPAWN_LABEL,
                    );
                    // This timer measures how long it took for the task to go from scheduled
                    // to started.
                    let task_start_timer = counters::task_spawn_latency_timer(
                        counters::PEER_BROADCAST_EVENT_LABEL,
                        counters::START_LABEL,
                    );
                    bounded_executor
                        .spawn(tasks::process_transaction_broadcast(
                            smp_clone,
                            transactions,
                            request_id,
                            timeline_state,
                            peer,
                            task_start_timer,
                        ))
                        .await;
                },
                MempoolSyncMsg::BroadcastTransactionsResponse {
                    request_id,
                    retry,
                    backoff,
                } => {
                    let ack_timestamp = SystemTime::now();
                    smp.network_interface.process_broadcast_ack(
                        PeerNetworkId::new(network_id, peer_id),
                        request_id,
                        retry,
                        backoff,
                        ack_timestamp,
                    );
                },
            }
        },
        Event::RpcRequest(peer_id, _msg, _, _res_tx) => {
            counters::unexpected_msg_count_inc(&network_id);
            sample!(
                SampleRate::Duration(Duration::from_secs(60)),
                warn!(LogSchema::new(LogEntry::UnexpectedNetworkMsg)
                    .peer(&PeerNetworkId::new(network_id, peer_id)))
            );
        },
    }
}

/// Garbage collect all expired transactions by SystemTTL.
pub(crate) async fn gc_coordinator(mempool: Arc<Mutex<CoreMempool>>, gc_interval_ms: u64) {
    debug!(LogSchema::event_log(LogEntry::GCRuntime, LogEvent::Start));
    let mut interval = IntervalStream::new(interval(Duration::from_millis(gc_interval_ms)));
    while let Some(_interval) = interval.next().await {
        sample!(
            SampleRate::Duration(Duration::from_secs(60)),
            debug!(LogSchema::event_log(LogEntry::GCRuntime, LogEvent::Live))
        );
        mempool.lock().gc();
    }

    error!(LogSchema::event_log(
        LogEntry::GCRuntime,
        LogEvent::Terminated
    ));
}

/// Periodically logs a snapshot of transactions in core mempool.
/// In the future we may want an interactive way to directly query mempool's internal state.
/// For now, we will rely on this periodic snapshot to observe the internal state.
pub(crate) async fn snapshot_job(mempool: Arc<Mutex<CoreMempool>>, snapshot_interval_secs: u64) {
    let mut interval = IntervalStream::new(interval(Duration::from_secs(snapshot_interval_secs)));
    while let Some(_interval) = interval.next().await {
        let snapshot = mempool.lock().gen_snapshot();
        trace!(LogSchema::new(LogEntry::MempoolSnapshot).txns(snapshot));
    }
}
