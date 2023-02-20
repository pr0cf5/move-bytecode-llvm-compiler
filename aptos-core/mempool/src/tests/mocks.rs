// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{
    core_mempool::{CoreMempool, TimelineState},
    shared_mempool::start_shared_mempool,
    MempoolClientSender, QuorumStoreRequest,
};
use anyhow::{format_err, Result};
use aptos_channels::{self, aptos_channel, message_queues::QueueStyle};
use aptos_config::{
    config::{NetworkConfig, NodeConfig},
    network_id::NetworkId,
};
use aptos_event_notifications::{ReconfigNotification, ReconfigNotificationListener};
use aptos_infallible::{Mutex, RwLock};
use aptos_mempool_notifications::{self, MempoolNotifier};
use aptos_network::{
    application::{
        interface::{NetworkClient, NetworkServiceEvents},
        storage::PeerMetadataStorage,
    },
    peer_manager::{conn_notifs_channel, ConnectionRequestSender, PeerManagerRequestSender},
    protocols::{
        network::{NetworkEvents, NetworkSender, NewNetworkEvents, NewNetworkSender},
        wire::handshake::v1::ProtocolId::MempoolDirectSend,
    },
};
use aptos_storage_interface::{mock::MockDbReaderWriter, DbReaderWriter};
use aptos_types::{
    account_config::AccountSequenceInfo, mempool_status::MempoolStatusCode,
    on_chain_config::OnChainConfigPayload, transaction::SignedTransaction,
};
use aptos_vm_validator::{
    mocks::mock_vm_validator::MockVMValidator, vm_validator::TransactionValidation,
};
use futures::channel::mpsc;
use maplit::hashmap;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::runtime::{Handle, Runtime};

/// Mock of a running instance of shared mempool.
pub struct MockSharedMempool {
    _runtime: Option<Runtime>,
    _handle: Option<Handle>,
    pub ac_client: MempoolClientSender,
    pub mempool: Arc<Mutex<CoreMempool>>,
    pub consensus_to_mempool_sender: mpsc::Sender<QuorumStoreRequest>,
    pub mempool_notifier: MempoolNotifier,
}

impl MockSharedMempool {
    /// Creates a mock of a running instance of shared mempool.
    /// Returns the runtime on which the shared mempool is running
    /// and the channel through which shared mempool receives client events.
    pub fn new() -> Self {
        let runtime = aptos_runtimes::spawn_named_runtime("shared-mem".into(), None);
        let (ac_client, mempool, quorum_store_sender, mempool_notifier) = Self::start(
            runtime.handle(),
            &DbReaderWriter::new(MockDbReaderWriter),
            MockVMValidator,
        );
        Self {
            _runtime: Some(runtime),
            _handle: None,
            ac_client,
            mempool,
            consensus_to_mempool_sender: quorum_store_sender,
            mempool_notifier,
        }
    }

    /// Creates a mock of a running instance of shared mempool inside a tokio runtime;
    /// Holds a runtime handle instead.
    pub fn new_in_runtime<V: TransactionValidation + 'static>(
        db: &DbReaderWriter,
        validator: V,
    ) -> Self {
        let handle = Handle::current();
        let (ac_client, mempool, quorum_store_sender, mempool_notifier) =
            Self::start(&handle, db, validator);
        Self {
            _runtime: None,
            _handle: Some(handle),
            ac_client,
            mempool,
            consensus_to_mempool_sender: quorum_store_sender,
            mempool_notifier,
        }
    }

    pub fn start<V: TransactionValidation + 'static>(
        handle: &Handle,
        db: &DbReaderWriter,
        validator: V,
    ) -> (
        MempoolClientSender,
        Arc<Mutex<CoreMempool>>,
        mpsc::Sender<QuorumStoreRequest>,
        MempoolNotifier,
    ) {
        let mut config = NodeConfig::random();
        config.validator_network = Some(NetworkConfig::network_with_id(NetworkId::Validator));

        let mempool = Arc::new(Mutex::new(CoreMempool::new(&config)));
        let (network_reqs_tx, _network_reqs_rx) = aptos_channel::new(QueueStyle::FIFO, 8, None);
        let (connection_reqs_tx, _) = aptos_channel::new(QueueStyle::FIFO, 8, None);
        let (_network_notifs_tx, network_notifs_rx) = aptos_channel::new(QueueStyle::FIFO, 8, None);
        let (_, conn_notifs_rx) = conn_notifs_channel::new();
        let network_sender = NetworkSender::new(
            PeerManagerRequestSender::new(network_reqs_tx),
            ConnectionRequestSender::new(connection_reqs_tx),
        );
        let network_events = NetworkEvents::new(network_notifs_rx, conn_notifs_rx);
        let (ac_client, client_events) = mpsc::channel(1_024);
        let (quorum_store_sender, quorum_store_receiver) = mpsc::channel(1_024);
        let (mempool_notifier, mempool_listener) =
            aptos_mempool_notifications::new_mempool_notifier_listener_pair();
        let (reconfig_sender, reconfig_events) = aptos_channel::new(QueueStyle::LIFO, 1, None);
        let reconfig_event_subscriber = ReconfigNotificationListener {
            notification_receiver: reconfig_events,
        };
        reconfig_sender
            .push((), ReconfigNotification {
                version: 1,
                on_chain_configs: OnChainConfigPayload::new(1, Arc::new(HashMap::new())),
            })
            .unwrap();
        let peer_metadata_storage = PeerMetadataStorage::new(&[NetworkId::Validator]);
        let network_senders = hashmap! {NetworkId::Validator => network_sender};
        let network_client = NetworkClient::new(
            vec![MempoolDirectSend],
            vec![],
            network_senders,
            peer_metadata_storage,
        );
        let network_and_events = hashmap! {NetworkId::Validator => network_events};
        let network_service_events = NetworkServiceEvents::new(network_and_events);

        start_shared_mempool(
            handle,
            &config,
            mempool.clone(),
            network_client,
            network_service_events,
            client_events,
            quorum_store_receiver,
            mempool_listener,
            reconfig_event_subscriber,
            db.reader.clone(),
            Arc::new(RwLock::new(validator)),
            vec![],
        );

        (ac_client, mempool, quorum_store_sender, mempool_notifier)
    }

    pub fn add_txns(&self, txns: Vec<SignedTransaction>) -> Result<()> {
        {
            let mut pool = self.mempool.lock();
            for txn in txns {
                if pool
                    .add_txn(
                        txn.clone(),
                        txn.gas_unit_price(),
                        AccountSequenceInfo::Sequential(0),
                        TimelineState::NotReady,
                    )
                    .code
                    != MempoolStatusCode::Accepted
                {
                    return Err(format_err!("failed to insert into mock mempool"));
                };
            }
        }
        Ok(())
    }

    pub fn get_txns(&self, size: u64) -> Vec<SignedTransaction> {
        let pool = self.mempool.lock();
        // assume txn size is less than 100kb
        pool.get_batch(size, size * 102400, HashSet::new())
    }

    pub fn remove_txn(&self, txn: &SignedTransaction) {
        let mut pool = self.mempool.lock();
        pool.commit_transaction(&txn.sender(), txn.sequence_number())
    }
}

impl Default for MockSharedMempool {
    fn default() -> Self {
        Self::new()
    }
}
