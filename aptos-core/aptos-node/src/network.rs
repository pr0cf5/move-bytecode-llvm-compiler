// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use aptos_channels::{self, aptos_channel, message_queues::QueueStyle};
use aptos_config::{
    config::{NetworkConfig, NodeConfig},
    network_id::NetworkId,
};
use aptos_consensus::network_interface::{ConsensusMsg, DIRECT_SEND, RPC};
use aptos_event_notifications::EventSubscriptionService;
use aptos_logger::debug;
use aptos_mempool::network::MempoolSyncMsg;
use aptos_network::{
    application::{
        interface::{NetworkClient, NetworkServiceEvents},
        storage::PeerMetadataStorage,
    },
    protocols::network::{
        NetworkApplicationConfig, NetworkClientConfig, NetworkEvents, NetworkSender,
        NetworkServiceConfig,
    },
    ProtocolId,
};
use aptos_network_builder::builder::NetworkBuilder;
use aptos_storage_service_types::StorageServiceMessage;
use aptos_time_service::TimeService;
use aptos_types::chain_id::ChainId;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::runtime::Runtime;

// TODO: should the applications not make these configurable?
// Application network channel sizes
const CONSENSUS_NETWORK_CHANNEL_BUFFER_SIZE: usize = 1024;
const MEMPOOL_NETWORK_CHANNEL_BUFFER_SIZE: usize = 1024;

/// A simple struct that holds both the network client
/// and receiving interfaces for an application.
pub struct ApplicationNetworkInterfaces<T> {
    pub network_client: NetworkClient<T>,
    pub network_service_events: NetworkServiceEvents<T>,
}

/// A simple struct that holds an individual application
/// network handle (i.e., network id, sender and receiver).
struct ApplicationNetworkHandle<T> {
    pub network_id: NetworkId,
    pub network_sender: NetworkSender<T>,
    pub network_events: NetworkEvents<T>,
}

/// TODO: make this configurable (e.g., for compression)
/// Returns the network application config for the consensus client and service
pub fn consensus_network_configuration() -> NetworkApplicationConfig {
    let direct_send_protocols: Vec<ProtocolId> = DIRECT_SEND.into();
    let rpc_protocols: Vec<ProtocolId> = RPC.into();

    let network_client_config =
        NetworkClientConfig::new(direct_send_protocols.clone(), rpc_protocols.clone());
    let network_service_config = NetworkServiceConfig::new(
        direct_send_protocols,
        rpc_protocols,
        aptos_channel::Config::new(CONSENSUS_NETWORK_CHANNEL_BUFFER_SIZE)
            .queue_style(QueueStyle::FIFO)
            .counters(&aptos_consensus::counters::PENDING_CONSENSUS_NETWORK_EVENTS),
    );
    NetworkApplicationConfig::new(network_client_config, network_service_config)
}

/// Returns the network application config for the mempool client and service
pub fn mempool_network_configuration() -> NetworkApplicationConfig {
    let direct_send_protocols = vec![ProtocolId::MempoolDirectSend];
    let rpc_protocols = vec![]; // Mempool does not use RPC

    let network_client_config =
        NetworkClientConfig::new(direct_send_protocols.clone(), rpc_protocols.clone());
    let network_service_config = NetworkServiceConfig::new(
        direct_send_protocols,
        rpc_protocols,
        aptos_channel::Config::new(MEMPOOL_NETWORK_CHANNEL_BUFFER_SIZE)
            .queue_style(QueueStyle::KLAST) // TODO: why is this not FIFO?
            .counters(&aptos_mempool::counters::PENDING_MEMPOOL_NETWORK_EVENTS),
    );
    NetworkApplicationConfig::new(network_client_config, network_service_config)
}

/// Returns the network application config for the storage service client and server
pub fn storage_service_network_configuration(node_config: &NodeConfig) -> NetworkApplicationConfig {
    let direct_send_protocols = vec![]; // The storage service does not use direct send
    let rpc_protocols = vec![ProtocolId::StorageServiceRpc];
    let max_network_channel_size = node_config
        .state_sync
        .storage_service
        .max_network_channel_size as usize;

    let network_client_config =
        NetworkClientConfig::new(direct_send_protocols.clone(), rpc_protocols.clone());
    let network_service_config = NetworkServiceConfig::new(
        direct_send_protocols,
        rpc_protocols,
        aptos_channel::Config::new(max_network_channel_size)
            .queue_style(QueueStyle::FIFO)
            .counters(
                &aptos_storage_service_server::metrics::PENDING_STORAGE_SERVER_NETWORK_EVENTS,
            ),
    );
    NetworkApplicationConfig::new(network_client_config, network_service_config)
}

/// Extracts all network configs and ids from the given node config.
/// This method also does some basic verification of the network configs.
fn extract_network_configs_and_ids(
    node_config: &NodeConfig,
) -> (Vec<NetworkConfig>, Vec<NetworkId>) {
    // Extract all network configs
    let mut network_configs: Vec<NetworkConfig> = node_config.full_node_networks.to_vec();
    if let Some(network_config) = node_config.validator_network.as_ref() {
        // Ensure that mutual authentication is enabled by default!
        if !network_config.mutual_authentication {
            panic!("Validator networks must always have mutual_authentication enabled!");
        }
        network_configs.push(network_config.clone());
    }

    // Extract all network IDs
    let mut network_ids = vec![];
    for network_config in &network_configs {
        // Guarantee there is only one of this network
        let network_id = network_config.network_id;
        if network_ids.contains(&network_id) {
            panic!(
                "Duplicate NetworkId: '{}'. Can't start node with duplicate networks! Check the node config!",
                network_id
            );
        }
        network_ids.push(network_id);
    }

    (network_configs, network_ids)
}

/// Sets up all networks and returns the appropriate application network interfaces
pub fn setup_networks_and_get_interfaces(
    node_config: &NodeConfig,
    chain_id: ChainId,
    event_subscription_service: &mut EventSubscriptionService,
) -> (
    Vec<Runtime>,
    Option<ApplicationNetworkInterfaces<ConsensusMsg>>,
    ApplicationNetworkInterfaces<MempoolSyncMsg>,
    ApplicationNetworkInterfaces<StorageServiceMessage>,
) {
    // Gather all network configs and network ids
    let (network_configs, network_ids) = extract_network_configs_and_ids(node_config);

    // Create the global peer metadata storage
    let peer_metadata_storage = PeerMetadataStorage::new(&network_ids);

    // Create each network and register the application handles
    let mut network_runtimes = vec![];
    let mut consensus_network_handle = None;
    let mut mempool_network_handles = vec![];
    let mut storage_service_network_handles = vec![];
    for network_config in network_configs.into_iter() {
        // Create a network runtime for the config
        let runtime = create_network_runtime(&network_config);

        // Entering gives us a runtime to instantiate all the pieces of the builder
        let _enter = runtime.enter();

        // Create a new network builder
        let mut network_builder = NetworkBuilder::create(
            chain_id,
            node_config.base.role,
            &network_config,
            TimeService::real(),
            Some(event_subscription_service),
            peer_metadata_storage.clone(),
        );

        // Register consensus (both client and server) with the network
        let network_id = network_config.network_id;
        if network_id.is_validator_network() {
            // A validator node must have only a single consensus network handle
            if consensus_network_handle.is_some() {
                panic!("There can be at most one validator network!");
            } else {
                consensus_network_handle = Some(register_client_and_service_with_network(
                    &mut network_builder,
                    network_id,
                    consensus_network_configuration(),
                ));
            }
        }

        // Register mempool (both client and server) with the network
        let mempool_network_handle = register_client_and_service_with_network(
            &mut network_builder,
            network_id,
            mempool_network_configuration(),
        );
        mempool_network_handles.push(mempool_network_handle);

        // Register the storage service (both client and server) with the network
        let storage_service_network_handle = register_client_and_service_with_network(
            &mut network_builder,
            network_id,
            storage_service_network_configuration(node_config),
        );
        storage_service_network_handles.push(storage_service_network_handle);

        // Build and start the network on the runtime
        network_builder.build(runtime.handle().clone());
        network_builder.start();
        network_runtimes.push(runtime);
        debug!(
            "Network built for the network context: {}",
            network_builder.network_context()
        );
    }

    // Transform all network handles into application interfaces
    let (consensus_interfaces, mempool_interfaces, storage_service_interfaces) =
        transform_network_handles_into_interfaces(
            node_config,
            consensus_network_handle,
            mempool_network_handles,
            storage_service_network_handles,
            peer_metadata_storage,
        );

    (
        network_runtimes,
        consensus_interfaces,
        mempool_interfaces,
        storage_service_interfaces,
    )
}

/// Creates a network runtime for the given network config
fn create_network_runtime(network_config: &NetworkConfig) -> Runtime {
    let network_id = network_config.network_id;
    debug!("Creating runtime for network ID: {}", network_id);

    // Create the runtime
    let thread_name = format!(
        "network-{}",
        network_id.as_str().chars().take(3).collect::<String>()
    );
    aptos_runtimes::spawn_named_runtime(thread_name, network_config.runtime_threads)
}

/// Registers a new application client and service with the network
fn register_client_and_service_with_network<T: Serialize + for<'de> Deserialize<'de>>(
    network_builder: &mut NetworkBuilder,
    network_id: NetworkId,
    application_config: NetworkApplicationConfig,
) -> ApplicationNetworkHandle<T> {
    let (network_sender, network_events) =
        network_builder.add_client_and_service(&application_config);
    ApplicationNetworkHandle {
        network_id,
        network_sender,
        network_events,
    }
}

/// Tranforms the given network handles into interfaces that can
/// be used by the applications themselves.
fn transform_network_handles_into_interfaces(
    node_config: &NodeConfig,
    consensus_network_handle: Option<ApplicationNetworkHandle<ConsensusMsg>>,
    mempool_network_handles: Vec<ApplicationNetworkHandle<MempoolSyncMsg>>,
    storage_service_network_handles: Vec<ApplicationNetworkHandle<StorageServiceMessage>>,
    peer_metadata_storage: Arc<PeerMetadataStorage>,
) -> (
    Option<ApplicationNetworkInterfaces<ConsensusMsg>>,
    ApplicationNetworkInterfaces<MempoolSyncMsg>,
    ApplicationNetworkInterfaces<StorageServiceMessage>,
) {
    let consensus_interfaces = consensus_network_handle.map(|consensus_network_handle| {
        create_network_interfaces(
            vec![consensus_network_handle],
            consensus_network_configuration(),
            peer_metadata_storage.clone(),
        )
    });
    let mempool_interfaces = create_network_interfaces(
        mempool_network_handles,
        mempool_network_configuration(),
        peer_metadata_storage.clone(),
    );
    let storage_service_interfaces = create_network_interfaces(
        storage_service_network_handles,
        storage_service_network_configuration(node_config),
        peer_metadata_storage,
    );

    (
        consensus_interfaces,
        mempool_interfaces,
        storage_service_interfaces,
    )
}

/// Creates an application network inteface using the given
/// handles and config.
fn create_network_interfaces<
    T: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone + 'static,
>(
    network_handles: Vec<ApplicationNetworkHandle<T>>,
    network_application_config: NetworkApplicationConfig,
    peer_metadata_storage: Arc<PeerMetadataStorage>,
) -> ApplicationNetworkInterfaces<T> {
    // Gather the network senders and events
    let mut network_senders = HashMap::new();
    let mut network_and_events = HashMap::new();
    for network_handle in network_handles {
        let network_id = network_handle.network_id;
        network_senders.insert(network_id, network_handle.network_sender);
        network_and_events.insert(network_id, network_handle.network_events);
    }

    // Create the network client
    let network_client_config = network_application_config.network_client_config;
    let network_client = NetworkClient::new(
        network_client_config.direct_send_protocols_and_preferences,
        network_client_config.rpc_protocols_and_preferences,
        network_senders,
        peer_metadata_storage,
    );

    // Create the network service events
    let network_service_events = NetworkServiceEvents::new(network_and_events);

    // Create and return the new network interfaces
    ApplicationNetworkInterfaces {
        network_client,
        network_service_events,
    }
}
