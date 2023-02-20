// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for validator_network.

use crate::builder::NetworkBuilder;
use aptos_channels::aptos_channel;
use aptos_config::{
    config::{Peer, PeerRole, PeerSet, RoleType, NETWORK_CHANNEL_SIZE},
    network_id::{NetworkContext, NetworkId, PeerNetworkId},
};
use aptos_crypto::{test_utils::TEST_SEED, x25519, Uniform};
use aptos_infallible::RwLock;
use aptos_netcore::transport::ConnectionOrigin;
use aptos_network::{
    application::{interface::NetworkClient, storage::PeerMetadataStorage},
    peer_manager::builder::AuthenticationMode,
    protocols::network::{
        Event, NetworkApplicationConfig, NetworkClientConfig, NetworkEvents, NetworkServiceConfig,
    },
    ProtocolId,
};
use aptos_time_service::TimeService;
use aptos_types::{chain_id::ChainId, network_address::NetworkAddress, PeerId};
use futures::{executor::block_on, StreamExt};
use maplit::hashmap;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::runtime::Runtime;

const TEST_RPC_PROTOCOL: ProtocolId = ProtocolId::ConsensusRpcBcs;
const TEST_DIRECT_SEND_PROTOCOL: ProtocolId = ProtocolId::ConsensusDirectSendBcs;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct DummyMsg(pub Vec<u8>);

pub fn dummy_network_config() -> NetworkApplicationConfig {
    let direct_send_protocols = vec![TEST_DIRECT_SEND_PROTOCOL];
    let rpc_protocls = vec![TEST_RPC_PROTOCOL];

    let network_client_config =
        NetworkClientConfig::new(direct_send_protocols.clone(), rpc_protocls.clone());
    let network_service_config = NetworkServiceConfig::new(
        direct_send_protocols,
        rpc_protocls,
        aptos_channel::Config::new(NETWORK_CHANNEL_SIZE),
    );
    NetworkApplicationConfig::new(network_client_config, network_service_config)
}

/// TODO(davidiw): In DummyNetwork, replace DummyMsg with a Serde compatible type once migration
/// is complete
pub type DummyNetworkEvents = NetworkEvents<DummyMsg>;

pub struct DummyNetwork {
    pub runtime: Runtime,
    pub dialer_peer: PeerNetworkId,
    pub dialer_events: DummyNetworkEvents,
    pub dialer_network_client: NetworkClient<DummyMsg>,
    pub listener_peer: PeerNetworkId,
    pub listener_events: DummyNetworkEvents,
    pub listener_network_client: NetworkClient<DummyMsg>,
}

/// The following sets up a 2 peer network and verifies connectivity.
pub fn setup_network() -> DummyNetwork {
    let runtime = Runtime::new().unwrap();
    let role = RoleType::Validator;
    let network_id = NetworkId::Validator;
    let chain_id = ChainId::default();
    let dialer_peer = PeerNetworkId::new(network_id, PeerId::random());
    let listener_peer = PeerNetworkId::new(network_id, PeerId::random());

    // Setup keys for dialer.
    let mut rng = StdRng::from_seed(TEST_SEED);
    let dialer_identity_private_key = x25519::PrivateKey::generate(&mut rng);
    let dialer_identity_public_key = dialer_identity_private_key.public_key();
    let dialer_pubkeys: HashSet<_> = vec![dialer_identity_public_key].into_iter().collect();

    // Setup keys for listener.
    let listener_identity_private_key = x25519::PrivateKey::generate(&mut rng);

    // Setup listen addresses
    let dialer_addr: NetworkAddress = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
    let listener_addr: NetworkAddress = "/ip4/127.0.0.1/tcp/0".parse().unwrap();

    // Setup seed peers
    let mut seeds = PeerSet::new();
    seeds.insert(
        dialer_peer.peer_id(),
        Peer::new(vec![], dialer_pubkeys, PeerRole::Validator),
    );

    let trusted_peers = Arc::new(RwLock::new(HashMap::new()));
    let authentication_mode = AuthenticationMode::Mutual(listener_identity_private_key);
    let peer_metadata_storage = PeerMetadataStorage::new(&[network_id]);
    // Set up the listener network
    let network_context = NetworkContext::new(role, network_id, listener_peer.peer_id());
    let mut network_builder = NetworkBuilder::new_for_test(
        chain_id,
        seeds.clone(),
        trusted_peers,
        network_context,
        TimeService::real(),
        listener_addr,
        authentication_mode,
        peer_metadata_storage.clone(),
    );

    let (listener_sender, mut listener_events) =
        network_builder.add_client_and_service::<_, DummyNetworkEvents>(&dummy_network_config());
    network_builder.build(runtime.handle().clone()).start();
    let listener_network_client = NetworkClient::new(
        vec![TEST_DIRECT_SEND_PROTOCOL],
        vec![TEST_RPC_PROTOCOL],
        hashmap! {network_id => listener_sender},
        peer_metadata_storage,
    );

    // Add the listener address with port
    let listener_addr = network_builder.listen_address();
    seeds.insert(
        listener_peer.peer_id(),
        Peer::from_addrs(PeerRole::Validator, vec![listener_addr]),
    );

    let authentication_mode = AuthenticationMode::Mutual(dialer_identity_private_key);

    let peer_metadata_storage = PeerMetadataStorage::new(&[network_id]);
    // Set up the dialer network
    let network_context = NetworkContext::new(role, network_id, dialer_peer.peer_id());

    let trusted_peers = Arc::new(RwLock::new(HashMap::new()));

    let mut network_builder = NetworkBuilder::new_for_test(
        chain_id,
        seeds,
        trusted_peers,
        network_context,
        TimeService::real(),
        dialer_addr,
        authentication_mode,
        peer_metadata_storage.clone(),
    );

    let (dialer_sender, mut dialer_events) =
        network_builder.add_client_and_service::<_, DummyNetworkEvents>(&dummy_network_config());
    network_builder.build(runtime.handle().clone()).start();
    let dialer_network_client = NetworkClient::new(
        vec![TEST_DIRECT_SEND_PROTOCOL],
        vec![TEST_RPC_PROTOCOL],
        hashmap! {network_id => dialer_sender},
        peer_metadata_storage,
    );

    // Wait for establishing connection
    let first_dialer_event = block_on(dialer_events.next()).unwrap();
    if let Event::NewPeer(metadata) = first_dialer_event {
        assert_eq!(metadata.remote_peer_id, listener_peer.peer_id());
        assert_eq!(metadata.origin, ConnectionOrigin::Outbound);
        assert_eq!(metadata.role, PeerRole::Validator);
    } else {
        panic!(
            "No NewPeer event on dialer received instead: {:?}",
            first_dialer_event
        );
    }

    let first_listener_event = block_on(listener_events.next()).unwrap();
    if let Event::NewPeer(metadata) = first_listener_event {
        assert_eq!(metadata.remote_peer_id, dialer_peer.peer_id());
        assert_eq!(metadata.origin, ConnectionOrigin::Inbound);
        assert_eq!(metadata.role, PeerRole::Validator);
    } else {
        panic!(
            "No NewPeer event on listener received instead: {:?}",
            first_listener_event
        );
    }

    DummyNetwork {
        runtime,
        dialer_peer,
        dialer_events,
        dialer_network_client,
        listener_peer,
        listener_events,
        listener_network_client,
    }
}
