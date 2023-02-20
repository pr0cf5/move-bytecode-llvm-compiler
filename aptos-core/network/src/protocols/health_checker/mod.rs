// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

//! Protocol used to ensure peer liveness
//!
//! The HealthChecker is responsible for ensuring liveness of all peers of a node.
//! It does so by periodically selecting a random connected peer and sending a Ping probe. A
//! healthy peer is expected to respond with a corresponding Pong message.
//!
//! If a certain number of successive liveness probes for a peer fail, the HealthChecker initiates a
//! disconnect from the peer. It relies on ConnectivityManager or the remote peer to re-establish
//! the connection.
//!
//! Future Work
//! -----------
//! We can make a few other improvements to the health checker. These are:
//! - Make the policy for interpreting ping failures pluggable
//! - Use successful inbound pings as a sign of remote note being healthy
//! - Ping a peer only in periods of no application-level communication with the peer
use crate::{
    application::interface::NetworkClientInterface,
    constants::NETWORK_CHANNEL_SIZE,
    counters,
    logging::NetworkSchema,
    protocols::{
        health_checker::interface::{HealthCheckData, HealthCheckNetworkInterface},
        network::{
            Event, NetworkApplicationConfig, NetworkClientConfig, NetworkEvents,
            NetworkServiceConfig,
        },
        rpc::error::RpcError,
    },
    ProtocolId,
};
use aptos_channels::{aptos_channel, message_queues::QueueStyle};
use aptos_config::network_id::{NetworkContext, PeerNetworkId};
use aptos_logger::prelude::*;
use aptos_short_hex_str::AsShortHexStr;
use aptos_time_service::{TimeService, TimeServiceTrait};
use aptos_types::PeerId;
use bytes::Bytes;
use futures::{
    channel::oneshot,
    stream::{FuturesUnordered, StreamExt},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::hash_map::Entry, time::Duration};

pub mod builder;
mod interface;
#[cfg(test)]
mod test;

/// The interface from Network to HealthChecker layer.
///
/// `HealthCheckerNetworkEvents` is a `Stream` of `PeerManagerNotification` where the
/// raw `Bytes` rpc messages are deserialized into
/// `HealthCheckerMsg` types. `HealthCheckerNetworkEvents` is a thin wrapper
/// around an `channel::Receiver<PeerManagerNotification>`.
pub type HealthCheckerNetworkEvents = NetworkEvents<HealthCheckerMsg>;

/// Returns a network application config for the health check client and service
pub fn health_checker_network_config() -> NetworkApplicationConfig {
    let direct_send_protocols = vec![]; // Health checker doesn't use direct send
    let rpc_protocols = vec![ProtocolId::HealthCheckerRpc];

    let network_client_config =
        NetworkClientConfig::new(direct_send_protocols.clone(), rpc_protocols.clone());
    let network_service_config = NetworkServiceConfig::new(
        direct_send_protocols,
        rpc_protocols,
        aptos_channel::Config::new(NETWORK_CHANNEL_SIZE)
            .queue_style(QueueStyle::LIFO)
            .counters(&counters::PENDING_HEALTH_CHECKER_NETWORK_EVENTS),
    );
    NetworkApplicationConfig::new(network_client_config, network_service_config)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum HealthCheckerMsg {
    Ping(Ping),
    Pong(Pong),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Ping(u32);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pong(u32);

/// The actor performing health checks by running the Ping protocol
pub struct HealthChecker<NetworkClient> {
    network_context: NetworkContext,
    /// A handle to a time service for easily mocking time-related operations.
    time_service: TimeService,
    /// Network interface to send requests to the Network Layer
    network_interface: HealthCheckNetworkInterface<NetworkClient>,
    /// Random-number generator.
    rng: SmallRng,
    /// Time we wait between each set of pings.
    ping_interval: Duration,
    /// Ping timeout duration.
    ping_timeout: Duration,
    /// Number of successive ping failures we tolerate before declaring a node as unhealthy and
    /// disconnecting from it. In the future, this can be replaced with a more general failure
    /// detection policy.
    ping_failures_tolerated: u64,
    /// Counter incremented in each round of health checks
    round: u64,
}

impl<NetworkClient: NetworkClientInterface<HealthCheckerMsg> + Unpin> HealthChecker<NetworkClient> {
    /// Create new instance of the [`HealthChecker`] actor.
    pub fn new(
        network_context: NetworkContext,
        time_service: TimeService,
        network_interface: HealthCheckNetworkInterface<NetworkClient>,
        ping_interval: Duration,
        ping_timeout: Duration,
        ping_failures_tolerated: u64,
    ) -> Self {
        HealthChecker {
            network_context,
            time_service,
            network_interface,
            rng: SmallRng::from_entropy(),
            ping_interval,
            ping_timeout,
            ping_failures_tolerated,
            round: 0,
        }
    }

    pub async fn start(mut self) {
        let mut tick_handlers = FuturesUnordered::new();
        info!(
            NetworkSchema::new(&self.network_context),
            "{} Health checker actor started", self.network_context
        );

        let ticker = self.time_service.interval(self.ping_interval);
        tokio::pin!(ticker);

        loop {
            futures::select! {
                maybe_event = self.network_interface.next() => {
                    // Shutdown the HealthChecker when this network instance shuts
                    // down. This happens when the `PeerManager` drops.
                    let event = match maybe_event {
                        Some(event) => event,
                        None => break,
                    };

                    match event {
                        Event::NewPeer(metadata) => {
                            self.network_interface.health_check_data().insert(
                                metadata.remote_peer_id,
                                HealthCheckData::new(self.round)
                            );
                        }
                        Event::LostPeer(metadata) => {
                            self.network_interface.health_check_data().remove(
                                &metadata.remote_peer_id
                            );
                        }
                        Event::RpcRequest(peer_id, msg, protocol, res_tx) => {
                            match msg {
                                HealthCheckerMsg::Ping(ping) => self.handle_ping_request(peer_id, ping, protocol, res_tx),
                                _ => {
                                    warn!(
                                        SecurityEvent::InvalidHealthCheckerMsg,
                                        NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
                                        rpc_message = msg,
                                        "{} Unexpected RPC message from {}",
                                        self.network_context,
                                        peer_id
                                    );
                                    debug_assert!(false, "Unexpected rpc request");
                                }
                            };
                        }
                        Event::Message(peer_id, msg) => {
                            error!(
                                SecurityEvent::InvalidNetworkEventHC,
                                NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
                                "{} Unexpected direct send from {} msg {:?}",
                                self.network_context,
                                peer_id,
                                msg,
                            );
                            debug_assert!(false, "Unexpected network event");
                        }
                    }
                }
                _ = ticker.select_next_some() => {
                    self.round += 1;
                    let connected = self.network_interface.connected_peers();
                    if connected.is_empty() {
                        trace!(
                            NetworkSchema::new(&self.network_context),
                            round = self.round,
                            "{} No connected peer to ping round: {}",
                            self.network_context,
                            self.round
                        );
                        continue
                    }

                    for peer_id in connected {
                        let nonce = self.rng.gen::<u32>();
                        trace!(
                            NetworkSchema::new(&self.network_context),
                            round = self.round,
                            "{} Will ping: {} for round: {} nonce: {}",
                            self.network_context,
                            peer_id.short_str(),
                            self.round,
                            nonce
                        );

                        tick_handlers.push(Self::ping_peer(
                            self.network_context,
                            self.network_interface.network_client(),
                            peer_id,
                            self.round,
                            nonce,
                            self.ping_timeout,
                        ));
                    }
                }
                res = tick_handlers.select_next_some() => {
                    let (peer_id, round, nonce, ping_result) = res;
                    self.handle_ping_response(peer_id, round, nonce, ping_result).await;
                }
            }
        }
        warn!(
            NetworkSchema::new(&self.network_context),
            "{} Health checker actor terminated", self.network_context
        );
    }

    fn handle_ping_request(
        &mut self,
        peer_id: PeerId,
        ping: Ping,
        protocol: ProtocolId,
        res_tx: oneshot::Sender<Result<Bytes, RpcError>>,
    ) {
        let message = match protocol.to_bytes(&HealthCheckerMsg::Pong(Pong(ping.0))) {
            Ok(msg) => msg,
            Err(e) => {
                warn!(
                    NetworkSchema::new(&self.network_context),
                    error = ?e,
                    "{} Unable to serialize pong response: {}", self.network_context, e
                );
                return;
            },
        };
        trace!(
            NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
            "{} Sending Pong response to peer: {} with nonce: {}",
            self.network_context,
            peer_id.short_str(),
            ping.0,
        );
        // Record Ingress HC here and reset failures.
        let _ = self
            .network_interface
            .health_check_data()
            .write(peer_id, |entry| {
                match entry {
                    Entry::Vacant(..) => {
                        // Don't do anything if there isn't an entry
                    },
                    Entry::Occupied(inner) => {
                        let data = inner.get_mut();
                        data.failures = 0;
                    },
                };
                Ok(())
            });

        let _ = res_tx.send(Ok(message.into()));
    }

    async fn handle_ping_response(
        &mut self,
        peer_id: PeerId,
        round: u64,
        req_nonce: u32,
        ping_result: Result<Pong, RpcError>,
    ) {
        match ping_result {
            Ok(pong) => {
                if pong.0 == req_nonce {
                    trace!(
                        NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
                        rount = round,
                        "{} Ping successful for peer: {} round: {}",
                        self.network_context,
                        peer_id.short_str(),
                        round
                    );
                    // Update last successful ping to current round.
                    // If it's not in storage, don't bother updating it
                    let _ = self
                        .network_interface
                        .health_check_data()
                        .write(peer_id, |entry| {
                            match entry {
                                Entry::Vacant(..) => {
                                    // Don't do anything if there isn't an entry
                                },
                                Entry::Occupied(inner) => {
                                    let data = inner.get_mut();
                                    // Update state if it's a newer round
                                    if round > data.round {
                                        data.round = round;
                                        data.failures = 0;
                                    }
                                },
                            };
                            Ok(())
                        });
                } else {
                    warn!(
                        SecurityEvent::InvalidHealthCheckerMsg,
                        NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
                        "{} Pong nonce doesn't match Ping nonce. Round: {}, Pong: {}, Ping: {}",
                        self.network_context,
                        round,
                        pong.0,
                        req_nonce
                    );
                    debug_assert!(false, "Pong nonce doesn't match our challenge Ping nonce");
                }
            },
            Err(err) => {
                warn!(
                    NetworkSchema::new(&self.network_context)
                        .remote_peer(&peer_id),
                    error = ?err,
                    round = round,
                    "{} Ping failed for peer: {} round: {} with error: {:?}",
                    self.network_context,
                    peer_id.short_str(),
                    round,
                    err
                );
                let _ = self
                    .network_interface
                    .health_check_data()
                    .write(peer_id, |entry| {
                        // Don't add in a default in case it's already disconnected
                        match entry {
                            Entry::Vacant(..) => {
                                // Don't do anything if there isn't an entry
                            },
                            Entry::Occupied(inner) => {
                                // If this is the result of an older ping, we ignore it.
                                // Increment num of failures.
                                let data = inner.get_mut();
                                if data.round <= round {
                                    data.failures += 1;
                                }
                            },
                        }
                        Ok(())
                    });

                // If the ping failures are now more than
                // `self.ping_failures_tolerated`, we disconnect from the node.
                // The HealthChecker only performs the disconnect. It relies on
                // ConnectivityManager or the remote peer to re-establish the connection.
                let failures = self
                    .network_interface
                    .health_check_data()
                    .read(&peer_id)
                    .map(|data| data.failures)
                    .unwrap_or(0);
                if failures > self.ping_failures_tolerated {
                    info!(
                        NetworkSchema::new(&self.network_context).remote_peer(&peer_id),
                        "{} Disconnecting from peer: {}",
                        self.network_context,
                        peer_id.short_str()
                    );
                    let peer_network_id =
                        PeerNetworkId::new(self.network_context.network_id(), peer_id);
                    if let Err(err) = self
                        .network_interface
                        .disconnect_peer(peer_network_id)
                        .await
                    {
                        warn!(
                            NetworkSchema::new(&self.network_context)
                                .remote_peer(&peer_id),
                            error = ?err,
                            "{} Failed to disconnect from peer: {} with error: {:?}",
                            self.network_context,
                            peer_id.short_str(),
                            err
                        );
                    }
                }
            },
        }
    }

    async fn ping_peer(
        network_context: NetworkContext,
        network_client: NetworkClient, // TODO: we shouldn't need to pass the client directly
        peer_id: PeerId,
        round: u64,
        nonce: u32,
        ping_timeout: Duration,
    ) -> (PeerId, u64, u32, Result<Pong, RpcError>) {
        trace!(
            NetworkSchema::new(&network_context).remote_peer(&peer_id),
            round = round,
            "{} Sending Ping request to peer: {} for round: {} nonce: {}",
            network_context,
            peer_id.short_str(),
            round,
            nonce
        );
        let peer_network_id = PeerNetworkId::new(network_context.network_id(), peer_id);
        let res_pong_msg = network_client
            .send_to_peer_rpc(
                HealthCheckerMsg::Ping(Ping(nonce)),
                ping_timeout,
                peer_network_id,
            )
            .await
            .map_err(|error| RpcError::Error(error.into()))
            .and_then(|msg| match msg {
                HealthCheckerMsg::Pong(res) => Ok(res),
                _ => Err(RpcError::InvalidRpcResponse),
            });
        (peer_id, round, nonce, res_pong_msg)
    }
}
