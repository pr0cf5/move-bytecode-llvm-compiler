// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{LoadDestination, NetworkLoadTest};
use aptos_forge::{
    GroupNetworkDelay, NetworkContext, NetworkTest, Swarm, SwarmChaos, SwarmExt,
    SwarmNetworkBandwidth, SwarmNetworkDelay, Test,
};
use aptos_logger::info;
use rand::Rng;
use tokio::runtime::Runtime;

/// Config for additing variable processing overhead/delay into
/// execution, to make different nodes have different processing speed.
pub struct ExecutionDelayConfig {
    /// Fraction (0.0 - 1.0) of nodes on which any delay will be introduced
    pub inject_delay_node_fraction: f64,
    /// For nodes with delay, what percentage (0-100) of transaction will be delayed.
    /// (this is needed because delay that can be introduced is integer number of ms)
    /// Different node speed come from this setting, each node is selected a number
    /// between 1 and given max.
    pub inject_delay_max_transaction_percentage: u32,
    /// Fixed busy-loop delay applied to each transaction that is delayed,
    /// before it is executed.
    pub inject_delay_per_transaction_ms: u32,
}

pub struct ThreeRegionSimulationTest {
    pub add_execution_delay: Option<ExecutionDelayConfig>,
}

impl Test for ThreeRegionSimulationTest {
    fn name(&self) -> &'static str {
        "network::three-region-simulation"
    }
}

/// Create a SwarmNetworkDelay with the following topology:
/// 1. 3 equal size group of nodes, each in a different region
/// 2. Each region has minimal network delay amongst its nodes
/// 3. Each region has a network delay to the other two regions, as estimated by https://www.cloudping.co/grid
/// 4. Currently simulating a 50 percentile network delay between us-west <--> af-south <--> eu-north
fn create_three_region_swarm_network_delay(swarm: &dyn Swarm) -> SwarmNetworkDelay {
    let all_validators = swarm.validators().map(|v| v.peer_id()).collect::<Vec<_>>();

    // each region has 1/3 of the validators
    let region_size = all_validators.len() / 3;
    let mut us_west = all_validators;
    let mut af_south = us_west.split_off(region_size);
    let eu_north = af_south.split_off(region_size);

    let group_network_delays = vec![
        GroupNetworkDelay {
            name: "us-west-to-af-south".to_string(),
            source_nodes: us_west.clone(),
            target_nodes: af_south.clone(),
            latency_ms: 300,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
        GroupNetworkDelay {
            name: "af-south-to-us-west".to_string(),
            source_nodes: af_south.clone(),
            target_nodes: us_west.clone(),
            latency_ms: 300,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
        GroupNetworkDelay {
            name: "us-west-to-eu-north".to_string(),
            source_nodes: us_west.clone(),
            target_nodes: eu_north.clone(),
            latency_ms: 150,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
        GroupNetworkDelay {
            name: "eu-north-to-us-west".to_string(),
            source_nodes: eu_north.clone(),
            target_nodes: us_west.clone(),
            latency_ms: 150,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
        GroupNetworkDelay {
            name: "eu-north-to-af-south".to_string(),
            source_nodes: eu_north.clone(),
            target_nodes: af_south.clone(),
            latency_ms: 200,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
        GroupNetworkDelay {
            name: "af-south-to-eu-north".to_string(),
            source_nodes: af_south.clone(),
            target_nodes: eu_north.clone(),
            latency_ms: 200,
            jitter_ms: 50,
            correlation_percentage: 50,
        },
    ];

    info!("US_WEST: {:?}", us_west);
    info!("AF_SOUTH B: {:?}", af_south);
    info!("EU_NORTH C: {:?}", eu_north);

    SwarmNetworkDelay {
        group_network_delays,
    }
}

// 1 Gbps
fn create_bandwidth_limit() -> SwarmNetworkBandwidth {
    SwarmNetworkBandwidth {
        rate: 1000,
        limit: 20971520,
        buffer: 10000,
    }
}

fn add_execution_delay(swarm: &mut dyn Swarm, config: &ExecutionDelayConfig) -> anyhow::Result<()> {
    let runtime = Runtime::new().unwrap();
    let validators = swarm.get_validator_clients_with_names();

    runtime.block_on(async {
        let mut rng = rand::thread_rng();
        for (name, validator) in validators {
            let sleep_fraction = if rng.gen_bool(config.inject_delay_node_fraction) {
                rng.gen_range(1_u32, config.inject_delay_max_transaction_percentage)
            } else {
                0
            };
            let name = name.clone();
            info!(
                "Validator {} adding {}% of transactions with 1ms execution delay",
                name, sleep_fraction
            );
            validator
                .set_failpoint(
                    "aptos_vm::execution::user_transaction".to_string(),
                    format!(
                        "{}%delay({})",
                        sleep_fraction, config.inject_delay_per_transaction_ms
                    ),
                )
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "set_failpoint to add execution delay on {} failed, {:?}",
                        name,
                        e
                    )
                })?;
        }
        Ok::<(), anyhow::Error>(())
    })
}

fn remove_execution_delay(swarm: &mut dyn Swarm) -> anyhow::Result<()> {
    let runtime = Runtime::new().unwrap();
    let validators = swarm.get_validator_clients_with_names();

    runtime.block_on(async {
        for (name, validator) in validators {
            let name = name.clone();

            validator
                .set_failpoint(
                    "aptos_vm::execution::block_metadata".to_string(),
                    "off".to_string(),
                )
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "set_failpoint to remove execution delay on {} failed, {:?}",
                        name,
                        e
                    )
                })?;
        }
        Ok::<(), anyhow::Error>(())
    })
}

impl NetworkLoadTest for ThreeRegionSimulationTest {
    fn setup(&self, ctx: &mut NetworkContext) -> anyhow::Result<LoadDestination> {
        // inject network delay
        let delay = create_three_region_swarm_network_delay(ctx.swarm());
        let chaos = SwarmChaos::Delay(delay);
        ctx.swarm().inject_chaos(chaos)?;

        // inject bandwidth limit
        let bandwidth = create_bandwidth_limit();
        let chaos = SwarmChaos::Bandwidth(bandwidth);
        ctx.swarm().inject_chaos(chaos)?;

        if let Some(config) = &self.add_execution_delay {
            add_execution_delay(ctx.swarm(), config)?;
        }

        Ok(LoadDestination::AllNodes)
    }

    fn finish(&self, swarm: &mut dyn Swarm) -> anyhow::Result<()> {
        if self.add_execution_delay.is_some() {
            remove_execution_delay(swarm)?;
        }

        swarm.remove_all_chaos()
    }
}

impl NetworkTest for ThreeRegionSimulationTest {
    fn run<'t>(&self, ctx: &mut NetworkContext<'t>) -> anyhow::Result<()> {
        <dyn NetworkLoadTest>::run(self, ctx)
    }
}
