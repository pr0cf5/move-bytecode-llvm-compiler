// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

mod indexer;
mod logger;
mod network;
mod services;
mod state_sync;
mod storage;
pub mod utils;

#[cfg(test)]
mod tests;

use anyhow::anyhow;
use aptos_api::bootstrap as bootstrap_api;
use aptos_config::config::{NodeConfig, PersistableConfig};
use aptos_framework::ReleaseBundle;
use aptos_logger::{prelude::*, telemetry_log_writer::TelemetryLog, Level, LoggerFilterUpdater};
use aptos_state_sync_driver::driver_factory::StateSyncRuntimes;
use aptos_types::chain_id::ChainId;
use clap::Parser;
use futures::channel::mpsc;
use hex::FromHex;
use rand::{rngs::StdRng, SeedableRng};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};
use tokio::runtime::Runtime;

const EPOCH_LENGTH_SECS: u64 = 60;

/// Runs an Aptos validator or fullnode
#[derive(Clone, Debug, Parser)]
#[clap(name = "Aptos Node", author, version)]
pub struct AptosNodeArgs {
    /// Path to node configuration file (or template for local test mode).
    #[clap(short = 'f', long, parse(from_os_str), required_unless = "test")]
    config: Option<PathBuf>,

    /// Directory to run the test mode in.
    ///
    /// Repeated runs will start up from previous state.
    #[clap(long, parse(from_os_str), requires("test"))]
    test_dir: Option<PathBuf>,

    /// Run only a single validator node testnet.
    #[clap(long)]
    test: bool,

    /// Random number generator seed for starting a single validator testnet.
    #[clap(long, parse(try_from_str = FromHex::from_hex), requires("test"))]
    seed: Option<[u8; 32]>,

    /// Use random ports instead of ports from the node configuration.
    #[clap(long, requires("test"))]
    random_ports: bool,

    /// Paths to the Aptos framework release package to be used for genesis.
    #[clap(long, requires("test"))]
    genesis_framework: Option<PathBuf>,

    /// Enable lazy mode.
    ///
    /// Setting this flag will set `consensus#mempool_poll_count` config to `u64::MAX` and
    /// only commit a block when there are user transactions in mempool.
    #[clap(long, requires("test"))]
    lazy: bool,
}

impl AptosNodeArgs {
    /// Runs an Aptos node based on the given command line arguments and config flags
    pub fn run(self) {
        if self.test {
            println!("WARNING: Entering test mode! This should never be used in production!");

            // Set the genesis framework
            let genesis_framework = if let Some(path) = self.genesis_framework {
                ReleaseBundle::read(path).unwrap()
            } else {
                aptos_cached_packages::head_release_bundle().clone()
            };

            // Create a seeded RNG, setup the test environment and start the node
            let rng = self
                .seed
                .map(StdRng::from_seed)
                .unwrap_or_else(StdRng::from_entropy);
            setup_test_environment_and_start_node(
                self.config,
                self.test_dir,
                self.random_ports,
                self.lazy,
                &genesis_framework,
                rng,
            )
            .expect("Test node should start correctly!");
        } else {
            // Get the config file path
            let config_path = self.config.expect("Config is required to launch node");
            if !config_path.exists() {
                panic!(
                    "The node config file could not be found! Ensure the given path is correct: {:?}",
                    config_path.display()
                )
            }

            // A config file exists, attempt to parse the config
            let config = NodeConfig::load(config_path.clone()).unwrap_or_else(|error| {
                panic!(
                    "Failed to parse node config file! Given file path: {:?}. Error: {:?}",
                    config_path.display(),
                    error
                )
            });
            println!("Using node config {:?}", &config);

            // Start the node
            start(config, None, true).expect("Node should start correctly");
        };
    }
}

/// Runtime handle to ensure that all inner runtimes stay in scope
pub struct AptosHandle {
    _api_runtime: Option<Runtime>,
    _backup_runtime: Option<Runtime>,
    _consensus_runtime: Option<Runtime>,
    _mempool_runtime: Runtime,
    _network_runtimes: Vec<Runtime>,
    _index_runtime: Option<Runtime>,
    _state_sync_runtimes: StateSyncRuntimes,
    _telemetry_runtime: Option<Runtime>,
}

/// Start an Aptos node
pub fn start(
    config: NodeConfig,
    log_file: Option<PathBuf>,
    create_global_rayon_pool: bool,
) -> anyhow::Result<()> {
    // Setup panic handler
    aptos_crash_handler::setup_panic_handler();

    // Create global rayon thread pool
    utils::create_global_rayon_pool(create_global_rayon_pool);

    // Instantiate the global logger
    let (remote_log_receiver, logger_filter_update) = logger::create_logger(&config, log_file);

    // Ensure failpoints are configured correctly
    if fail::has_failpoints() {
        warn!("Failpoints are enabled!");

        // Set all of the failpoints
        if let Some(failpoints) = &config.failpoints {
            for (point, actions) in failpoints {
                fail::cfg(point, actions).unwrap_or_else(|_| {
                    panic!(
                        "Failed to set actions for failpoint! Failpoint: {:?}, Actions: {:?}",
                        point, actions
                    )
                });
            }
        }
    } else if config.failpoints.is_some() {
        warn!("Failpoints is set in the node config, but the binary didn't compile with this feature!");
    }

    // Set up the node environment and start it
    let _node_handle =
        setup_environment_and_start_node(config, remote_log_receiver, Some(logger_filter_update))?;
    let term = Arc::new(AtomicBool::new(false));
    while !term.load(Ordering::Acquire) {
        thread::park();
    }

    Ok(())
}

/// Creates a simple test environment and starts the node
pub fn setup_test_environment_and_start_node<R>(
    config_path: Option<PathBuf>,
    test_dir: Option<PathBuf>,
    random_ports: bool,
    enable_lazy_mode: bool,
    framework: &ReleaseBundle,
    rng: R,
) -> anyhow::Result<()>
where
    R: rand::RngCore + rand::CryptoRng,
{
    // If there wasn't a test directory specified, create a temporary one
    let test_dir =
        test_dir.unwrap_or_else(|| aptos_temppath::TempPath::new().as_ref().to_path_buf());

    // Create the directories for the node
    fs::DirBuilder::new().recursive(true).create(&test_dir)?;
    let test_dir = test_dir.canonicalize()?;

    // The validator builder puts the first node in the 0 directory
    let validator_config_path = test_dir.join("0").join("node.yaml");
    let aptos_root_key_path = test_dir.join("mint.key");

    // If there's already a config, use it. Otherwise create a test one.
    let config = if validator_config_path.exists() {
        NodeConfig::load(&validator_config_path)
            .map_err(|error| anyhow!("Unable to load config: {:?}", error))?
    } else {
        // Create a test only config for a single validator node
        let node_config = create_single_node_test_config(config_path, enable_lazy_mode);

        // Build genesis and the validator node
        let builder = aptos_genesis::builder::Builder::new(&test_dir, framework.clone())?
            .with_init_config(Some(Arc::new(move |_, config, _| {
                *config = node_config.clone();
            })))
            .with_init_genesis_config(Some(Arc::new(|genesis_config| {
                genesis_config.allow_new_validators = true;
                genesis_config.epoch_duration_secs = EPOCH_LENGTH_SECS;
                genesis_config.recurring_lockup_duration_secs = 7200;
            })))
            .with_randomize_first_validator_ports(random_ports);
        let (root_key, _genesis, genesis_waypoint, validators) = builder.build(rng)?;

        // Write the mint key to disk
        let serialized_keys = bcs::to_bytes(&root_key)?;
        let mut key_file = fs::File::create(&aptos_root_key_path)?;
        key_file.write_all(&serialized_keys)?;

        // Build a waypoint file so that clients / docker can grab it easily
        let waypoint_file_path = test_dir.join("waypoint.txt");
        Write::write_all(
            &mut fs::File::create(waypoint_file_path)?,
            genesis_waypoint.to_string().as_bytes(),
        )?;

        // Return the validator config
        validators[0].config.clone()
    };

    // Prepare log file since we cannot automatically route logs to stderr
    let log_file = test_dir.join("validator.log");

    // Print out useful information about the environment and the node
    println!("Completed generating configuration:");
    println!("\tLog file: {:?}", log_file);
    println!("\tTest dir: {:?}", test_dir);
    println!("\tAptos root key path: {:?}", aptos_root_key_path);
    println!("\tWaypoint: {}", config.base.waypoint.genesis_waypoint());
    println!("\tChainId: {}", ChainId::test());
    println!("\tREST API endpoint: http://{}", &config.api.address);
    println!(
        "\tMetrics endpoint: http://{}:{}/metrics",
        &config.inspection_service.address, &config.inspection_service.port
    );
    println!(
        "\tAptosnet fullnode network endpoint: {}",
        &config.full_node_networks[0].listen_address
    );
    if enable_lazy_mode {
        println!("\tLazy mode is enabled");
    }
    println!("\nAptos is running, press ctrl-c to exit\n");

    start(config, Some(log_file), false)
}

/// Creates a single node test config, with a few config tweaks to reduce
/// the overhead of running the node on a local machine.
fn create_single_node_test_config(
    config_path: Option<PathBuf>,
    enable_lazy_mode: bool,
) -> NodeConfig {
    // Build a single validator network with a generated config
    let mut node_config = NodeConfig::default_for_validator();

    // Adjust some fields in the default template to lower the overhead of
    // running on a local machine.
    node_config.execution.concurrency_level = 1;
    node_config.execution.num_proof_reading_threads = 1;
    node_config.peer_monitoring_service.max_concurrent_requests = 1;
    node_config
        .mempool
        .shared_mempool_max_concurrent_inbound_syncs = 1;
    node_config
        .state_sync
        .state_sync_driver
        .enable_auto_bootstrapping = true;

    // Configure the validator network
    let validator_network = node_config.validator_network.as_mut().unwrap();
    validator_network.max_concurrent_network_reqs = 1;
    validator_network.connectivity_check_interval_ms = 10000;
    validator_network.max_connection_delay_ms = 10000;
    validator_network.ping_interval_ms = 10000;
    validator_network.runtime_threads = Some(1);

    // Configure the fullnode network
    let fullnode_network = node_config.full_node_networks.get_mut(0).unwrap();
    fullnode_network.max_concurrent_network_reqs = 1;
    fullnode_network.connectivity_check_interval_ms = 10000;
    fullnode_network.max_connection_delay_ms = 10000;
    fullnode_network.ping_interval_ms = 10000;
    fullnode_network.runtime_threads = Some(1);

    // If a config path was provided, use that as the template
    if let Some(config_path) = config_path {
        node_config = NodeConfig::load_config(&config_path).unwrap_or_else(|error| {
            panic!(
                "Failed to load config at path: {:?}. Error: {:?}",
                config_path, error
            )
        });
    }

    // Change the default log level
    node_config.logger.level = Level::Debug;

    // Enable the REST API
    node_config.api.address = format!("0.0.0.0:{}", node_config.api.address.port())
        .parse()
        .expect("Unable to set the REST API address!");

    // Set the correct poll count for mempool
    if enable_lazy_mode {
        node_config.consensus.quorum_store_poll_count = u64::MAX;
    }

    node_config
}

/// Initializes the node environment and starts the node
pub fn setup_environment_and_start_node(
    mut node_config: NodeConfig,
    remote_log_rx: Option<mpsc::Receiver<TelemetryLog>>,
    logger_filter_update_job: Option<LoggerFilterUpdater>,
) -> anyhow::Result<AptosHandle> {
    // Start the node inspection service
    services::start_node_inspection_service(&node_config);

    // Set up the storage database and any RocksDB checkpoints
    let (aptos_db, db_rw, backup_service, genesis_waypoint) =
        storage::initialize_database_and_checkpoints(&mut node_config)?;

    // Set the Aptos VM configurations
    utils::set_aptos_vm_configurations(&node_config);

    // Start the telemetry service (as early as possible and before any blocking calls)
    let chain_id = utils::fetch_chain_id(&db_rw)?;
    let telemetry_runtime = services::start_telemetry_service(
        &node_config,
        remote_log_rx,
        logger_filter_update_job,
        chain_id,
    );

    // Create an event subscription service (and reconfig subscriptions for consensus and mempool)
    let (
        mut event_subscription_service,
        mempool_reconfig_subscription,
        consensus_reconfig_subscription,
    ) = state_sync::create_event_subscription_service(&node_config, &db_rw);

    // Set up the networks and gather the application network handles
    let (
        network_runtimes,
        consensus_network_interfaces,
        mempool_network_interfaces,
        storage_service_network_interfaces,
    ) = network::setup_networks_and_get_interfaces(
        &node_config,
        chain_id,
        &mut event_subscription_service,
    );

    // Start state sync and get the notification endpoints for mempool and consensus
    let (state_sync_runtimes, mempool_listener, consensus_notifier) =
        state_sync::start_state_sync_and_get_notification_handles(
            &node_config,
            storage_service_network_interfaces,
            genesis_waypoint,
            event_subscription_service,
            db_rw.clone(),
        )?;

    // Bootstrap the API and indexer
    let (mempool_client_receiver, api_runtime, index_runtime) =
        services::bootstrap_api_and_indexer(&node_config, aptos_db, chain_id)?;

    // Create mempool and get the consensus to mempool sender
    let (mempool_runtime, consensus_to_mempool_sender) =
        services::start_mempool_runtime_and_get_consensus_sender(
            &mut node_config,
            &db_rw,
            mempool_reconfig_subscription,
            mempool_network_interfaces,
            mempool_listener,
            mempool_client_receiver,
        );

    // Create the consensus runtime (this blocks on state sync first)
    let consensus_runtime = consensus_network_interfaces.map(|consensus_network_interfaces| {
        // Wait until state sync has been initialized
        debug!("Waiting until state sync is initialized!");
        state_sync_runtimes.block_until_initialized();
        debug!("State sync initialization complete.");

        // Initialize and start consensus
        services::start_consensus_runtime(
            &mut node_config,
            db_rw,
            consensus_reconfig_subscription,
            consensus_network_interfaces,
            consensus_notifier,
            consensus_to_mempool_sender,
        )
    });

    Ok(AptosHandle {
        _api_runtime: api_runtime,
        _backup_runtime: backup_service,
        _consensus_runtime: consensus_runtime,
        _mempool_runtime: mempool_runtime,
        _network_runtimes: network_runtimes,
        _index_runtime: index_runtime,
        _state_sync_runtimes: state_sync_runtimes,
        _telemetry_runtime: telemetry_runtime,
    })
}
