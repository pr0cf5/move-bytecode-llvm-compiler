// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::{smoke_test_environment::new_local_swarm_with_aptos, txn_emitter::generate_traffic};
use aptos_forge::TransactionType;
use std::time::Duration;

#[ignore]
#[tokio::test]
// Assumes that the consensus-only-perf-test feature is enabled.
async fn test_consensus_only_with_txn_emitter() {
    let mut swarm = new_local_swarm_with_aptos(1).await;

    let all_validators = swarm.validators().map(|v| v.peer_id()).collect::<Vec<_>>();

    let txn_stat = generate_traffic(
        &mut swarm,
        &all_validators,
        Duration::from_secs(10),
        1,
        vec![
            (TransactionType::P2P, 70),
            (TransactionType::AccountGeneration, 20),
        ],
    )
    .await
    .unwrap();
    println!("{:?}", txn_stat.rate(Duration::from_secs(10)));
    // assert some much smaller number than expected, so it doesn't fail under contention
    assert!(txn_stat.submitted > 30);
    assert!(txn_stat.committed > 30);
}
