// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use crate::{metrics, network::StorageServiceNetworkEvents, StorageReader, StorageServiceServer};
use anyhow::{format_err, Result};
use aptos_bitvec::BitVec;
use aptos_channels::{aptos_channel, message_queues::QueueStyle};
use aptos_config::{
    config::StorageServiceConfig,
    network_id::{NetworkId, PeerNetworkId},
};
use aptos_crypto::{ed25519::Ed25519PrivateKey, HashValue, PrivateKey, SigningKey, Uniform};
use aptos_logger::Level;
use aptos_network::{
    application::interface::NetworkServiceEvents,
    peer_manager::PeerManagerNotification,
    protocols::{
        network::{NetworkEvents, NewNetworkEvents},
        rpc::InboundRpcRequest,
        wire::handshake::v1::ProtocolId,
    },
};
use aptos_storage_interface::{DbReader, ExecutedTrees, Order};
use aptos_storage_service_types::{
    requests::{
        DataRequest, EpochEndingLedgerInfoRequest, NewTransactionOutputsWithProofRequest,
        NewTransactionsOrOutputsWithProofRequest, NewTransactionsWithProofRequest,
        StateValuesWithProofRequest, StorageServiceRequest, TransactionOutputsWithProofRequest,
        TransactionsOrOutputsWithProofRequest, TransactionsWithProofRequest,
    },
    responses::{
        CompleteDataRange, DataResponse, DataSummary, ProtocolMetadata, ServerProtocolVersion,
        StorageServerSummary, StorageServiceResponse,
    },
    Epoch, StorageServiceError, StorageServiceMessage,
};
use aptos_time_service::{MockTimeService, TimeService};
use aptos_types::{
    account_address::AccountAddress,
    aggregate_signature::AggregateSignature,
    block_info::BlockInfo,
    chain_id::ChainId,
    contract_event::EventWithVersion,
    epoch_change::EpochChangeProof,
    event::EventKey,
    ledger_info::{LedgerInfo, LedgerInfoWithSignatures},
    proof::{
        AccumulatorConsistencyProof, SparseMerkleProof, SparseMerkleRangeProof,
        TransactionAccumulatorSummary,
    },
    state_proof::StateProof,
    state_store::{
        state_key::StateKey,
        state_value::{StateValue, StateValueChunkWithProof},
    },
    transaction::{
        AccountTransactionsWithProof, ExecutionStatus, RawTransaction, Script, SignedTransaction,
        Transaction, TransactionInfo, TransactionListWithProof, TransactionOutput,
        TransactionOutputListWithProof, TransactionPayload, TransactionStatus,
        TransactionWithProof, Version,
    },
    write_set::WriteSet,
    PeerId,
};
use claims::{assert_matches, assert_none};
use futures::channel::{oneshot, oneshot::Receiver};
use mockall::{
    mock,
    predicate::{always, eq},
    Sequence,
};
use rand::{rngs::OsRng, Rng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::time::timeout;

/// Various test constants for storage
const MAX_RESPONSE_TIMEOUT_SECS: u64 = 40;
const PROTOCOL_VERSION: u64 = 1;

#[tokio::test]
async fn test_cachable_requests_compression() {
    // Create test data
    let start_version = 0;
    let end_version = 454;
    let proof_version = end_version;
    let include_events = false;
    let compression_options = [true, false];

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    let mut expectation_sequence = Sequence::new();
    let mut transaction_lists_with_proof = vec![];
    for _ in compression_options {
        // Create and save test transaction lists
        let transaction_list_with_proof = create_transaction_list_with_proof(
            start_version,
            end_version,
            proof_version,
            include_events,
        );
        transaction_lists_with_proof.push(transaction_list_with_proof.clone());

        // Expect the data to be fetched from storage exactly once
        db_reader
            .expect_get_transactions()
            .times(1)
            .with(
                eq(start_version),
                eq(end_version - start_version + 1),
                eq(proof_version),
                eq(include_events),
            )
            .return_once(move |_, _, _, _| Ok(transaction_list_with_proof))
            .in_sequence(&mut expectation_sequence);
    }

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Repeatedly fetch the data and verify the responses
    for (i, use_compression) in compression_options.iter().enumerate() {
        for _ in 0..10 {
            let response = get_transactions_with_proof(
                &mut mock_client,
                start_version,
                end_version,
                proof_version,
                include_events,
                *use_compression,
            )
            .await
            .unwrap();

            // Verify the response is correct
            assert_eq!(response.is_compressed(), *use_compression);
            match response.get_data_response().unwrap() {
                DataResponse::TransactionsWithProof(response) => {
                    assert_eq!(response, transaction_lists_with_proof[i]);
                },
                _ => panic!("Expected transactions with proof but got: {:?}", response),
            };
        }
    }
}

#[tokio::test]
async fn test_cachable_requests_eviction() {
    // Create test data
    let max_lru_cache_size = StorageServiceConfig::default().max_lru_cache_size;
    let version = 101;
    let start_index = 100;
    let end_index = 199;
    let state_value_chunk_with_proof = StateValueChunkWithProof {
        first_index: start_index,
        last_index: end_index,
        first_key: HashValue::random(),
        last_key: HashValue::random(),
        raw_values: vec![],
        proof: SparseMerkleRangeProof::new(vec![]),
        root_hash: HashValue::random(),
    };

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    let mut expectation_sequence = Sequence::new();
    db_reader
        .expect_get_state_leaf_count()
        .times(max_lru_cache_size as usize)
        .with(always())
        .returning(move |_| Ok(165));
    for _ in 0..2 {
        let state_value_chunk_with_proof_clone = state_value_chunk_with_proof.clone();
        db_reader
            .expect_get_state_value_chunk_with_proof()
            .times(1)
            .with(
                eq(version),
                eq(start_index as usize),
                eq((end_index - start_index + 1) as usize),
            )
            .return_once(move |_, _, _| Ok(state_value_chunk_with_proof_clone))
            .in_sequence(&mut expectation_sequence);
    }

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Process a request to fetch a state chunk. This should cache and serve the response.
    for _ in 0..2 {
        let _ =
            get_state_values_with_proof(&mut mock_client, version, start_index, end_index, true)
                .await;
    }

    // Process enough requests to evict the previously cached response
    for version in 0..max_lru_cache_size {
        let _ = get_number_of_states(&mut mock_client, version, true).await;
    }

    // Process a request to fetch the state chunk again. This requires refetching the data.
    let _ =
        get_state_values_with_proof(&mut mock_client, version, start_index, end_index, true).await;
}

#[tokio::test]
async fn test_cachable_requests_data_versions() {
    // Create test data
    let start_versions = [0, 76, 101, 230, 300, 454];
    let end_version = 454;
    let proof_version = end_version;
    let include_events = false;

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    let mut expectation_sequence = Sequence::new();
    let mut transaction_lists_with_proof = vec![];
    for start_version in start_versions {
        // Create and save test transaction lists
        let transaction_list_with_proof = create_transaction_list_with_proof(
            start_version,
            end_version,
            proof_version,
            include_events,
        );
        transaction_lists_with_proof.push(transaction_list_with_proof.clone());

        // Expect the data to be fetched from storage once
        db_reader
            .expect_get_transactions()
            .times(1)
            .with(
                eq(start_version),
                eq(end_version - start_version + 1),
                eq(proof_version),
                eq(include_events),
            )
            .return_once(move |_, _, _, _| Ok(transaction_list_with_proof))
            .in_sequence(&mut expectation_sequence);
    }

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Repeatedly fetch the data and verify the responses
    for (i, start_version) in start_versions.iter().enumerate() {
        for _ in 0..10 {
            let response = get_transactions_with_proof(
                &mut mock_client,
                *start_version,
                end_version,
                proof_version,
                include_events,
                true,
            )
            .await
            .unwrap();

            // Verify the response is correct
            match response {
                StorageServiceResponse::CompressedResponse(_, _) => {
                    match response.get_data_response().unwrap() {
                        DataResponse::TransactionsWithProof(transactions_with_proof) => {
                            assert_eq!(transactions_with_proof, transaction_lists_with_proof[i])
                        },
                        _ => panic!(
                            "Expected compressed transactions with proof but got: {:?}",
                            response
                        ),
                    }
                },
                _ => panic!("Expected compressed response but got: {:?}", response),
            };
        }
    }
}

#[tokio::test]
async fn test_get_server_protocol_version() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Process a request to fetch the protocol version
    let response = get_protocol_version(&mut mock_client, true).await;

    // Verify the response is correct
    let expected_data_response = DataResponse::ServerProtocolVersion(ServerProtocolVersion {
        protocol_version: PROTOCOL_VERSION,
    });
    assert_matches!(response, StorageServiceResponse::CompressedResponse(_, _));
    assert_eq!(
        response.get_data_response().unwrap(),
        expected_data_response
    );
}

#[tokio::test]
async fn test_get_states_with_proof() {
    // Test small and large chunk requests
    let max_state_chunk_size = StorageServiceConfig::default().max_state_chunk_size;
    for chunk_size in [1, 100, max_state_chunk_size] {
        // Create test data
        let version = 101;
        let start_index = 100;
        let end_index = start_index + chunk_size - 1;
        let state_value_chunk_with_proof = StateValueChunkWithProof {
            first_index: start_index,
            last_index: end_index,
            first_key: HashValue::random(),
            last_key: HashValue::random(),
            raw_values: vec![],
            proof: SparseMerkleRangeProof::new(vec![]),
            root_hash: HashValue::random(),
        };

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        expect_get_state_values_with_proof(
            &mut db_reader,
            version,
            start_index,
            chunk_size,
            state_value_chunk_with_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Process a request to fetch a states chunk with a proof
        let response =
            get_state_values_with_proof(&mut mock_client, version, start_index, end_index, false)
                .await
                .unwrap();

        // Verify the response is correct
        assert_matches!(response, StorageServiceResponse::RawResponse(_));
        assert_eq!(
            response.get_data_response().unwrap(),
            DataResponse::StateValueChunkWithProof(state_value_chunk_with_proof)
        );
    }
}

#[tokio::test]
async fn test_get_states_with_proof_chunk_limit() {
    // Create test data
    let max_state_chunk_size = StorageServiceConfig::default().max_state_chunk_size;
    let chunk_size = max_state_chunk_size * 10; // Set a chunk request larger than the max
    let version = 101;
    let start_index = 100;
    let state_value_chunk_with_proof = StateValueChunkWithProof {
        first_index: start_index,
        last_index: start_index + max_state_chunk_size - 1,
        first_key: HashValue::random(),
        last_key: HashValue::random(),
        raw_values: vec![],
        proof: SparseMerkleRangeProof::new(vec![]),
        root_hash: HashValue::random(),
    };

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    expect_get_state_values_with_proof(
        &mut db_reader,
        version,
        start_index,
        max_state_chunk_size,
        state_value_chunk_with_proof.clone(),
    );

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Process a request to fetch a states chunk with a proof
    let response = get_state_values_with_proof(
        &mut mock_client,
        version,
        start_index,
        start_index + chunk_size - 1,
        false,
    )
    .await
    .unwrap();

    // Verify the response is correct
    assert_matches!(response, StorageServiceResponse::RawResponse(_));
    assert_eq!(
        response.get_data_response().unwrap(),
        DataResponse::StateValueChunkWithProof(state_value_chunk_with_proof)
    );
}

#[tokio::test]
async fn test_get_states_with_proof_network_limit() {
    // Test different byte limits
    for network_limit_bytes in [1, 512, 1024, 10 * 1024] {
        get_states_with_proof_network_limit(network_limit_bytes).await;
    }
}

#[tokio::test]
async fn test_get_states_with_proof_invalid() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Test invalid ranges
    let start_index = 100;
    for end_index in [0, 99] {
        let response =
            get_state_values_with_proof(&mut mock_client, 0, start_index, end_index, false)
                .await
                .unwrap_err();
        assert_matches!(response, StorageServiceError::InvalidRequest(_));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions() {
    // Test small and large chunk sizes
    let max_transaction_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
    for chunk_size in [1, 100, max_transaction_chunk_size] {
        // Test event inclusion
        for include_events in [true, false] {
            // Create test data
            let highest_version = 45576;
            let highest_epoch = 43;
            let lowest_version = 4566;
            let peer_version = highest_version - chunk_size;
            let highest_ledger_info =
                create_test_ledger_info_with_sigs(highest_epoch, highest_version);
            let transaction_list_with_proof = create_transaction_list_with_proof(
                peer_version + 1,
                highest_version,
                highest_version,
                include_events,
            );

            // Create the mock db reader
            let mut db_reader =
                create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
            expect_get_transactions(
                &mut db_reader,
                peer_version + 1,
                highest_version - peer_version,
                highest_version,
                include_events,
                transaction_list_with_proof.clone(),
            );

            // Create the storage client and server
            let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
            tokio::spawn(service.start());

            // Send a request to subscribe to new transactions
            let mut response_receiver = get_new_transactions_with_proof(
                &mut mock_client,
                peer_version,
                highest_epoch,
                include_events,
            )
            .await;

            // Verify no subscription response has been received yet
            assert_none!(response_receiver.try_recv().unwrap());

            // Elapse enough time to force the subscription thread to work
            wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

            // Verify a response is received and that it contains the correct data
            verify_new_transactions_with_proof(
                &mut mock_client,
                response_receiver,
                transaction_list_with_proof,
                highest_ledger_info,
            )
            .await;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_epoch_change() {
    // Test event inclusion
    for include_events in [true, false] {
        // Create test data
        let highest_version = 45576;
        let highest_epoch = 1032;
        let lowest_version = 4566;
        let peer_version = highest_version - 100;
        let peer_epoch = highest_epoch - 20;
        let epoch_change_version = peer_version + 45;
        let epoch_change_proof = EpochChangeProof {
            ledger_info_with_sigs: vec![create_test_ledger_info_with_sigs(
                peer_epoch,
                epoch_change_version,
            )],
            more: false,
        };
        let transaction_list_with_proof = create_transaction_list_with_proof(
            peer_version + 1,
            epoch_change_version,
            epoch_change_version,
            include_events,
        );

        // Create the mock db reader
        let mut db_reader = create_mock_db_for_subscription(
            create_test_ledger_info_with_sigs(highest_epoch, highest_version),
            lowest_version,
        );
        expect_get_transactions(
            &mut db_reader,
            peer_version + 1,
            epoch_change_version - peer_version,
            epoch_change_version,
            include_events,
            transaction_list_with_proof.clone(),
        );
        expect_get_epoch_ending_ledger_infos(
            &mut db_reader,
            peer_epoch,
            peer_epoch + 1,
            epoch_change_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Send a request to subscribe to new transactions
        let response_receiver = get_new_transactions_with_proof(
            &mut mock_client,
            peer_version,
            peer_epoch,
            include_events,
        )
        .await;

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        verify_new_transactions_with_proof(
            &mut mock_client,
            response_receiver,
            transaction_list_with_proof,
            epoch_change_proof.ledger_info_with_sigs[0].clone(),
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_different_networks() {
    // Test small and large chunk sizes
    let max_transaction_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
    for chunk_size in [100, max_transaction_chunk_size] {
        // Test event inclusion
        for include_events in [true, false] {
            // Create test data
            let highest_version = 45576;
            let highest_epoch = 43;
            let lowest_version = 4566;
            let peer_version_1 = highest_version - chunk_size;
            let peer_version_2 = highest_version - (chunk_size - 10);
            let highest_ledger_info =
                create_test_ledger_info_with_sigs(highest_epoch, highest_version);
            let transaction_list_with_proof_1 = create_transaction_list_with_proof(
                peer_version_1 + 1,
                highest_version,
                highest_version,
                include_events,
            );
            let transaction_list_with_proof_2 = create_transaction_list_with_proof(
                peer_version_2 + 1,
                highest_version,
                highest_version,
                include_events,
            );

            // Create the mock db reader
            let mut db_reader =
                create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
            expect_get_transactions(
                &mut db_reader,
                peer_version_1 + 1,
                highest_version - peer_version_1,
                highest_version,
                include_events,
                transaction_list_with_proof_1.clone(),
            );
            expect_get_transactions(
                &mut db_reader,
                peer_version_2 + 1,
                highest_version - peer_version_2,
                highest_version,
                include_events,
                transaction_list_with_proof_2.clone(),
            );

            // Create the storage client and server
            let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
            tokio::spawn(service.start());

            // Send a request to subscribe to new transactions for peer 1
            let peer_id = PeerId::random();
            let peer_network_1 = PeerNetworkId::new(NetworkId::Public, peer_id);
            let mut response_receiver_1 = get_new_transactions_with_proof_for_peer(
                &mut mock_client,
                peer_version_1,
                highest_epoch,
                include_events,
                Some(peer_network_1),
            )
            .await;

            // Send a request to subscribe to new transactions for peer 2
            let peer_network_2 = PeerNetworkId::new(NetworkId::Vfn, peer_id);
            let mut response_receiver_2 = get_new_transactions_with_proof_for_peer(
                &mut mock_client,
                peer_version_2,
                highest_epoch,
                include_events,
                Some(peer_network_2),
            )
            .await;

            // Verify no subscription response has been received yet
            assert_none!(response_receiver_1.try_recv().unwrap());
            assert_none!(response_receiver_2.try_recv().unwrap());

            // Elapse enough time to force the subscription thread to work
            wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

            // Verify a response is received and that it contains the correct data for both peers
            verify_new_transactions_with_proof(
                &mut mock_client,
                response_receiver_1,
                transaction_list_with_proof_1,
                highest_ledger_info.clone(),
            )
            .await;
            verify_new_transactions_with_proof(
                &mut mock_client,
                response_receiver_2,
                transaction_list_with_proof_2,
                highest_ledger_info,
            )
            .await;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_max_chunk() {
    // Test event inclusion
    for include_events in [true, false] {
        // Create test data
        let highest_version = 1034556;
        let highest_epoch = 343;
        let lowest_version = 3453;
        let max_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
        let requested_chunk_size = max_chunk_size + 1;
        let peer_version = highest_version - requested_chunk_size;
        let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);
        let transaction_list_with_proof = create_transaction_list_with_proof(
            peer_version + 1,
            peer_version + requested_chunk_size,
            peer_version + requested_chunk_size,
            include_events,
        );

        // Create the mock db reader
        let mut db_reader =
            create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
        expect_get_transactions(
            &mut db_reader,
            peer_version + 1,
            max_chunk_size,
            highest_version,
            include_events,
            transaction_list_with_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Send a request to subscribe to new transactions
        let response_receiver = get_new_transactions_with_proof(
            &mut mock_client,
            peer_version,
            highest_epoch,
            include_events,
        )
        .await;

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        verify_new_transactions_with_proof(
            &mut mock_client,
            response_receiver,
            transaction_list_with_proof,
            highest_ledger_info,
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transaction_outputs() {
    // Test small and large chunk sizes
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [1, 100, max_output_chunk_size] {
        // Create test data
        let highest_version = 5060;
        let highest_epoch = 30;
        let lowest_version = 101;
        let peer_version = highest_version - chunk_size;
        let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);
        let output_list_with_proof =
            create_output_list_with_proof(peer_version + 1, highest_version, highest_version);

        // Create the mock db reader
        let mut db_reader =
            create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
        expect_get_transaction_outputs(
            &mut db_reader,
            peer_version + 1,
            highest_version - peer_version,
            highest_version,
            output_list_with_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Send a request to subscribe to new transaction outputs
        let mut response_receiver =
            get_new_outputs_with_proof(&mut mock_client, peer_version, highest_epoch).await;

        // Verify no subscription response has been received yet
        assert_none!(response_receiver.try_recv().unwrap());

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        verify_new_transaction_outputs_with_proof(
            &mut mock_client,
            response_receiver,
            output_list_with_proof,
            highest_ledger_info,
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transaction_outputs_different_networks() {
    // Test small and large chunk sizes
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [100, max_output_chunk_size] {
        // Create test data
        let highest_version = 5060;
        let highest_epoch = 30;
        let lowest_version = 101;
        let peer_version_1 = highest_version - chunk_size;
        let peer_version_2 = highest_version - (chunk_size - 50);
        let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);
        let output_list_with_proof_1 =
            create_output_list_with_proof(peer_version_1 + 1, highest_version, highest_version);
        let output_list_with_proof_2 =
            create_output_list_with_proof(peer_version_2 + 1, highest_version, highest_version);

        // Create the mock db reader
        let mut db_reader =
            create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
        expect_get_transaction_outputs(
            &mut db_reader,
            peer_version_1 + 1,
            highest_version - peer_version_1,
            highest_version,
            output_list_with_proof_1.clone(),
        );
        expect_get_transaction_outputs(
            &mut db_reader,
            peer_version_2 + 1,
            highest_version - peer_version_2,
            highest_version,
            output_list_with_proof_2.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Send a request to subscribe to new transaction outputs for peer 1
        let peer_id = PeerId::random();
        let peer_network_1 = PeerNetworkId::new(NetworkId::Validator, peer_id);
        let mut response_receiver_1 = get_new_outputs_with_proof_for_peer(
            &mut mock_client,
            peer_version_1,
            highest_epoch,
            Some(peer_network_1),
        )
        .await;

        // Send a request to subscribe to new transaction outputs for peer 2
        let peer_network_2 = PeerNetworkId::new(NetworkId::Vfn, peer_id);
        let mut response_receiver_2 = get_new_outputs_with_proof_for_peer(
            &mut mock_client,
            peer_version_2,
            highest_epoch,
            Some(peer_network_2),
        )
        .await;

        // Verify no subscription response has been received yet
        assert_none!(response_receiver_1.try_recv().unwrap());
        assert_none!(response_receiver_2.try_recv().unwrap());

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        verify_new_transaction_outputs_with_proof(
            &mut mock_client,
            response_receiver_1,
            output_list_with_proof_1,
            highest_ledger_info.clone(),
        )
        .await;
        verify_new_transaction_outputs_with_proof(
            &mut mock_client,
            response_receiver_2,
            output_list_with_proof_2,
            highest_ledger_info,
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transaction_outputs_epoch_change() {
    // Create test data
    let highest_version = 10000;
    let highest_epoch = 10000;
    let lowest_version = 0;
    let peer_version = highest_version - 1000;
    let peer_epoch = highest_epoch - 1000;
    let epoch_change_version = peer_version + 1;
    let epoch_change_proof = EpochChangeProof {
        ledger_info_with_sigs: vec![create_test_ledger_info_with_sigs(
            peer_epoch,
            epoch_change_version,
        )],
        more: false,
    };
    let output_list_with_proof =
        create_output_list_with_proof(peer_version + 1, epoch_change_version, epoch_change_version);

    // Create the mock db reader
    let mut db_reader = create_mock_db_for_subscription(
        create_test_ledger_info_with_sigs(highest_epoch, highest_version),
        lowest_version,
    );
    expect_get_transaction_outputs(
        &mut db_reader,
        peer_version + 1,
        epoch_change_version - peer_version,
        epoch_change_version,
        output_list_with_proof.clone(),
    );
    expect_get_epoch_ending_ledger_infos(
        &mut db_reader,
        peer_epoch,
        peer_epoch + 1,
        epoch_change_proof.clone(),
    );

    // Create the storage client and server
    let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Send a request to subscribe to new transaction outputs
    let response_receiver =
        get_new_outputs_with_proof(&mut mock_client, peer_version, peer_epoch).await;

    // Elapse enough time to force the subscription thread to work
    wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

    // Verify a response is received and that it contains the correct data
    verify_new_transaction_outputs_with_proof(
        &mut mock_client,
        response_receiver,
        output_list_with_proof,
        epoch_change_proof.ledger_info_with_sigs[0].clone(),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transaction_outputs_max_chunk() {
    // Create test data
    let highest_version = 65660;
    let highest_epoch = 30;
    let lowest_version = 101;
    let max_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    let requested_chunk_size = max_chunk_size + 1;
    let peer_version = highest_version - requested_chunk_size;
    let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);
    let output_list_with_proof = create_output_list_with_proof(
        peer_version + 1,
        peer_version + requested_chunk_size,
        highest_version,
    );

    // Create the mock db reader
    let mut db_reader =
        create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
    expect_get_transaction_outputs(
        &mut db_reader,
        peer_version + 1,
        max_chunk_size,
        highest_version,
        output_list_with_proof.clone(),
    );

    // Create the storage client and server
    let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Send a request to subscribe to new transaction outputs
    let response_receiver =
        get_new_outputs_with_proof(&mut mock_client, peer_version, highest_epoch).await;

    // Elapse enough time to force the subscription thread to work
    wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

    // Verify a response is received and that it contains the correct data
    verify_new_transaction_outputs_with_proof(
        &mut mock_client,
        response_receiver,
        output_list_with_proof,
        highest_ledger_info,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_or_outputs() {
    // Test small and large chunk sizes
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [1, 100, max_output_chunk_size] {
        // Test fallback to transaction syncing
        for fallback_to_transactions in [false, true] {
            // Create test data
            let highest_version = 5060;
            let highest_epoch = 30;
            let lowest_version = 101;
            let peer_version = highest_version - chunk_size;
            let highest_ledger_info =
                create_test_ledger_info_with_sigs(highest_epoch, highest_version);
            let output_list_with_proof =
                create_output_list_with_proof(peer_version + 1, highest_version, highest_version);
            let transaction_list_with_proof = create_transaction_list_with_proof(
                highest_version,
                highest_version,
                highest_version,
                false,
            ); // Creates a small transaction list

            // Create the mock db reader
            let mut db_reader =
                create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
            expect_get_transaction_outputs(
                &mut db_reader,
                peer_version + 1,
                highest_version - peer_version,
                highest_version,
                output_list_with_proof.clone(),
            );
            if fallback_to_transactions {
                expect_get_transactions(
                    &mut db_reader,
                    peer_version + 1,
                    highest_version - peer_version,
                    highest_version,
                    false,
                    transaction_list_with_proof.clone(),
                );
            }

            // Create the storage client and server
            let storage_config = configure_network_chunk_limit(
                fallback_to_transactions,
                &output_list_with_proof,
                &transaction_list_with_proof,
            );
            let (mut mock_client, service, mock_time) =
                MockClient::new(Some(db_reader), Some(storage_config));
            tokio::spawn(service.start());

            // Send a request to subscribe to new transactions or outputs
            let mut response_receiver = get_new_transactions_or_outputs_with_proof(
                &mut mock_client,
                peer_version,
                highest_epoch,
                false,
                0, // Outputs cannot be reduced and will fallback to transactions
            )
            .await;

            // Verify no subscription response has been received yet
            assert_none!(response_receiver.try_recv().unwrap());

            // Elapse enough time to force the subscription thread to work
            wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

            // Verify a response is received and that it contains the correct data
            if fallback_to_transactions {
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver,
                    Some(transaction_list_with_proof),
                    None,
                    highest_ledger_info,
                )
                .await;
            } else {
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver,
                    None,
                    Some(output_list_with_proof),
                    highest_ledger_info,
                )
                .await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_or_outputs_different_network() {
    // Test small and large chunk sizes
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [100, max_output_chunk_size] {
        // Test fallback to transaction syncing
        for fallback_to_transactions in [false, true] {
            // Create test data
            let highest_version = 5060;
            let highest_epoch = 30;
            let lowest_version = 101;
            let peer_version_1 = highest_version - chunk_size;
            let peer_version_2 = highest_version - (chunk_size - 50);
            let highest_ledger_info =
                create_test_ledger_info_with_sigs(highest_epoch, highest_version);
            let output_list_with_proof_1 =
                create_output_list_with_proof(peer_version_1 + 1, highest_version, highest_version);
            let output_list_with_proof_2 =
                create_output_list_with_proof(peer_version_2 + 1, highest_version, highest_version);
            let transaction_list_with_proof = create_transaction_list_with_proof(
                highest_version,
                highest_version,
                highest_version,
                false,
            ); // Creates a small transaction list

            // Create the mock db reader
            let mut db_reader =
                create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
            expect_get_transaction_outputs(
                &mut db_reader,
                peer_version_1 + 1,
                highest_version - peer_version_1,
                highest_version,
                output_list_with_proof_1.clone(),
            );
            expect_get_transaction_outputs(
                &mut db_reader,
                peer_version_2 + 1,
                highest_version - peer_version_2,
                highest_version,
                output_list_with_proof_2.clone(),
            );
            if fallback_to_transactions {
                expect_get_transactions(
                    &mut db_reader,
                    peer_version_1 + 1,
                    highest_version - peer_version_1,
                    highest_version,
                    false,
                    transaction_list_with_proof.clone(),
                );
                expect_get_transactions(
                    &mut db_reader,
                    peer_version_2 + 1,
                    highest_version - peer_version_2,
                    highest_version,
                    false,
                    transaction_list_with_proof.clone(),
                );
            }

            // Create the storage client and server
            let storage_config = configure_network_chunk_limit(
                fallback_to_transactions,
                &output_list_with_proof_1,
                &transaction_list_with_proof,
            );
            let (mut mock_client, service, mock_time) =
                MockClient::new(Some(db_reader), Some(storage_config));
            tokio::spawn(service.start());

            // Send a request to subscribe to new transactions or outputs for peer 1
            let peer_id = PeerId::random();
            let peer_network_1 = PeerNetworkId::new(NetworkId::Public, peer_id);
            let mut response_receiver_1 = get_new_transactions_or_outputs_with_proof_for_peer(
                &mut mock_client,
                peer_version_1,
                highest_epoch,
                false,
                0, // Outputs cannot be reduced and will fallback to transactions
                Some(peer_network_1),
            )
            .await;

            // Send a request to subscribe to new transactions or outputs for peer 1
            let peer_network_2 = PeerNetworkId::new(NetworkId::Validator, peer_id);
            let mut response_receiver_2 = get_new_transactions_or_outputs_with_proof_for_peer(
                &mut mock_client,
                peer_version_2,
                highest_epoch,
                false,
                0, // Outputs cannot be reduced and will fallback to transactions
                Some(peer_network_2),
            )
            .await;

            // Verify no subscription response has been received yet
            assert_none!(response_receiver_1.try_recv().unwrap());
            assert_none!(response_receiver_2.try_recv().unwrap());

            // Elapse enough time to force the subscription thread to work
            wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

            // Verify a response is received and that it contains the correct data
            if fallback_to_transactions {
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver_1,
                    Some(transaction_list_with_proof.clone()),
                    None,
                    highest_ledger_info.clone(),
                )
                .await;
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver_2,
                    Some(transaction_list_with_proof),
                    None,
                    highest_ledger_info,
                )
                .await;
            } else {
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver_1,
                    None,
                    Some(output_list_with_proof_1.clone()),
                    highest_ledger_info.clone(),
                )
                .await;
                verify_new_transactions_or_outputs_with_proof(
                    &mut mock_client,
                    response_receiver_2,
                    None,
                    Some(output_list_with_proof_2.clone()),
                    highest_ledger_info,
                )
                .await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_or_outputs_epoch_change() {
    // Test fallback to transaction syncing
    for fallback_to_transactions in [false, true] {
        // Create test data
        let highest_version = 10000;
        let highest_epoch = 10000;
        let lowest_version = 0;
        let peer_version = highest_version - 1000;
        let peer_epoch = highest_epoch - 1000;
        let epoch_change_version = peer_version + 1;
        let epoch_change_proof = EpochChangeProof {
            ledger_info_with_sigs: vec![create_test_ledger_info_with_sigs(
                peer_epoch,
                epoch_change_version,
            )],
            more: false,
        };
        let output_list_with_proof = create_output_list_with_proof(
            peer_version + 1,
            epoch_change_version,
            epoch_change_version,
        );
        let transaction_list_with_proof = create_transaction_list_with_proof(
            peer_version + 1,
            peer_version + 1,
            epoch_change_version,
            false,
        ); // Creates a small transaction list

        // Create the mock db reader
        let mut db_reader = create_mock_db_for_subscription(
            create_test_ledger_info_with_sigs(highest_epoch, highest_version),
            lowest_version,
        );
        expect_get_epoch_ending_ledger_infos(
            &mut db_reader,
            peer_epoch,
            peer_epoch + 1,
            epoch_change_proof.clone(),
        );
        expect_get_transaction_outputs(
            &mut db_reader,
            peer_version + 1,
            epoch_change_version - peer_version,
            epoch_change_version,
            output_list_with_proof.clone(),
        );
        if fallback_to_transactions {
            expect_get_transactions(
                &mut db_reader,
                peer_version + 1,
                epoch_change_version - peer_version,
                epoch_change_version,
                false,
                transaction_list_with_proof.clone(),
            );
        }

        // Create the storage client and server
        let storage_config = configure_network_chunk_limit(
            fallback_to_transactions,
            &output_list_with_proof,
            &transaction_list_with_proof,
        );
        let (mut mock_client, service, mock_time) =
            MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Send a request to subscribe to new transaction outputs
        let response_receiver = get_new_transactions_or_outputs_with_proof(
            &mut mock_client,
            peer_version,
            peer_epoch,
            false,
            5,
        )
        .await;

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        if fallback_to_transactions {
            verify_new_transactions_or_outputs_with_proof(
                &mut mock_client,
                response_receiver,
                Some(transaction_list_with_proof),
                None,
                epoch_change_proof.ledger_info_with_sigs[0].clone(),
            )
            .await;
        } else {
            verify_new_transactions_or_outputs_with_proof(
                &mut mock_client,
                response_receiver,
                None,
                Some(output_list_with_proof),
                epoch_change_proof.ledger_info_with_sigs[0].clone(),
            )
            .await;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_new_transactions_or_outputs_max_chunk() {
    // Test fallback to transaction syncing
    for fallback_to_transactions in [false, true] {
        // Create test data
        let highest_version = 65660;
        let highest_epoch = 30;
        let lowest_version = 101;
        let max_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
        let requested_chunk_size = max_chunk_size + 1;
        let peer_version = highest_version - requested_chunk_size;
        let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);
        let output_list_with_proof = create_output_list_with_proof(
            peer_version + 1,
            peer_version + requested_chunk_size,
            highest_version,
        );
        let transaction_list_with_proof = create_transaction_list_with_proof(
            peer_version + 1,
            peer_version + 1,
            peer_version + requested_chunk_size,
            false,
        ); // Creates a small transaction list

        // Create the mock db reader
        let max_num_output_reductions = 5;
        let mut db_reader =
            create_mock_db_for_subscription(highest_ledger_info.clone(), lowest_version);
        for i in 0..=max_num_output_reductions {
            expect_get_transaction_outputs(
                &mut db_reader,
                peer_version + 1,
                (max_chunk_size as u32 / (u32::pow(2, i as u32))) as u64,
                highest_version,
                output_list_with_proof.clone(),
            );
        }
        if fallback_to_transactions {
            expect_get_transactions(
                &mut db_reader,
                peer_version + 1,
                max_chunk_size,
                highest_version,
                false,
                transaction_list_with_proof.clone(),
            );
        }

        // Create the storage client and server
        let storage_config = configure_network_chunk_limit(
            fallback_to_transactions,
            &output_list_with_proof,
            &transaction_list_with_proof,
        );
        let (mut mock_client, service, mock_time) =
            MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Send a request to subscribe to new transaction outputs
        let response_receiver = get_new_transactions_or_outputs_with_proof(
            &mut mock_client,
            peer_version,
            highest_epoch,
            false,
            max_num_output_reductions,
        )
        .await;

        // Elapse enough time to force the subscription thread to work
        wait_for_subscription_service_to_refresh(&mut mock_client, &mock_time).await;

        // Verify a response is received and that it contains the correct data
        if fallback_to_transactions {
            verify_new_transactions_or_outputs_with_proof(
                &mut mock_client,
                response_receiver,
                Some(transaction_list_with_proof),
                None,
                highest_ledger_info,
            )
            .await;
        } else {
            verify_new_transactions_or_outputs_with_proof(
                &mut mock_client,
                response_receiver,
                None,
                Some(output_list_with_proof),
                highest_ledger_info,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn test_get_number_of_states_at_version() {
    // Create test data
    let version = 101;
    let number_of_states: u64 = 560;

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    db_reader
        .expect_get_state_leaf_count()
        .times(1)
        .with(eq(version))
        .returning(move |_| Ok(number_of_states as usize));

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Process a request to fetch the number of states at a version
    let response = get_number_of_states(&mut mock_client, version, false)
        .await
        .unwrap();

    // Verify the response is correct
    assert_matches!(response, StorageServiceResponse::RawResponse(_));
    assert_eq!(
        response.get_data_response().unwrap(),
        DataResponse::NumberOfStatesAtVersion(number_of_states)
    );
}

#[tokio::test]
async fn test_get_number_of_states_at_version_invalid() {
    // Create test data
    let version = 1;

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    db_reader
        .expect_get_state_leaf_count()
        .times(1)
        .with(eq(version))
        .returning(move |_| Err(format_err!("Version does not exist!")));

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Process a request to fetch the number of states at a version
    let response = get_number_of_states(&mut mock_client, version, false)
        .await
        .unwrap_err();

    // Verify the response is correct
    assert_matches!(response, StorageServiceError::InternalError(_));
}

#[tokio::test]
async fn test_get_storage_server_summary() {
    // Create test data
    let highest_version = 506;
    let highest_epoch = 30;
    let lowest_version = 101;
    let state_prune_window = 50;
    let highest_ledger_info = create_test_ledger_info_with_sigs(highest_epoch, highest_version);

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    let highest_ledger_info_clone = highest_ledger_info.clone();
    db_reader
        .expect_get_latest_ledger_info()
        .times(1)
        .returning(move || Ok(highest_ledger_info_clone.clone()));
    db_reader
        .expect_get_first_txn_version()
        .times(1)
        .returning(move || Ok(Some(lowest_version)));
    db_reader
        .expect_get_first_write_set_version()
        .times(1)
        .returning(move || Ok(Some(lowest_version)));
    db_reader
        .expect_get_epoch_snapshot_prune_window()
        .times(1)
        .returning(move || Ok(state_prune_window));
    db_reader
        .expect_is_state_pruner_enabled()
        .returning(move || Ok(true));

    // Create the storage client and server
    let (mut mock_client, service, mock_time) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Fetch the storage summary and verify we get a default summary response
    let response = get_storage_server_summary(&mut mock_client, true)
        .await
        .unwrap();
    let default_response = StorageServiceResponse::new(
        DataResponse::StorageServerSummary(StorageServerSummary::default()),
        true,
    )
    .unwrap();
    assert_eq!(response, default_response);

    // Elapse enough time to force a cache update
    advance_storage_refresh_time(&mock_time).await;

    // Process another request to fetch the storage summary
    let response = get_storage_server_summary(&mut mock_client, true)
        .await
        .unwrap();
    // Verify the response is correct (after the cache update)
    let default_storage_config = StorageServiceConfig::default();
    let expected_server_summary = StorageServerSummary {
        protocol_metadata: ProtocolMetadata {
            max_epoch_chunk_size: default_storage_config.max_epoch_chunk_size,
            max_state_chunk_size: default_storage_config.max_state_chunk_size,
            max_transaction_chunk_size: default_storage_config.max_transaction_chunk_size,
            max_transaction_output_chunk_size: default_storage_config
                .max_transaction_output_chunk_size,
        },
        data_summary: DataSummary {
            synced_ledger_info: Some(highest_ledger_info),
            epoch_ending_ledger_infos: Some(CompleteDataRange::from_genesis(highest_epoch - 1)),
            transactions: Some(CompleteDataRange::new(lowest_version, highest_version).unwrap()),
            transaction_outputs: Some(
                CompleteDataRange::new(lowest_version, highest_version).unwrap(),
            ),
            states: Some(
                CompleteDataRange::new(
                    highest_version - state_prune_window as u64 + 1,
                    highest_version,
                )
                .unwrap(),
            ),
        },
    };
    assert_eq!(
        response,
        StorageServiceResponse::new(
            DataResponse::StorageServerSummary(expected_server_summary),
            true
        )
        .unwrap()
    );
}

#[tokio::test]
async fn test_get_transactions_with_proof() {
    // Test small and large chunk requests
    let max_transaction_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
    for chunk_size in [1, 100, max_transaction_chunk_size] {
        // Test event inclusion
        for include_events in [true, false] {
            // Create test data
            let start_version = 0;
            let end_version = start_version + chunk_size - 1;
            let proof_version = end_version;
            let transaction_list_with_proof = create_transaction_list_with_proof(
                start_version,
                end_version,
                proof_version,
                include_events,
            );

            // Create the mock db reader
            let mut db_reader = create_mock_db_reader();
            expect_get_transactions(
                &mut db_reader,
                start_version,
                chunk_size,
                proof_version,
                include_events,
                transaction_list_with_proof.clone(),
            );

            // Create the storage client and server
            let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
            tokio::spawn(service.start());

            // Create a request to fetch transactions with a proof
            let response = get_transactions_with_proof(
                &mut mock_client,
                start_version,
                end_version,
                proof_version,
                include_events,
                true,
            )
            .await
            .unwrap();

            // Verify the response is correct
            match response.get_data_response().unwrap() {
                DataResponse::TransactionsWithProof(transactions_with_proof) => {
                    assert_eq!(transactions_with_proof, transaction_list_with_proof)
                },
                _ => panic!("Expected transactions with proof but got: {:?}", response),
            };
        }
    }
}

#[tokio::test]
async fn test_get_transactions_with_chunk_limit() {
    // Test event inclusion
    for include_events in [true, false] {
        // Create test data
        let max_transaction_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
        let chunk_size = max_transaction_chunk_size * 10; // Set a chunk request larger than the max
        let start_version = 0;
        let end_version = start_version + max_transaction_chunk_size - 1;
        let proof_version = end_version;
        let transaction_list_with_proof = create_transaction_list_with_proof(
            start_version,
            end_version,
            proof_version,
            include_events,
        );

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        expect_get_transactions(
            &mut db_reader,
            start_version,
            max_transaction_chunk_size,
            proof_version,
            include_events,
            transaction_list_with_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Create a request to fetch transactions with a proof
        let response = get_transactions_with_proof(
            &mut mock_client,
            start_version,
            start_version + chunk_size - 1,
            proof_version,
            include_events,
            true,
        )
        .await
        .unwrap();

        // Verify the response is correct
        match response.get_data_response().unwrap() {
            DataResponse::TransactionsWithProof(transactions_with_proof) => {
                assert_eq!(transactions_with_proof, transaction_list_with_proof)
            },
            _ => panic!("Expected transactions with proof but got: {:?}", response),
        };
    }
}

#[tokio::test]
async fn test_get_transactions_with_proof_network_limit() {
    // Test different byte limits
    for network_limit_bytes in [1, 1024, 10 * 1024, 100 * 1024] {
        get_transactions_with_proof_network_limit(network_limit_bytes).await;
    }
}

#[tokio::test]
async fn test_get_transactions_with_proof_invalid() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Test invalid ranges
    let start_version = 1000;
    for end_version in [0, 999] {
        let response = get_transactions_with_proof(
            &mut mock_client,
            start_version,
            end_version,
            end_version,
            true,
            true,
        )
        .await
        .unwrap_err();
        assert_matches!(response, StorageServiceError::InvalidRequest(_));
    }
}

#[tokio::test]
async fn test_get_transaction_outputs_with_proof() {
    // Test small and large chunk requests
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [1, 100, max_output_chunk_size] {
        // Create test data
        let start_version = 0;
        let end_version = start_version + chunk_size - 1;
        let proof_version = end_version;
        let output_list_with_proof =
            create_output_list_with_proof(start_version, end_version, proof_version);

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        expect_get_transaction_outputs(
            &mut db_reader,
            start_version,
            chunk_size,
            proof_version,
            output_list_with_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Create a request to fetch transactions outputs with a proof
        let response = get_outputs_with_proof(
            &mut mock_client,
            start_version,
            end_version,
            end_version,
            true,
        )
        .await
        .unwrap();

        // Verify the response is correct
        match response.get_data_response().unwrap() {
            DataResponse::TransactionOutputsWithProof(outputs_with_proof) => {
                assert_eq!(outputs_with_proof, output_list_with_proof)
            },
            _ => panic!(
                "Expected transaction outputs with proof but got: {:?}",
                response
            ),
        };
    }
}

#[tokio::test]
async fn test_get_transaction_outputs_with_proof_chunk_limit() {
    // Create test data
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    let chunk_size = max_output_chunk_size * 10; // Set a chunk request larger than the max
    let start_version = 0;
    let end_version = start_version + max_output_chunk_size - 1;
    let proof_version = end_version;
    let output_list_with_proof =
        create_output_list_with_proof(start_version, end_version, proof_version);

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    expect_get_transaction_outputs(
        &mut db_reader,
        start_version,
        max_output_chunk_size,
        proof_version,
        output_list_with_proof.clone(),
    );

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Create a request to fetch transactions outputs with a proof
    let response = get_outputs_with_proof(
        &mut mock_client,
        start_version,
        start_version + chunk_size - 1,
        end_version,
        true,
    )
    .await
    .unwrap();

    // Verify the response is correct
    match response.get_data_response().unwrap() {
        DataResponse::TransactionOutputsWithProof(outputs_with_proof) => {
            assert_eq!(outputs_with_proof, output_list_with_proof)
        },
        _ => panic!(
            "Expected transaction outputs with proof but got: {:?}",
            response
        ),
    };
}

#[tokio::test]
async fn test_get_transaction_outputs_with_proof_network_limit() {
    // Test different byte limits
    for network_limit_bytes in [1, 5 * 1024, 50 * 1024, 100 * 1024] {
        get_outputs_with_proof_network_limit(network_limit_bytes).await;
    }
}

#[tokio::test]
async fn test_get_transaction_outputs_with_proof_invalid() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Test invalid ranges
    let start_version = 1000;
    for end_version in [0, 999] {
        let response = get_outputs_with_proof(
            &mut mock_client,
            start_version,
            end_version,
            end_version,
            true,
        )
        .await
        .unwrap_err();
        assert_matches!(response, StorageServiceError::InvalidRequest(_));
    }
}

#[tokio::test]
async fn test_get_transactions_or_outputs_with_proof() {
    // Test small and large chunk requests
    let max_output_chunk_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
    for chunk_size in [1, 100, max_output_chunk_size] {
        // Test fallback to transaction syncing
        for fallback_to_transactions in [false, true] {
            // Create test data
            let start_version = 0;
            let end_version = start_version + chunk_size - 1;
            let proof_version = end_version;
            let output_list_with_proof =
                create_output_list_with_proof(start_version, end_version, proof_version);
            let transaction_list_with_proof = create_transaction_list_with_proof(
                start_version,
                start_version,
                proof_version,
                false,
            ); // Creates a small transaction list

            // Create the mock db reader
            let max_num_output_reductions = 5;
            let mut db_reader = create_mock_db_reader();
            for i in 0..=max_num_output_reductions {
                expect_get_transaction_outputs(
                    &mut db_reader,
                    start_version,
                    (chunk_size as u32 / (u32::pow(2, i as u32))) as u64,
                    proof_version,
                    output_list_with_proof.clone(),
                );
            }
            if fallback_to_transactions {
                expect_get_transactions(
                    &mut db_reader,
                    start_version,
                    chunk_size,
                    proof_version,
                    false,
                    transaction_list_with_proof.clone(),
                );
            }

            // Create the storage client and server
            let storage_config = configure_network_chunk_limit(
                fallback_to_transactions,
                &output_list_with_proof,
                &transaction_list_with_proof,
            );
            let (mut mock_client, service, _) =
                MockClient::new(Some(db_reader), Some(storage_config));
            tokio::spawn(service.start());

            // Create a request to fetch transactions or outputs with a proof
            let response = get_transactions_or_outputs_with_proof(
                &mut mock_client,
                start_version,
                end_version,
                end_version,
                false,
                max_num_output_reductions,
                true,
            )
            .await
            .unwrap();

            // Verify the response is correct
            verify_transactions_or_output_response(
                fallback_to_transactions,
                &output_list_with_proof,
                &transaction_list_with_proof,
                &response,
            );
        }
    }
}

#[tokio::test]
async fn test_get_transactions_or_outputs_with_proof_network_limit() {
    // Test different byte limits
    for network_limit_bytes in [1, 2 * 1024, 10 * 1024, 30 * 1024] {
        get_transactions_or_outputs_with_proof_network_limit(network_limit_bytes).await;
    }
}

#[tokio::test]
async fn test_get_transactions_or_outputs_with_proof_invalid() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Test invalid ranges
    let start_version = 1000;
    for end_version in [0, 999] {
        let response = get_transactions_or_outputs_with_proof(
            &mut mock_client,
            start_version,
            end_version,
            end_version,
            false,
            3,
            true,
        )
        .await
        .unwrap_err();
        assert_matches!(response, StorageServiceError::InvalidRequest(_));
    }
}

#[tokio::test]
async fn test_get_transactions_or_outputs_with_proof_chunk_limit() {
    // Test fallback to transaction syncing
    for fallback_to_transactions in [false, true] {
        // Create test data
        let max_output_chunk_size =
            StorageServiceConfig::default().max_transaction_output_chunk_size;
        let max_transaction_chunk_size = StorageServiceConfig::default().max_transaction_chunk_size;
        let chunk_size = max_output_chunk_size * 10; // Set a chunk request larger than the max
        let start_version = 0;
        let end_version = start_version + max_output_chunk_size - 1;
        let proof_version = end_version;
        let output_list_with_proof =
            create_output_list_with_proof(start_version, end_version, proof_version);
        let transaction_list_with_proof =
            create_transaction_list_with_proof(start_version, start_version, proof_version, false); // Creates a small transaction list

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        expect_get_transaction_outputs(
            &mut db_reader,
            start_version,
            max_output_chunk_size,
            proof_version,
            output_list_with_proof.clone(),
        );
        if fallback_to_transactions {
            expect_get_transactions(
                &mut db_reader,
                start_version,
                max_transaction_chunk_size,
                proof_version,
                false,
                transaction_list_with_proof.clone(),
            );
        }

        // Create the storage client and server
        let storage_config = configure_network_chunk_limit(
            fallback_to_transactions,
            &output_list_with_proof,
            &transaction_list_with_proof,
        );
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Create a request to fetch transactions outputs with a proof
        let response = get_transactions_or_outputs_with_proof(
            &mut mock_client,
            start_version,
            start_version + chunk_size - 1,
            end_version,
            false,
            0,
            false,
        )
        .await
        .unwrap();

        // Verify the response is correct
        verify_transactions_or_output_response(
            fallback_to_transactions,
            &output_list_with_proof,
            &transaction_list_with_proof,
            &response,
        );
    }
}

#[tokio::test]
async fn test_get_epoch_ending_ledger_infos() {
    // Test small and large chunk requests
    let max_epoch_chunk_size = StorageServiceConfig::default().max_epoch_chunk_size;
    for chunk_size in [1, 100, max_epoch_chunk_size] {
        // Create test data
        let start_epoch = 11;
        let expected_end_epoch = start_epoch + chunk_size - 1;
        let epoch_change_proof = EpochChangeProof {
            ledger_info_with_sigs: create_epoch_ending_ledger_infos(
                start_epoch,
                expected_end_epoch,
            ),
            more: false,
        };

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        expect_get_epoch_ending_ledger_infos(
            &mut db_reader,
            start_epoch,
            expected_end_epoch + 1,
            epoch_change_proof.clone(),
        );

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
        tokio::spawn(service.start());

        // Create a request to fetch epoch ending ledger infos
        let data_request = DataRequest::GetEpochEndingLedgerInfos(EpochEndingLedgerInfoRequest {
            start_epoch,
            expected_end_epoch,
        });
        let storage_request = StorageServiceRequest::new(data_request, true);

        // Process the request
        let response = mock_client.process_request(storage_request).await.unwrap();

        // Verify the response is correct
        match response.get_data_response().unwrap() {
            DataResponse::EpochEndingLedgerInfos(response_epoch_change_proof) => {
                assert_eq!(response_epoch_change_proof, epoch_change_proof)
            },
            _ => panic!("Expected epoch ending ledger infos but got: {:?}", response),
        };
    }
}

#[tokio::test]
async fn test_get_epoch_ending_ledger_infos_chunk_limit() {
    // Create test data
    let max_epoch_chunk_size = StorageServiceConfig::default().max_epoch_chunk_size;
    let chunk_size = max_epoch_chunk_size * 10; // Set a chunk request larger than the max
    let start_epoch = 11;
    let expected_end_epoch = start_epoch + max_epoch_chunk_size - 1;
    let epoch_change_proof = EpochChangeProof {
        ledger_info_with_sigs: create_epoch_ending_ledger_infos(start_epoch, expected_end_epoch),
        more: false,
    };

    // Create the mock db reader
    let mut db_reader = create_mock_db_reader();
    expect_get_epoch_ending_ledger_infos(
        &mut db_reader,
        start_epoch,
        expected_end_epoch + 1,
        epoch_change_proof.clone(),
    );

    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(Some(db_reader), None);
    tokio::spawn(service.start());

    // Create a request to fetch epoch ending ledger infos
    let data_request = DataRequest::GetEpochEndingLedgerInfos(EpochEndingLedgerInfoRequest {
        start_epoch,
        expected_end_epoch: start_epoch + chunk_size - 1,
    });
    let storage_request = StorageServiceRequest::new(data_request, true);

    // Process the request
    let response = mock_client.process_request(storage_request).await.unwrap();

    // Verify the response is correct
    match response.get_data_response().unwrap() {
        DataResponse::EpochEndingLedgerInfos(response_epoch_change_proof) => {
            assert_eq!(response_epoch_change_proof, epoch_change_proof)
        },
        _ => panic!("Expected epoch ending ledger infos but got: {:?}", response),
    };
}

#[tokio::test]
async fn test_get_epoch_ending_ledger_infos_network_limit() {
    // Test different byte limits
    for network_limit_bytes in [1, 10 * 1024, 50 * 1024, 100 * 1024, 1024 * 1024] {
        get_epoch_ending_ledger_infos_network_limit(network_limit_bytes).await;
    }
}

#[tokio::test]
async fn test_get_epoch_ending_ledger_infos_invalid() {
    // Create the storage client and server
    let (mut mock_client, service, _) = MockClient::new(None, None);
    tokio::spawn(service.start());

    // Test invalid ranges
    let start_epoch = 11;
    for expected_end_epoch in [0, 10] {
        let data_request = DataRequest::GetEpochEndingLedgerInfos(EpochEndingLedgerInfoRequest {
            start_epoch,
            expected_end_epoch,
        });
        let storage_request = StorageServiceRequest::new(data_request, true);

        // Process and verify the response
        let response = mock_client
            .process_request(storage_request)
            .await
            .unwrap_err();
        assert_matches!(response, StorageServiceError::InvalidRequest(_));
    }
}

/// A wrapper around the inbound network interface/channel for easily sending
/// mock client requests to a [`StorageServiceServer`].
struct MockClient {
    peer_manager_notifiers:
        HashMap<NetworkId, aptos_channel::Sender<(PeerId, ProtocolId), PeerManagerNotification>>,
}

impl MockClient {
    fn new(
        db_reader: Option<MockDatabaseReader>,
        storage_config: Option<StorageServiceConfig>,
    ) -> (Self, StorageServiceServer<StorageReader>, MockTimeService) {
        initialize_logger();

        // Create the storage reader
        let storage_config = storage_config.unwrap_or_default();
        let storage_reader = StorageReader::new(
            storage_config,
            Arc::new(db_reader.unwrap_or_else(create_mock_db_reader)),
        );

        // Setup the networks and the network events
        let network_ids = vec![NetworkId::Validator, NetworkId::Vfn, NetworkId::Public];
        let mut network_and_events = HashMap::new();
        let mut peer_manager_notifiers = HashMap::new();
        for network_id in network_ids {
            let queue_cfg =
                aptos_channel::Config::new(storage_config.max_network_channel_size as usize)
                    .queue_style(QueueStyle::FIFO)
                    .counters(&metrics::PENDING_STORAGE_SERVER_NETWORK_EVENTS);
            let (peer_manager_notifier, peer_manager_notification_receiver) = queue_cfg.build();
            let (_, connection_notification_receiver) = queue_cfg.build();

            let network_events = NetworkEvents::new(
                peer_manager_notification_receiver,
                connection_notification_receiver,
            );
            network_and_events.insert(network_id, network_events);
            peer_manager_notifiers.insert(network_id, peer_manager_notifier);
        }
        let storage_service_network_events =
            StorageServiceNetworkEvents::new(NetworkServiceEvents::new(network_and_events));

        // Create the storage service
        let executor = tokio::runtime::Handle::current();
        let mock_time_service = TimeService::mock();
        let storage_server = StorageServiceServer::new(
            StorageServiceConfig::default(),
            executor,
            storage_reader,
            mock_time_service.clone(),
            storage_service_network_events,
        );

        // Return the client and service
        let mock_client = Self {
            peer_manager_notifiers,
        };
        (mock_client, storage_server, mock_time_service.into_mock())
    }

    /// Send the given storage request and wait for a response
    async fn process_request(
        &mut self,
        request: StorageServiceRequest,
    ) -> Result<StorageServiceResponse, StorageServiceError> {
        let receiver = self.send_request(request, None, None).await;
        self.wait_for_response(receiver).await
    }

    /// Send the specified storage request and return the receiver on which to
    /// expect a result.
    pub async fn send_request(
        &mut self,
        request: StorageServiceRequest,
        peer_id: Option<AccountAddress>,
        network_id: Option<NetworkId>,
    ) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
        // Create the inbound rpc request
        let peer_id = peer_id.unwrap_or_else(PeerId::random);
        let network_id = network_id.unwrap_or_else(get_random_network_id);
        let protocol_id = ProtocolId::StorageServiceRpc;
        let data = protocol_id
            .to_bytes(&StorageServiceMessage::Request(request))
            .unwrap();
        let (res_tx, res_rx) = oneshot::channel();
        let inbound_rpc = InboundRpcRequest {
            protocol_id,
            data: data.into(),
            res_tx,
        };
        let notification = PeerManagerNotification::RecvRpc(peer_id, inbound_rpc);

        // Push the request up to the storage service
        self.peer_manager_notifiers
            .get(&network_id)
            .unwrap()
            .push((peer_id, protocol_id), notification)
            .unwrap();

        res_rx
    }

    /// Helper method to wait for and deserialize a response on the specified receiver
    async fn wait_for_response(
        &mut self,
        receiver: Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>>,
    ) -> Result<StorageServiceResponse, StorageServiceError> {
        if let Ok(response) =
            timeout(Duration::from_secs(MAX_RESPONSE_TIMEOUT_SECS), receiver).await
        {
            let response = ProtocolId::StorageServiceRpc
                .from_bytes::<StorageServiceMessage>(&response.unwrap().unwrap())
                .unwrap();
            match response {
                StorageServiceMessage::Response(response) => response,
                _ => panic!("Unexpected response message: {:?}", response),
            }
        } else {
            panic!("Timed out while waiting for a response from the storage service!")
        }
    }
}

/// A helper method to request a states with proof chunk using the
/// the specified network limit.
async fn get_epoch_ending_ledger_infos_network_limit(network_limit_bytes: u64) {
    for use_compression in [true, false] {
        // Create test data
        let max_epoch_chunk_size = StorageServiceConfig::default().max_epoch_chunk_size;
        let min_bytes_per_ledger_info = 5000;
        let start_epoch = 98754;

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        let mut expectation_sequence = Sequence::new();
        let mut chunk_size = max_epoch_chunk_size;
        while chunk_size >= 1 {
            let ledger_info_with_sigs =
                create_epoch_ending_ledger_infos_using_sizes(chunk_size, min_bytes_per_ledger_info);
            let epoch_change_proof = EpochChangeProof {
                ledger_info_with_sigs,
                more: false,
            };
            db_reader
                .expect_get_epoch_ending_ledger_infos()
                .times(1)
                .with(eq(start_epoch), eq(start_epoch + chunk_size))
                .in_sequence(&mut expectation_sequence)
                .returning(move |_, _| Ok(epoch_change_proof.clone()));
            chunk_size /= 2;
        }

        // Create a storage config with the specified max network byte limit
        let storage_config = StorageServiceConfig {
            max_network_chunk_bytes: network_limit_bytes,
            ..Default::default()
        };

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Process a request to fetch epoch ending ledger infos
        let data_request = DataRequest::GetEpochEndingLedgerInfos(EpochEndingLedgerInfoRequest {
            start_epoch,
            expected_end_epoch: start_epoch + max_epoch_chunk_size - 1,
        });
        let storage_request = StorageServiceRequest::new(data_request, use_compression);
        let response = mock_client.process_request(storage_request).await.unwrap();

        // Verify the response adheres to the network limits
        match response.get_data_response().unwrap() {
            DataResponse::EpochEndingLedgerInfos(epoch_change_proof) => {
                let num_response_bytes = bcs::to_bytes(&response).unwrap().len() as u64;
                let num_ledger_infos = epoch_change_proof.ledger_info_with_sigs.len() as u64;
                if num_response_bytes > network_limit_bytes {
                    assert_eq!(num_ledger_infos, 1); // Data cannot be reduced more than a single item
                } else {
                    let max_num_ledger_infos = network_limit_bytes / min_bytes_per_ledger_info;
                    assert!(num_ledger_infos <= max_num_ledger_infos); // Verify data fits correctly into the limit
                }
            },
            _ => panic!("Expected epoch ending ledger infos but got: {:?}", response),
        }
    }
}

/// A helper method to request a states with proof chunk using the
/// the specified network limit.
async fn get_states_with_proof_network_limit(network_limit_bytes: u64) {
    for use_compression in [true, false] {
        // Create test data
        let max_state_chunk_size = StorageServiceConfig::default().max_state_chunk_size;
        let min_bytes_per_state_value = 100;
        let version = 101;
        let start_index = 100;

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        let mut expectation_sequence = Sequence::new();
        let mut chunk_size = max_state_chunk_size;
        while chunk_size >= 1 {
            let state_value_chunk_with_proof = StateValueChunkWithProof {
                first_index: start_index,
                last_index: start_index + chunk_size - 1,
                first_key: HashValue::random(),
                last_key: HashValue::random(),
                raw_values: create_state_keys_and_values(chunk_size, min_bytes_per_state_value),
                proof: SparseMerkleRangeProof::new(vec![]),
                root_hash: HashValue::random(),
            };
            db_reader
                .expect_get_state_value_chunk_with_proof()
                .times(1)
                .with(
                    eq(version),
                    eq(start_index as usize),
                    eq(chunk_size as usize),
                )
                .in_sequence(&mut expectation_sequence)
                .returning(move |_, _, _| Ok(state_value_chunk_with_proof.clone()));
            chunk_size /= 2;
        }

        // Create a storage config with the specified max network byte limit
        let storage_config = StorageServiceConfig {
            max_network_chunk_bytes: network_limit_bytes,
            ..Default::default()
        };

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Process a request to fetch a states chunk with a proof
        let response = get_state_values_with_proof(
            &mut mock_client,
            version,
            start_index,
            start_index + max_state_chunk_size + 1000, // Request more than the max chunk
            use_compression,
        )
        .await
        .unwrap();

        // Verify the response adheres to the network limits
        match response.get_data_response().unwrap() {
            DataResponse::StateValueChunkWithProof(state_value_chunk_with_proof) => {
                let num_response_bytes = bcs::to_bytes(&response).unwrap().len() as u64;
                let num_state_values = state_value_chunk_with_proof.raw_values.len() as u64;
                if num_response_bytes > network_limit_bytes {
                    assert_eq!(num_state_values, 1); // Data cannot be reduced more than a single item
                } else {
                    let max_num_state_values = network_limit_bytes / min_bytes_per_state_value;
                    assert!(num_state_values <= max_num_state_values); // Verify data fits correctly into the limit
                }
            },
            _ => panic!("Expected state values with proof but got: {:?}", response),
        }
    }
}

/// A helper method to request a transaction outputs with proof chunk using the
/// the specified network limit.
async fn get_outputs_with_proof_network_limit(network_limit_bytes: u64) {
    for use_compression in [true, false] {
        // Create test data
        let max_output_chunk_size =
            StorageServiceConfig::default().max_transaction_output_chunk_size;
        let min_bytes_per_output = 1536; // 1.5 KB
        let start_version = 455;
        let proof_version = 1000000;

        // Create the mock db reader
        let mut db_reader = create_mock_db_reader();
        let mut expectation_sequence = Sequence::new();
        let mut chunk_size = max_output_chunk_size;
        while chunk_size >= 1 {
            let output_list_with_proof =
                create_output_list_using_sizes(start_version, chunk_size, min_bytes_per_output);
            db_reader
                .expect_get_transaction_outputs()
                .times(1)
                .with(eq(start_version), eq(chunk_size), eq(proof_version))
                .in_sequence(&mut expectation_sequence)
                .returning(move |_, _, _| Ok(output_list_with_proof.clone()));
            chunk_size /= 2;
        }

        // Create a storage config with the specified max network byte limit
        let storage_config = StorageServiceConfig {
            max_network_chunk_bytes: network_limit_bytes,
            ..Default::default()
        };

        // Create the storage client and server
        let (mut mock_client, service, _) = MockClient::new(Some(db_reader), Some(storage_config));
        tokio::spawn(service.start());

        // Process a request to fetch outputs with a proof
        let response = get_outputs_with_proof(
            &mut mock_client,
            start_version,
            start_version + (max_output_chunk_size * 10), // Request more than the max chunk
            proof_version,
            use_compression,
        )
        .await
        .unwrap();

        // Verify the response is correct
        match response.get_data_response().unwrap() {
            DataResponse::TransactionOutputsWithProof(outputs_with_proof) => {
                let num_response_bytes = bcs::to_bytes(&response).unwrap().len() as u64;
                let num_outputs = outputs_with_proof.transactions_and_outputs.len() as u64;
                if num_response_bytes > network_limit_bytes {
                    assert_eq!(num_outputs, 1); // Data cannot be reduced more than a single item
                } else {
                    let max_outputs = network_limit_bytes / min_bytes_per_output;
                    assert!(num_outputs <= max_outputs); // Verify data fits correctly into the limit
                }
            },
            _ => panic!("Expected outputs with proof but got: {:?}", response),
        };
    }
}

/// A helper method to request a transactions with proof chunk using the
/// the specified network limit.
async fn get_transactions_with_proof_network_limit(network_limit_bytes: u64) {
    for use_compression in [true, false] {
        for include_events in [true, false] {
            // Create test data
            let max_transaction_chunk_size =
                StorageServiceConfig::default().max_transaction_chunk_size;
            let min_bytes_per_transaction = 512; // 0.5 KB
            let start_version = 121245;
            let proof_version = 202020;

            // Create the mock db reader
            let mut db_reader = create_mock_db_reader();
            let mut expectation_sequence = Sequence::new();
            let mut chunk_size = max_transaction_chunk_size;
            while chunk_size >= 1 {
                let transaction_list_with_proof = create_transaction_list_using_sizes(
                    start_version,
                    chunk_size,
                    min_bytes_per_transaction,
                    include_events,
                );
                db_reader
                    .expect_get_transactions()
                    .times(1)
                    .with(
                        eq(start_version),
                        eq(chunk_size),
                        eq(proof_version),
                        eq(include_events),
                    )
                    .in_sequence(&mut expectation_sequence)
                    .returning(move |_, _, _, _| Ok(transaction_list_with_proof.clone()));
                chunk_size /= 2;
            }

            // Create a storage config with the specified max network byte limit
            let storage_config = StorageServiceConfig {
                max_network_chunk_bytes: network_limit_bytes,
                ..Default::default()
            };

            // Create the storage client and server
            let (mut mock_client, service, _) =
                MockClient::new(Some(db_reader), Some(storage_config));
            tokio::spawn(service.start());

            // Process a request to fetch transactions with a proof
            let response = get_transactions_with_proof(
                &mut mock_client,
                start_version,
                start_version + max_transaction_chunk_size - 1,
                proof_version,
                include_events,
                use_compression,
            )
            .await
            .unwrap();

            // Verify the response is correct
            match response.get_data_response().unwrap() {
                DataResponse::TransactionsWithProof(transactions_with_proof) => {
                    let num_response_bytes = bcs::to_bytes(&response).unwrap().len() as u64;
                    let num_transactions = transactions_with_proof.transactions.len() as u64;
                    if num_response_bytes > network_limit_bytes {
                        assert_eq!(num_transactions, 1); // Data cannot be reduced more than a single item
                    } else {
                        let max_transactions = network_limit_bytes / min_bytes_per_transaction;
                        assert!(num_transactions <= max_transactions); // Verify data fits correctly into the limit
                    }
                },
                _ => panic!("Expected transactions with proof but got: {:?}", response),
            };
        }
    }
}

/// A helper method to request transactions or outputs with proof using the
/// the specified network limit.
async fn get_transactions_or_outputs_with_proof_network_limit(network_limit_bytes: u64) {
    for use_compression in [true, false] {
        for include_events in [true, false] {
            // Create test data
            let min_bytes_per_output = 2500; // 2.5 KB
            let min_bytes_per_transaction = 499; // 0.5 KB
            let start_version = 455;
            let proof_version = 1000000;
            let max_output_size = StorageServiceConfig::default().max_transaction_output_chunk_size;
            let max_transaction_size = StorageServiceConfig::default().max_transaction_chunk_size;

            // Create the mock db reader
            let mut db_reader = create_mock_db_reader();
            let mut expectation_sequence = Sequence::new();
            let mut chunk_size = max_output_size;
            let mut max_num_output_reductions = 0;
            while chunk_size >= 1 {
                let output_list_with_proof =
                    create_output_list_using_sizes(start_version, chunk_size, min_bytes_per_output);
                db_reader
                    .expect_get_transaction_outputs()
                    .times(1)
                    .with(eq(start_version), eq(chunk_size), eq(proof_version))
                    .in_sequence(&mut expectation_sequence)
                    .returning(move |_, _, _| Ok(output_list_with_proof.clone()));
                chunk_size /= 2;
                max_num_output_reductions += 1;
            }
            let mut chunk_size = max_transaction_size;
            while chunk_size >= 1 {
                let transaction_list_with_proof = create_transaction_list_using_sizes(
                    start_version,
                    chunk_size,
                    min_bytes_per_transaction,
                    include_events,
                );
                db_reader
                    .expect_get_transactions()
                    .times(1)
                    .with(
                        eq(start_version),
                        eq(chunk_size),
                        eq(proof_version),
                        eq(include_events),
                    )
                    .in_sequence(&mut expectation_sequence)
                    .returning(move |_, _, _, _| Ok(transaction_list_with_proof.clone()));
                chunk_size /= 2;
            }

            // Create the storage client and server
            let storage_config = StorageServiceConfig {
                max_network_chunk_bytes: network_limit_bytes,
                ..Default::default()
            };
            let (mut mock_client, service, _) =
                MockClient::new(Some(db_reader), Some(storage_config));
            tokio::spawn(service.start());

            // Process a request to fetch transactions or outputs with a proof
            let response = get_transactions_or_outputs_with_proof(
                &mut mock_client,
                start_version,
                start_version + (max_output_size * 10), // Request more than the max chunk
                proof_version,
                include_events,
                max_num_output_reductions,
                use_compression,
            )
            .await
            .unwrap();

            // Verify the response is correct
            match response.get_data_response().unwrap() {
                DataResponse::TransactionsOrOutputsWithProof(
                    transactions_or_outputs_with_proof,
                ) => {
                    let (transactions_with_proof, outputs_with_proof) =
                        transactions_or_outputs_with_proof;

                    if let Some(transactions_with_proof) = transactions_with_proof {
                        let num_response_bytes =
                            bcs::to_bytes(&transactions_with_proof).unwrap().len() as u64;
                        let num_transactions = transactions_with_proof.transactions.len() as u64;
                        if num_response_bytes > network_limit_bytes {
                            assert_eq!(num_transactions, 1); // Data cannot be reduced more than a single item
                        } else {
                            let max_transactions = network_limit_bytes / min_bytes_per_transaction;
                            assert!(num_transactions <= max_transactions);
                        }
                    } else if let Some(outputs_with_proof) = outputs_with_proof {
                        let num_response_bytes =
                            bcs::to_bytes(&outputs_with_proof).unwrap().len() as u64;
                        let num_outputs = outputs_with_proof.transactions_and_outputs.len() as u64;
                        if num_response_bytes > network_limit_bytes {
                            assert_eq!(num_outputs, 1); // Data cannot be reduced more than a single item
                        } else {
                            let max_outputs = network_limit_bytes / min_bytes_per_output;
                            assert!(num_outputs <= max_outputs);
                        }
                    } else {
                        panic!("No transactions or outputs were returned!");
                    }
                },
                _ => panic!(
                    "Expected transactions or outputs with proof but got: {:?}",
                    response
                ),
            };
        }
    }
}

/// Waits until the storage summary has refreshed for the first time
async fn wait_for_storage_to_refresh(mock_client: &mut MockClient, mock_time: &MockTimeService) {
    let storage_request = StorageServiceRequest::new(DataRequest::GetStorageServerSummary, true);
    while mock_client
        .process_request(storage_request.clone())
        .await
        .unwrap()
        == StorageServiceResponse::new(
            DataResponse::StorageServerSummary(StorageServerSummary::default()),
            true,
        )
        .unwrap()
    {
        advance_storage_refresh_time(mock_time).await;
    }
}

/// Advances enough time that the subscription service is able to refresh
async fn wait_for_subscription_service_to_refresh(
    mock_client: &mut MockClient,
    mock_time: &MockTimeService,
) {
    // Elapse enough time to force storage to be updated
    wait_for_storage_to_refresh(mock_client, mock_time).await;

    // Elapse enough time to force the subscription thread to work
    advance_storage_refresh_time(mock_time).await;
}

/// Advances the given timer by the amount of time it takes to refresh storage
async fn advance_storage_refresh_time(mock_time: &MockTimeService) {
    let default_storage_config = StorageServiceConfig::default();
    let cache_update_freq_ms = default_storage_config.storage_summary_refresh_interval_ms;
    mock_time.advance_ms_async(cache_update_freq_ms).await;
}

/// Sends a transactions with proof request and processes the response
async fn get_transactions_with_proof(
    mock_client: &mut MockClient,
    start_version: u64,
    end_version: u64,
    proof_version: u64,
    include_events: bool,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request = DataRequest::GetTransactionsWithProof(TransactionsWithProofRequest {
        proof_version,
        start_version,
        end_version,
        include_events,
    });
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Sends a transaction outputs with proof request and processes the response
async fn get_outputs_with_proof(
    mock_client: &mut MockClient,
    start_version: u64,
    end_version: u64,
    proof_version: u64,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request =
        DataRequest::GetTransactionOutputsWithProof(TransactionOutputsWithProofRequest {
            proof_version,
            start_version,
            end_version,
        });
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Sends a transaction or outputs with proof request and processes the response
async fn get_transactions_or_outputs_with_proof(
    mock_client: &mut MockClient,
    start_version: u64,
    end_version: u64,
    proof_version: u64,
    include_events: bool,
    max_num_output_reductions: u64,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request =
        DataRequest::GetTransactionsOrOutputsWithProof(TransactionsOrOutputsWithProofRequest {
            proof_version,
            start_version,
            end_version,
            include_events,
            max_num_output_reductions,
        });
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Sends a state values with proof request and processes the response
async fn get_state_values_with_proof(
    mock_client: &mut MockClient,
    version: u64,
    start_index: u64,
    end_index: u64,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request = DataRequest::GetStateValuesWithProof(StateValuesWithProofRequest {
        version,
        start_index,
        end_index,
    });
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Sends a number of states request and processes the response
async fn get_number_of_states(
    mock_client: &mut MockClient,
    version: u64,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request = DataRequest::GetNumberOfStatesAtVersion(version);
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Sends a protocol version request and processes the response
async fn get_protocol_version(
    mock_client: &mut MockClient,
    use_compression: bool,
) -> StorageServiceResponse {
    let data_request = DataRequest::GetServerProtocolVersion;
    send_storage_request(mock_client, use_compression, data_request)
        .await
        .unwrap()
}

/// Sends a storage summary request and processes the response
async fn get_storage_server_summary(
    mock_client: &mut MockClient,
    use_compression: bool,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let data_request = DataRequest::GetStorageServerSummary;
    send_storage_request(mock_client, use_compression, data_request).await
}

/// Creates and sends a request for new transaction outputs
async fn get_new_outputs_with_proof(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    get_new_outputs_with_proof_for_peer(mock_client, known_version, known_epoch, None).await
}

/// Creates and sends a request for new transaction outputs for the specified peer
async fn get_new_outputs_with_proof_for_peer(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
    peer_network_id: Option<PeerNetworkId>,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    // Create the data request
    let data_request =
        DataRequest::GetNewTransactionOutputsWithProof(NewTransactionOutputsWithProofRequest {
            known_version,
            known_epoch,
        });
    let storage_request = StorageServiceRequest::new(data_request, true);

    // Send the request
    let (peer_id, network_id) = extract_peer_and_network_id(peer_network_id);
    mock_client
        .send_request(storage_request, peer_id, network_id)
        .await
}

/// Creates and sends a request for new transactions
async fn get_new_transactions_with_proof(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
    include_events: bool,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    get_new_transactions_with_proof_for_peer(
        mock_client,
        known_version,
        known_epoch,
        include_events,
        None,
    )
    .await
}

/// Creates and sends a request for new transactions for the specified peer
async fn get_new_transactions_with_proof_for_peer(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
    include_events: bool,
    peer_network_id: Option<PeerNetworkId>,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    // Create the data request
    let data_request = DataRequest::GetNewTransactionsWithProof(NewTransactionsWithProofRequest {
        known_version,
        known_epoch,
        include_events,
    });
    let storage_request = StorageServiceRequest::new(data_request, true);

    // Send the request
    let (peer_id, network_id) = extract_peer_and_network_id(peer_network_id);
    mock_client
        .send_request(storage_request, peer_id, network_id)
        .await
}

/// Creates and sends a request for new transactions or outputs
async fn get_new_transactions_or_outputs_with_proof(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
    include_events: bool,
    max_num_output_reductions: u64,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    get_new_transactions_or_outputs_with_proof_for_peer(
        mock_client,
        known_version,
        known_epoch,
        include_events,
        max_num_output_reductions,
        None,
    )
    .await
}

/// Creates and sends a request for new transactions or outputs for the specified peer
async fn get_new_transactions_or_outputs_with_proof_for_peer(
    mock_client: &mut MockClient,
    known_version: u64,
    known_epoch: u64,
    include_events: bool,
    max_num_output_reductions: u64,
    peer_network_id: Option<PeerNetworkId>,
) -> Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>> {
    // Create the data request
    let data_request = DataRequest::GetNewTransactionsOrOutputsWithProof(
        NewTransactionsOrOutputsWithProofRequest {
            known_version,
            known_epoch,
            include_events,
            max_num_output_reductions,
        },
    );
    let storage_request = StorageServiceRequest::new(data_request, true);

    // Send the request
    let (peer_id, network_id) = extract_peer_and_network_id(peer_network_id);
    mock_client
        .send_request(storage_request, peer_id, network_id)
        .await
}

/// Extracts the peer and network ids from an optional peer network id
fn extract_peer_and_network_id(
    peer_network_id: Option<PeerNetworkId>,
) -> (Option<AccountAddress>, Option<NetworkId>) {
    if let Some(peer_network_id) = peer_network_id {
        (
            Some(peer_network_id.peer_id()),
            Some(peer_network_id.network_id()),
        )
    } else {
        (None, None)
    }
}

/// Sends the given storage request to the given client
async fn send_storage_request(
    mock_client: &mut MockClient,
    use_compression: bool,
    data_request: DataRequest,
) -> Result<StorageServiceResponse, StorageServiceError> {
    let storage_request = StorageServiceRequest::new(data_request, use_compression);
    mock_client.process_request(storage_request).await
}

/// Creates a mock db with the basic expectations required to handle subscription requests
fn create_mock_db_for_subscription(
    highest_ledger_info_clone: LedgerInfoWithSignatures,
    lowest_version: Version,
) -> MockDatabaseReader {
    let mut db_reader = create_mock_db_reader();
    db_reader
        .expect_get_latest_ledger_info()
        .returning(move || Ok(highest_ledger_info_clone.clone()));
    db_reader
        .expect_get_first_txn_version()
        .returning(move || Ok(Some(lowest_version)));
    db_reader
        .expect_get_first_write_set_version()
        .returning(move || Ok(Some(lowest_version)));
    db_reader
        .expect_get_epoch_snapshot_prune_window()
        .returning(move || Ok(100));
    db_reader
        .expect_is_state_pruner_enabled()
        .returning(move || Ok(true));
    db_reader
}

/// Sets an expectation on the given mock db for a call to fetch transactions
fn expect_get_transactions(
    mock_db: &mut MockDatabaseReader,
    start_version: u64,
    num_items: u64,
    proof_version: u64,
    include_events: bool,
    transaction_list: TransactionListWithProof,
) {
    mock_db
        .expect_get_transactions()
        .times(1)
        .with(
            eq(start_version),
            eq(num_items),
            eq(proof_version),
            eq(include_events),
        )
        .returning(move |_, _, _, _| Ok(transaction_list.clone()));
}

/// Sets an expectation on the given mock db for a call to fetch transaction outputs
fn expect_get_transaction_outputs(
    mock_db: &mut MockDatabaseReader,
    start_version: u64,
    num_items: u64,
    proof_version: u64,
    output_list: TransactionOutputListWithProof,
) {
    mock_db
        .expect_get_transaction_outputs()
        .times(1)
        .with(eq(start_version), eq(num_items), eq(proof_version))
        .returning(move |_, _, _| Ok(output_list.clone()));
}

/// Sets an expectation on the given mock db for a call to fetch an epoch change proof
fn expect_get_epoch_ending_ledger_infos(
    mock_db: &mut MockDatabaseReader,
    start_epoch: u64,
    expected_end_epoch: u64,
    epoch_change_proof: EpochChangeProof,
) {
    mock_db
        .expect_get_epoch_ending_ledger_infos()
        .times(1)
        .with(eq(start_epoch), eq(expected_end_epoch))
        .returning(move |_, _| Ok(epoch_change_proof.clone()));
}

/// Sets an expectation on the given mock db for a call to fetch state values with proof
fn expect_get_state_values_with_proof(
    mock_db: &mut MockDatabaseReader,
    version: u64,
    start_index: u64,
    chunk_size: u64,
    state_value_chunk_with_proof: StateValueChunkWithProof,
) {
    mock_db
        .expect_get_state_value_chunk_with_proof()
        .times(1)
        .with(
            eq(version),
            eq(start_index as usize),
            eq(chunk_size as usize),
        )
        .returning(move |_, _, _| Ok(state_value_chunk_with_proof.clone()));
}

/// Creates a new storage service config with the limit
/// configured to be the size of an output list or transaction
/// list (depending on if `fallback_to_transactions` is set).
fn configure_network_chunk_limit(
    fallback_to_transactions: bool,
    output_list_with_proof: &TransactionOutputListWithProof,
    transaction_list_with_proof: &TransactionListWithProof,
) -> StorageServiceConfig {
    let max_network_chunk_bytes = if fallback_to_transactions {
        // Network limit is only big enough for the transaction list
        bcs::to_bytes(&transaction_list_with_proof).unwrap().len() as u64 + 1
    } else {
        // Network limit is big enough for the output list
        bcs::to_bytes(&output_list_with_proof).unwrap().len() as u64 + 1
    };
    StorageServiceConfig {
        max_network_chunk_bytes,
        ..Default::default()
    }
}

/// Creates a test epoch change proof
fn create_epoch_ending_ledger_infos(
    start_epoch: Epoch,
    end_epoch: Epoch,
) -> Vec<LedgerInfoWithSignatures> {
    let mut ledger_info_with_sigs = vec![];
    for epoch in start_epoch..end_epoch {
        ledger_info_with_sigs.push(create_test_ledger_info_with_sigs(epoch, 0));
    }
    ledger_info_with_sigs
}

/// Creates a test transaction output list with proof
fn create_output_list_with_proof(
    start_version: u64,
    end_version: u64,
    proof_version: u64,
) -> TransactionOutputListWithProof {
    let transaction_list_with_proof =
        create_transaction_list_with_proof(start_version, end_version, proof_version, false);
    let transactions_and_outputs = transaction_list_with_proof
        .transactions
        .iter()
        .map(|txn| (txn.clone(), create_test_transaction_output()))
        .collect();

    TransactionOutputListWithProof::new(
        transactions_and_outputs,
        Some(start_version),
        transaction_list_with_proof.proof,
    )
}

/// Creates a set of state keys and values using the specified number and size
fn create_state_keys_and_values(
    num_keys_and_values: u64,
    min_bytes_per_key_value: u64,
) -> Vec<(StateKey, StateValue)> {
    // Generate random bytes of the given size
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..min_bytes_per_key_value)
        .map(|_| rng.gen::<u8>())
        .collect();

    // Create the requested keys and values
    (0..num_keys_and_values)
        .map(|_| {
            let state_value = StateValue::new(random_bytes.clone());
            (StateKey::Raw(vec![]), state_value)
        })
        .collect()
}

/// Creates a test ledger info with signatures
fn create_test_ledger_info_with_sigs(epoch: u64, version: u64) -> LedgerInfoWithSignatures {
    // Create a mock ledger info with signatures
    let ledger_info = LedgerInfo::new(
        BlockInfo::new(
            epoch,
            0,
            HashValue::zero(),
            HashValue::zero(),
            version,
            0,
            None,
        ),
        HashValue::zero(),
    );
    LedgerInfoWithSignatures::new(ledger_info, AggregateSignature::empty())
}

/// Creates a test transaction output
fn create_test_transaction_output() -> TransactionOutput {
    TransactionOutput::new(
        WriteSet::default(),
        vec![],
        0,
        TransactionStatus::Keep(ExecutionStatus::MiscellaneousError(None)),
    )
}

/// Creates a test user transaction
fn create_test_transaction(sequence_number: u64, code_bytes: Vec<u8>) -> Transaction {
    let private_key = Ed25519PrivateKey::generate_for_testing();
    let public_key = private_key.public_key();

    let transaction_payload = TransactionPayload::Script(Script::new(code_bytes, vec![], vec![]));
    let raw_transaction = RawTransaction::new(
        AccountAddress::random(),
        sequence_number,
        transaction_payload,
        0,
        0,
        0,
        ChainId::new(10),
    );
    let signed_transaction = SignedTransaction::new(
        raw_transaction.clone(),
        public_key,
        private_key.sign(&raw_transaction).unwrap(),
    );

    Transaction::UserTransaction(signed_transaction)
}

/// Creates a test transaction output list with proof
fn create_transaction_list_with_proof(
    start_version: u64,
    end_version: u64,
    _proof_version: u64,
    include_events: bool,
) -> TransactionListWithProof {
    // Include events if required
    let events = if include_events { Some(vec![]) } else { None };

    // Create the requested transactions
    let mut transactions = vec![];
    for sequence_number in start_version..=end_version {
        transactions.push(create_test_transaction(sequence_number, vec![]));
    }

    // Create a transaction list with an empty proof
    let mut transaction_list_with_proof = TransactionListWithProof::new_empty();
    transaction_list_with_proof.first_transaction_version = Some(start_version);
    transaction_list_with_proof.events = events;
    transaction_list_with_proof.transactions = transactions;

    transaction_list_with_proof
}

/// Creates a test epoch change proof with the given sizes
fn create_epoch_ending_ledger_infos_using_sizes(
    num_ledger_infos: u64,
    min_bytes_per_ledger_info: u64,
) -> Vec<LedgerInfoWithSignatures> {
    // Create a mock ledger info
    let ledger_info = LedgerInfo::new(
        BlockInfo::new(0, 0, HashValue::zero(), HashValue::zero(), 0, 0, None),
        HashValue::zero(),
    );

    // Generate random bytes of the given size
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..min_bytes_per_ledger_info)
        .map(|_| rng.gen::<u8>())
        .collect();

    // Create the ledger infos with signatures
    (0..num_ledger_infos)
        .map(|_| {
            let multi_signatures =
                AggregateSignature::new(BitVec::from(random_bytes.clone()), None);
            LedgerInfoWithSignatures::new(ledger_info.clone(), multi_signatures)
        })
        .collect()
}

/// Creates a test transaction output list with proof with the given sizes
fn create_output_list_using_sizes(
    start_version: u64,
    num_outputs: u64,
    min_bytes_per_output: u64,
) -> TransactionOutputListWithProof {
    // Create a test transaction list that enforces the given size requirements
    let transaction_list_with_proof = create_transaction_list_using_sizes(
        start_version,
        num_outputs,
        min_bytes_per_output,
        false,
    );

    // Create a test transaction and output list
    let transactions_and_outputs = transaction_list_with_proof
        .transactions
        .iter()
        .map(|txn| (txn.clone(), create_test_transaction_output()))
        .collect();

    TransactionOutputListWithProof::new(
        transactions_and_outputs,
        Some(start_version),
        transaction_list_with_proof.proof,
    )
}

/// Creates a test transaction list with proof with the given sizes
fn create_transaction_list_using_sizes(
    start_version: u64,
    num_transactions: u64,
    min_bytes_per_transaction: u64,
    include_events: bool,
) -> TransactionListWithProof {
    // Generate random bytes of the given size
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..min_bytes_per_transaction)
        .map(|_| rng.gen::<u8>())
        .collect();

    // Include events if required
    let events = if include_events { Some(vec![]) } else { None };

    // Create the requested transactions
    let mut transactions = vec![];
    for sequence_number in start_version..=start_version + num_transactions - 1 {
        transactions.push(create_test_transaction(
            sequence_number,
            random_bytes.clone(),
        ));
    }

    // Create a transaction list with an empty proof
    let mut transaction_list_with_proof = TransactionListWithProof::new_empty();
    transaction_list_with_proof.first_transaction_version = Some(start_version);
    transaction_list_with_proof.events = events;
    transaction_list_with_proof.transactions = transactions;

    transaction_list_with_proof
}

/// Verifies that a new transaction outputs with proof response is received
/// and that the response contains the correct data.
async fn verify_new_transaction_outputs_with_proof(
    mock_client: &mut MockClient,
    receiver: Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>>,
    output_list_with_proof: TransactionOutputListWithProof,
    expected_ledger_info: LedgerInfoWithSignatures,
) {
    match mock_client
        .wait_for_response(receiver)
        .await
        .unwrap()
        .get_data_response()
        .unwrap()
    {
        DataResponse::NewTransactionOutputsWithProof((outputs_with_proof, ledger_info)) => {
            assert_eq!(outputs_with_proof, output_list_with_proof);
            assert_eq!(ledger_info, expected_ledger_info);
        },
        response => panic!(
            "Expected new transaction outputs with proof but got: {:?}",
            response
        ),
    };
}

/// Verifies that a new transactions with proof response is received
/// and that the response contains the correct data.
async fn verify_new_transactions_with_proof(
    mock_client: &mut MockClient,
    receiver: Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>>,
    expected_transactions_with_proof: TransactionListWithProof,
    expected_ledger_info: LedgerInfoWithSignatures,
) {
    match mock_client
        .wait_for_response(receiver)
        .await
        .unwrap()
        .get_data_response()
        .unwrap()
    {
        DataResponse::NewTransactionsWithProof((transactions_with_proof, ledger_info)) => {
            assert_eq!(transactions_with_proof, expected_transactions_with_proof);
            assert_eq!(ledger_info, expected_ledger_info);
        },
        response => panic!(
            "Expected new transaction with proof but got: {:?}",
            response
        ),
    };
}

/// Verifies that a new transactions or outputs with proof response is received
/// and that the response contains the correct data.
async fn verify_new_transactions_or_outputs_with_proof(
    mock_client: &mut MockClient,
    receiver: Receiver<Result<bytes::Bytes, aptos_network::protocols::network::RpcError>>,
    expected_transaction_list_with_proof: Option<TransactionListWithProof>,
    expected_output_list_with_proof: Option<TransactionOutputListWithProof>,
    expected_ledger_info: LedgerInfoWithSignatures,
) {
    let response = mock_client.wait_for_response(receiver).await.unwrap();
    match response.get_data_response().unwrap() {
        DataResponse::NewTransactionsOrOutputsWithProof((
            transactions_or_outputs_with_proof,
            ledger_info,
        )) => {
            let (transactions_with_proof, outputs_with_proof) = transactions_or_outputs_with_proof;
            if let Some(transactions_with_proof) = transactions_with_proof {
                assert_eq!(
                    transactions_with_proof,
                    expected_transaction_list_with_proof.unwrap()
                );
            } else {
                assert_eq!(
                    outputs_with_proof.unwrap(),
                    expected_output_list_with_proof.unwrap()
                );
            }
            assert_eq!(ledger_info, expected_ledger_info);
        },
        response => panic!(
            "Expected new transaction outputs with proof but got: {:?}",
            response
        ),
    };
}

/// Verifies that a transactions or outputs with proof response is received
/// and that the response contains the correct data.
fn verify_transactions_or_output_response(
    fallback_to_transactions: bool,
    output_list_with_proof: &TransactionOutputListWithProof,
    transaction_list_with_proof: &TransactionListWithProof,
    response: &StorageServiceResponse,
) {
    match response.get_data_response().unwrap() {
        DataResponse::TransactionsOrOutputsWithProof(transactions_or_outputs_with_proof) => {
            let (transactions_with_proof, outputs_with_proof) = transactions_or_outputs_with_proof;
            if fallback_to_transactions {
                assert_eq!(
                    transactions_with_proof.unwrap(),
                    transaction_list_with_proof.clone()
                );
            } else {
                assert_eq!(outputs_with_proof.unwrap(), output_list_with_proof.clone());
            }
        },
        _ => panic!(
            "Expected transactions or outputs with proof but got: {:?}",
            response
        ),
    };
}

/// Initializes the Aptos logger for tests
pub fn initialize_logger() {
    aptos_logger::Logger::builder()
        .is_async(false)
        .level(Level::Debug)
        .build();
}

/// Returns a random network ID
fn get_random_network_id() -> NetworkId {
    let mut rng = OsRng;
    let random_number: u8 = rng.gen();
    match random_number % 3 {
        0 => NetworkId::Validator,
        1 => NetworkId::Vfn,
        2 => NetworkId::Public,
        num => panic!("This shouldn't be possible! Got num: {:?}", num),
    }
}

/// Creates a mock database reader
pub fn create_mock_db_reader() -> MockDatabaseReader {
    MockDatabaseReader::new()
}

// This automatically creates a MockDatabaseReader.
// TODO(joshlind): if we frequently use these mocks, we should define a single
// mock test crate to be shared across the codebase.
mock! {
    pub DatabaseReader {}
    impl DbReader for DatabaseReader {
        fn get_epoch_ending_ledger_infos(
            &self,
            start_epoch: u64,
            end_epoch: u64,
        ) -> Result<EpochChangeProof>;

        fn get_transactions(
            &self,
            start_version: Version,
            batch_size: u64,
            ledger_version: Version,
            fetch_events: bool,
        ) -> Result<TransactionListWithProof>;

        fn get_transaction_by_hash(
            &self,
            hash: HashValue,
            ledger_version: Version,
            fetch_events: bool,
        ) -> Result<Option<TransactionWithProof>>;

        fn get_transaction_by_version(
            &self,
            version: Version,
            ledger_version: Version,
            fetch_events: bool,
        ) -> Result<TransactionWithProof>;

        fn get_first_txn_version(&self) -> Result<Option<Version>>;

        fn get_first_write_set_version(&self) -> Result<Option<Version>>;

        fn get_transaction_outputs(
            &self,
            start_version: Version,
            limit: u64,
            ledger_version: Version,
        ) -> Result<TransactionOutputListWithProof>;

        fn get_events(
            &self,
            event_key: &EventKey,
            start: u64,
            order: Order,
            limit: u64,
            ledger_version: Version,
        ) -> Result<Vec<EventWithVersion>>;

        fn get_block_timestamp(&self, version: u64) -> Result<u64>;

        fn get_last_version_before_timestamp(
            &self,
            _timestamp: u64,
            _ledger_version: Version,
        ) -> Result<Version>;

        fn get_latest_ledger_info_option(&self) -> Result<Option<LedgerInfoWithSignatures>>;

        fn get_latest_ledger_info(&self) -> Result<LedgerInfoWithSignatures>;

        fn get_latest_version(&self) -> Result<Version>;

        fn get_latest_commit_metadata(&self) -> Result<(Version, u64)>;

        fn get_account_transaction(
            &self,
            address: AccountAddress,
            seq_num: u64,
            include_events: bool,
            ledger_version: Version,
        ) -> Result<Option<TransactionWithProof>>;

        fn get_account_transactions(
            &self,
            address: AccountAddress,
            seq_num: u64,
            limit: u64,
            include_events: bool,
            ledger_version: Version,
        ) -> Result<AccountTransactionsWithProof>;

        fn get_state_proof_with_ledger_info(
            &self,
            known_version: u64,
            ledger_info: LedgerInfoWithSignatures,
        ) -> Result<StateProof>;

        fn get_state_proof(&self, known_version: u64) -> Result<StateProof>;

        fn get_state_value_with_proof_by_version(
            &self,
            state_key: &StateKey,
            version: Version,
        ) -> Result<(Option<StateValue>, SparseMerkleProof)>;

        fn get_latest_executed_trees(&self) -> Result<ExecutedTrees>;

        fn get_epoch_ending_ledger_info(&self, known_version: u64) -> Result<LedgerInfoWithSignatures>;

        fn get_latest_transaction_info_option(&self) -> Result<Option<(Version, TransactionInfo)>>;

        fn get_accumulator_root_hash(&self, _version: Version) -> Result<HashValue>;

        fn get_accumulator_consistency_proof(
            &self,
            _client_known_version: Option<Version>,
            _ledger_version: Version,
        ) -> Result<AccumulatorConsistencyProof>;

        fn get_accumulator_summary(
            &self,
            ledger_version: Version,
        ) -> Result<TransactionAccumulatorSummary>;

        fn get_state_leaf_count(&self, version: Version) -> Result<usize>;

        fn get_state_value_chunk_with_proof(
            &self,
            version: Version,
            start_idx: usize,
            chunk_size: usize,
        ) -> Result<StateValueChunkWithProof>;

        fn get_epoch_snapshot_prune_window(&self) -> Result<usize>;

        fn is_state_pruner_enabled(&self) -> Result<bool>;
    }
}
