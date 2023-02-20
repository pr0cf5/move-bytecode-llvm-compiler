// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use crate::{
    block_executor::BlockExecutor,
    chunk_executor::ChunkExecutor,
    db_bootstrapper::{generate_waypoint, maybe_bootstrap},
    mock_vm::{encode_mint_transaction, MockVM},
    tests,
};
use aptos_crypto::HashValue;
use aptos_db::AptosDB;
use aptos_executor_types::{BlockExecutorTrait, ChunkExecutorTrait};
use aptos_storage_interface::DbReaderWriter;
use aptos_types::{
    ledger_info::LedgerInfoWithSignatures,
    test_helpers::transaction_test_helpers::block,
    transaction::{TransactionListWithProof, TransactionOutputListWithProof},
};
use rand::Rng;

pub struct TestExecutor {
    _path: aptos_temppath::TempPath,
    pub db: DbReaderWriter,
    pub executor: ChunkExecutor<MockVM>,
}

impl TestExecutor {
    pub fn new() -> TestExecutor {
        let path = aptos_temppath::TempPath::new();
        path.create_as_dir().unwrap();
        let db = DbReaderWriter::new(AptosDB::new_for_test(path.path()));
        let genesis = aptos_vm_genesis::test_genesis_transaction();
        let waypoint = generate_waypoint::<MockVM>(&db, &genesis).unwrap();
        maybe_bootstrap::<MockVM>(&db, &genesis, waypoint).unwrap();
        let executor = ChunkExecutor::new(db.clone());

        TestExecutor {
            _path: path,
            db,
            executor,
        }
    }
}

fn execute_and_commit_chunk(
    chunks: Vec<TransactionListWithProof>,
    ledger_info: LedgerInfoWithSignatures,
    db: &DbReaderWriter,
    executor: &ChunkExecutor<MockVM>,
) {
    // Execute the first chunk. After that we should still get the genesis ledger info from DB.
    executor
        .execute_chunk(chunks[0].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the second chunk. After that we should still get the genesis ledger info from DB.
    executor
        .execute_chunk(chunks[1].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute an empty chunk. After that we should still get the genesis ledger info from DB.
    executor
        .execute_chunk(TransactionListWithProof::new_empty(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the second chunk again. After that we should still get the same thing.
    executor
        .execute_chunk(chunks[1].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the third chunk. After that we should get the new ledger info.
    executor
        .execute_chunk(chunks[2].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li, ledger_info);
}

#[test]
#[cfg_attr(feature = "consensus-only-perf-test", ignore)]
fn test_executor_execute_or_apply_and_commit_chunk() {
    let first_batch_size = 30;
    let second_batch_size = 40;
    let third_batch_size = 20;
    let overlapping_size = 5;

    let first_batch_start = 1;
    let second_batch_start = first_batch_start + first_batch_size;
    let third_batch_start = second_batch_start + second_batch_size - overlapping_size;

    let (chunks, ledger_info) = {
        tests::create_transaction_chunks(vec![
            first_batch_start..first_batch_start + first_batch_size,
            second_batch_start..second_batch_start + second_batch_size,
            third_batch_start..third_batch_start + third_batch_size,
        ])
    };
    // First test with transactions only and reset chunks to be `Vec<TransactionOutputListWithProof>`.
    let chunks = {
        let TestExecutor {
            _path,
            db,
            executor,
        } = TestExecutor::new();
        execute_and_commit_chunk(chunks, ledger_info.clone(), &db, &executor);

        let ledger_version = db.reader.get_latest_version().unwrap();
        let output1 = db
            .reader
            .get_transaction_outputs(first_batch_start, first_batch_size, ledger_version)
            .unwrap();
        let output2 = db
            .reader
            .get_transaction_outputs(second_batch_start, second_batch_size, ledger_version)
            .unwrap();
        let output3 = db
            .reader
            .get_transaction_outputs(third_batch_start, third_batch_size, ledger_version)
            .unwrap();
        vec![output1, output2, output3]
    };

    // Test with transaction outputs.
    let TestExecutor {
        _path,
        db,
        executor,
    } = TestExecutor::new();
    // Execute the first chunk. After that we should still get the genesis ledger info from DB.
    executor.reset().unwrap();
    executor
        .apply_chunk(chunks[0].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the second chunk. After that we should still get the genesis ledger info from DB.
    executor
        .apply_chunk(chunks[1].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute an empty chunk. After that we should still get the genesis ledger info from DB.
    executor
        .apply_chunk(
            TransactionOutputListWithProof::new_empty(),
            &ledger_info,
            None,
        )
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the second chunk again. After that we should still get the same thing.
    executor
        .apply_chunk(chunks[1].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li.ledger_info().version(), 0);
    assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());

    // Execute the third chunk. After that we should get the new ledger info.
    executor
        .apply_chunk(chunks[2].clone(), &ledger_info, None)
        .unwrap();
    executor.commit_chunk().unwrap();
    let li = db.reader.get_latest_ledger_info().unwrap();
    assert_eq!(li, ledger_info);
}

#[test]
fn test_executor_execute_and_commit_chunk_restart() {
    let first_batch_size = 30;
    let second_batch_size = 40;

    let (chunks, ledger_info) = {
        let first_batch_start = 1;
        let second_batch_start = first_batch_start + first_batch_size;
        tests::create_transaction_chunks(vec![
            first_batch_start..first_batch_start + first_batch_size,
            second_batch_start..second_batch_start + second_batch_size,
        ])
    };

    let TestExecutor {
        _path,
        db,
        executor,
    } = TestExecutor::new();

    // First we simulate syncing the first chunk of transactions.
    {
        executor
            .execute_chunk(chunks[0].clone(), &ledger_info, None)
            .unwrap();
        executor.commit_chunk().unwrap();
        let li = db.reader.get_latest_ledger_info().unwrap();
        assert_eq!(li.ledger_info().version(), 0);
        assert_eq!(li.ledger_info().consensus_block_id(), HashValue::zero());
    }

    // Then we restart executor and resume to the next chunk.
    {
        let executor = ChunkExecutor::<MockVM>::new(db.clone());

        executor
            .execute_chunk(chunks[1].clone(), &ledger_info, None)
            .unwrap();
        executor.commit_chunk().unwrap();
        let li = db.reader.get_latest_ledger_info().unwrap();
        assert_eq!(li, ledger_info);
    }
}

#[test]
#[cfg_attr(feature = "consensus-only-perf-test", ignore)]
fn test_executor_execute_and_commit_chunk_local_result_mismatch() {
    let first_batch_size = 10;
    let second_batch_size = 10;

    let (chunks, ledger_info) = {
        let first_batch_start = 1;
        let second_batch_start = first_batch_start + first_batch_size;
        tests::create_transaction_chunks(vec![
            first_batch_start..first_batch_start + first_batch_size,
            second_batch_start..second_batch_start + second_batch_size,
        ])
    };

    let TestExecutor {
        _path,
        db,
        executor: chunk_manager,
    } = TestExecutor::new();

    // commit 5 txns first.
    {
        let executor = BlockExecutor::<MockVM>::new(db);
        let parent_block_id = executor.committed_block_id();
        let block_id = tests::gen_block_id(1);

        let mut rng = rand::thread_rng();
        let txns = (0..5)
            .map(|_| encode_mint_transaction(tests::gen_address(rng.gen::<u64>()), 100))
            .collect::<Vec<_>>();
        let output = executor
            .execute_block((block_id, block(txns)), parent_block_id)
            .unwrap();
        let ledger_info = tests::gen_ledger_info(6, output.root_hash(), block_id, 1);
        executor.commit_blocks(vec![block_id], ledger_info).unwrap();
    }

    // Fork starts. Should fail.
    chunk_manager.finish();
    chunk_manager.reset().unwrap();

    assert!(chunk_manager
        .execute_chunk(chunks[1].clone(), &ledger_info, None)
        .is_err());
}

#[cfg(feature = "consensus-only-perf-test")]
#[test]
fn test_executor_execute_and_commit_chunk_without_verify() {
    let first_batch_size = 10;
    let second_batch_size = 10;

    let (chunks, ledger_info) = {
        let first_batch_start = 1;
        let second_batch_start = first_batch_start + first_batch_size;
        tests::create_transaction_chunks(vec![
            first_batch_start..first_batch_start + first_batch_size,
            second_batch_start..second_batch_start + second_batch_size,
        ])
    };

    let TestExecutor {
        _path,
        db,
        executor: chunk_manager,
    } = TestExecutor::new();

    // commit 5 txns first.
    {
        let executor = BlockExecutor::<MockVM>::new(db);
        let parent_block_id = executor.committed_block_id();
        let block_id = tests::gen_block_id(1);

        let mut rng = rand::thread_rng();
        let txns = (0..5)
            .map(|_| encode_mint_transaction(tests::gen_address(rng.gen::<u64>()), 100))
            .collect::<Vec<_>>();
        let output = executor
            .execute_block((block_id, block(txns)), parent_block_id)
            .unwrap();
        let ledger_info = tests::gen_ledger_info(6, output.root_hash(), block_id, 1);
        executor.commit_blocks(vec![block_id], ledger_info).unwrap();
    }

    // Fork starts. Should fail.
    chunk_manager.finish();
    chunk_manager.reset().unwrap();

    assert!(chunk_manager
        .execute_chunk(chunks[1].clone(), &ledger_info, None)
        .is_ok());
}
