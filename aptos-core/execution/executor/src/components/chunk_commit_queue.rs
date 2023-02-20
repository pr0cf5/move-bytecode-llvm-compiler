// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use anyhow::{anyhow, Result};
use aptos_executor_types::ExecutedChunk;
use aptos_storage_interface::{DbReader, ExecutedTrees};
use std::{collections::VecDeque, sync::Arc};

pub struct ChunkCommitQueue {
    persisted_view: ExecutedTrees,
    chunks_to_commit: VecDeque<Arc<ExecutedChunk>>,
}

impl ChunkCommitQueue {
    pub fn new_from_db(db: &Arc<dyn DbReader>) -> Result<Self> {
        let persisted_view = db.get_latest_executed_trees()?;
        Ok(Self::new(persisted_view))
    }

    pub fn new(persisted_view: ExecutedTrees) -> Self {
        Self {
            persisted_view,
            chunks_to_commit: VecDeque::new(),
        }
    }

    pub fn persisted_and_latest_view(&self) -> (ExecutedTrees, ExecutedTrees) {
        (self.persisted_view.clone(), self.latest_view())
    }

    pub fn latest_view(&self) -> ExecutedTrees {
        self.chunks_to_commit
            .back()
            .map(|chunk| chunk.result_view.clone())
            .unwrap_or_else(|| self.persisted_view.clone())
    }

    pub fn next_chunk_to_commit(&self) -> Result<(ExecutedTrees, Arc<ExecutedChunk>)> {
        Ok((
            self.persisted_view.clone(),
            self.chunks_to_commit
                .front()
                .ok_or_else(|| anyhow!("Commit queue is empty."))
                .map(Arc::clone)?,
        ))
    }

    pub fn enqueue(&mut self, chunk: ExecutedChunk) {
        self.chunks_to_commit.push_back(Arc::new(chunk))
    }

    pub fn dequeue(&mut self) -> Result<()> {
        let committed_chunk = self
            .chunks_to_commit
            .pop_front()
            .ok_or_else(|| anyhow!("Commit queue is empty."))?;
        self.persisted_view = committed_chunk.result_view.clone();
        Ok(())
    }
}
