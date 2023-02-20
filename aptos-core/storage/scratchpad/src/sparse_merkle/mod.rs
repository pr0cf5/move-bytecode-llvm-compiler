// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

//! This module implements an in-memory Sparse Merkle Tree that is similar to what we use in
//! storage to represent world state. This tree will store only a small portion of the state -- the
//! part of accounts that have been modified by uncommitted transactions. For example, if we
//! execute a transaction T_i on top of committed state and it modified account A, we will end up
//! having the following tree:
//! ```text
//!              S_i
//!             /   \
//!            o     y
//!           / \
//!          x   A
//! ```
//! where A has the new state of the account, and y and x are the siblings on the path from root to
//! A in the tree.
//!
//! This Sparse Merkle Tree is immutable once constructed. If the next transaction T_{i+1} modified
//! another account B that lives in the subtree at y, a new tree will be constructed and the
//! structure will look like the following:
//! ```text
//!                 S_i        S_{i+1}
//!                /   \      /       \
//!               /     y   /          \
//!              / _______/             \
//!             //                       \
//!            o                          y'
//!           / \                        / \
//!          x   A                      z   B
//! ```
//!
//! Using this structure, we are able to query the global state, taking into account the output of
//! uncommitted transactions. For example, if we want to execute another transaction T_{i+1}', we
//! can use the tree S_i. If we look for account A, we can find its new value in the tree.
//! Otherwise, we know the account does not exist in the tree, and we can fall back to storage. As
//! another example, if we want to execute transaction T_{i+2}, we can use the tree S_{i+1} that
//! has updated values for both account A and B.
//!
//! Each version of the tree holds a strong reference (an Arc<Node>) to its root as well as one to
//! its base tree (S_i is the base tree of S_{i+1} in the above example). The root node in turn,
//! recursively holds all descendant nodes created in the same version, and weak references
//! (a Weak<Node>) to all descendant nodes that was created from previous versions.
//! With this construction:
//!     1. Even if a reference to a specific tree is dropped, the nodes belonging to it won't be
//! dropped as long as trees depending on it still hold strong references to it via the chain of
//! "base trees".
//!     2. Even if a tree is not dropped, when nodes it created are persisted to DB, all of them
//! and those created by its previous versions can be dropped, which we express by calling "prune()"
//! on it which replaces the strong references to its root and its base tree with weak references.
//!     3. We can hold strong references to recently accessed nodes that have already been persisted
//! in an LRU flavor cache for less DB reads.
//!
//! This Sparse Merkle Tree serves a dual purpose. First, to support a leader based consensus
//! algorithm, we need to build a tree of transactions like the following:
//! ```text
//! Committed -> T5 -> T6  -> T7
//!              └---> T6' -> T7'
//!                    └----> T7"
//! ```
//! Once T5 is executed, we will have a tree that stores the modified portion of the state. Later
//! when we execute T6 on top of T5, the output of T5 can be visible to T6.
//!
//! Second, given this tree representation it is straightforward to compute the root hash of S_i
//! once T_i is executed. This allows us to verify the proofs we need when executing T_{i+1}.

// See https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=e9c4c53eb80b30d09112fcfb07d481e7
#![allow(clippy::let_and_return)]
// See https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=795cd4f459f1d4a0005a99650726834b
#![allow(clippy::while_let_loop)]

mod metrics;
mod node;
mod updater;
mod utils;

#[cfg(test)]
mod sparse_merkle_test;
#[cfg(any(test, feature = "bench", feature = "fuzzing"))]
pub mod test_utils;

use crate::sparse_merkle::{
    metrics::{LATEST_GENERATION, OLDEST_GENERATION, TIMER},
    node::{NodeInner, SubTree},
    updater::SubTreeUpdater,
};
use aptos_crypto::{
    hash::{CryptoHash, SPARSE_MERKLE_PLACEHOLDER_HASH},
    HashValue,
};
use aptos_infallible::Mutex;
use aptos_types::{
    nibble::nibble_path::NibblePath, proof::SparseMerkleProofExt,
    state_store::state_storage_usage::StateStorageUsage,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    sync::{Arc, MutexGuard, Weak},
};
use thiserror::Error;

type NodePosition = bitvec::vec::BitVec<bitvec::order::Msb0, u8>;
const BITS_IN_NIBBLE: usize = 4;
const BITS_IN_BYTE: usize = 8;

/// To help finding the oldest ancestor of any SMT, a branch tracker is created each time
/// the chain of SMTs forked (two or more SMTs updating the same parent).
#[derive(Debug)]
struct BranchTracker<V> {
    /// Current branch head, n.b. when the head just started dropping, this weak link becomes
    /// invalid, we fall back to the `next`
    head: Weak<Inner<V>>,
    /// Dealing with the edge case where the branch head just started dropping, but the branch
    /// tracker hasn't been locked and updated yet.
    next: Weak<Inner<V>>,
    /// Parent branch, if any.
    parent: Option<Arc<Mutex<BranchTracker<V>>>>,
}

impl<V> BranchTracker<V> {
    fn new_head_unknown(
        parent: Option<Arc<Mutex<Self>>>,
        _locked_family: &MutexGuard<()>,
    ) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            head: Weak::new(),
            next: Weak::new(),
            parent,
        }))
    }

    fn set_head(
        &mut self,
        head: &Arc<Inner<V>>,
        next: Option<&Arc<Inner<V>>>,
        _locked_family: &MutexGuard<()>,
    ) {
        // Detach from parent
        // n.b. the parent branch might not be dropped after this, because whenever a fork
        //      happens, the first branch shares the parent branch tracker.
        self.parent = None;

        self.head = Arc::downgrade(head);
        self.next = next.map_or_else(Weak::new, Arc::downgrade)
    }

    fn parent(&self, _locked_family: &MutexGuard<()>) -> Option<Arc<Mutex<Self>>> {
        self.parent.clone()
    }

    fn head(&self, _locked_family: &MutexGuard<()>) -> Option<Arc<Inner<V>>> {
        // if `head.upgrade()` failed, it's that the head is being dropped.
        //
        // Notice the starting of the drop a SMT is not protected by the family lock -- but
        // change of the links between the branch trackers and SMTs are always protected by the
        // family lock.
        // see `impl<V> Drop for Inner<V>`
        self.head.upgrade().or_else(|| self.next.upgrade())
    }
}

/// Keeps track of references of children and the branch tracker of the current branch.
#[derive(Debug)]
struct InnerLinks<V> {
    children: Vec<Arc<Inner<V>>>,
    branch_tracker: Arc<Mutex<BranchTracker<V>>>,
}

impl<V> InnerLinks<V> {
    fn new(branch_tracker: Arc<Mutex<BranchTracker<V>>>) -> Mutex<Self> {
        Mutex::new(Self {
            children: Vec::new(),
            branch_tracker,
        })
    }
}

/// The inner content of a sparse merkle tree, we have this so that even if a tree is dropped, the
/// INNER of it can still live if referenced by a previous version.
#[derive(Debug)]
struct Inner<V> {
    root: SubTree<V>,
    usage: StateStorageUsage,
    links: Mutex<InnerLinks<V>>,
    generation: u64,
    family_lock: Arc<Mutex<()>>,
}

impl<V> Drop for Inner<V> {
    fn drop(&mut self) {
        // To prevent recursively locking the family, buffer all descendants outside.
        let mut processed_descendants = Vec::new();

        {
            let locked_family = self.family_lock.lock();

            let mut stack = self.drain_children_for_drop(&locked_family);

            while let Some(descendant) = stack.pop() {
                if Arc::strong_count(&descendant) == 1 {
                    // The only ref is the one we are now holding, and there's no weak ref that can
                    // upgrade because the only `Weak<Inner<V>>`s are held by `BranchTracker`s and
                    // they try to upgrade only when under the protection of the family lock. So the
                    // descendant will be dropped after we free the `Arc`, which results in a chain
                    // of such structures being dropped recursively and that might trigger a stack
                    // overflow. To prevent that we follow the chain further to disconnect things
                    // beforehand.
                    stack.extend(descendant.drain_children_for_drop(&locked_family));
                    // Note: After the above call, there is not even weak refs to `descendant`
                    // because all relevant `BranchTrackers` now point their heads to one of the
                    // children.
                }
                // All descendants process must be pushed, because they can become droppable after
                // the ref count check above, since the family lock doesn't protect de-refs to the
                // SMTs. -- all drops must NOT be recursive because we will be trying to lock the
                // family again.
                processed_descendants.push(descendant);
            }
        };
        // Now that the lock is released, those in `processed_descendants` will be dropped in turn
        // if applicable.
    }
}

impl<V> Inner<V> {
    fn new(root: SubTree<V>, usage: StateStorageUsage) -> Arc<Self> {
        let family_lock = Arc::new(Mutex::new(()));
        let branch_tracker = BranchTracker::new_head_unknown(None, &family_lock.lock());
        let me = Arc::new(Self {
            root,
            usage,
            links: InnerLinks::new(branch_tracker.clone()),
            generation: 0,
            family_lock,
        });
        branch_tracker.lock().head = Arc::downgrade(&me);

        me
    }

    fn become_oldest(self: Arc<Self>, locked_family: &MutexGuard<()>) -> Arc<Self> {
        {
            let links_locked = self.links.lock();
            let mut branch_tracker_locked = links_locked.branch_tracker.lock();
            branch_tracker_locked.set_head(
                &self,                         /* head */
                links_locked.children.first(), /* next */
                locked_family,
            );
        }
        self
    }

    fn spawn_impl(
        &self,
        child_root: SubTree<V>,
        child_usage: StateStorageUsage,
        branch_tracker: Arc<Mutex<BranchTracker<V>>>,
        family_lock: Arc<Mutex<()>>,
    ) -> Arc<Self> {
        LATEST_GENERATION.set(self.generation as i64 + 1);
        Arc::new(Self {
            root: child_root,
            usage: child_usage,
            links: InnerLinks::new(branch_tracker),
            generation: self.generation + 1,
            family_lock,
        })
    }

    fn spawn(
        self: &Arc<Self>,
        child_root: SubTree<V>,
        child_usage: StateStorageUsage,
    ) -> Arc<Self> {
        let locked_family = self.family_lock.lock();
        let mut links_locked = self.links.lock();

        let child = if links_locked.children.is_empty() {
            let child = self.spawn_impl(
                child_root,
                child_usage,
                links_locked.branch_tracker.clone(),
                self.family_lock.clone(),
            );
            let mut branch_tracker_locked = links_locked.branch_tracker.lock();
            if branch_tracker_locked.next.upgrade().is_none() {
                branch_tracker_locked.next = Arc::downgrade(&child);
            }
            child
        } else {
            // forking a new branch
            let branch_tracker = BranchTracker::new_head_unknown(
                Some(links_locked.branch_tracker.clone()),
                &locked_family,
            );
            let child = self.spawn_impl(
                child_root,
                child_usage,
                branch_tracker.clone(),
                self.family_lock.clone(),
            );
            branch_tracker.lock().head = Arc::downgrade(&child);
            child
        };
        links_locked.children.push(child.clone());

        child
    }

    fn get_oldest_ancestor(self: &Arc<Self>) -> Arc<Self> {
        // Under the protection of family_lock, the branching structure won't change,
        // so we can follow the links and find the head of the oldest branch tracker.
        let locked_family = self.family_lock.lock();
        let (mut ret, mut parent) = {
            let branch_tracker = self.links.lock().branch_tracker.clone();
            let branch_tracker_locked = branch_tracker.lock();
            (
                branch_tracker_locked
                    .head(&locked_family)
                    .expect("Leaf must have a head."),
                branch_tracker_locked.parent(&locked_family),
            )
        };

        while let Some(parent_bt) = parent {
            let parent_bt_locked = parent_bt.lock();
            if let Some(parent_bt_head) = parent_bt_locked.head(&locked_family) {
                ret = parent_bt_head;
                parent = parent_bt_locked.parent(&locked_family);
                continue;
            }
            break;
        }

        OLDEST_GENERATION.set(ret.generation as i64);
        ret
    }

    fn drain_children_for_drop(&self, locked_family: &MutexGuard<()>) -> Vec<Arc<Self>> {
        self.links
            .lock()
            .children
            .drain(..)
            .map(|child| child.become_oldest(locked_family))
            .collect()
    }
}

/// The Sparse Merkle Tree implementation.
#[derive(Clone, Debug)]
pub struct SparseMerkleTree<V> {
    inner: Arc<Inner<V>>,
}

impl<V> SparseMerkleTree<V>
where
    V: Clone + CryptoHash + Send + Sync,
{
    /// Constructs a Sparse Merkle Tree with a root hash. This is often used when we restart and
    /// the scratch pad and the storage have identical state, so we use a single root hash to
    /// represent the entire state.
    pub fn new(root_hash: HashValue, usage: StateStorageUsage) -> Self {
        let root = if root_hash != *SPARSE_MERKLE_PLACEHOLDER_HASH {
            SubTree::new_unknown(root_hash)
        } else {
            assert!(usage.is_untracked() || usage == StateStorageUsage::zero());
            SubTree::new_empty()
        };

        Self {
            inner: Inner::new(root, usage),
        }
    }

    #[cfg(test)]
    fn new_test(root_hash: HashValue) -> Self {
        Self::new(root_hash, StateStorageUsage::new_untracked())
    }

    pub fn new_empty() -> Self {
        Self {
            inner: Inner::new(SubTree::new_empty(), StateStorageUsage::zero()),
        }
    }

    pub fn has_same_root_hash(&self, other: &Self) -> bool {
        self.root_hash() == other.root_hash()
    }

    fn get_oldest_ancestor(&self) -> Self {
        Self {
            inner: self.inner.get_oldest_ancestor(),
        }
    }

    pub fn freeze(self) -> FrozenSparseMerkleTree<V> {
        let base_smt = self.get_oldest_ancestor();
        let base_generation = base_smt.inner.generation;

        FrozenSparseMerkleTree {
            base_smt,
            base_generation,
            smt: self,
        }
    }

    #[cfg(test)]
    fn new_with_root(root: SubTree<V>) -> Self {
        Self {
            inner: Inner::new(root, StateStorageUsage::new_untracked()),
        }
    }

    fn root_weak(&self) -> SubTree<V> {
        self.inner.root.weak()
    }

    /// Returns the root hash of this tree.
    pub fn root_hash(&self) -> HashValue {
        self.inner.root.hash()
    }

    fn generation(&self) -> u64 {
        self.inner.generation
    }

    fn is_the_same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn usage(&self) -> StateStorageUsage {
        self.inner.usage
    }
}

/// In tests and benchmark, reference to ancestors are manually managed
#[cfg(any(feature = "fuzzing", feature = "bench", test))]
impl<V> SparseMerkleTree<V>
where
    V: Clone + CryptoHash + Send + Sync,
{
    pub fn batch_update(
        &self,
        updates: Vec<(HashValue, Option<&V>)>,
        proof_reader: &impl ProofRead,
    ) -> Result<Self, UpdateError> {
        self.clone()
            .freeze()
            .batch_update(updates, StateStorageUsage::zero(), proof_reader)
            .map(FrozenSparseMerkleTree::unfreeze)
    }

    pub fn get(&self, key: HashValue) -> StateStoreStatus<V> {
        self.clone().freeze().get(key)
    }
}

impl<V> Default for SparseMerkleTree<V>
where
    V: Clone + CryptoHash + Send + Sync,
{
    fn default() -> Self {
        SparseMerkleTree::new_empty()
    }
}

/// `AccountStatus` describes the result of querying an account from this SparseMerkleTree.
#[derive(Debug, Eq, PartialEq)]
pub enum StateStoreStatus<V> {
    /// The entry exists in the tree, therefore we can give its value.
    ExistsInScratchPad(V),

    /// The entry does not exist in the tree, but exists in DB. This happens when the search
    /// reaches a leaf node that has the requested account, but the node has only the value hash
    /// because it was loaded into memory as part of a non-inclusion proof. When we go to DB we
    /// don't need to traverse the tree to find the same leaf, instead we can use the value hash to
    /// look up the entry content directly.
    ExistsInDB,

    /// The entry does not exist in either the tree or DB. This happens when the search reaches
    /// an empty node, or a leaf node that has a different account.
    DoesNotExist,

    /// We do not know if this entry exists or not and need to go to DB to find out. This happens
    /// when the search reaches a subtree node.
    Unknown,
}

/// In the entire lifetime of this, in-mem nodes won't be dropped because a reference to the oldest
/// SMT is held inside.
#[derive(Clone, Debug)]
pub struct FrozenSparseMerkleTree<V> {
    base_smt: SparseMerkleTree<V>,
    base_generation: u64,
    smt: SparseMerkleTree<V>,
}

impl<V> FrozenSparseMerkleTree<V>
where
    V: Clone + CryptoHash + Send + Sync,
{
    fn spawn(&self, child_root: SubTree<V>, child_usage: StateStorageUsage) -> Self {
        Self {
            base_smt: self.base_smt.clone(),
            base_generation: self.base_generation,
            smt: SparseMerkleTree {
                inner: self.smt.inner.spawn(child_root, child_usage),
            },
        }
    }

    pub fn unfreeze(self) -> SparseMerkleTree<V> {
        self.smt
    }

    pub fn root_hash(&self) -> HashValue {
        self.smt.root_hash()
    }

    /// Compares an old and a new SMTs and return the newly created node hashes in between.
    pub fn new_node_hashes_since(&self, since_smt: &Self) -> HashMap<NibblePath, HashValue> {
        let _timer = TIMER
            .with_label_values(&["new_node_hashes_since"])
            .start_timer();

        assert!(self.base_smt.is_the_same(&since_smt.base_smt));
        let mut node_hashes = HashMap::new();
        Self::new_node_hashes_since_impl(
            self.smt.root_weak(),
            since_smt.smt.generation() + 1,
            &mut NodePosition::with_capacity(HashValue::LENGTH_IN_BITS),
            &mut node_hashes,
        );
        node_hashes
    }

    /// Recursively generate the partial node update batch of jellyfish merkle
    fn new_node_hashes_since_impl(
        subtree: SubTree<V>,
        since_generation: u64,
        pos: &mut NodePosition,
        node_hashes: &mut HashMap<NibblePath, HashValue>,
    ) {
        if let Some(node) = subtree.get_node_if_in_mem(since_generation) {
            let is_nibble = if let Some(path) = Self::maybe_to_nibble_path(pos) {
                node_hashes.insert(path, subtree.hash());
                true
            } else {
                false
            };
            match node.inner().borrow() {
                NodeInner::Internal(internal_node) => {
                    let depth = pos.len();
                    pos.push(false);
                    Self::new_node_hashes_since_impl(
                        internal_node.left.weak(),
                        since_generation,
                        pos,
                        node_hashes,
                    );
                    *pos.get_mut(depth).unwrap() = true;
                    Self::new_node_hashes_since_impl(
                        internal_node.right.weak(),
                        since_generation,
                        pos,
                        node_hashes,
                    );
                    pos.pop();
                },
                NodeInner::Leaf(leaf_node) => {
                    let mut path = NibblePath::new_even(leaf_node.key.to_vec());
                    if !is_nibble {
                        path.truncate(pos.len() / BITS_IN_NIBBLE + 1);
                    }
                    node_hashes.insert(path, subtree.hash());
                },
            }
        }
    }

    fn maybe_to_nibble_path(pos: &NodePosition) -> Option<NibblePath> {
        assert!(pos.len() <= HashValue::LENGTH_IN_BITS);

        if pos.len() % BITS_IN_NIBBLE == 0 {
            let mut bytes = pos.clone().into_vec();
            if pos.len() % BITS_IN_BYTE == 0 {
                Some(NibblePath::new_even(bytes))
            } else {
                // Unused bits in `BitVec` is uninitialized, setting to 0 to make sure.
                if let Some(b) = bytes.last_mut() {
                    *b &= 0xF0
                }

                Some(NibblePath::new_odd(bytes))
            }
        } else {
            None
        }
    }

    /// Constructs a new Sparse Merkle Tree by applying `updates`, which are considered to happen
    /// all at once.
    /// Since the tree is immutable, existing tree remains the same and may share parts with the
    /// new, returned tree.
    pub fn batch_update(
        &self,
        updates: Vec<(HashValue, Option<&V>)>,
        usage: StateStorageUsage,
        proof_reader: &impl ProofRead,
    ) -> Result<Self, UpdateError> {
        // Flatten, dedup and sort the updates with a btree map since the updates between different
        // versions may overlap on the same address in which case the latter always overwrites.
        let kvs = updates
            .into_iter()
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>();

        if kvs.is_empty() {
            assert_eq!(self.smt.inner.usage, usage);
            Ok(self.clone())
        } else {
            let current_root = self.smt.root_weak();
            let root = SubTreeUpdater::update(
                current_root,
                &kvs[..],
                proof_reader,
                self.smt.inner.generation + 1,
            )?;
            Ok(self.spawn(root, usage))
        }
    }

    /// Queries a `key` in this `SparseMerkleTree`.
    pub fn get(&self, key: HashValue) -> StateStoreStatus<V> {
        let mut subtree = self.smt.root_weak();
        let mut bits = key.iter_bits();

        loop {
            match subtree {
                SubTree::Empty => return StateStoreStatus::DoesNotExist,
                SubTree::NonEmpty { .. } => {
                    match subtree.get_node_if_in_mem(self.base_generation) {
                        None => return StateStoreStatus::Unknown,
                        Some(node) => match node.inner() {
                            NodeInner::Internal(internal_node) => {
                                subtree = if bits.next().expect("Tree is too deep.") {
                                    internal_node.right.weak()
                                } else {
                                    internal_node.left.weak()
                                };
                                continue;
                            }, // end NodeInner::Internal
                            NodeInner::Leaf(leaf_node) => {
                                return if leaf_node.key == key {
                                    match &leaf_node.value.data.get_if_in_mem() {
                                        Some(value) => StateStoreStatus::ExistsInScratchPad(
                                            value.as_ref().clone(),
                                        ),
                                        None => StateStoreStatus::ExistsInDB,
                                    }
                                } else {
                                    StateStoreStatus::DoesNotExist
                                };
                            }, // end NodeInner::Leaf
                        }, // end Some(node) got from mem
                    }
                }, // end SubTree::NonEmpty
            }
        } // end loop
    }

    pub fn usage(&self) -> StateStorageUsage {
        self.smt.usage()
    }
}

/// A type that implements `ProofRead` can provide proof for keys in persistent storage.
pub trait ProofRead: Sync {
    /// Gets verified proof for this key in persistent storage.
    fn get_proof(&self, key: HashValue) -> Option<&SparseMerkleProofExt>;
}

/// All errors `update` can possibly return.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum UpdateError {
    /// The update intends to insert a key that does not exist in the tree, so the operation needs
    /// proof to get more information about the tree, but no proof is provided.
    #[error("Missing Proof")]
    MissingProof,
    /// At `depth` a persisted subtree was encountered and a proof was requested to assist finding
    /// details about the subtree, but the result proof indicates the subtree is empty.
    #[error(
        "Short proof: key: {}, num_siblings: {}, depth: {}",
        key,
        num_siblings,
        depth
    )]
    ShortProof {
        key: HashValue,
        num_siblings: usize,
        depth: usize,
    },
}
