---
title: "Running Validator Node"
slug: "running-validator-node"
---

# Running Validator Node

:::tip Deploying a validator node? Read this first
If you are deploying a validator node, then make sure to read the [Node Requirements](nodes/validator-node/operator/node-requirements.md) first.
:::

## Install Validator node

### Deploy

The following guides provide step-by-step instructions for running public fullnode, validator node, and validator fullnode for the Aptos blockchain. 

- ### [On AWS](nodes/validator-node/operator/running-validator-node/using-aws.md)
- ### [On Azure](nodes/validator-node/operator/running-validator-node/using-azure.md)
- ### [On GCP](nodes/validator-node/operator/running-validator-node/using-gcp.md)
- ### [Using Docker](nodes/validator-node/operator/running-validator-node/using-docker.md)
- ### [Using Aptos Source](nodes/validator-node/operator/running-validator-node/using-source-code.md)

### Configure Validator node

### Connect to Aptos network

After deploying your nodes, [connect to the Aptos Network](../connect-to-aptos-network.md).

### Set up staking pool operations

After connecting your nodes to the Aptos network, [establish staking pool operations](../staking-pool-operations.md).

## Test Validator node

After your nodes are deployed and configure, make sure they meet [node liveness criteria](../node-liveness-criteria.md).

## Install Validator fullnode

Note that many of the same instructions can be used to run a validator fullnode in Aptos:

-  If you use the provided reference Kubernetes deployments (i.e. for cloud-managed kubernetes on AWS, Azure, or GCP), then one validator node and one validator fullnode are deployed by default.
- When using the Docker or the source code, the `fullnode.yaml` will enable you to run a validator fullnode. 
  - See [Step 11](nodes/validator-node/operator/running-validator-node/using-docker.md#docker-vfn) in the Docker-based instructions. 
  - Similarly, if you use source code, see from [Step 13](run-validator-node-using-source#source-code-vfn) in the source code instructions. 
:::