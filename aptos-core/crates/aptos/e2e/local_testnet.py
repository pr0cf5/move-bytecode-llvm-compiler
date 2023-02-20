# Copyright (c) Aptos
# SPDX-License-Identifier: Apache-2.0

# This file contains functions for running the local testnet.

import logging
import subprocess
import time

from urllib.request import urlopen

from common import FAUCET_PORT, NODE_PORT, Network

LOG = logging.getLogger(__name__)

# Run a local testnet in a docker container. We choose to detach here and we'll
# stop running it later using the container name.
def run_node(network: Network, image_repo: str):
    image_name = build_image_name(network, image_repo)
    container_name = f"aptos-tools-{network}"
    LOG.info(f"Trying to run aptos CLI local testnet from image: {image_name}")

    # First delete the existing container if there is one with the same name.
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Run the container.
    subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--name",
            container_name,
            "-p",
            f"{NODE_PORT}:{NODE_PORT}",
            "-p",
            f"{FAUCET_PORT}:{FAUCET_PORT}",
            image_name,
            "aptos",
            "node",
            "run-local-testnet",
            "--with-faucet",
        ],
    )
    LOG.info(f"Running aptos CLI local testnet from image: {image_name}")
    return container_name


# Stop running the detached node.
def stop_node(container_name: str):
    LOG.info(f"Stopping container: {container_name}")
    subprocess.check_output(["docker", "stop", container_name])
    LOG.info(f"Stopped container: {container_name}")


# Query the node and faucet APIs until they start up or we timeout.
def wait_for_startup(container_name: str, timeout: int):
    LOG.info(f"Waiting for node and faucet APIs for {container_name} to come up")
    count = 0
    api_response = None
    faucet_response = None
    while True:
        try:
            api_response = urlopen(f"http://127.0.0.1:{NODE_PORT}/v1")
            faucet_response = urlopen(f"http://127.0.0.1:{FAUCET_PORT}/health")
            if api_response.status != 200 or faucet_response.status != 200:
                raise RuntimeError(
                    f"API or faucet not ready. API response: {api_response}. "
                    f"Faucet response: {faucet_response}"
                )
            break
        except Exception:
            if count >= timeout:
                LOG.error(f"Timeout while waiting for node / faucet to come up")
                raise
            count += 1
            time.sleep(1)
    LOG.info(f"Node and faucet APIs for {container_name} came up")


def build_image_name(network: Network, image_repo: str):
    return f"{image_repo}aptoslabs/tools:{network}"
