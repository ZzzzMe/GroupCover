# GroupCover Implementation

This repository contains the implementation of GroupCover, with a primary focus on augmenting experiments running in a real Trusted Execution Environment (TEE).

## Experiment Deployment

We have deployed our experiments on both **Intel TDX** and **AMD SEV** platforms. Our machines are equipped with **256 cores and 1TB of memory**. To simulate the distributed nature of multiple pods, we configured the generated trusted VMs with **1 core and 2GB of memory**.

## Intel TDX and AMD SEV Confidential Virtual Machines

This section provides access credentials for the confidential virtual machines (VMs) powered by Intel TDX and AMD SEV, along with guidance on monitoring machine performance using `htop` within the VMs.
![image](https://github.com/ZzzzMe/GroupCover/assets/49647496/d0c0df7f-c45e-403d-8e1b-0901cae63c15)
![image](https://github.com/ZzzzMe/GroupCover/assets/49647496/bd479b8b-f5ee-495d-a926-7074c830dbb2)
![image](https://github.com/ZzzzMe/GroupCover/assets/49647496/61d479ce-504d-4e5b-96d4-5b28b184f622)

## Objective

The goal of these experiments is to assess the performance implications of GroupCover in genuine TEE settings. By deploying on Intel TDX and AMD SEV, we aim to provide insights into how GroupCover behaves across different trusted execution environments, offering valuable data for further optimizations and security assessments.


# Establishing RPC Connections for Confidential Computing

In our project, we explore two approaches to establish RPC (Remote Procedure Call) connections for secure and efficient communication between components in a Trusted Execution Environment (TEE). Below, we detail the `runv` and `runc` modes, each with its setup and considerations for establishing RPC connections.

## 1. `runv` Mode

In the `runv` mode, we initiate the trusted VM using qemu with a bridge mode configuration. This setup allows the trusted VM and the host machine to be on the same network segment under the `virbr0` network interface, facilitating RPC connections between them.

### Important Setup Steps:

- **Network Configuration**: The bridge mode is set through qemu on the `virbr0` network interface, which is network bridge's name.
- **Environment Variables**: It's crucial to export `GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` environment variables, setting them to `virbr0`, which is the network interface in use.

This approach ensures a direct and efficient RPC connection setup, bypassing the need for complex networking configurations.

## 2. `runc` Mode

The `runc` mode leverages the confidential container - COCO, within a Kubernetes environment. Here, we deploy the master program on one Kubernetes pod and the worker program on another, enabling RPC connections through the containers' network.

### Considerations:

- **Network Communication**: Given that container-to-container communication often involves multiple network proxies (including ingress, nginx, etc.), this method may introduce additional layers and potential bottlenecks.
- **Deployment for Optimal Performance**: To mitigate networking overhead and achieve optimal performance, we employ the `runv` mode for deploying our RPC-based secure inference tasks.


## Scripts
In master VM:
```
RANK=i WORLD_SIZE=j WORKER=k tee_test_transformer_master.py
```
in which $i$ is the master's rank, j is the total VM's number, and k is the rank of worker which you want to offload the linear operations.
In worker VM:
```
RANK=i WORLD_SIZE=j  tee_test_transformer_worker.py
```


