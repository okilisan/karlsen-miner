# Karlsen-miner
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/karlsen-network/karlsen-miner/ci.yaml)](https://github.com/karlsen-network/karlsen-miner/actions)
[![Latest Release](https://img.shields.io/github/v/release/karlsen-network/karlsen-miner?display_name=tag&style=flat-square)](https://github.com/karlsen-network/karlsen-miner/releases)
[![Downloads Latest](https://img.shields.io/github/downloads/karlsen-network/karlsen-miner/latest/total?style=flat-square)](https://github.com/karlsen-network/karlsen-miner/releases/latest)
[![Join the Karlsen Discord Server](https://img.shields.io/discord/1169939685280337930.svg?label=&logo=discord&logoColor=ffffff)](https://discord.gg/ZPZRvgMJDT)

This is a modification of [Kaspa GPU Miner](https://github.com/tmrlvi/kaspa-miner) for Karlsen compatible with KarlsenhashV2 based on 
[FishHashPlus](https://github.com/karlsen-network/karlsend/blob/mainnet_karlsenhashv2/domain/consensus/utils/pow/fishhashplus_kernel.go) by [Lolliedieb](https://github.com/Lolliedieb). 
We use the improved FishHashPlus version that underwent security auditing, with a smaller header size. KarlsenhashV2 is an ASIC-resistant, memory-intensive algorithm that generates a DAG requiring sufficient GPU VRAM.

## Installation

### From Git Sources

If you are looking to build from the repository (for debug / extension), note that the plugins are additional
packages in the workspace. To compile a specific package, you run the following command or any subset of it

```sh
git clone https://github.com/karlsen-network/karlsen-miner
cd karlsen-miner
cargo build --release --all
```
And, the miner (and plugins) will be in `targets/release`.

### From Binaries
The [release page](https://github.com/karlsen-network/karlsen-miner/releases/latest) includes precompiled binaries for Linux, and Windows (for the GPU version).

### Removing Plugins
To remove a plugin, you simply remove the corresponding `dll`/`so` for the directory of the miner. 

* `libkarlsencuda.so`, `libkarlsencuda.dll`: Cuda support for karlsen-miner
* `libkarlsenopencl.so`, `libkarlsenopencl.dll`: OpenCL support for karlsen-miner (currently disabled)

# Usage
To start mining, you need to run [rusty-karlsen](https://github.com/karlsen-network/rusty-karlsen) and have an address to send the rewards to.
Here is a [guidance](https://github.com/karlsen-network/docs/blob/main/Getting%20Started/Rust%20Full%20Node%20Installation.md) on how to run a full node and how to generate addresses.

Help:
```
karlsen-miner 
A Karlsen high performance CPU/GPU miner

USAGE:
    karlsen-miner [OPTIONS] --mining-address <MINING_ADDRESS>

OPTIONS:
    -a, --mining-address <MINING_ADDRESS>                  The Karlsen address for the miner reward
        --cuda-device <CUDA_DEVICE>                        Which CUDA GPUs to use [default: all]
        --cuda-disable                                     Disable cuda workers
        --cuda-lock-core-clocks <CUDA_LOCK_CORE_CLOCKS>    Lock core clocks eg: ,1200, [default: 0]
        --cuda-lock-mem-clocks <CUDA_LOCK_MEM_CLOCKS>      Lock mem clocks eg: ,810, [default: 0]
        --cuda-no-blocking-sync                            Actively wait for result. Higher CPU usage, but less red blocks. Can have lower workload.
        --cuda-power-limits <CUDA_POWER_LIMITS>            Lock power limits eg: ,150, [default: 0]
        --cuda-workload <CUDA_WORKLOAD>                    Ratio of nonces to GPU possible parrallel run [default: 64]
        --cuda-workload-absolute                           The values given by workload are not ratio, but absolute number of nonces [default: false]
    -d, --debug                                            Enable debug logging level
        --devfund-percent <DEVFUND_PERCENT>                The percentage of blocks to send to the devfund (minimum 0%) [default: 0]
    -h, --help                                             Print help information
        --mine-when-not-synced                             Mine even when karlsend says it is not synced
        --nonce-gen <NONCE_GEN>                            The random method used to generate nonces. Options: (i) xoshiro (ii) lean [default: lean]
    -p, --port <PORT>                                      karlsend port [default: Mainnet = 42110, Testnet = 42210, Devnet = 42610]
    -s, --karlsend-address <karlsend_ADDRESS>              IP, pool, or node address of the Karlsend instance. Use stratum+tcp:// for stratum or grpc:// for Karlsend (default: grpc://127.0.0.1)
    -t, --threads <NUM_THREADS>                            Amount of CPU miner threads to launch [default: 0]
        --testnet                                          Use testnet instead of mainnet [default: false]
        --devnet                                           Use devnet instead of mainnet [default: false]
```

To start mining, you just need to run the following:
```
./karlsen-miner --mining-address karlsen:XXXXX
```

This will run the miner on all the available GPU devcies.

## Devfund

The devfund is a fund managed by the Karlsen community in order to fund Karlsen development <br>
A miner that wants to mine higher percentage into the dev-fund can pass the following flags: <br>
`--devfund-precent=XX.YY` to mine only XX.YY% of the blocks into the devfund.

**This version automatically sets the devfund donation to the Karlsen Devfund, with a default donation rate of 0%**

If you would like to support us, run the miner with the following command:
```
./karlsen-miner --devfund-percent <DEVFUND_PERCENT> --mining-address karlsen:XXXXX
```

## Karlsen Dev Fund
```
karlsen:qzrq7v5jhsc5znvtfdg6vxg7dz5x8dqe4wrh90jkdnwehp6vr8uj7csdss2l7
```

## Please consider donating to the original dev:

**Elichai**: `kaspa:qzvqtx5gkvl3tc54up6r8pk5mhuft9rtr0lvn624w9mtv4eqm9rvc9zfdmmpu`

**HauntedCook**: `kaspa:qz4jdyu04hv4hpyy00pl6trzw4gllnhnwy62xattejv2vaj5r0p5quvns058f`
