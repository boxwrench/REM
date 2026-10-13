# Strix Halo: keeping the NPU available without slowing the iGPU

Running the NPU (XDNA 2) on Linux has a cost that is easy to miss: **the IOMMU
must be on**, and on a stock kernel that slows the iGPU's prompt processing by
about 7.5%. This page explains why, shows how a kernel feature arriving in Linux
7.4 ("PerfOpt") removes the cost, and gives the settings we use to run the GPU
and the NPU together without hanging the machine. It is measured on one box (below)
and written so you can reproduce it.

Context for REM: REM puts background memory work on the NPU while a model runs
on the iGPU. That is exactly the "both busy at once" case covered here. See
[hardware-context.md](hardware-context.md) for the platform itself.

---

## The box

AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB LPDDR5x, Radeon 8060S (`gfx1151`),
NPU `RyzenAI-npu5` (firmware 1.1.2.65), XRT 2.25, ROCm 7.2.2, Ubuntu 24.04 base.
BIOS power mode "Performance". The GPU workload throughout is a 321B MoE language model
(GLM-5.3-Flash via [Project Maya](https://github.com/mw00/project-maya), HIP
build) doing 32K- and 64K-token prompt prefills.

## 1. The NPU needs the IOMMU, and the IOMMU costs the iGPU

The `amdxdna` driver gives each NPU context a shared virtual address space
(SVA/PASID). That requires the AMD IOMMU. With `amd_iommu=off`, opening the NPU
fails with `ENODEV`.

Turning the IOMMU back on with `iommu=pt` (passthrough: devices get identity
mappings, no translation) unblocks the NPU, but the iGPU slows down anyway. In
passthrough mode the GPU still has ATS/PRI/PASID enabled, and its DMA still goes through
the IOMMU:

| Kernel 6.17, GPU prefill | `amd_iommu=off` | `iommu=pt` | cost |
| --- | ---: | ---: | ---: |
| ~32K-token prompt | 387.0 tok/s | 356.4 tok/s | **−7.9%** |
| ~64K-token prompt | 377.3 tok/s | 350.0 tok/s | **−7.3%** |

Decode and clocks were unchanged, with the same sclk and package power. The cost is
in the DMA path, not in power. So on a stock kernel you choose between the NPU
and roughly 7.5% of your iGPU prefill.

`pci=noats` is **not** a workaround on Strix Halo. It breaks `amdgpu` init (PSP
`SETUP_TMR` fails and every firmware load fails). The NPU's SMU also times out. Don't try it.

## 2. PerfOpt removes the cost

AMD's IOMMU "PerfOpt" series by **Mario Limonciello** (AMD) lets an iGPU in an identity
domain bypass the IOMMU, by switching off its ATS/PRI/PASID after init. It is
queued in the IOMMU tree for **Linux 7.4**, on by default, with
`amdgpu.iommu_perfopt=0` to disable it. The two commits we built from:
[`6e87850d1a8f`](https://git.kernel.org/pub/scm/linux/kernel/git/iommu/linux.git/commit/?id=6e87850d1a8f)
("iommu/amd: Add PerfOpt IOMMU performance optimization support") and
[`e74db71a3530`](https://git.kernel.org/pub/scm/linux/kernel/git/iommu/linux.git/commit/?id=e74db71a3530)
("drm/amdgpu: Enable PerfOpt IOMMU perf optimization when GPU in identity domain"). Public benchmarks so far measured the iGPU
alone. We tested it with the NPU in use.

| GPU prefill | `amd_iommu=off` (6.17) | `iommu=pt` (6.17) | **`iommu=pt` + PerfOpt (7.3-rc6 + iommu `next`)** |
| --- | ---: | ---: | ---: |
| ~32K-token prompt | 387.0 | 356.4 | **390.3 tok/s** |
| ~64K-token prompt | 377.3 | 350.0 | **378.3 tok/s** |
| decode, no speculation (ms/token) | 54.11 | 54.43 | 54.19 |

- dmesg shows `amdgpu 0000:c5:00.0: AMD-Vi: PerfOpt armed on IOMMU0`, and
  `lspci -vvv` on the GPU shows `ATSCtl/PRICtl/PASIDCtl: Enable-`. On 6.17 with
  `iommu=pt` all three were `Enable+`.
- The NPU still opens and runs normally.
- Caveat: the kernel version also changed (6.17 → 7.3-rc6), so the gain is PerfOpt
  plus the newer kernel. In practice the IOMMU tax is gone.

**The result: with PerfOpt, the NPU is available at no cost to the iGPU.**

### Building the test kernel (until 7.4 ships)

Nothing here is patched. It is the IOMMU maintainers' `next` branch, built as-is:

```sh
git clone --branch next https://git.kernel.org/pub/scm/linux/kernel/git/iommu/linux.git
cd linux    # we used c233d59828627bfaab750ec5546e81970475f951 (7.3.0-rc6)
cp /boot/config-$(uname -r) .config
scripts/config --set-str LOCALVERSION -perfopt --enable DEBUG_INFO_NONE \
  --set-str SYSTEM_TRUSTED_KEYS "" --set-str SYSTEM_REVOCATION_KEYS ""
make olddefconfig && make -j$(nproc) bindeb-pkg
```

The out-of-tree `amdxdna` DKMS package refuses kernels other than 6.17–6.19, and
its 7.0.0-rc1 source does not build against 7.3 as shipped. Its trace include
picks up the new in-tree header. Adding `-I$(src)` ahead of `-I$(src)/trace` in its
Makefile fixes that. The `amd/xdna-driver` main branch (2.21.0) builds unmodified
with `ARCH=x86_64`. Install the module under
`/lib/modules/<release>/updates/dkms/` and run `depmod`. Secure Boot must be off
or the modules signed. Keep your distro kernel installed and the GRUB menu
visible (`GRUB_TIMEOUT_STYLE=menu`, `GRUB_TIMEOUT=5`) for rollback.

## 3. Running the GPU and the NPU at the same time

GPU + NPU load together has been reported to **hard-hang** Strix Halo. NPU firmware timeouts and
Data Fabric deferred-error machine checks (MC28) are described in
[amd/xdna-driver#1605](https://github.com/amd/xdna-driver/issues/1605), where a
newer kernel alone did not help. Mike Alani's [gufo fork](https://github.com/mikealanni/gufo)
(NPU prefill offload for Qwen3.8 Flash-Next, being upstreamed as
[gufo-org/gufo#568](https://github.com/gufo-org/gufo/pull/568)) found and documents the workaround. **Hold the
fabric clock (FCLK) at a fixed level**, set the NPU's power mode to `performance`
(not `turbo`), and keep the NPU out of runtime suspend:

```sh
echo manual | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
echo 7      | sudo tee /sys/class/drm/card1/device/pp_dpm_fclk     # 7 = 2000 MHz here
sudo xrt-smi configure --pmode performance
echo on     | sudo tee /sys/bus/pci/devices/0000:c6:00.1/power/control
```

(Your card number and NPU PCI address may differ. Use the `card*` that has
`pp_dpm_fclk`, and the NPU address from `lspci | grep -i npu`.) All four reset at reboot.
[`scripts/npu-hold/`](../scripts/npu-hold/) applies them at boot and after resume
via a systemd unit and a system-sleep hook.

Holding FCLK at the top level also gives the GPU **+1.6 to 2.2%** prefill on its
own, at no measurable power cost.

### Our soak (PerfOpt kernel, settings above)

71 minutes of real GPU work: the model writing a game over 8 turns, 9K–85K-token
prompts. An NPU bfp8 GEMM (M4096 K2560 N1024, Mike Alani's kernel) ran at 100% duty for
the first 64.5 minutes alongside it.

| | result |
| --- | --- |
| machine checks / `Hardware Error` | **0** |
| `amdxdna` timeouts or errors | **0** |
| `amdgpu` faults, timeouts, resets | **0** |
| IOMMU events / page faults | **0** |
| NPU GEMM result check | PASS at start and end |
| NPU throughput with the GPU busy | ~21 TFLOPS steady (27.5 alone) |
| GPU throughput with the NPU busy | prefill −5 to −7% from shared memory bandwidth |

A text defect in some of the model's output during that run (a newline after the
first digit of numbers) turned out to be unrelated to the kernel or NPU. The box's
serving layer was an old checkout that tokenized GLM's numbers the Qwen way, a bug
already fixed upstream. With the current serving layer, the same requests come out
clean, with or without the NPU running.

## What's next

- Whether the hang trigger is FCLK *switching* or a low FCLK *level*. We're testing
  a lower fixed FCLK, which would make the workaround cheaper and give AMD a precise
  target.
- Using the NPU for real work during prefill. It runs the GEMMs a prefill
  needs at ~21 TFLOPS under load, against ~15 TFLOPS effective for the iGPU on the same
  shapes. REM's compaction on the NPU and prompt offload can take turns on it.

## Credits

- **Mario Limonciello** (AMD) wrote the PerfOpt series. The IOMMU result above is
  his work. We only built it and measured it with the NPU in use.
- **Mike Alani** ([mikealanni/gufo](https://github.com/mikealanni/gufo), a fork of
  [gufo-org/gufo](https://github.com/gufo-org/gufo)) built the NPU prefill offload,
  found and documented the fabric-clock hold, and wrote the NPU bfp8 GEMM used in our
  soak. It is being upstreamed as
  [gufo-org/gufo#568](https://github.com/gufo-org/gufo/pull/568).
- The reporters in [amd/xdna-driver#1605](https://github.com/amd/xdna-driver/issues/1605)
  wrote down the GPU+NPU hang in detail.
