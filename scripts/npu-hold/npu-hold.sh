#!/bin/sh
# npu-hold.sh: GPU+NPU safety settings for Strix Halo (fabric clock held at its top level, NPU performance mode,
# NPU kept out of runtime suspend). Run as root at boot and after resume; all three reset otherwise.
set -u
dev=""
for d in /sys/class/drm/card*/device; do [ -e "$d/pp_dpm_fclk" ] && dev=$d && break; done
[ -n "$dev" ] || { echo "npu-hold: no amdgpu pp_dpm_fclk" >&2; exit 1; }
top=$(($(grep -c . "$dev/pp_dpm_fclk") - 1))
echo manual > "$dev/power_dpm_force_performance_level"
echo "$top" > "$dev/pp_dpm_fclk"
for n in /sys/bus/pci/drivers/amdxdna/0000:*; do [ -e "$n/power/control" ] && echo on > "$n/power/control"; done
XRT=/opt/xilinx/xrt/bin/xrt-smi
[ -x "$XRT" ] && "$XRT" configure --pmode performance >/dev/null 2>&1
grep '\*' "$dev/pp_dpm_fclk" | grep -q "^$top:" && echo "npu-hold: fclk held at level $top" || { echo "npu-hold: fclk NOT held" >&2; exit 1; }
