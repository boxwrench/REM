#!/bin/sh
# npu-hold.sh: GPU+NPU safety settings for Strix Halo (fabric clock held at its top level, NPU performance mode,
# NPU kept out of runtime suspend). Run as root at boot and after resume; all three reset otherwise.
#   npu-hold.sh            apply the hold (root), then verify it
#   npu-hold.sh --check    verify only (no writes, no root needed); exit 1 if the hold is not in place.
#                          Run this before starting any GPU+NPU job and refuse the job on failure.
set -u
dev=""
for d in /sys/class/drm/card*/device; do [ -e "$d/pp_dpm_fclk" ] && dev=$d && break; done
[ -n "$dev" ] || { echo "npu-hold: no amdgpu pp_dpm_fclk" >&2; exit 1; }
top=$(($(grep -c . "$dev/pp_dpm_fclk") - 1))

# Held = performance level "manual" AND exactly one starred fclk level, the top one (under "auto" a star can
# also sit on the top level, so the star alone is not proof) AND every NPU kept out of runtime suspend.
check() {
  perf=$(cat "$dev/power_dpm_force_performance_level")
  stars=$(grep -c '\*' "$dev/pp_dpm_fclk")
  topstar=$(grep -c "^$top:.*\*" "$dev/pp_dpm_fclk")
  if [ "$perf" != manual ] || [ "$stars" != 1 ] || [ "$topstar" != 1 ]; then
    echo "npu-hold: NOT held (perf=$perf, starred levels=$stars, top starred=$topstar)" >&2; return 1
  fi
  for n in /sys/bus/pci/drivers/amdxdna/0000:*; do
    [ -e "$n/power/control" ] || continue
    [ "$(cat "$n/power/control")" = on ] || { echo "npu-hold: NPU $n power/control is not on" >&2; return 1; }
  done
  return 0
}

if [ "${1:-}" = "--check" ] || [ "${1:-}" = "--status" ]; then
  check || exit 1
  echo "npu-hold: held at fclk level $top"; exit 0
fi

echo manual > "$dev/power_dpm_force_performance_level"
echo "$top" > "$dev/pp_dpm_fclk"
for n in /sys/bus/pci/drivers/amdxdna/0000:*; do [ -e "$n/power/control" ] && echo on > "$n/power/control"; done
XRT=$(PATH=/usr/bin:/usr/local/bin:/opt/xilinx/xrt/bin:$PATH command -v xrt-smi)
if [ -n "$XRT" ]; then
  "$XRT" configure --pmode performance >/dev/null 2>&1 || echo "npu-hold: xrt-smi pmode failed" >&2
else
  echo "npu-hold: xrt-smi not found, NPU pmode not set" >&2
fi
# the starred fclk level can lag the write by a moment; give it up to 5 s
i=0
while ! check 2>/dev/null; do
  i=$((i + 1)); [ $i -gt 10 ] && { check; echo "npu-hold: fclk NOT held" >&2; exit 1; }
  sleep 0.5
done
echo "npu-hold: fclk held at level $top"
