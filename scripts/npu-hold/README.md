# npu-hold: GPU+NPU safety settings at boot and after resume (Strix Halo)

Holds the GPU fabric clock (FCLK) at its top level, sets the NPU to `performance` power mode,
and keeps the NPU out of runtime suspend. Without these, running the iGPU and the NPU together
can hang the machine; all three reset at every boot. See
[docs/strix-halo-iommu-npu.md](../../docs/strix-halo-iommu-npu.md).

Install (sudo):

    sudo install -m 755 npu-hold.sh /usr/local/sbin/npu-hold.sh
    sudo install -m 644 npu-hold.service /etc/systemd/system/npu-hold.service
    sudo install -m 755 npu-hold-sleep-hook /usr/lib/systemd/system-sleep/npu-hold
    sudo systemctl daemon-reload && sudo systemctl enable --now npu-hold.service

Check: `npu-hold.sh --check` (no root, no writes) exits 0 only if the hold is fully in place:
performance level `manual`, exactly one starred `pp_dpm_fclk` level and it is the top one (under
`auto` the top level can also show a star, so the star alone is not proof), and every NPU's
`power/control` is `on`. Run it before starting any GPU+NPU job and refuse the job if it fails;
the settings can be reset after boot (for example by a driver reload or another tool).
`systemctl status npu-hold` shows whether the boot-time hold ran. Undo: `sudo systemctl disable npu-hold`, remove the three files, reboot.
