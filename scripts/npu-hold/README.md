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

Check: `systemctl status npu-hold`, and `pp_dpm_fclk` on the card that has it shows the top
level starred. Undo: `sudo systemctl disable npu-hold`, remove the three files, reboot.
