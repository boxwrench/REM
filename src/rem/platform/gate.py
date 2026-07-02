"""Reachability probe to check NPU, driver, XRT, and IOMMU state."""

import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str) -> tuple[int, str]:
    """Run a shell command and return its exit code and stdout/stderr combined."""
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
        output = (res.stdout or "") + (res.stderr or "")
        return res.returncode, output.strip()
    except Exception as e:
        return -1, str(e)


def run_checks() -> dict:
    """Check system environment for NPU access and return a verdict dict."""
    checks = {}

    # Check 1: accel node present?
    accel_code, accel_out = run_command("ls /dev/accel/accel0")
    accel_passed = accel_code == 0
    checks["accel_node"] = {
        "verdict": "pass" if accel_passed else "fail",
        "detail": accel_out if accel_passed else f"Error (code {accel_code}): {accel_out}",
    }

    # Check 2: driver loaded?
    driver_code, driver_out = run_command("lsmod | grep -i amdxdna")
    driver_passed = driver_code == 0
    checks["driver_loaded"] = {
        "verdict": "pass" if driver_passed else "fail",
        "detail": driver_out if driver_passed else "amdxdna driver module not found in lsmod",
    }

    # Check 3: kernel version
    kernel_code, kernel_out = run_command("uname -r")
    kernel_passed = kernel_code == 0
    checks["kernel_version"] = {
        "verdict": "pass" if kernel_passed else "fail",
        "detail": kernel_out,
    }

    # Check 4: XRT examine
    xrt_code, xrt_out = run_command("xrt-smi examine")
    xrt_passed = xrt_code == 0
    checks["xrt_examine"] = {
        "verdict": "pass" if xrt_passed else "fail",
        "detail": xrt_out if xrt_passed else f"xrt-smi examine failed or not installed (code {xrt_code}): {xrt_out}",
    }

    # Check 5: IOMMU state
    iommu_code, iommu_out = run_command("cat /proc/cmdline | tr ' ' '\\n' | grep -i iommu")
    # We want to check if iommu is configured. Specifically, verify that amd_iommu=off is NOT in the cmdline.
    # If cmdline has amd_iommu=off, it's a failure. If amd_iommu=on, it's a pass.
    # Let's inspect the output.
    cmdline_code, cmdline_out = run_command("cat /proc/cmdline")
    iommu_off = "amd_iommu=off" in cmdline_out
    iommu_on = "amd_iommu=on" in cmdline_out
    
    if iommu_off:
        iommu_verdict = "fail"
        iommu_detail = f"IOMMU explicitly disabled: {cmdline_out.strip()}"
    elif iommu_on:
        iommu_verdict = "pass"
        iommu_detail = f"IOMMU enabled: {iommu_out.strip()}"
    else:
        # If neither is found, let's report what was found or default
        iommu_verdict = "pass" if iommu_code == 0 else "fail"
        iommu_detail = iommu_out if iommu_code == 0 else "No iommu boot flags found"

    checks["iommu_state"] = {
        "verdict": iommu_verdict,
        "detail": iommu_detail,
    }

    # Determine overall verdict
    # PASS: accel node present + driver loaded + iommu state pass + XRT present
    # PARTIAL: accel node present but driver or XRT missing -> installable
    # FAIL: no accel node present, or IOMMU explicitly disabled (amd_iommu=off)
    if not accel_passed or iommu_verdict == "fail":
        overall = "FAIL"
    elif not driver_passed or not xrt_passed:
        overall = "PARTIAL"
    else:
        overall = "PASS"

    result = {
        "verdict": overall,
        "checks": checks,
    }

    # Write JSON to bench/e0_gate.json
    bench_dir = Path("bench")
    bench_dir.mkdir(exist_ok=True)
    with open(bench_dir / "e0_gate.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = run_checks()
    
    # Print human-readable summary to stderr
    sys.stderr.write("=" * 50 + "\n")
    sys.stderr.write(f"REM Reachability Probe Overall Verdict: {res['verdict']}\n")
    sys.stderr.write("=" * 50 + "\n")
    for check_name, info in res["checks"].items():
        sys.stderr.write(f"- {check_name}: {info['verdict'].upper()}\n")
        indent_detail = "\n  ".join(info["detail"].splitlines())
        sys.stderr.write(f"  Detail: {indent_detail}\n")
    sys.stderr.write("=" * 50 + "\n")
    
    if res["verdict"] == "FAIL":
        sys.exit(1)
    elif res["verdict"] == "PARTIAL":
        sys.exit(0)
    else:
        sys.exit(0)
