#!/usr/bin/env python3
"""Pre-flight check for PERSIST Phase I-A on NMSU Discovery HPC."""

import os
import sys
import shutil
from pathlib import Path


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return ok


def main():
    print("PERSIST Phase I-A: Pre-Flight Check")
    print("=" * 50)
    failures = 0

    # 1. Python environment
    print("\n1. Python Environment")
    check("Python version", sys.version_info >= (3, 10), sys.version.split()[0])

    try:
        import torch
        check("PyTorch", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
        if not check("CUDA available", cuda_ok, gpu_name):
            failures += 1
            print("         Note: CUDA not available on login node. This is OK.")
            print("         GPU will be available when jobs run on compute nodes.")
    except ImportError:
        check("PyTorch", False, "not installed")
        failures += 1

    try:
        import torchvision
        check("torchvision", True, torchvision.__version__)
    except ImportError:
        check("torchvision", False, "not installed")
        failures += 1

    try:
        import ripser
        check("ripser", True)
    except ImportError:
        check("ripser", False, "not installed")
        failures += 1

    try:
        import gudhi
        check("gudhi", True)
    except ImportError:
        check("gudhi", False, "not installed")
        failures += 1

    try:
        import scipy
        check("scipy", True, scipy.__version__)
    except ImportError:
        check("scipy", False, "not installed")
        failures += 1

    # 2. Configs
    print("\n2. Experiment Configs")
    config_dir = Path("configs")
    imagenet_configs = sorted(config_dir.glob("exp01_*_imagenet100.yaml"))
    if not check("ImageNet-100 configs", len(imagenet_configs) == 10,
                  f"found {len(imagenet_configs)}/10"):
        failures += 1
        for c in imagenet_configs:
            print(f"         {c.name}")

    # 3. SLURM scripts
    print("\n3. SLURM Scripts")
    slurm_run = Path("slurm/run_experiment.sh")
    slurm_submit = Path("slurm/submit_all.sh")
    if not check("run_experiment.sh", slurm_run.exists()):
        failures += 1
    if not check("submit_all.sh", slurm_submit.exists()):
        failures += 1

    slurm_logs = Path("slurm/logs")
    check("slurm/logs/ directory", slurm_logs.exists())
    if not slurm_logs.exists():
        slurm_logs.mkdir(parents=True, exist_ok=True)
        print("         Created slurm/logs/")

    # Check run_experiment.sh is executable
    if slurm_run.exists():
        is_exec = os.access(slurm_run, os.X_OK)
        if not is_exec:
            os.chmod(slurm_run, 0o755)
            print("         Made run_experiment.sh executable")

    # 4. ImageNet data
    print("\n4. ImageNet Data")
    data_dir = Path("data/imagenet")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        check("ImageNet train/", False, f"{train_dir} not found")
        failures += 1
        print("         See docs/HPC_WORKFLOW.md for download instructions.")
    else:
        n_train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        if not check("ImageNet train/ classes", n_train_classes >= 100,
                      f"{n_train_classes} class folders"):
            failures += 1

        # Check a class folder has actual images
        sample_class = next((d for d in train_dir.iterdir() if d.is_dir()), None)
        if sample_class:
            n_images = len(list(sample_class.glob("*.JPEG")))
            check("Sample class images", n_images > 0,
                  f"{sample_class.name}: {n_images} images")

    if not val_dir.exists():
        check("ImageNet val/", False, f"{val_dir} not found")
        failures += 1
    else:
        n_val_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
        if not check("ImageNet val/ classes", n_val_classes >= 100,
                      f"{n_val_classes} class folders"):
            failures += 1

    # 5. Disk space
    print("\n5. Storage")
    scratch = Path("/fs1/scratch/cag1145")
    if scratch.exists():
        usage = shutil.disk_usage(scratch)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        check("Scratch space", free_gb > 50,
              f"{used_gb:.0f}GB used / {total_gb:.0f}GB total / {free_gb:.0f}GB free")
    else:
        check("Scratch path", False, "/fs1/scratch/cag1145 not found (not on cluster?)")

    # 6. SLURM availability
    print("\n6. SLURM")
    squeue = shutil.which("squeue")
    sbatch = shutil.which("sbatch")
    check("squeue", squeue is not None)
    check("sbatch", sbatch is not None)
    if not squeue:
        print("         Not on cluster? SLURM commands only work on Discovery.")

    # Summary
    print("\n" + "=" * 50)
    if failures == 0:
        print("All checks passed. Ready to submit.")
        print("Run: bash slurm/submit_all.sh")
    else:
        print(f"{failures} check(s) failed. See above for details.")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
