"""
EXP-01 Experiment Dashboard — Local web interface for managing experiments.

Run: python dashboard/app.py
Open: http://localhost:5050
"""

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import math

import psutil
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


def sanitize_for_json(obj):
    """Replace inf/NaN floats with None so JSON.parse() won't choke."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj

# ─── Configuration ───
PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# ─── Experiment definitions per dataset ───

def _make_exp(exp_id, name, config, params, dataset):
    return {"id": exp_id, "name": name, "config": config, "params": params, "dataset": dataset}

# 14 original architectures + 5 WRN width ladder = 19 per dataset
_ARCH_DEFS = [
    ("", "ResNet-18", "exp01", "~11M"),
    ("_resnet50", "ResNet-50", "exp01_resnet50", "~23.6M"),
    ("_vit", "ViT-Small", "exp01_vit", "~3M"),
    ("_wrn2810", "WRN-28-10", "exp01_wrn2810", "~36.5M"),
    ("_wrn281", "WRN-28-1", "exp01_wrn281", "~0.4M"),
    ("_wrn282", "WRN-28-2", "exp01_wrn282", "~1.5M"),
    ("_wrn284", "WRN-28-4", "exp01_wrn284", "~5.9M"),
    ("_wrn286", "WRN-28-6", "exp01_wrn286", "~13.0M"),
    ("_wrn288", "WRN-28-8", "exp01_wrn288", "~23.4M"),
    ("_mlpmixer", "MLP-Mixer", "exp01_mlpmixer", "~2.3M"),
    ("_resnet18wide", "ResNet-18 Wide", "exp01_resnet18wide", "~44.7M"),
    ("_densenet121", "DenseNet-121", "exp01_densenet121", "~7M"),
    ("_efficientnet", "EfficientNet-B0", "exp01_efficientnet", "~4.1M"),
    ("_vgg16bn", "VGG-16-BN", "exp01_vgg16bn", "~14.7M"),
    ("_convnext", "ConvNeXt-Tiny", "exp01_convnext", "~27.9M"),
    ("_mobilenetv3", "MobileNet-V3-S", "exp01_mobilenetv3", "~1.1M"),
    ("_vittiny", "ViT-Tiny", "exp01_vittiny", "~0.3M"),
    ("_shufflenet", "ShuffleNet-V2", "exp01_shufflenet", "~1.3M"),
    ("_regnet", "RegNet-Y-400MF", "exp01_regnet", "~3.9M"),
]

EXPERIMENTS_CIFAR100 = [
    _make_exp(
        base_id,
        name,
        f"{base_id}.yaml",
        params,
        "cifar100",
    )
    for suffix, name, base_id, params in _ARCH_DEFS
]

EXPERIMENTS_CUB200 = [
    _make_exp(
        f"{base_id}_cub200",
        name,
        f"{base_id}_cub200.yaml",
        params,
        "cub200",
    )
    for suffix, name, base_id, params in _ARCH_DEFS
]

EXPERIMENTS_RESISC45 = [
    _make_exp(
        f"{base_id}_resisc45",
        name,
        f"{base_id}_resisc45.yaml",
        params,
        "resisc45",
    )
    for suffix, name, base_id, params in _ARCH_DEFS
]

ALL_EXPERIMENTS = {
    "cifar100": EXPERIMENTS_CIFAR100,
    "cub200": EXPERIMENTS_CUB200,
    "resisc45": EXPERIMENTS_RESISC45,
}

# Active dataset (switchable via API)
active_dataset = "cifar100"

def get_active_experiments():
    return ALL_EXPERIMENTS.get(active_dataset, EXPERIMENTS_CIFAR100)

def get_all_experiments_flat():
    """Return all experiments across all datasets."""
    result = []
    for exps in ALL_EXPERIMENTS.values():
        result.extend(exps)
    return result

PHASES = [
    {"id": "phase1", "name": "Train Task A", "module": "phase1_train_task_a", "check_dir": "checkpoints", "check_file": "task_a_best.pt", "auto": True},
    {"id": "phase2", "name": "Landscape Topology", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary.json", "auto": True},
    {"id": "phase2_run1", "name": "Landscape Slice 2", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary_run1.json", "auto": True, "run_id": "1"},
    {"id": "phase2_run2", "name": "Landscape Slice 3", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary_run2.json", "auto": True, "run_id": "2"},
    {"id": "phase2_run3", "name": "Landscape Slice 4", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary_run3.json", "auto": True, "run_id": "3"},
    {"id": "phase2_run4", "name": "Landscape Slice 5", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary_run4.json", "auto": True, "run_id": "4"},
    {"id": "phase3", "name": "Sequential Forgetting", "module": "phase3_sequential_forgetting", "check_dir": "forgetting", "check_file": "forgetting_curve.json", "auto": True},
    {"id": "phase2b", "name": "Displacement Analysis", "module": "phase2b_displacement_analysis", "check_dir": "displacement", "check_file": "displacement_summary.json", "auto": False},
]

# ─── State ───
runner_state = {
    "running": False,
    "paused": False,
    "current_experiment": None,
    "current_phase": None,
    "queue": [],
    "logs": [],
    "process": None,
    "started_at": None,
}
runner_lock = threading.Lock()


def get_experiment_status(exp_id):
    """Check which phases are complete for an experiment."""
    result_dir = RESULTS_DIR / exp_id
    status = {}
    for phase in PHASES:
        check_path = result_dir / phase["check_dir"] / phase["check_file"]
        status[phase["id"]] = "complete" if check_path.exists() else "pending"
    return status


def get_experiment_results(exp_id):
    """Load results from completed phases."""
    result_dir = RESULTS_DIR / exp_id
    results = {}

    # Phase 1: accuracy
    ckpt_dir = result_dir / "checkpoints"
    if (ckpt_dir / "task_a_best.pt").exists():
        # Try to read accuracy from topology summary (has checkpoint_accuracy)
        topo_path = result_dir / "topology" / "topology_summary.json"
        if topo_path.exists():
            with open(topo_path) as f:
                data = json.load(f)
                results["accuracy"] = data.get("checkpoint_accuracy", None)

    # Phase 2: topology + baseline metrics
    topo_path = result_dir / "topology" / "topology_summary.json"
    if topo_path.exists():
        with open(topo_path) as f:
            data = json.load(f)
            results["h0_persistence"] = data.get("H0", None)
            results["h0_max_lifetime"] = data.get("H0_max_lifetime", None)
            results["h1_persistence"] = data.get("H1", None)
            results["hessian_trace"] = data.get("hessian_trace_mean", None)
            results["max_eigenvalue"] = data.get("max_eigenvalue", None)
            results["fisher_trace"] = data.get("fisher_trace", None)
            results["max_barrier"] = data.get("max_barrier", None)
            results["loss_range"] = f"[{data.get('loss_min', 0):.2f}, {data.get('loss_max', 0):.2f}]"
            if results.get("accuracy") is None:
                results["accuracy"] = data.get("checkpoint_accuracy", None)

    # Phase 3: forgetting
    forget_path = result_dir / "forgetting" / "forgetting_curve.json"
    if forget_path.exists():
        with open(forget_path) as f:
            data = json.load(f)
            results["initial_acc"] = data.get("initial_task_a_acc", None)
            if results.get("accuracy") is None:
                results["accuracy"] = results["initial_acc"]
            curve = data.get("curve", [])
            for point in curve:
                step = point["step"]
                if step > 0:
                    results[f"ret_{step}"] = point["task_a_acc"]

    return results


def archive_phase_results(exp_id, phase_id):
    """Back up existing result files before a re-run."""
    result_dir = RESULTS_DIR / exp_id
    backup_map = {
        "phase1": [("checkpoints", "task_a_best.pt")],
        "phase2": [("topology", "topology_summary.json")],
        "phase2_run1": [("topology", "topology_summary_run1.json")],
        "phase2_run2": [("topology", "topology_summary_run2.json")],
        "phase2_run3": [("topology", "topology_summary_run3.json")],
        "phase2_run4": [("topology", "topology_summary_run4.json")],
        "phase3": [("forgetting", "forgetting_curve.json")],
    }
    for subdir, filename in backup_map.get(phase_id, []):
        src = result_dir / subdir / filename
        if src.exists():
            dst = src.with_suffix(src.suffix + ".bak")
            shutil.copy2(str(src), str(dst))


def run_phase(exp, phase, run_id=None):
    """Run a single phase for an experiment."""
    config_path = CONFIGS_DIR / exp["config"]
    module = f"experiments.exp01_topological_persistence.{phase['module']}"

    cmd = [str(VENV_PYTHON), "-m", module, "--config", str(config_path)]
    # Multi-slice: pass --run-id from phase definition or explicit run_id arg
    effective_run_id = run_id if run_id is not None else phase.get("run_id")
    if effective_run_id is not None and "phase2" in phase["id"]:
        cmd.extend(["--run-id", str(effective_run_id)])

    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Starting {exp['name']} — {phase['name']}"
    with runner_lock:
        runner_state["logs"].append(log_entry)
        runner_state["current_experiment"] = exp["name"]
        runner_state["current_phase"] = phase["name"]

    try:
        env = os.environ.copy()
        env["CLEARML_OFF"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with runner_lock:
            runner_state["process"] = process

        for line in process.stdout:
            line = line.rstrip()
            if line:
                with runner_lock:
                    runner_state["logs"].append(line)
                    # Keep last 500 lines
                    if len(runner_state["logs"]) > 500:
                        runner_state["logs"] = runner_state["logs"][-500:]

        process.wait()

        status = "completed" if process.returncode == 0 else f"FAILED (code {process.returncode})"
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {exp['name']} — {phase['name']}: {status}"
        with runner_lock:
            runner_state["logs"].append(log_entry)

        return process.returncode == 0

    except Exception as e:
        with runner_lock:
            runner_state["logs"].append(f"ERROR: {e}")
        return False


def runner_worker(queue, force=None):
    """Background worker that processes the experiment queue.

    Args:
        queue: list of (exp_id, phase_id) tuples to run.
        force: optional set of (exp_id, phase_id) tuples that should skip
               the completion check and archive existing results first.
    """
    if force is None:
        force = set()

    with runner_lock:
        runner_state["running"] = True
        runner_state["started_at"] = datetime.now().isoformat()

    for exp_id, phase_id in queue:
        # Find experiment and phase
        exp = next((e for e in get_all_experiments_flat() if e["id"] == exp_id), None)
        phase = next((p for p in PHASES if p["id"] == phase_id), None)
        if not exp or not phase:
            continue

        is_forced = (exp_id, phase_id) in force

        if is_forced:
            # Archive existing results before re-running
            archive_phase_results(exp_id, phase_id)
            with runner_lock:
                runner_state["logs"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Re-running {exp['name']} — {phase['name']} (existing results backed up)"
                )
        else:
            # Check if already complete
            status = get_experiment_status(exp_id)
            if status[phase_id] == "complete":
                with runner_lock:
                    runner_state["logs"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Skipping {exp['name']} — {phase['name']} (already complete)"
                    )
                continue

        success = run_phase(exp, phase)
        if not success:
            with runner_lock:
                runner_state["logs"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Stopping queue due to failure."
                )

    with runner_lock:
        runner_state["running"] = False
        runner_state["paused"] = False
        runner_state["current_experiment"] = None
        runner_state["current_phase"] = None
        runner_state["process"] = None
        runner_state["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Queue complete."
        )


# ─── Routes ───

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dataset", methods=["POST"])
def api_switch_dataset():
    """Switch active dataset (cifar100 or cifar10)."""
    global active_dataset
    data = request.json or {}
    ds = data.get("dataset", "cifar100")
    if ds not in ALL_EXPERIMENTS:
        return jsonify({"error": f"Unknown dataset: {ds}. Available: {list(ALL_EXPERIMENTS.keys())}"}), 400
    active_dataset = ds
    return jsonify({"dataset": active_dataset})


@app.route("/api/status")
def api_status():
    """Get full dashboard status."""
    experiments = []
    for exp in get_active_experiments():
        phase_status = get_experiment_status(exp["id"])
        results = get_experiment_results(exp["id"])
        experiments.append({
            **exp,
            "phases": phase_status,
            "results": results,
        })

    with runner_lock:
        return jsonify(sanitize_for_json({
            "experiments": experiments,
            "active_dataset": active_dataset,
            "runner": {
                "running": runner_state["running"],
                "paused": runner_state["paused"],
                "current_experiment": runner_state["current_experiment"],
                "current_phase": runner_state["current_phase"],
                "started_at": runner_state["started_at"],
            },
        }))


@app.route("/api/logs")
def api_logs():
    """Get recent logs."""
    offset = request.args.get("offset", 0, type=int)
    with runner_lock:
        logs = runner_state["logs"][offset:]
        return jsonify({
            "logs": logs,
            "total": len(runner_state["logs"]),
            "offset": offset,
        })


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start running experiments."""
    with runner_lock:
        if runner_state["running"]:
            return jsonify({"error": "Already running"}), 400

    data = request.json or {}
    mode = data.get("mode", "all_remaining")

    queue = []

    if mode == "all_remaining":
        # Queue all incomplete auto-phases for all experiments (no reruns!)
        for exp in get_active_experiments():
            status = get_experiment_status(exp["id"])
            for phase in PHASES:
                if not phase.get("auto", True):
                    continue  # skip non-auto phases (e.g. displacement)
                if status[phase["id"]] != "complete":
                    queue.append((exp["id"], phase["id"]))

    elif mode == "all_remaining_all":
        # Queue across ALL datasets, no reruns
        for exp in get_all_experiments_flat():
            status = get_experiment_status(exp["id"])
            for phase in PHASES:
                if not phase.get("auto", True):
                    continue
                if status[phase["id"]] != "complete":
                    queue.append((exp["id"], phase["id"]))

    elif mode == "single":
        exp_id = data.get("experiment")
        phase_id = data.get("phase")
        if exp_id and phase_id:
            queue.append((exp_id, phase_id))

    elif mode == "experiment":
        exp_id = data.get("experiment")
        if exp_id:
            status = get_experiment_status(exp_id)
            for phase in PHASES:
                if status[phase["id"]] != "complete":
                    queue.append((exp_id, phase["id"]))

    if not queue:
        return jsonify({"message": "Nothing to run — all complete!"})

    with runner_lock:
        runner_state["queue"] = queue
        runner_state["logs"] = [f"[{datetime.now().strftime('%H:%M:%S')}] Queued {len(queue)} tasks."]

    thread = threading.Thread(target=runner_worker, args=(queue,), daemon=True)
    thread.start()

    return jsonify({"message": f"Started {len(queue)} tasks", "queue": queue})


@app.route("/api/rerun", methods=["POST"])
def api_rerun():
    """Re-run a phase for one or all experiments, archiving existing results.

    For phase2, automatically expands to all slice phases (phase2, phase2_run1-4).
    """
    with runner_lock:
        if runner_state["running"]:
            return jsonify({"error": "Already running"}), 400

    data = request.json or {}
    exp_id = data.get("experiment")  # optional — if omitted, re-run for all
    phase_id = data.get("phase")

    if not phase_id:
        return jsonify({"error": "phase is required"}), 400

    # Validate phase
    if not any(p["id"] == phase_id for p in PHASES):
        return jsonify({"error": f"Unknown phase: {phase_id}"}), 400

    # For phase2, expand to include all slice phases
    phase_ids = [phase_id]
    if phase_id == "phase2":
        phase_ids = [p["id"] for p in PHASES if p["id"].startswith("phase2") and p["id"] != "phase2b"]

    experiments = ([next(e for e in get_all_experiments_flat() if e["id"] == exp_id)]
                   if exp_id else list(get_active_experiments()))

    queue = []
    force = set()
    for exp in experiments:
        for pid in phase_ids:
            queue.append((exp["id"], pid))
            force.add((exp["id"], pid))

    with runner_lock:
        runner_state["queue"] = queue
        runner_state["logs"] = [
            f"[{datetime.now().strftime('%H:%M:%S')}] Queued {len(queue)} re-run task(s) for {'/'.join(phase_ids)}."
        ]

    thread = threading.Thread(target=runner_worker, args=(queue,), kwargs={"force": force}, daemon=True)
    thread.start()

    return jsonify({"message": f"Re-running {len(queue)} task(s)", "queue": queue})


@app.route("/api/clean_rebuild", methods=["POST"])
def api_clean_rebuild():
    """Clean invalid Phase 2 multi-slice data (seed bug) and Phase 3 forgetting
    data (eval_steps changed), then queue all remaining phases for both datasets.

    This is the one-click fix for the seed bug + early eval steps changes.
    """
    with runner_lock:
        if runner_state["running"]:
            return jsonify({"error": "Cannot clean while running"}), 400

    cleaned = {"phase2_slices": 0, "phase3_forgetting": 0}

    for exp in get_all_experiments_flat():
        result_dir = RESULTS_DIR / exp["id"]
        topo_dir = result_dir / "topology"
        forget_dir = result_dir / "forgetting"

        # Delete invalid multi-slice Phase 2 files (run1/run2 had same seed as default)
        for run_id in ["1", "2"]:
            for fname in [
                f"topology_summary_run{run_id}.json",
                f"loss_landscape_run{run_id}.npz",
                f"landscape_directions_run{run_id}.pt",
                f"persistence_diagram_H0_run{run_id}.npy",
                f"persistence_diagram_H1_run{run_id}.npy",
            ]:
                f = topo_dir / fname
                if f.exists():
                    f.unlink()
                    cleaned["phase2_slices"] += 1

        # Archive and delete Phase 3 forgetting files (eval_steps changed)
        fc = forget_dir / "forgetting_curve.json"
        if fc.exists():
            bak = fc.with_suffix(".json.bak")
            shutil.copy2(str(fc), str(bak))
            fc.unlink()
            cleaned["phase3_forgetting"] += 1

    # Queue all remaining phases for both datasets
    queue = []
    for exp in get_all_experiments_flat():
        status = get_experiment_status(exp["id"])
        for phase in PHASES:
            if not phase.get("auto", True):
                continue
            if status[phase["id"]] != "complete":
                queue.append((exp["id"], phase["id"]))

    if not queue:
        return jsonify({
            "message": f"Cleaned {cleaned['phase2_slices']} slice files, archived {cleaned['phase3_forgetting']} forgetting curves. Nothing to re-run.",
            **cleaned,
        })

    with runner_lock:
        runner_state["queue"] = queue
        runner_state["logs"] = [
            f"[{datetime.now().strftime('%H:%M:%S')}] Cleaned {cleaned['phase2_slices']} invalid slice files, archived {cleaned['phase3_forgetting']} forgetting curves.",
            f"[{datetime.now().strftime('%H:%M:%S')}] Queued {len(queue)} tasks for rebuild.",
        ]

    thread = threading.Thread(target=runner_worker, args=(queue,), daemon=True)
    thread.start()

    return jsonify({
        "message": f"Cleaned and queued {len(queue)} tasks",
        "cleaned": cleaned,
        "queue_size": len(queue),
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the current run."""
    with runner_lock:
        if runner_state["process"]:
            # Resume first if paused, so terminate actually works
            if runner_state["paused"]:
                try:
                    os.kill(runner_state["process"].pid, signal.SIGCONT)
                except OSError:
                    pass
            runner_state["process"].terminate()
            runner_state["logs"].append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Stopped by user."
            )
            runner_state["running"] = False
            runner_state["paused"] = False
            runner_state["current_experiment"] = None
            runner_state["current_phase"] = None
            return jsonify({"message": "Stopped"})
        return jsonify({"error": "Nothing running"}), 400


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Pause the current run (SIGSTOP)."""
    with runner_lock:
        if runner_state["process"] and runner_state["running"] and not runner_state["paused"]:
            try:
                os.kill(runner_state["process"].pid, signal.SIGSTOP)
                runner_state["paused"] = True
                runner_state["logs"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Paused by user."
                )
                return jsonify({"message": "Paused"})
            except OSError as e:
                return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Nothing to pause"}), 400


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Resume a paused run (SIGCONT)."""
    with runner_lock:
        if runner_state["process"] and runner_state["paused"]:
            try:
                os.kill(runner_state["process"].pid, signal.SIGCONT)
                runner_state["paused"] = False
                runner_state["logs"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Resumed by user."
                )
                return jsonify({"message": "Resumed"})
            except OSError as e:
                return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Nothing to resume"}), 400


@app.route("/api/correlation")
def api_correlation():
    """Get correlation results if available."""
    # Try dataset-specific file first (from fixed Phase 4)
    ds_tag = f"_{active_dataset}"
    path = RESULTS_DIR / f"correlation_results{ds_tag}.json"
    if not path.exists():
        # Fall back to legacy non-tagged file
        path = RESULTS_DIR / "correlation_results.json"
    if path.exists():
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No correlation results yet"}), 404


@app.route("/api/run_correlation", methods=["POST"])
def api_run_correlation():
    """Run Phase 4 cross-architecture correlation."""
    # Find all experiments with both topology and forgetting data
    result_dirs = []
    for exp in get_active_experiments():
        status = get_experiment_status(exp["id"])
        if status["phase2"] == "complete" and status["phase3"] == "complete":
            result_dirs.append(str(RESULTS_DIR / exp["id"]))

    if len(result_dirs) < 3:
        return jsonify({"error": f"Need >= 3 complete experiments, have {len(result_dirs)}"}), 400

    cmd = [
        str(VENV_PYTHON), "-m",
        "experiments.exp01_topological_persistence.phase4_correlation",
        "--results-dirs", *result_dirs,
    ]

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
        return jsonify({
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/system")
def api_system():
    """Get GPU/CPU/RAM stats."""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    # CPU temp
    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        for name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            if name in temps and temps[name]:
                cpu_temp = int(temps[name][0].current)
                break
    except Exception:
        pass

    # RAM
    mem = psutil.virtual_memory()

    # GPU via nvidia-smi
    gpu = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 7:
                gpu = {
                    "name": parts[0],
                    "temp_c": int(parts[1]),
                    "util_percent": int(parts[2]),
                    "mem_used_mb": int(parts[3]),
                    "mem_total_mb": int(parts[4]),
                    "power_w": float(parts[5]),
                    "power_limit_w": float(parts[6]),
                }
    except Exception:
        pass

    # GPU processes
    gpu_processes = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpu_processes.append({
                            "pid": int(parts[0]),
                            "mem_mb": int(parts[1]),
                            "name": parts[2].split("/")[-1],
                        })
    except Exception:
        pass

    return jsonify({
        "cpu": {
            "percent": cpu_percent,
            "count": cpu_count,
            "freq_mhz": int(cpu_freq.current) if cpu_freq else 0,
            "temp_c": cpu_temp,
        },
        "ram": {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        },
        "gpu": gpu,
        "gpu_processes": gpu_processes,
    })


if __name__ == "__main__":
    print(f"\n  EXP-01 Experiment Dashboard")
    print(f"  http://localhost:5050")
    print(f"  Project: {PROJECT_ROOT}\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
