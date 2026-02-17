"""
EXP-01 Experiment Dashboard — Local web interface for managing experiments.

Run: python dashboard/app.py
Open: http://localhost:5050
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ─── Configuration ───
PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

EXPERIMENTS = [
    {"id": "exp01", "name": "ResNet-18", "config": "exp01.yaml", "params": "~11M"},
    {"id": "exp01_resnet50", "name": "ResNet-50", "config": "exp01_resnet50.yaml", "params": "~23.6M"},
    {"id": "exp01_vit", "name": "ViT-Small", "config": "exp01_vit.yaml", "params": "~3M"},
    {"id": "exp01_wrn2810", "name": "WRN-28-10", "config": "exp01_wrn2810.yaml", "params": "~36.5M"},
    {"id": "exp01_mlpmixer", "name": "MLP-Mixer", "config": "exp01_mlpmixer.yaml", "params": "~2.3M"},
    {"id": "exp01_resnet18wide", "name": "ResNet-18 Wide", "config": "exp01_resnet18wide.yaml", "params": "~44.7M"},
    {"id": "exp01_densenet121", "name": "DenseNet-121", "config": "exp01_densenet121.yaml", "params": "~7M"},
    {"id": "exp01_efficientnet", "name": "EfficientNet-B0", "config": "exp01_efficientnet.yaml", "params": "~4.1M"},
]

PHASES = [
    {"id": "phase1", "name": "Train Task A", "module": "phase1_train_task_a", "check_dir": "checkpoints", "check_file": "task_a_best.pt"},
    {"id": "phase2", "name": "Landscape Topology", "module": "phase2_landscape_topology", "check_dir": "topology", "check_file": "topology_summary.json"},
    {"id": "phase3", "name": "Sequential Forgetting", "module": "phase3_sequential_forgetting", "check_dir": "forgetting", "check_file": "forgetting_curve.json"},
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


def run_phase(exp, phase):
    """Run a single phase for an experiment."""
    config_path = CONFIGS_DIR / exp["config"]
    module = f"experiments.exp01_topological_persistence.{phase['module']}"

    cmd = [str(VENV_PYTHON), "-m", module, "--config", str(config_path)]

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


def runner_worker(queue):
    """Background worker that processes the experiment queue."""
    with runner_lock:
        runner_state["running"] = True
        runner_state["started_at"] = datetime.now().isoformat()

    for exp_id, phase_id in queue:
        # Find experiment and phase
        exp = next((e for e in EXPERIMENTS if e["id"] == exp_id), None)
        phase = next((p for p in PHASES if p["id"] == phase_id), None)
        if not exp or not phase:
            continue

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
            break

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


@app.route("/api/status")
def api_status():
    """Get full dashboard status."""
    experiments = []
    for exp in EXPERIMENTS:
        phase_status = get_experiment_status(exp["id"])
        results = get_experiment_results(exp["id"])
        experiments.append({
            **exp,
            "phases": phase_status,
            "results": results,
        })

    with runner_lock:
        return jsonify({
            "experiments": experiments,
            "runner": {
                "running": runner_state["running"],
                "paused": runner_state["paused"],
                "current_experiment": runner_state["current_experiment"],
                "current_phase": runner_state["current_phase"],
                "started_at": runner_state["started_at"],
            },
        })


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
        # Queue all incomplete phases for all experiments
        for exp in EXPERIMENTS:
            status = get_experiment_status(exp["id"])
            for phase in PHASES:
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
    for exp in EXPERIMENTS:
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
