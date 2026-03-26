"""
EXP-04 Pilot Dashboard — Grokking Topology

Run:   python dashboard_exp04/app.py
Open:  http://localhost:5051

Reads from results/exp04_pilot/seed_*/
Shows: training curves, PH dynamics, baselines, grokking onset, cross-seed view.
"""

import json
import os
import math
import subprocess
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "exp04_pilot"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp04_pilot.yaml"


def sanitize(obj):
    """Replace inf/NaN with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


def load_seed_data(seed_dir):
    """Load all JSON results for one seed."""
    data = {"seed": seed_dir.name}

    training_path = seed_dir / "training_metrics.json"
    if training_path.exists():
        with open(training_path) as f:
            raw = json.load(f)
            data["training"] = raw.get("metrics", [])
            data["seed_value"] = raw.get("seed")

    topo_path = seed_dir / "topology_metrics.json"
    if topo_path.exists():
        with open(topo_path) as f:
            data["topology"] = json.load(f)

    baseline_path = seed_dir / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data["baselines"] = json.load(f)

    return data


def detect_grokking_onset(training_metrics, test_thresh=0.9, train_thresh=0.99):
    """Find first step where test_acc > test_thresh and train_acc > train_thresh."""
    for entry in training_metrics:
        if entry.get("train_acc", 0) > train_thresh and entry.get("test_acc", 0) > test_thresh:
            return entry["step"]
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/seeds")
def api_seeds():
    """List available seeds and their status."""
    if not RESULTS_DIR.exists():
        return jsonify([])

    seeds = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("seed_"):
            has_training = (d / "training_metrics.json").exists()
            has_topo = (d / "topology_metrics.json").exists()
            has_baselines = (d / "baseline_metrics.json").exists()
            n_ckpts = len(list((d / "checkpoints").glob("*.pt"))) if (d / "checkpoints").exists() else 0
            seeds.append({
                "name": d.name,
                "has_training": has_training,
                "has_topology": has_topo,
                "has_baselines": has_baselines,
                "n_checkpoints": n_ckpts,
            })
    return jsonify(seeds)


@app.route("/api/seed/<seed_name>")
def api_seed_data(seed_name):
    """Full data for one seed."""
    seed_dir = RESULTS_DIR / seed_name
    if not seed_dir.exists():
        return jsonify({"error": "not found"}), 404

    data = load_seed_data(seed_dir)

    # Detect grokking onset
    if "training" in data:
        data["grokking_onset"] = detect_grokking_onset(data["training"])

    return jsonify(sanitize(data))


@app.route("/api/overview")
def api_overview():
    """Cross-seed summary for pilot gate evaluation."""
    if not RESULTS_DIR.exists():
        return jsonify({"seeds": [], "pilot_gate": None})

    all_seeds = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not (d.is_dir() and d.name.startswith("seed_")):
            continue

        seed_data = load_seed_data(d)
        onset = detect_grokking_onset(seed_data.get("training", []))

        summary = {
            "name": d.name,
            "grokking_onset": onset,
            "grokked": onset is not None,
            "final_test_acc": seed_data["training"][-1]["test_acc"] if seed_data.get("training") else None,
            "final_train_acc": seed_data["training"][-1]["train_acc"] if seed_data.get("training") else None,
        }

        # If topology data exists, grab the last pre-onset H0 feature count trend
        if "topology" in seed_data and onset is not None:
            pre_onset = [t for t in seed_data["topology"] if t["step"] < onset]
            if len(pre_onset) >= 3:
                recent = pre_onset[-3:]
                h0_values = [t.get("h0_feature_count", 0) for t in recent]
                # Direction: increasing, decreasing, or flat
                diffs = [h0_values[i+1] - h0_values[i] for i in range(len(h0_values)-1)]
                if all(d > 0 for d in diffs):
                    summary["h0_trend_pre_onset"] = "increasing"
                elif all(d < 0 for d in diffs):
                    summary["h0_trend_pre_onset"] = "decreasing"
                else:
                    summary["h0_trend_pre_onset"] = "mixed"
            else:
                summary["h0_trend_pre_onset"] = "insufficient_data"
        else:
            summary["h0_trend_pre_onset"] = "no_data"

        all_seeds.append(summary)

    # Pilot gate check
    consistent_count = sum(
        1 for s in all_seeds
        if s["h0_trend_pre_onset"] in ("increasing", "decreasing")
    )
    first_direction = None
    for s in all_seeds:
        if s["h0_trend_pre_onset"] in ("increasing", "decreasing"):
            if first_direction is None:
                first_direction = s["h0_trend_pre_onset"]
            elif s["h0_trend_pre_onset"] != first_direction:
                consistent_count = 0  # Directions disagree
                break

    pilot_gate = {
        "consistent_seeds": consistent_count,
        "required": 3,
        "passed": consistent_count >= 3,
        "direction": first_direction,
    }

    return jsonify(sanitize({"seeds": all_seeds, "pilot_gate": pilot_gate}))


@app.route("/api/system")
def api_system():
    """GPU/CPU/RAM stats — same as PERSIST dashboard."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        for name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            if name in temps and temps[name]:
                cpu_temp = int(temps[name][0].current)
                break
    except Exception:
        pass

    mem = psutil.virtual_memory()

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
    print("EXP-04 Dashboard: http://localhost:5051")
    print(f"Results dir: {RESULTS_DIR}")
    app.run(host="0.0.0.0", port=5051, debug=True)
