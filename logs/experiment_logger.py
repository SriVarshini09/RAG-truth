"""
Experiment Logger
Auto-logs every experiment run to logs/runs/ as JSON and updates logs/summary.csv.

Usage:
    from logs.experiment_logger import ExperimentLogger
    logger = ExperimentLogger()
    run_id = logger.start_run("Self-Verification GPT-4o-mini", config={...})
    logger.finish_run(run_id, metrics={...}, predictions_file="results/...")
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent
RUNS_DIR = LOGS_DIR / "runs"
SUMMARY_CSV = LOGS_DIR / "summary.csv"

SUMMARY_FIELDS = [
    "run_id", "method", "timestamp", "n_samples",
    "overall_f1", "overall_precision", "overall_recall",
    "qa_f1", "summary_f1", "data2txt_f1",
    "ci_lower", "ci_upper",
    "runtime_seconds", "cost_usd",
    "predictions_file", "notes",
]


class ExperimentLogger:
    def __init__(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._active_runs = {}

    # ─── Start / Finish ───────────────────────────────────────────────────────

    def start_run(self, method: str, config: dict = None, notes: str = "") -> str:
        """Begin timing a run. Returns run_id."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_method = method.lower().replace(" ", "_").replace("-", "_")[:30]
        run_id = f"{ts}_{safe_method}"

        self._active_runs[run_id] = {
            "run_id": run_id,
            "method": method,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config or {},
            "notes": notes,
            "_start_time": time.time(),
        }
        print(f"[Logger] Started run: {run_id}")
        return run_id

    def finish_run(
        self,
        run_id: str,
        metrics: dict,
        predictions_file: str = "",
        cost_usd: float = 0.0,
    ) -> dict:
        """
        Finish a run, save JSON log, update summary.csv.

        metrics dict should contain:
            overall_f1, overall_precision, overall_recall
            per_task: {QA: {f1, precision, recall}, Summary: {...}, Data2txt: {...}}
            bootstrap_ci: {mean_f1, ci_lower, ci_upper}
            n_samples: int
        """
        if run_id not in self._active_runs:
            raise ValueError(f"Unknown run_id: {run_id}")

        run = self._active_runs.pop(run_id)
        runtime = round(time.time() - run.pop("_start_time"), 2)

        per_task = metrics.get("per_task", {})
        ci = metrics.get("bootstrap_ci", {})

        log = {
            "run_id": run["run_id"],
            "method": run["method"],
            "timestamp": run["timestamp"],
            "config": run["config"],
            "notes": run["notes"],
            "metrics": {
                "n_samples": metrics.get("n_samples", 0),
                "overall_f1": metrics.get("overall_f1", 0.0),
                "overall_precision": metrics.get("overall_precision", 0.0),
                "overall_recall": metrics.get("overall_recall", 0.0),
                "per_task": per_task,
                "bootstrap_ci": ci,
            },
            "runtime_seconds": runtime,
            "cost_usd": cost_usd,
            "predictions_file": predictions_file,
        }

        # Save JSON
        log_path = RUNS_DIR / f"{run_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        # Update CSV
        self._append_to_csv(log, per_task, ci)

        print(f"[Logger] Run saved: {log_path}")
        print(f"[Logger] F1={metrics.get('overall_f1', 0):.4f}  "
              f"Runtime={runtime}s  Cost=${cost_usd:.3f}")
        return log

    def log_run(
        self,
        method: str,
        metrics: dict,
        config: dict = None,
        predictions_file: str = "",
        cost_usd: float = 0.0,
        notes: str = "",
    ) -> dict:
        """Convenience: start + finish in one call (for already-completed runs)."""
        run_id = self.start_run(method, config=config, notes=notes)
        return self.finish_run(run_id, metrics, predictions_file=predictions_file, cost_usd=cost_usd)

    # ─── CSV ──────────────────────────────────────────────────────────────────

    def _append_to_csv(self, log: dict, per_task: dict, ci: dict) -> None:
        write_header = not SUMMARY_CSV.exists()

        row = {
            "run_id": log["run_id"],
            "method": log["method"],
            "timestamp": log["timestamp"],
            "n_samples": log["metrics"].get("n_samples", ""),
            "overall_f1": log["metrics"].get("overall_f1", ""),
            "overall_precision": log["metrics"].get("overall_precision", ""),
            "overall_recall": log["metrics"].get("overall_recall", ""),
            "qa_f1": per_task.get("QA", {}).get("f1", ""),
            "summary_f1": per_task.get("Summary", {}).get("f1", ""),
            "data2txt_f1": per_task.get("Data2txt", {}).get("f1", ""),
            "ci_lower": ci.get("ci_lower", ""),
            "ci_upper": ci.get("ci_upper", ""),
            "runtime_seconds": log.get("runtime_seconds", ""),
            "cost_usd": log.get("cost_usd", ""),
            "predictions_file": log.get("predictions_file", ""),
            "notes": log.get("notes", ""),
        }

        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ─── Read ─────────────────────────────────────────────────────────────────

    def list_runs(self) -> list:
        """Return list of all logged run dicts."""
        runs = []
        for path in sorted(RUNS_DIR.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        return runs

    def get_run(self, run_id: str) -> dict:
        path = RUNS_DIR / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No log for run_id: {run_id}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def print_summary(self) -> None:
        """Print all runs as a comparison table."""
        runs = self.list_runs()
        if not runs:
            print("[Logger] No runs logged yet.")
            return

        print("\n" + "=" * 80)
        print(f"{'Method':<35} {'F1':>6} {'P':>6} {'R':>6} "
              f"{'QA':>6} {'Sum':>6} {'D2T':>6} {'CI':>16} {'n':>5}")
        print("-" * 80)
        for r in runs:
            m = r["metrics"]
            ci = m.get("bootstrap_ci", {})
            pt = m.get("per_task", {})
            ci_str = f"[{ci.get('ci_lower', '')},{ci.get('ci_upper', '')}]"
            print(
                f"{r['method']:<35} "
                f"{m.get('overall_f1', 0):>6.4f} "
                f"{m.get('overall_precision', 0):>6.4f} "
                f"{m.get('overall_recall', 0):>6.4f} "
                f"{pt.get('QA', {}).get('f1', 0):>6.4f} "
                f"{pt.get('Summary', {}).get('f1', 0):>6.4f} "
                f"{pt.get('Data2txt', {}).get('f1', 0):>6.4f} "
                f"{ci_str:>16} "
                f"{m.get('n_samples', 0):>5}"
            )
        print("=" * 80)


if __name__ == "__main__":
    logger = ExperimentLogger()
    logger.print_summary()
