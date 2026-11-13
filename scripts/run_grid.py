#!/usr/bin/env python
"""Train and evaluate every condition x seed: the main experiment.

    python scripts/run_grid.py --data-dir data --seeds 0 1 2 \
        --conditions none random swap swap_nosym swap_fluent

For each run, named <condition>_seed<k>, three scripts are called in order:
  04_train.py                 -> <runs-dir>/<name>/model.pt, train_summary.json, train_log.jsonl
  01_eval_compositional.py    -> <results-dir>/<name>/summary.json and per-item files
  05_eval_downstream.py       -> <results-dir>/<name>/downstream.json
Then train_summary.json and train_log.jsonl are copied to <results-dir>/<name>/ and the
checkpoint is deleted (keep it with --keep-checkpoints).

A step whose output already exists is skipped, so an interrupted grid can simply be
started again. With --gpus 0 1 the runs are spread over two GPUs, one run per GPU at
a time; every run still uses a single GPU, as in the NegCLIP recipe.

Seeds are the outer loop: all conditions finish seed 0 before seed 1 starts, so a
grid that is stopped early still has complete seeds. With --time-budget-hours the
grid stops starting new runs when the next one would not finish in time, so that the
tables can still be built from the finished runs before a session limit is reached.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from compbind import CONDITIONS  # noqa: E402
from compbind.benchmarks import BENCHMARKS  # noqa: E402
from compbind.utils import add_config_option, ensure_dir, load_json  # noqa: E402

print_lock = threading.Lock()


def say(message: str) -> None:
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluation_done(results_dir: str, name: str) -> bool:
    path = os.path.join(results_dir, name, "summary.json")
    return os.path.isfile(path) and all(b in load_json(path).get("benchmarks", {}) for b in BENCHMARKS)


def downstream_done(results_dir: str, name: str, need_retrieval: bool, datasets: list) -> bool:
    path = os.path.join(results_dir, name, "downstream.json")
    if not os.path.isfile(path):
        return False
    saved = load_json(path)
    return all(d in saved.get("zero_shot", {}) for d in datasets) and (
        not need_retrieval or "coco_retrieval" in saved)


def missing_parts(args, name: str) -> tuple:
    """(evaluation missing, downstream missing, training record present) for one run."""
    need_eval = not evaluation_done(args.results_dir, name)
    need_downstream = not args.skip_downstream and not downstream_done(
        args.results_dir, name, not args.skip_retrieval, args.zero_shot)
    have_training_record = os.path.isfile(os.path.join(args.results_dir, name, "train_summary.json"))
    return need_eval, need_downstream, have_training_record


def run_one(args, condition: str, seed: int, gpu) -> str:
    """Train and evaluate one run. Returns "done", "skipped" or "FAILED"."""
    name = f"{condition}_seed{seed}"
    run_dir = ensure_dir(os.path.join(args.runs_dir, name))
    result_dir = ensure_dir(os.path.join(args.results_dir, name))
    model = os.path.join(run_dir, "model.pt")
    need_eval, need_downstream, have_training_record = missing_parts(args, name)
    if not need_eval and not need_downstream and have_training_record:
        return "skipped"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    common = ["--data-dir", args.data_dir, "--batch-size", str(args.batch_size), "--workers", str(args.workers)]
    if args.coco_roots:
        common += ["--coco-roots"] + list(args.coco_roots)
    steps = []
    if have_training_record and not os.path.isfile(model):
        say(f"{name}: the checkpoint was deleted after an earlier evaluation, so the model is trained again "
            "for the missing evaluations. GPU training is not exactly repeatable; use --keep-checkpoints "
            "if you plan to add evaluations later.")
    if not os.path.isfile(model) or not have_training_record:
        train = [sys.executable, os.path.join(HERE, "04_train.py"), *common, "--negatives", condition,
                 "--seed", str(seed), "--epochs", str(args.epochs), "--lr", str(args.lr),
                 "--warmup", str(args.warmup), "--runs-dir", args.runs_dir]
        if args.max_steps is not None:
            train += ["--max-steps", str(args.max_steps)]
        if args.no_amp:
            train += ["--no-amp"]
        steps.append(("train", train))
    if need_eval:
        evaluate = [sys.executable, os.path.join(HERE, "01_eval_compositional.py"), *common, "--model", model,
                    "--name", name, "--out-dir", args.results_dir, "--order-seeds",
                    *[str(s) for s in args.order_seeds]]
        if args.limit is not None:
            evaluate += ["--limit", str(args.limit)]
        steps.append(("evaluate", evaluate))
    if need_downstream:
        downstream = [sys.executable, os.path.join(HERE, "05_eval_downstream.py"), *common, "--model", model,
                      "--name", name, "--out-dir", args.results_dir, "--zero-shot", *args.zero_shot]
        if args.imagenet_root:
            downstream += ["--imagenet-root", args.imagenet_root]
        if args.imagenet_limit is not None:
            downstream += ["--imagenet-limit", str(args.imagenet_limit)]
        if args.skip_retrieval:
            downstream += ["--skip-retrieval"]
        steps.append(("downstream", downstream))

    log_path = os.path.join(run_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as log:
        for label, command in steps:
            say(f"{name}: {label} (GPU {gpu if gpu is not None else 'none'})")
            log.write(f"\n===== {label}: {' '.join(command)}\n")
            log.flush()
            code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code != 0:
                log.flush()
                with open(log_path, "r", encoding="utf-8", errors="replace") as written:
                    tail = "".join(written.readlines()[-25:])
                say(f"{name}: {label} FAILED (exit code {code}). Last lines of {log_path}:\n{tail}")
                return "FAILED"
            if label == "train":   # keep the small training records with the results
                for filename in ("train_summary.json", "train_log.jsonl"):
                    shutil.copy2(os.path.join(run_dir, filename), os.path.join(result_dir, filename))
    if not args.keep_checkpoints and os.path.isfile(model):
        os.remove(model)
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("COMPBIND_DATA", "data"))
    parser.add_argument("--coco-roots", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="*", type=int, default=None,
                        help="GPU ids to use (default: all visible GPUs; none = CPU).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=None, help="Image-loading processes per run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Quick test: stop training early.")
    parser.add_argument("--limit", type=int, default=None, help="Quick test: evaluate on N items.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--order-seeds", nargs="*", type=int, default=[0],
                        help="COCO-Order shuffle seeds for the fine-tuned models.")
    parser.add_argument("--zero-shot", nargs="*", default=["cifar10", "cifar100"])
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--imagenet-limit", type=int, default=None)
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep model.pt after evaluation (about 600 MB per run).")
    parser.add_argument("--time-budget-hours", type=float, default=None,
                        help="Do not start a run that is not expected to finish within this many hours\n"
                             "after the grid started. Protects against the platform's session limit:\n"
                             "runs that are not started are simply missing from the results.")
    parser.add_argument("--expected-run-minutes", type=float, default=45,
                        help="First guess of how long one run takes (used until a run has finished).")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    add_config_option(parser, "grid")
    args = parser.parse_args()

    if args.gpus is None:
        try:
            import torch

            args.gpus = list(range(torch.cuda.device_count()))
        except ImportError:
            args.gpus = []
    slots = args.gpus or [None]
    if args.workers is None:
        args.workers = max(2, (os.cpu_count() or 4) // len(slots))

    jobs: queue.Queue = queue.Queue()
    for seed in args.seeds:
        for condition in args.conditions:
            jobs.put((condition, seed))
    total = jobs.qsize()
    say(f"{total} runs on {'GPU ' + ', '.join(map(str, args.gpus)) if args.gpus else 'CPU'}; "
        f"{args.workers} image-loading processes per run")
    outcome: dict = {}
    grid_start, durations = time.time(), []

    def worker(gpu) -> None:
        while True:
            try:
                condition, seed = jobs.get_nowait()
            except queue.Empty:
                return
            start = time.time()
            need_eval, need_downstream, have_record = missing_parts(args, f"{condition}_seed{seed}")
            pending = need_eval or need_downstream or not have_record
            if pending and args.time_budget_hours is not None:
                expected = max(durations) if durations else 60 * args.expected_run_minutes
                if start - grid_start + expected > 3600 * args.time_budget_hours:
                    outcome[(condition, seed)] = "not started"
                    say(f"{condition}_seed{seed}: not started, it would not finish within the time budget "
                        f"of {args.time_budget_hours:g} h")
                    continue
            try:
                status = run_one(args, condition, seed, gpu)
            except Exception as error:  # keep the other runs going
                status = "FAILED"
                say(f"{condition}_seed{seed}: {type(error).__name__}: {error}")
            outcome[(condition, seed)] = status
            if status == "done":
                durations.append(time.time() - start)
            say(f"{condition}_seed{seed}: {status} in {(time.time() - start) / 60:.1f} min "
                f"({len(outcome)}/{total} runs finished)")

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [f"{c}_seed{s}" for (c, s), status in outcome.items() if status == "FAILED"]
    not_started = [f"{c}_seed{s}" for (c, s), status in outcome.items() if status == "not started"]
    say(f"Grid finished: {sum(v == 'done' for v in outcome.values())} done, "
        f"{sum(v == 'skipped' for v in outcome.values())} skipped, {len(failed)} failed"
        + (f" ({', '.join(failed)})" if failed else "")
        + (f", {len(not_started)} not started because of the time budget ({', '.join(not_started)})"
           if not_started else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
