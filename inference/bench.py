#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_MODEL_REPOS = [
    "unsloth/gemma-3-270M-it-GGUF",
    "unsloth/gemma-3-1B-it-GGUF",
    "unsloth/gemma-3-4B-it-GGUF",
    "unsloth/gemma-3-12B-it-GGUF",
]

DEFAULT_QUANTIZATIONS = ["Q4_K_M", "Q8_0", "Q2_K_L"]
DEFAULT_BATCH_SIZES = [512, 1024, 2048]
DEFAULT_UBATCH_SIZES = [128, 256, 512]
DEFAULT_PROMPT_TOKENS = [128, 512]
DEFAULT_GEN_TOKENS = [128, 512]


def parse_int_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer in {raw!r}") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gather_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        out = subprocess.run(
            ["lscpu", "-J"],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(out.stdout)
    except Exception:
        try:
            out = subprocess.run(
                ["lscpu"],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in out.stdout.splitlines():
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()
        except Exception:
            info = {}
    return info


def gather_system_context() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_info": gather_cpu_info(),
    }


def default_thread_sweep() -> list[int]:
    cpu_count = os.cpu_count() or 1
    candidates = [1, max(1, cpu_count // 2), cpu_count]
    seen = set()
    deduped: list[int] = []
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def cat_frames() -> list[str]:
    # Simple 2-frame cat walk animation.
    return [
        " /\\_/\\",
        " =^.^=",
        "  > <",
    ]


def render_cat(step: int) -> str:
    frames = [
        [" /\\_/\\", " (=^.^)", "  /   "],
        [" /\\_/\\", " (=^.^)", "   \\  "],
    ]
    frame = frames[step % len(frames)]
    return "\n".join(frame)


def ensure_binary(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"llama-bench not found at {path}")
    return path


@dataclass
class BenchParams:
    model_path: Path
    model_repo: str | None
    quantization: str | None
    threads: int
    batch_size: int
    ubatch_size: int
    n_prompt: int
    n_gen: int
    repetitions: int
    numa: str | None
    priority: int
    progress: bool

    def as_cli_args(self) -> list[str]:
        args = [
            "--model",
            str(self.model_path),
            "--threads",
            str(self.threads),
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--n-prompt",
            str(self.n_prompt),
            "--n-gen",
            str(self.n_gen),
            "--repetitions",
            str(self.repetitions),
        ]
        if self.numa:
            args.extend(["--numa", self.numa])
        if self.priority != 0:
            args.extend(["--prio", str(self.priority)])
        if self.progress:
            args.append("--progress")
        args.extend(["--output", "json"])
        return args


def run_benchmark(binary: Path, params: BenchParams, env: dict[str, str]) -> dict[str, Any]:
    cmd = [str(binary), *params.as_cli_args()]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    parsed_output: Any
    try:
        parsed_output = json.loads(proc.stdout)
    except Exception:
        parsed_output = None

    return {
        "timestamp_utc": utc_now(),
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "parsed": parsed_output,
        "params": asdict(params),
    }


def build_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["LLAMA_THREAD_COUNT"] = str(threads)
    env["GGML_NUM_THREADS"] = str(threads)
    return env


def model_filename(repo_id: str, quantization: str) -> str:
    base = repo_id.split("/")[-1]
    return f"{base}-{quantization}.gguf"


def find_matching_file(folder: Path, pattern: str) -> Path | None:
    matches = list(folder.rglob(pattern))
    if not matches:
        return None
    return sorted(matches)[0]


def resolve_repo_model(
    repo_id: str,
    quantization: str,
    cache_dir: Path,
    download: bool,
) -> Path:
    target_dir = cache_dir / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = model_filename(repo_id, quantization)
    direct_path = target_dir / filename
    if direct_path.exists():
        return direct_path

    if not download:
        raise FileNotFoundError(
            f"Model for {repo_id} with {quantization} not found in {target_dir}; "
            "enable --download-models or place the file manually."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download models automatically. "
            "Install it in the venv and rerun with --download-models."
        ) from exc

    # Download only matching quantization files to limit bandwidth.
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"*{quantization}.gguf"],
    )
    found = find_matching_file(target_dir, f"*{quantization}.gguf")
    if not found:
        raise FileNotFoundError(
            f"Downloaded {repo_id} but did not find a *{quantization}.gguf file in {target_dir}"
        )
    return found


def safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def flatten_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten run results into a list of records with numeric fields ready for plotting."""
    records: list[dict[str, Any]] = []
    for run in result.get("runs", []):
        parsed = run.get("parsed")
        if parsed is None:
            continue
        parsed_list = parsed if isinstance(parsed, list) else [parsed]
        for parsed_entry in parsed_list:
            if not isinstance(parsed_entry, dict):
                continue
            rec = {}
            rec.update(run.get("params", {}))
            model_val = run.get("params", {}).get("model_repo") or run.get("params", {}).get("model_path")
            rec["model_repo"] = str(model_val) if model_val is not None else None
            quant_val = run.get("params", {}).get("quantization")
            rec["quantization"] = str(quant_val) if quant_val is not None else None
            rec["returncode"] = run.get("returncode")
            rec["avg_ts"] = safe_float(parsed_entry.get("avg_ts"))
            rec["avg_ns"] = safe_float(parsed_entry.get("avg_ns"))
            rec["n_threads"] = safe_float(parsed_entry.get("n_threads"))
            rec["n_batch"] = safe_float(parsed_entry.get("n_batch"))
            rec["n_ubatch"] = safe_float(parsed_entry.get("n_ubatch"))
            rec["n_prompt"] = safe_float(parsed_entry.get("n_prompt"))
            rec["n_gen"] = safe_float(parsed_entry.get("n_gen"))
            records.append(rec)
    return records


def aggregate_best(records: list[dict[str, Any]], x_key: str, metric_key: str, group_keys: list[str]) -> dict[tuple[Any, ...], dict[Any, float]]:
    lines: dict[tuple[Any, ...], dict[Any, float]] = {}
    for rec in records:
        metric = rec.get(metric_key)
        x_val = rec.get(x_key)
        if metric is None or x_val is None:
            continue
        group = tuple(rec.get(k) for k in group_keys)
        by_x = lines.setdefault(group, {})
        current = by_x.get(x_val)
        if current is None or metric > current:
            by_x[x_val] = metric
    return lines


def plot_lines(lines: dict[tuple[Any, ...], dict[Any, float]], xlabel: str, ylabel: str, title: str, out_path: Path) -> bool:
    if not lines:
        return False
    plt.figure()
    for group, points in lines.items():
        xs = sorted(points)
        ys = [points[x] for x in xs]
        label = " | ".join([str(g) for g in group if g not in (None, "")])
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True


def generate_plots(result: dict[str, Any], out_dir: Path) -> list[str]:
    if plt is None:
        return []
    records = [r for r in flatten_results(result) if r.get("returncode") == 0]
    plots: list[str] = []
    if not records:
        return plots

    group_keys = ["model_repo", "quantization"]
    throughput_vs_threads = aggregate_best(records, "n_threads", "avg_ts", group_keys)
    if plot_lines(
        throughput_vs_threads,
        xlabel="Threads",
        ylabel="Tokens/s (best across batches)",
        title="Throughput vs Threads",
        out_path=out_dir / "throughput_vs_threads.png",
    ):
        plots.append(str(out_dir / "throughput_vs_threads.png"))

    throughput_vs_batch = aggregate_best(records, "n_batch", "avg_ts", group_keys)
    if plot_lines(
        throughput_vs_batch,
        xlabel="Batch size",
        ylabel="Tokens/s (best across threads)",
        title="Throughput vs Batch size",
        out_path=out_dir / "throughput_vs_batch.png",
    ):
        plots.append(str(out_dir / "throughput_vs_batch.png"))

    latency_vs_threads = aggregate_best(records, "n_threads", "avg_ns", group_keys)
    if plot_lines(
        latency_vs_threads,
        xlabel="Threads",
        ylabel="Avg ns (lower is better, best across batches)",
        title="Latency vs Threads",
        out_path=out_dir / "latency_vs_threads.png",
    ):
        plots.append(str(out_dir / "latency_vs_threads.png"))

    return plots


def iter_model_paths(
    model_paths: list[Path] | None,
    model_repos: list[str] | None,
    quantizations: list[str] | None,
    cache_dir: Path,
    download_models: bool,
) -> Iterable[tuple[Path, str | None, str | None]]:
    if model_paths:
        for path in model_paths:
            yield (path, None, None)
        return

    repos = model_repos or DEFAULT_MODEL_REPOS
    quants = quantizations or DEFAULT_QUANTIZATIONS

    for repo in repos:
        for quant in quants:
            resolved = resolve_repo_model(repo, quant, cache_dir, download_models)
            yield (resolved, repo, quant)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run llama.cpp's llama-bench across parameter sweeps and store JSON results.",
    )
    parser.add_argument(
        "--bench-binary",
        type=Path,
        default=base_dir / "llama.cpp" / "build" / "bin" / "llama-bench",
        help="Path to llama-bench binary.",
    )
    parser.add_argument(
        "--model",
        dest="model_paths",
        action="append",
        type=Path,
        help="Path to a GGUF model; pass multiple times to sweep. If omitted, uses README defaults.",
    )
    parser.add_argument(
        "--model-repo",
        dest="model_repos",
        action="append",
        help="Hugging Face repo IDs to download; defaults to README list if no --model is provided.",
    )
    parser.add_argument(
        "--quantization",
        dest="quantizations",
        action="append",
        help="Quantizations to sweep for repo downloads (default README list).",
    )
    parser.add_argument(
        "--download-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download models via huggingface_hub when missing (default: True).",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=base_dir / "models-cache",
        help="Where to store/download models when using repo IDs.",
    )
    parser.add_argument(
        "--threads",
        type=parse_int_list,
        default=None,
        help="Comma-separated thread counts (default: 1, half, all cores).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=parse_int_list,
        default=None,
        help="Comma-separated batch sizes to test (default sweep).",
    )
    parser.add_argument(
        "--ubatch-size",
        dest="ubatch_sizes",
        type=parse_int_list,
        default=None,
        help="Comma-separated micro-batch sizes to test (default sweep).",
    )
    parser.add_argument(
        "--n-prompt",
        type=parse_int_list,
        default=None,
        help="Prompt token counts (default sweep).",
    )
    parser.add_argument(
        "--n-gen",
        type=parse_int_list,
        default=None,
        help="Generate token counts (default sweep).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="How many repetitions to pass to llama-bench.",
    )
    parser.add_argument(
        "--numa",
        choices=["distribute", "isolate", "numactl"],
        default=None,
        help="Optional NUMA strategy for llama-bench.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Process/thread priority for llama-bench (maps to --prio).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable llama-bench progress indicators.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=base_dir / "result" / "result.json",
        help="Where to write the aggregated JSON results (plots saved alongside).",
    )
    args = parser.parse_args()

    bench_bin = ensure_binary(args.bench_binary)
    threads = args.threads or default_thread_sweep()
    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES
    ubatch_sizes = args.ubatch_sizes or DEFAULT_UBATCH_SIZES
    prompts = args.n_prompt or DEFAULT_PROMPT_TOKENS
    gens = args.n_gen or DEFAULT_GEN_TOKENS

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "bench_binary": str(bench_bin),
        "system": gather_system_context(),
        "runs": [],
        "plots": [],
    }

    def persist() -> None:
        with args.result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
            f.write("\n")

    combos = [
        (model_path, model_repo, quantization, thr, batch, ubatch, prompt_tokens, gen_tokens)
        for model_path, model_repo, quantization in iter_model_paths(
            args.model_paths,
            args.model_repos,
            args.quantizations,
            args.model_cache,
            args.download_models,
        )
        for thr in threads
        for batch in batch_sizes
        for ubatch in ubatch_sizes
        for prompt_tokens in prompts
        for gen_tokens in gens
    ]

    total = len(combos)
    start_time = time.time()

    for idx, (model_path, model_repo, quantization, thr, batch, ubatch, prompt_tokens, gen_tokens) in enumerate(combos, start=1):
        # animate cat walking inline in the console
        cat_frame = render_cat(idx)
        print(cat_frame, end="\r", flush=True)

        params = BenchParams(
            model_path=model_path,
            model_repo=model_repo,
            quantization=quantization,
            threads=thr,
            batch_size=batch,
            ubatch_size=ubatch,
            n_prompt=prompt_tokens,
            n_gen=gen_tokens,
            repetitions=args.repetitions,
            numa=args.numa,
            priority=args.priority,
            progress=args.progress,
        )
        env = build_env(thr)
        run_result = run_benchmark(bench_bin, params, env)
        run_result["run_index"] = len(result["runs"])
        result["runs"].append(run_result)
        persist()

        elapsed = time.time() - start_time
        avg_per = elapsed / idx
        remaining = (total - idx) * avg_per
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        cat = render_cat(idx)
        print(
            f"[{idx}/{total} | ETA {eta_str}] "
            f"{model_repo or model_path.name} {quantization or ''} "
            f"threads={thr} batch={batch} ubatch={ubatch} "
            f"prompt={prompt_tokens} gen={gen_tokens} "
            f"rc={run_result['returncode']}\n{cat}\n"
        )

    result["plots"] = generate_plots(result, args.result_path.parent)
    persist()


if __name__ == "__main__":
    main()
import time
