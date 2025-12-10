#!/usr/bin/env python3
"""
Plot benchmark relationships from result/bench_results.csv (or another CSV).

Generates a handful of PNGs under result/plots/:
  - mean_step_time vs num_threads (by framework/precision)
  - mean_step_time vs seq_len (faceted by framework, hue=precision)
  - mean_step_time vs precision (bar, faceted by framework)
  - tokens_per_sec vs num_threads (by framework/precision)

You can point to a different CSV with --input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No results file at {path}")
    df = pd.read_csv(path)
    # durations may be JSON-encoded lists; keep as-is for now
    # compute tokens_per_sec if seq_len + batch_size are present
    if {"seq_len", "batch_size", "mean_step_time"}.issubset(df.columns):
        df["tokens_per_sec"] = (df["seq_len"] * df["batch_size"]) / df["mean_step_time"]
    return df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_threads(df: pd.DataFrame, outdir: Path) -> None:
    if "num_threads" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="num_threads",
        y="mean_step_time",
        hue="framework",
        style="precision" if "precision" in df.columns else None,
        markers=True,
    )
    plt.title("Mean step time vs threads")
    plt.ylabel("Mean step time (s)")
    plt.xlabel("num_threads")
    plt.legend(title="framework / precision", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "mean_step_time_vs_threads.png", dpi=180)
    plt.close()


def plot_seq_len(df: pd.DataFrame, outdir: Path) -> None:
    if "seq_len" not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x="seq_len",
        y="mean_step_time",
        hue="precision" if "precision" in df.columns else None,
        col="framework" if "framework" in df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("Sequence length", "Mean step time (s)")
    g.figure.suptitle("Mean step time vs sequence length", y=1.02)
    g.savefig(outdir / "mean_step_time_vs_seq_len.png", dpi=180)
    plt.close(g.figure)


def plot_precision(df: pd.DataFrame, outdir: Path) -> None:
    if "precision" not in df.columns:
        return
    g = sns.catplot(
        data=df,
        x="precision",
        y="mean_step_time",
        col="framework" if "framework" in df.columns else None,
        kind="bar",
        order=sorted(df["precision"].unique()),
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("Precision/Quant", "Mean step time (s)")
    g.figure.suptitle("Mean step time vs precision", y=1.02)
    g.savefig(outdir / "mean_step_time_vs_precision.png", dpi=180)
    plt.close(g.figure)


def plot_tokens(df: pd.DataFrame, outdir: Path) -> None:
    if "tokens_per_sec" not in df.columns or "num_threads" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="num_threads",
        y="tokens_per_sec",
        hue="framework",
        style="precision" if "precision" in df.columns else None,
        markers=True,
    )
    plt.title("Tokens/sec vs threads")
    plt.ylabel("Tokens per second")
    plt.xlabel("num_threads")
    plt.legend(title="framework / precision", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "tokens_per_sec_vs_threads.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default="result/bench_results.csv", help="CSV results file")
    parser.add_argument("--output-dir", type=str, default="result/plots", help="Where to write PNGs")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    ensure_outdir(outdir)

    df = load_results(input_path)
    if df.empty:
        raise SystemExit(f"No data found in {input_path}")

    # basic sorting to make lines nicer
    if "num_threads" in df.columns:
        df = df.sort_values("num_threads")

    plot_threads(df, outdir)
    plot_seq_len(df, outdir)
    plot_precision(df, outdir)
    plot_tokens(df, outdir)

    print(f"[done] plots written to {outdir}")


if __name__ == "__main__":
    main()
