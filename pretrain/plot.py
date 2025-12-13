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
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No results file at {path}")
    df = pd.read_csv(path)
    # compute tokens_per_sec if seq_len + batch_size are present
    if {"seq_len", "batch_size", "mean_step_time"}.issubset(df.columns):
        df["tokens_per_sec"] = (df["seq_len"] * df["batch_size"]) / df["mean_step_time"]
        df["efficiency_tokens_per_sec_per_thread"] = df["tokens_per_sec"] / df["num_threads"].clip(lower=1)
    # add a simple config name for faceting
    if "depth" in df.columns and "width" in df.columns and "num_heads" in df.columns and "seq_len" in df.columns:
        df["config"] = (
            "seq"
            + df["seq_len"].astype(str)
            + "-d"
            + df["depth"].astype(str)
            + "-w"
            + df["width"].astype(str)
            + "-h"
            + df["num_heads"].astype(str)
            + (("-kv" + df["kv_heads"].astype(str)) if "kv_heads" in df.columns else "")
        )
    return df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_filters(df: pd.DataFrame, frameworks: Optional[List[str]], precisions: Optional[List[str]]) -> pd.DataFrame:
    if frameworks:
        df = df[df["framework"].isin(frameworks)]
    if precisions:
        df = df[df["precision"].isin(precisions)]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        col
        for col in [
            "framework",
            "precision",
            "compilation_mode",
            "attention_backend",
            "kv_heads",
            "num_threads",
            "seq_len",
            "depth",
            "width",
            "num_heads",
            "mlp_ratio",
            "dropout",
            "attention_variant",
            "batch_size",
            "config",
        ]
        if col in df.columns
    ]
    agg_dict = {
        "mean_step_time": ("mean_step_time", "mean"),
        "p95_step_time": ("p95_step_time", "mean"),
        "count": ("mean_step_time", "count"),
    }
    if "tokens_per_sec" in df.columns:
        agg_dict["tokens_per_sec"] = ("tokens_per_sec", "mean")
    if "efficiency_tokens_per_sec_per_thread" in df.columns:
        agg_dict["efficiency_tokens_per_sec_per_thread"] = ("efficiency_tokens_per_sec_per_thread", "mean")

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()
    return agg


def plot_threads(df: pd.DataFrame, outdir: Path) -> None:
    if "num_threads" not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x="num_threads",
        y="mean_step_time",
        hue="precision" if "precision" in df.columns else None,
        col="framework" if "framework" in df.columns else None,
        row="config" if "config" in df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("num_threads", "Mean step time (s)")
    g.figure.suptitle("Mean step time vs threads", y=1.02)
    g.savefig(outdir / "mean_step_time_vs_threads.png", dpi=180)
    plt.close(g.figure)


def plot_seq_len(df: pd.DataFrame, outdir: Path) -> None:
    if "seq_len" not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x="seq_len",
        y="mean_step_time",
        hue="precision" if "precision" in df.columns else None,
        col="framework" if "framework" in df.columns else None,
        row="config" if "config" in df.columns else None,
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
        row="config" if "config" in df.columns else None,
        kind="bar",
        order=sorted(df["precision"].unique()),
        sharey=False,
        sharex=True,
    )
    g.set_axis_labels("Precision/Quant", "Mean step time (s)")
    g.figure.suptitle("Mean step time vs precision", y=1.02)
    g.savefig(outdir / "mean_step_time_vs_precision.png", dpi=180)
    plt.close(g.figure)


def plot_tokens(df: pd.DataFrame, outdir: Path) -> None:
    if "tokens_per_sec" not in df.columns or "num_threads" not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x="num_threads",
        y="tokens_per_sec",
        hue="precision" if "precision" in df.columns else None,
        col="framework" if "framework" in df.columns else None,
        row="config" if "config" in df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("num_threads", "Tokens per second")
    g.figure.suptitle("Tokens/sec vs threads", y=1.02)
    g.savefig(outdir / "tokens_per_sec_vs_threads.png", dpi=180)
    plt.close(g.figure)


def plot_efficiency(df: pd.DataFrame, outdir: Path) -> None:
    if "efficiency_tokens_per_sec_per_thread" not in df.columns or "num_threads" not in df.columns:
        return
    g = sns.relplot(
        data=df,
        x="num_threads",
        y="efficiency_tokens_per_sec_per_thread",
        hue="precision" if "precision" in df.columns else None,
        col="framework" if "framework" in df.columns else None,
        row="config" if "config" in df.columns else None,
        kind="line",
        marker="o",
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("num_threads", "Tokens/sec per thread")
    g.figure.suptitle("Scaling efficiency vs threads", y=1.02)
    g.savefig(outdir / "efficiency_vs_threads.png", dpi=180)
    plt.close(g.figure)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default="result/bench_results.csv", help="CSV results file")
    parser.add_argument("--output-dir", type=str, default="result/plots", help="Where to write PNGs")
    parser.add_argument("--frameworks", type=str, default="", help="Comma-separated frameworks to include (default: all)")
    parser.add_argument("--precisions", type=str, default="", help="Comma-separated precisions to include (default: all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        candidate = input_path / "bench_results.csv"
        if candidate.exists():
            input_path = candidate
    outdir = Path(args.output_dir)
    ensure_outdir(outdir)

    df = load_results(input_path)
    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    if frameworks or precisions:
        df = apply_filters(df, frameworks, precisions)
    if df.empty:
        raise SystemExit(f"No data found in {input_path}")

    # basic sorting to make lines nicer
    if "num_threads" in df.columns:
        df = df.sort_values("num_threads")

    df_agg = aggregate(df)

    plot_threads(df_agg, outdir)
    plot_seq_len(df_agg, outdir)
    plot_precision(df_agg, outdir)
    plot_tokens(df_agg, outdir)
    plot_efficiency(df_agg, outdir)

    print(f"[done] plots written to {outdir}")


if __name__ == "__main__":
    main()
