# pandas использует динамические типы колонок; эти проверки отключены только
# для несовместимых с pandas-stubs операций DataFrame/Series.
# pyright: reportAssignmentType=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    
    try:
        df: pd.DataFrame = pd.read_csv("benchmark_results.csv", comment="#")
    except FileNotFoundError:
        print("Ошибка: Файл benchmark_results.csv не найден.")
        sys.exit(1)
    
    required_columns = {
        "variant",
        "gpus",
        "type",
        "N",
        "BS",
        "avg_ms",
        "ms_per_cell",
        "gflops",
    }
    missing = required_columns - set(df.columns)
    if missing:
        print(
            "Ошибка: CSV имеет старый формат. Не хватает колонок: "
            + ", ".join(sorted(missing))
        )
        print("Сначала запустите новый benchmark.")
        sys.exit(1)
    
    # Benchmark записывает, например:
    # multi_host,2,float,4096,256,...
    df["gpus"] = pd.to_numeric(df["gpus"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["BS"] = pd.to_numeric(df["BS"], errors="coerce")
    df = df.dropna(subset=["gpus", "N", "BS", "avg_ms", "ms_per_cell", "gflops"])
    df["gpus"] = df["gpus"].astype(int)
    df["N"] = df["N"].astype(int)
    df["BS"] = df["BS"].astype(int)
    
    # Для графиков фиксируем одну вычислительную конфигурацию:
    # графики показывают масштабирование по GPU и способ обмена halo-строками.
    PLOT_TYPE = "float"  # benchmark использует float + texture
    
    if PLOT_TYPE not in set(df["type"]):
        print(f"Ошибка: тип {PLOT_TYPE} отсутствует в benchmark_results.csv.")
        sys.exit(1)
    
    plot_df: pd.DataFrame = df[
        (df["type"] == PLOT_TYPE) & df["variant"].isin(["multi_host", "multi_direct"])
    ].copy()
    if plot_df.empty:
        print("Ошибка: в CSV нет вариантов multi_host/multi_direct.")
        sys.exit(1)
    
    df = plot_df
    v_types = sorted(df["type"].unique())
    n_vals = sorted(df["N"].unique())
    gpu_vals = sorted(df["gpus"].unique())
    bs_vals = sorted(df["BS"].unique())
    variants = sorted(df["variant"].unique())
    
    if not v_types or df.empty:
        print("Ошибка: в benchmark_results.csv нет данных для построения графиков.")
        sys.exit(1)
    
    
    # Стиль линии показывает только способ обмена данными.
    def variant_style(variant):
        transfer = "direct" if "direct" in variant else "host"
        linestyle = "--" if transfer == "direct" else "-"
        return linestyle, "o"
    
    
    def variant_label(variant, gpus):
        transfer = "GPU Direct" if "direct" in variant else "host copy"
        return f"{gpus} GPU: {transfer}"
    
    
    def configure_n_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
        ax.set_xlabel("N (сетка N×N)")
    
    
    def plot_by_n(metric, ylabel, filename, title):
        columns = 1 if len(v_types) == 1 else 2
        rows = (len(v_types) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows, columns, figsize=(15 if columns == 2 else 9, 5 * rows), squeeze=False
        )
        axes = axes.ravel()
    
        for index, vtype in enumerate(v_types):
            ax = axes[index]
            type_df = df[df["type"] == vtype]
    
            for gpus in gpu_vals:
                for variant in variants:
                    sub = type_df[
                        (type_df["gpus"] == gpus) & (type_df["variant"] == variant)
                    ].sort_values("N")
                    if sub.empty:
                        continue
                    linestyle, marker = variant_style(variant)
                    ax.plot(
                        sub["N"],
                        sub[metric],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.5,
                        markersize=4,
                        label=variant_label(variant, gpus),
                    )
    
            if len(v_types) > 1:
                ax.set_title(vtype, fontweight="bold")
            ax.set_ylabel(ylabel)
            configure_n_axis(ax)
            ax.grid(True, which="both", alpha=0.25)
    
        for ax in axes[len(v_types) :]:
            ax.remove()
    
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                fontsize=8,
                title="GPU / способ обмена",
            )
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(filename)
    
    
    plot_by_n(
        "gflops",
        "GFLOP/s",
        "plot_throughput.png",
        "CUDA Heat 2D: производительность для разного количества GPU",
    )
    
    max_n = max(n_vals)
    
    # ─── Метрики масштабирования ───
    # Для каждого варианта baseline — результат этого же варианта на 1 GPU.
    base_keys = ["type", "N", "BS", "variant"]
    baseline: pd.DataFrame = (
        df[df["gpus"] == 1][base_keys + ["avg_ms"]]
        .rename(columns={"avg_ms": "avg_ms_1gpu"})
        .drop_duplicates(base_keys)
    )
    scaling_metrics: pd.DataFrame = df.merge(baseline, on=base_keys, how="inner")
    scaling_metrics["speedup"] = scaling_metrics["avg_ms_1gpu"] / scaling_metrics["avg_ms"]
    scaling_metrics["efficiency"] = (
        scaling_metrics["speedup"] / scaling_metrics["gpus"] * 100.0
    )
    
    
    def plot_scaling_metric(metric, ylabel, filename, title, ideal=False):
        columns = 1 if len(v_types) == 1 else 2
        rows = (len(v_types) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows, columns, figsize=(15 if columns == 2 else 9, 5 * rows), squeeze=False
        )
        axes = axes.ravel()
    
        for index, vtype in enumerate(v_types):
            ax = axes[index]
            type_df = scaling_metrics[
                (scaling_metrics["type"] == vtype) & (scaling_metrics["N"] == max_n)
            ]
            for variant in variants:
                sub = type_df[type_df["variant"] == variant].sort_values("gpus")
                if sub.empty:
                    continue
                linestyle, marker = variant_style(variant)
                ax.plot(
                    sub["gpus"],
                    sub[metric],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.7,
                    label=variant_label(variant, 1).split(": ", 1)[1],
                )
    
            if ideal and gpu_vals:
                ax.plot(
                    gpu_vals,
                    gpu_vals,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    label="идеальное ускорение",
                )
            if len(v_types) > 1:
                ax.set_title(vtype, fontweight="bold")
            ax.set_xlabel("Количество GPU")
            ax.set_ylabel(ylabel)
            ax.set_xticks(gpu_vals)
            ax.grid(True, alpha=0.25)
    
        for ax in axes[len(v_types) :]:
            ax.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                fontsize=8,
                title="Способ обмена",
            )
        fig.suptitle(f"{title} (N={max_n})", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(filename)
    
    
    plot_scaling_metric(
        "speedup",
        "Ускорение S(p)",
        "plot_speedup.png",
        "CUDA Heat 2D: ускорение относительно 1 GPU",
        ideal=True,
    )
    plot_scaling_metric(
        "efficiency",
        "Эффективность E(p), %",
        "plot_efficiency.png",
        "CUDA Heat 2D: эффективность масштабирования",
    )
    
    # ─── Host copy против GPU Direct ───
    # Сравнение выполняется для одинаковых type/N/BS/gpus.
    transfer_df: pd.DataFrame = df.copy()
    transfer_df["transfer"] = transfer_df["variant"].map(
        lambda value: "direct" if "direct" in value else "host"
    )
    transfer_keys = ["type", "N", "BS", "gpus"]
    transfer_pivot: pd.DataFrame = (
        transfer_df.groupby(transfer_keys + ["transfer"], as_index=False)["avg_ms"]
        .mean()
        .pivot(index=transfer_keys, columns="transfer", values="avg_ms")
        .reset_index()
    )
    if "host" in transfer_pivot and "direct" in transfer_pivot:
        transfer_pivot["host_direct_speedup"] = (
            transfer_pivot["host"] / transfer_pivot["direct"]
        )
    else:
        transfer_pivot["host_direct_speedup"] = float("nan")
    
    comparison_df = transfer_pivot[
        (transfer_pivot["N"] == max_n) & (transfer_pivot["BS"] == min(bs_vals))
    ]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for vtype, sub in comparison_df.groupby("type"):
        sub = sub.sort_values("gpus")
        if sub["host_direct_speedup"].notna().any():
            ax.plot(
                sub["gpus"],
                sub["host_direct_speedup"],
                marker="o",
                linewidth=1.8,
                label=vtype if len(v_types) > 1 else "GPU Direct",
            )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title(f"Host copy / GPU Direct при N={max_n}", fontweight="bold")
    ax.set_xlabel("Количество GPU")
    ax.set_ylabel("Ускорение GPU Direct")
    ax.set_xticks(gpu_vals)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Тип")
    fig.savefig("plot_host_vs_direct.png", dpi=150)
    plt.close(fig)
    print("plot_host_vs_direct.png")

if __name__ == '__main__':
    main()