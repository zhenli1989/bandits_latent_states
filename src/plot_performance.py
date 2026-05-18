# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from src.export_table import export_summary_table


DISPLAY_NAME_MAP = {
    "LinUCB": "Plain LinUCB",
    "Box A": "LinUCB-Belief-Complex",
    "Box B": "LinUCB-Belief-Simplified",
}


def _display_algorithm_name(algo):
    return DISPLAY_NAME_MAP.get(algo, algo)


def _get_sample_indices(T_range, sampling=None):
    if sampling is None:
        return np.arange(len(T_range))

    if isinstance(sampling, (int, np.integer)):
        step = max(int(sampling), 1)
    elif sampling >= 1:
        step = max(int(sampling), 1)
    else:
        step = max(int(round(1 / sampling)), 1)

    index_choose = np.array([i for i, t in enumerate(T_range) if (t % step) == 0], dtype=int)
    if len(index_choose) == 0 or index_choose[0] != 0:
        index_choose = np.insert(index_choose, 0, 0)
    if index_choose[-1] != len(T_range) - 1:
        index_choose = np.append(index_choose, len(T_range) - 1)
    return np.unique(index_choose)


def _build_color_map(selected_df):
    color_map = {}

    if (selected_df["algorithm"] == "LinUCB").any():
        color_map[("LinUCB", 1)] = "#8c2d04"

    for algo, cmap_name in [("Box A", "Blues"), ("Box B", "Greens")]:
        batches = sorted(selected_df.loc[selected_df["algorithm"] == algo, "batch_size"].unique().tolist())
        if len(batches) == 0:
            continue
        cmap = cm.get_cmap(cmap_name)
        shades = np.linspace(0.58, 0.9, len(batches))
        for shade, batch in zip(shades, batches):
            color_map[(algo, batch)] = cmap(shade)

    return color_map


def _build_marker_map(selected_df):
    marker_map = {}
    if (selected_df["algorithm"] == "LinUCB").any():
        marker_map[("LinUCB", 1)] = "o"

    box_a_markers = ["^", "s", "D", "P", "X", "*"]
    box_b_markers = ["v", ">", "<", "h", "8", "p"]

    for algo, markers in [("Box A", box_a_markers), ("Box B", box_b_markers)]:
        batches = sorted(selected_df.loc[selected_df["algorithm"] == algo, "batch_size"].unique().tolist())
        for marker, batch in zip(markers, batches):
            marker_map[(algo, batch)] = marker

    return marker_map


def _get_panel_shape(n_panels):
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _build_c_style_maps(c_values, cmap_name="Blues"):
    c_values = sorted(c_values)
    cmap = cm.get_cmap(cmap_name)
    shades = np.linspace(0.35, 0.9, len(c_values))

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "*"]
    color_map = {c: cmap(shade) for c, shade in zip(c_values, shades)}
    marker_map = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(c_values)}
    return color_map, marker_map

def plot_optimal_regret_curve_by_strategy(path_models=".",
                                          df_summary=None,
                                          path_df_summary=None,
                                          seeds=None,
                                          number_rounds=50000,
                                          list_ucb_multipler=None,
                                          list_lmd=None,
                                          list_batch=None,
                                          algorithms=None,
                                          with_union_bound=False,
                                          sampling=None,
                                          include_ci=True,
                                          criterion_round=None,
                                          path_save=None,
                                          ax=None,
                                          maximum_y_value=400,
                                          line_width=1.25,
                                          use_parallel=True,
                                          n_jobs=-1,
                                          parallel_backend="loky"):
    """
    Plot one optimal regret curve per strategy.

    Selection rule:
    - LinUCB: best terminal average_belief_regret over (C, lambda)
    - Box A:  best terminal average_belief_regret over (batch_size, C, lambda)
    - Box B:  best terminal average_belief_regret over (batch_size, C, lambda)

    Output:
    - one curve per strategy
    - average belief regret
    - optional +/- 2 SE band
    """

    if algorithms is None:
        algorithms = ["LinUCB", "Box A", "Box B"]

    if criterion_round is None:
        criterion_round = number_rounds

    if df_summary is None:
        if path_df_summary is not None:
            df_summary = pd.read_csv(path_df_summary)
        else:
            df_summary = export_summary_table(
                path_models=path_models,
                seeds=seeds,
                number_rounds=number_rounds,
                list_ucb_multipler=list_ucb_multipler,
                list_lmd=list_lmd,
                list_batch=list_batch,
                algorithms=algorithms,
                with_union_bound=with_union_bound,
                path_save=None,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )

    df_summary = df_summary.copy()

    display_name_map = {
        "LinUCB": "Plain LinUCB",
        "Box A": "Staged LinUCB on Estimated Beliefs",
        "Box B": "LinUCB on Estimated Beliefs (without stages)",
    }

    color_map = {
        "LinUCB": "#8c2d04",
        "Box A": "steelblue",
        "Box B": "darkgreen",
    }

    marker_map = {
        "LinUCB": "o",
        "Box A": "D",
        "Box B": "v",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    else:
        fig = ax.figure

    df_terminal = df_summary[df_summary["round"] == criterion_round].copy()

    list_selected_rows = []
    plotted_lower = []
    plotted_upper = []

    for algo in algorithms:
        df_algo_terminal = df_terminal[df_terminal["algorithm"] == algo].copy()
        if len(df_algo_terminal) == 0:
            continue

        idx_best = df_algo_terminal["average_belief_regret"].idxmin()
        row_best = df_algo_terminal.loc[idx_best]
        list_selected_rows.append(row_best)

        batch_size = row_best["batch_size"]
        c_value = row_best["C"]
        lmd_value = row_best["lambda"]

        df_curve = df_summary.loc[
            (df_summary["algorithm"] == algo)
            & (df_summary["batch_size"] == batch_size)
            & (df_summary["C"] == c_value)
            & (df_summary["lambda"] == lmd_value)
        ].copy().sort_values("round")

        T_range_ = df_curve["round"].to_numpy()
        y_ = df_curve["average_belief_regret"].to_numpy()

        if include_ci:
            se_ = df_curve["se_belief_regret"].to_numpy()
            lower_ = y_ - 2 * se_
            upper_ = y_ + 2 * se_

        if sampling is not None:
            if isinstance(sampling, int):
                index_choose = np.arange(0, len(T_range_), sampling)
            elif isinstance(sampling, float):
                if (sampling <= 0) or (sampling > 1):
                    raise ValueError("If sampling is a float, it must be in (0,1].")
                n_keep = max(2, int(np.ceil(len(T_range_) * sampling)))
                index_choose = np.unique(np.linspace(0, len(T_range_) - 1, n_keep, dtype=int))
            else:
                raise ValueError("sampling must be None, int, or float.")

            T_range_ = T_range_[index_choose]
            y_ = y_[index_choose]
            if include_ci:
                lower_ = lower_[index_choose]
                upper_ = upper_[index_choose]

        if algo == "Box A":
            label = f"{display_name_map[algo]} ($\\ell$={int(batch_size)})"
        else:
            label = display_name_map.get(algo, algo)

        ax.plot(
            T_range_,
            y_,
            label=label,
            color=color_map.get(algo, None),
            marker=marker_map.get(algo, None),
            linewidth=line_width,
            markersize=5,
            markevery=max(1, len(T_range_) // 15),
        )

        if include_ci:
            ax.fill_between(
                T_range_,
                lower_,
                upper_,
                color=color_map.get(algo, None),
                alpha=0.18,
            )
            plotted_lower.append(lower_)
            plotted_upper.append(upper_)
        else:
            plotted_lower.append(y_)
            plotted_upper.append(y_)

    if len(plotted_lower) > 0:
        lower_all = np.concatenate(plotted_lower)
        upper_all = np.concatenate(plotted_upper)

        y_min = float(np.nanmin(lower_all))
        y_max = float(np.nanmax(upper_all))
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

        ax.set_xlim(int(df_summary["round"].min()), int(df_summary["round"].max()))
        y_upper = y_max + margin
        if maximum_y_value is not None:
            y_upper = min(y_upper, maximum_y_value)
        ax.set_ylim(max(0, y_min - margin), y_upper)

    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=1, frameon=True)

    fig.tight_layout()

    if path_save is not None:
        fig.savefig(path_save, dpi=300, bbox_inches="tight")

    df_selected = pd.DataFrame(list_selected_rows).reset_index(drop=True)

    return fig, ax, df_selected

def plot_best_regret_curves(number_rounds,
                            list_ucb_multipler,
                            list_lmd,
                            list_batch,
                            algorithms,
                            seeds,
                            df_summary=None,
                            path_df_summary=None,
                            with_union_bound=False,
                            path_models=".",
                            sampling=None,
                            include_ci=True,
                            path_save=None,
                            ax=None,
                            maximum_y_value=400,
                            line_width=1.25,
                            use_parallel=True,
                            n_jobs=-1,
                            parallel_backend="loky"):
    if algorithms is None:
        algorithms = ["LinUCB", "Box A", "Box B"]

    if df_summary is None:
        if path_df_summary is not None:
            df_summary = pd.read_csv(path_df_summary)
        else:
            df_summary = export_summary_table(
                path_models=path_models,
                seeds=seeds,
                number_rounds=number_rounds,
                list_ucb_multipler=list_ucb_multipler,
                list_lmd=list_lmd,
                list_batch=list_batch,
                algorithms=algorithms,
                with_union_bound=with_union_bound,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )

    required_cols = {
        "round", "algorithm", "batch_size", "lambda", "C",
        "average_belief_regret", "se_belief_regret"
    }
    missing = required_cols - set(df_summary.columns)

    if missing:
        raise ValueError(f"df_summary is missing required columns: {sorted(missing)}")

    df_terminal = df_summary.loc[
        (df_summary["round"] == number_rounds) &
        (df_summary["algorithm"].isin(algorithms))
    ].copy()

    selected_rows = []

    if "LinUCB" in algorithms:
        df_lin = df_terminal.loc[df_terminal["algorithm"] == "LinUCB"].copy()
        if not df_lin.empty:
            idx = df_lin["average_belief_regret"].idxmin()
            selected_rows.append(df_lin.loc[idx])

    for algo in ["Box A", "Box B"]:
        if algo not in algorithms:
            continue
        for batch in list_batch:
            df_sub = df_terminal.loc[
                (df_terminal["algorithm"] == algo) &
                (df_terminal["batch_size"] == batch)
            ].copy()
            if df_sub.empty:
                continue
            idx = df_sub["average_belief_regret"].idxmin()
            selected_rows.append(df_sub.loc[idx])

    if len(selected_rows) == 0:
        raise ValueError("No matching scenarios found for plotting.")

    df_selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure

    color_map = _build_color_map(df_selected)
    marker_map = _build_marker_map(df_selected)

    plotted_lower = []
    plotted_upper = []

    for _, row in df_selected.iterrows():
        mask = (
            (df_summary["algorithm"] == row["algorithm"]) &
            (df_summary["batch_size"] == row["batch_size"]) &
            (df_summary["lambda"] == row["lambda"]) &
            (df_summary["C"] == row["C"])
        )
        df_curve = df_summary.loc[mask].sort_values("round").reset_index(drop=True)

        T_range = df_curve["round"].to_numpy()
        y = df_curve["average_belief_regret"].to_numpy()

        if include_ci:
            se = df_curve["se_belief_regret"].to_numpy()
            lower = y - 2 * se
            upper = y + 2 * se

        index_choose = _get_sample_indices(T_range, sampling=sampling)
        T_range_ = T_range[index_choose]
        y_ = y[index_choose]
        if include_ci:
            lower_ = lower[index_choose]
            upper_ = upper[index_choose]
            plotted_lower.append(lower_)
            plotted_upper.append(upper_)
        else:
            plotted_lower.append(y_)
            plotted_upper.append(y_)

        algo = row["algorithm"]
        batch = int(row["batch_size"])
        c_value = row["C"]
        lmd_value = row["lambda"]
        algo_name = _display_algorithm_name(algo)

        if algo == "LinUCB":
            label = f"{algo_name} | C={c_value:g}, $\lambda$={lmd_value:g}"
        else:
            label = f"{algo_name} | $\ell$={batch}, C={c_value:g}, $\lambda$={lmd_value:g}"

        color = color_map[(algo, batch)]
        marker = marker_map[(algo, batch)]
        markevery = max(len(T_range_) // 8, 1)

        ax.plot(
            T_range_,
            y_,
            label=label,
            color=color,
            marker=marker,
            linewidth=line_width,
            markevery=markevery,
            markersize=5,
        )

        if include_ci:
            ax.fill_between(T_range_, lower_, upper_, color=color, alpha=0.15)

    lower_all = np.concatenate(plotted_lower)
    upper_all = np.concatenate(plotted_upper)
    y_min = float(np.nanmin(lower_all))
    y_max = float(np.nanmax(upper_all))
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    ax.set_xlim(int(df_summary["round"].min()), int(df_summary["round"].max()))
    y_upper = y_max + margin
    if maximum_y_value is not None:
        y_upper = min(y_upper, maximum_y_value)
    ax.set_ylim(max(0, y_min - margin), y_upper)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=12, ncol=1, frameon=False)

    if path_save is not None:
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        fig.savefig(path_save, dpi=300, bbox_inches="tight")

    return fig, ax, df_selected


def plot_best_regret_by_c_per_strategy(number_rounds,
                                       list_ucb_multipler,
                                       list_lmd,
                                       list_batch,
                                       algorithms,
                                       seeds,
                                       df_summary=None,
                                       path_df_summary=None,
                                       path_models=".",
                                       with_union_bound=False,
                                       sampling=None,
                                       include_ci=True,
                                       path_save_dir=None,
                                       file_ext="pdf",
                                       maximum_y_value=400,
                                       line_width=1.25,
                                       use_parallel=True,
                                       n_jobs=-1,
                                       parallel_backend="loky"):
    if df_summary is None:
        if path_df_summary is not None:
            df_summary = pd.read_csv(path_df_summary)
        else:
            df_summary = export_summary_table(
                path_models=path_models,
                seeds=seeds,
                number_rounds=number_rounds,
                list_ucb_multipler=list_ucb_multipler,
                list_lmd=list_lmd,
                list_batch=list_batch,
                algorithms=algorithms,
                with_union_bound=with_union_bound,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )

    required_cols = {
        "round", "algorithm", "batch_size", "lambda", "C",
        "average_belief_regret", "se_belief_regret"
    }
    missing = required_cols - set(df_summary.columns)
    if missing:
        raise ValueError(f"df_summary is missing required columns: {sorted(missing)}")

    c_values_global = sorted(list_ucb_multipler)
    color_map_c, marker_map_c = _build_c_style_maps(c_values_global, cmap_name="Blues")

    figs = {}
    selected_rows = []

    for algo in algorithms:
        df_algo = df_summary.loc[df_summary["algorithm"] == algo].copy()
        if df_algo.empty:
            continue

        if algo == "LinUCB":
            batches = [1]
        else:
            batches = [b for b in list_batch if b in df_algo["batch_size"].unique().tolist()]

        n_panels = len(batches)
        n_rows, n_cols = _get_panel_shape(n_panels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        legend_handles = None
        legend_labels = None

        for ax_idx, batch in enumerate(batches):
            ax = axes_flat[ax_idx]

            df_term = df_algo.loc[
                (df_algo["round"] == number_rounds) &
                (df_algo["batch_size"] == batch)
            ].copy()

            if df_term.empty:
                ax.set_visible(False)
                continue

            c_values_batch = sorted(df_term["C"].unique().tolist())
            plotted_lower = []
            plotted_upper = []

            for c_value in c_values_batch:
                df_c = df_term.loc[df_term["C"] == c_value].copy()
                idx = df_c["average_belief_regret"].idxmin()
                row = df_c.loc[idx]
                selected_rows.append(row)

                df_curve = df_algo.loc[
                    (df_algo["batch_size"] == batch) &
                    (df_algo["C"] == row["C"]) &
                    (df_algo["lambda"] == row["lambda"])
                ].sort_values("round").reset_index(drop=True)

                T_range = df_curve["round"].to_numpy()
                y = df_curve["average_belief_regret"].to_numpy()

                if include_ci:
                    se = df_curve["se_belief_regret"].to_numpy()
                    lower = y - 2 * se
                    upper = y + 2 * se

                index_choose = _get_sample_indices(T_range, sampling=sampling)
                T_range_ = T_range[index_choose]
                y_ = y[index_choose]

                if include_ci:
                    lower_ = lower[index_choose]
                    upper_ = upper[index_choose]
                    plotted_lower.append(lower_)
                    plotted_upper.append(upper_)
                else:
                    plotted_lower.append(y_)
                    plotted_upper.append(y_)

                color = color_map_c[c_value]
                marker = marker_map_c[c_value]
                markevery = max(len(T_range_) // 8, 1)

                label = f"C={c_value:g}"
                ax.plot(
                    T_range_,
                    y_,
                    color=color,
                    marker=marker,
                    linewidth=line_width,
                    markersize=4.5,
                    markevery=markevery,
                    label=label,
                )

                if include_ci:
                    ax.fill_between(T_range_, lower_, upper_, color=color, alpha=0.15)

            lower_all = np.concatenate(plotted_lower)
            upper_all = np.concatenate(plotted_upper)
            y_min = float(np.nanmin(lower_all))
            y_max = float(np.nanmax(upper_all))
            margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

            ax.set_xlim(int(df_algo["round"].min()), int(df_algo["round"].max()))
            y_upper = y_max + margin
            if maximum_y_value is not None:
                y_upper = min(y_upper, maximum_y_value)
            ax.set_ylim(max(0, y_min - margin), y_upper)

            if algo == "LinUCB":
                ax.text(
                    0.01, 0.98,
                    "Plain LinUCB",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="semibold",
                    va="top", ha="left",
                    color="dimgray",
                    bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.25"),
                )
            else:
                ax.text(
                    0.01, 0.98,
                    f"{_display_algorithm_name(algo)}, $\ell$ = {batch}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="semibold",
                    va="top", ha="left",
                    color="dimgray",
                    bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.25"),
                )
            ax.grid(alpha=0.2)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        if algo == "LinUCB":
            legend_y = 0.99
            rect_top = 0.92
            ncol_legend = min(len(legend_labels), 3)
        else:
            legend_y = 0.96
            rect_top = 0.92
            ncol_legend = min(len(legend_labels), 3)

        if legend_handles is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=ncol_legend,
                fontsize=10,
                frameon=False,
            )

        fig.tight_layout(rect=[0, 0, 1, rect_top])

        if path_save_dir is not None:
            os.makedirs(path_save_dir, exist_ok=True)
            algo_slug = _display_algorithm_name(algo).replace("-", "_").replace(" ", "_")
            fig.savefig(
                f"{path_save_dir}/best_regret_by_C_{algo_slug}.{file_ext}",
                dpi=300,
                bbox_inches="tight",
            )

        figs[algo] = fig

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return figs, selected_df
