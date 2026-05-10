# coding: utf-8
import os
import numpy as np
import pandas as pd
from joblib import load, Parallel, delayed
from openpyxl.styles import Font
from src.utils import format_lmd


DISPLAY_NAME_MAP = {
    "LinUCB": "Plain LinUCB",
    "Box A": "LinUCB-Belief-Complex",
    "Box B": "LinUCB-Belief-Simplified",
}


def _display_algorithm_name(algo):
    return DISPLAY_NAME_MAP.get(algo, algo)


def format_c(value):
    return "0" + "{:f}".format(value).rstrip("0").rstrip(".")[2:]


def _latex_escape(x):
    text = str(x)
    repl = {
        '&': r'\&',
        '%': r'\%',
        '_': r'\_',
        '#': r'\#',
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _latex_float(x):
    try:
        val = float(x)
    except Exception:
        return _latex_escape(x)
    return f"{val:.6g}"


def _format_terminal_cell(avg_value, se_value, decimals=0):
    ci_value = 2 * se_value
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(avg_value)} ({fmt.format(ci_value)})"


def _build_terminal_pivot(df_terminal, algorithm, batch_size=None, decimals=0):
    if batch_size is not None:
        df_terminal = df_terminal[df_terminal["batch_size"] == batch_size].copy()

    df_terminal = df_terminal.sort_values(["lambda", "C"]).copy()

    avg_table = (
        df_terminal
        .pivot(index="lambda", columns="C", values="average_belief_regret")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    se_table = (
        df_terminal
        .pivot(index="lambda", columns="C", values="se_belief_regret")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    best_mask = avg_table.copy().astype(bool)
    for col in avg_table.columns:
        col_min = avg_table[col].min()
        best_mask[col] = np.isclose(avg_table[col], col_min)

    table = avg_table.copy().astype(object)
    for idx in avg_table.index:
        for col in avg_table.columns:
            table.loc[idx, col] = _format_terminal_cell(
                avg_table.loc[idx, col],
                se_table.loc[idx, col],
                decimals=decimals
            )

    table.index.name = r"$\lambda$ / $C$"
    table.columns.name = None
    return table, best_mask


def load_one_seed_result(path_models,
                         seed,
                         algorithm,
                         batch_size,
                         c_value,
                         lmd_value,
                         prefix_union_bound=""):
    c_str = format_c(c_value)
    lmd_str = format_lmd(lmd_value)

    benchmark = load(f"{path_models}/random_seed_{seed}/dict_kpi_rewards.pkl")

    if algorithm == "LinUCB":
        obj = load(
            f"{path_models}/random_seed_{seed}/LinUCB_C{c_str}_Lmd{lmd_str}.pkl"
        )
        batch_size_out = 1

    elif algorithm == "Box A":
        obj = load(
            f"{path_models}/random_seed_{seed}/BoxA_B{batch_size}_C{c_str}_Lmd{lmd_str}{prefix_union_bound}.pkl"
        )
        batch_size_out = batch_size

    elif algorithm == "Box B":
        obj = load(
            f"{path_models}/random_seed_{seed}/BoxB_B{batch_size}_C{c_str}_Lmd{lmd_str}{prefix_union_bound}.pkl"
        )
        batch_size_out = batch_size

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    realized_reward = np.cumsum(np.array(obj.rewards))
    belief_regret = np.cumsum(
        benchmark["optim_believed_reward_belief"] - np.array(obj.believed_rewards)
    )

    return {
        "algorithm": algorithm,
        "batch_size": batch_size_out,
        "lambda": lmd_value,
        "C": c_value,
        "realized_reward": realized_reward,
        "belief_regret": belief_regret,
    }


def aggregate_one_scenario(path_models,
                           seeds,
                           algorithm,
                           batch_size,
                           c_value,
                           lmd_value,
                           number_rounds,
                           prefix_union_bound="",
                           use_parallel=True,
                           n_jobs=-1,
                           parallel_backend="loky"):
    if use_parallel:
        loaded = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(load_one_seed_result)(
                path_models=path_models,
                seed=seed,
                algorithm=algorithm,
                batch_size=batch_size,
                c_value=c_value,
                lmd_value=lmd_value,
                prefix_union_bound=prefix_union_bound,
            )
            for seed in seeds
        )
    else:
        loaded = [
            load_one_seed_result(
                path_models=path_models,
                seed=seed,
                algorithm=algorithm,
                batch_size=batch_size,
                c_value=c_value,
                lmd_value=lmd_value,
                prefix_union_bound=prefix_union_bound,
            )
            for seed in seeds
        ]

    mat_reward = np.column_stack([x["realized_reward"] for x in loaded])
    mat_regret = np.column_stack([x["belief_regret"] for x in loaded])

    n = len(seeds)

    df_out = pd.DataFrame({
        "round": np.arange(1, number_rounds + 1),
        "algorithm": algorithm,
        "batch_size": loaded[0]["batch_size"],
        "lambda": lmd_value,
        "C": c_value,
        "average_realized_reward": mat_reward.mean(axis=1),
        "se_realized_reward": mat_reward.std(axis=1, ddof=0) / np.sqrt(n),
        "average_belief_regret": mat_regret.mean(axis=1),
        "se_belief_regret": mat_regret.std(axis=1, ddof=0) / np.sqrt(n),
    })

    return df_out


def export_summary_table(path_models=".",
                         seeds=None,
                         number_rounds=50000,
                         list_ucb_multipler=None,
                         list_lmd=None,
                         list_batch=None,
                         algorithms=None,
                         with_union_bound=False,
                         path_save=None,
                         use_parallel=True,
                         n_jobs=-1,
                         parallel_backend="loky"):

    valid_algorithms = {"LinUCB", "Box A", "Box B"}
    invalid_algorithms = set(algorithms) - valid_algorithms
    if len(invalid_algorithms) > 0:
        raise ValueError(f"Unknown algorithms: {sorted(invalid_algorithms)}")

    prefix_union_bound = "_ub" if with_union_bound else ""

    scenario_list = []

    if "LinUCB" in algorithms:
        for c_value in list_ucb_multipler:
            for lmd_value in list_lmd:
                scenario_list.append(("LinUCB", 1, c_value, lmd_value))

    if "Box A" in algorithms:
        for batch_size in list_batch:
            for c_value in list_ucb_multipler:
                for lmd_value in list_lmd:
                    scenario_list.append(("Box A", batch_size, c_value, lmd_value))

    if "Box B" in algorithms:
        for batch_size in list_batch:
            for c_value in list_ucb_multipler:
                for lmd_value in list_lmd:
                    scenario_list.append(("Box B", batch_size, c_value, lmd_value))

    if use_parallel:
        df_list = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(aggregate_one_scenario)(
                path_models=path_models,
                seeds=seeds,
                algorithm=algorithm,
                batch_size=batch_size,
                c_value=c_value,
                lmd_value=lmd_value,
                number_rounds=number_rounds,
                prefix_union_bound=prefix_union_bound,
                use_parallel=False,
            )
            for algorithm, batch_size, c_value, lmd_value in scenario_list
        )
    else:
        df_list = [
            aggregate_one_scenario(
                path_models=path_models,
                seeds=seeds,
                algorithm=algorithm,
                batch_size=batch_size,
                c_value=c_value,
                lmd_value=lmd_value,
                number_rounds=number_rounds,
                prefix_union_bound=prefix_union_bound,
                use_parallel=False,
            )
            for algorithm, batch_size, c_value, lmd_value in scenario_list
        ]

    df_final = pd.concat(df_list, axis=0).reset_index(drop=True)

    if path_save is None:
        df_final[df_final['round'] == number_rounds].to_csv(path_save, index=False)
    return df_final


def _table_to_latex_block(table_df, best_mask, caption, label, size_command="\\small"):
    cols = table_df.columns.tolist()
    align = "l" + "c" * len(cols)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\centering")
    lines.append(size_command)
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")

    header = [table_df.index.name if table_df.index.name is not None else ""] + [_latex_float(c) for c in cols]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for idx, row in table_df.iterrows():
        row_vals = [_latex_float(idx)]
        for col in cols:
            cell_text = _latex_escape(row[col])
            if best_mask.loc[idx, col]:
                cell_text = rf"\textbf{{{cell_text}}}"
            row_vals.append(cell_text)
        lines.append(" & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def export_terminal_regret_tables(number_rounds,
                                  list_ucb_multipler,
                                  list_lmd,
                                  list_batch,
                                  algorithms,
                                  seeds,
                                  path_models=".",
                                  df_summary=None,
                                  path_df_summary=None,
                                  with_union_bound=False,
                                  decimals=0,
                                  path_excel=None,
                                  path_latex_txt=None,
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
                path_save=None,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )

    df_terminal = df_summary[df_summary["round"] == number_rounds].copy()
    if len(df_terminal) == 0:
        raise ValueError(f"No rows found at round={number_rounds}.")

    if list_batch is None:
        list_batch = sorted(
            df_terminal.loc[
                df_terminal["algorithm"].isin(["Box A", "Box B"]),
                "batch_size"
            ].dropna().unique().tolist()
        )

    dict_tables = {}
    dict_best_masks = {}
    latex_blocks = []

    if "LinUCB" in algorithms:
        df_lin = df_terminal[df_terminal["algorithm"] == "LinUCB"].copy()
        table_lin, best_mask_lin = _build_terminal_pivot(
            df_lin,
            algorithm="LinUCB",
            batch_size=1,
            decimals=decimals
        )
        dict_tables["LinUCB"] = table_lin
        dict_best_masks["LinUCB"] = best_mask_lin

        latex_blocks.append(
            _table_to_latex_block(
                table_lin,
                best_mask_lin,
                caption=r"Terminal cumulative belief regret for LinUCB. Entries report average regret with $2\times\mathrm{SE}$ in parentheses.",
                label="tab:linucb-terminal-regret",
            )
        )

    for algo in ["Box A", "Box B"]:
        if algo not in algorithms:
            continue

        dict_tables[algo] = {}
        dict_best_masks[algo] = {}

        for batch in list_batch:
            df_algo = df_terminal[df_terminal["algorithm"] == algo].copy()
            df_algo = df_algo[df_algo["batch_size"] == batch].copy()
            if len(df_algo) == 0:
                continue

            table_batch, best_mask_batch = _build_terminal_pivot(
                df_algo,
                algorithm=algo,
                batch_size=batch,
                decimals=decimals
            )

            dict_tables[algo][batch] = table_batch
            dict_best_masks[algo][batch] = best_mask_batch

            algo_tex = _display_algorithm_name(algo)
            label_prefix = "boxa" if algo == "Box A" else "boxb"

            latex_blocks.append(
                _table_to_latex_block(
                    table_batch,
                    best_mask_batch,
                    caption=rf"Terminal cumulative belief regret for {algo_tex} with batch size $\ell={batch}$. Entries report average regret with $2\times\mathrm{{SE}}$ in parentheses.",
                    label=rf"tab:{label_prefix}-terminal-regret-b{batch}",
                )
            )

    if path_excel is not None:
        parent_dir = os.path.dirname(path_excel)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)

        with pd.ExcelWriter(path_excel, engine="openpyxl") as writer:
            if "LinUCB" in dict_tables:
                dict_tables["LinUCB"].to_excel(writer, sheet_name="LinUCB")
                ws = writer.sheets["LinUCB"]

                table_df = dict_tables["LinUCB"]
                best_mask = dict_best_masks["LinUCB"]

                data_start_row = 2
                data_start_col = 2

                for i, idx in enumerate(table_df.index, start=data_start_row):
                    for j, col in enumerate(table_df.columns, start=data_start_col):
                        if best_mask.loc[idx, col]:
                            ws.cell(row=i, column=j).font = Font(bold=True)

            for algo in ["Box A", "Box B"]:
                if algo not in dict_tables:
                    continue

                sheet_name = _display_algorithm_name(algo).replace("-", "_")
                start_row = 0

                for batch in sorted(dict_tables[algo].keys()):
                    pd.DataFrame([[f"{_display_algorithm_name(algo)} - batch={batch}"]]).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                        header=False,
                        startrow=start_row,
                        startcol=0
                    )

                    table_df = dict_tables[algo][batch]
                    best_mask = dict_best_masks[algo][batch]

                    table_startrow = start_row + 2
                    table_df.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        startrow=table_startrow
                    )

                    ws = writer.sheets[sheet_name]

                    data_start_row = table_startrow + 2
                    data_start_col = 2

                    for i, idx in enumerate(table_df.index, start=data_start_row):
                        for j, col in enumerate(table_df.columns, start=data_start_col):
                            if best_mask.loc[idx, col]:
                                ws.cell(row=i, column=j).font = Font(bold=True)

                    start_row += table_df.shape[0] + 5

    latex_code = "\n\n".join(latex_blocks)
    if path_latex_txt is not None:
        parent_dir = os.path.dirname(path_latex_txt)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
        with open(path_latex_txt, "w", encoding="utf-8") as f:
            f.write(latex_code)

    return dict_tables, latex_code
