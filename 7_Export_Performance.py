from src.export_table import export_summary_table, export_terminal_regret_tables
from src.plot_performance import plot_optimal_regret_curve_by_strategy, plot_best_regret_curves, plot_best_regret_by_c_per_strategy

####################################################################################
####################################################################################
############################# AREA OF INPUT PARAMETERS #############################
####################################################################################
####################################################################################
##### Environment Parameters
PATH = "." # Root directory, should be the same path this "README.md" file locates
PATH_MODELS = f"{PATH}/models" # Path for models
PATH_PICS = f"{PATH}/pics" # Path for graphs
PATH_TABLE = f"{PATH}/tables" # Path for tabes

n_jobs = 10
##### Parameters for Simulated andits
list_seed_simulation = list(range(1951, 2051))  # To reproduce the results, use simulated seed from 1951 to 2051
number_rounds = 50000  # T, use 50000 to reproduce the results
with_union_bound = False
prefix_union_bound = 'ub' if with_union_bound else 'no_ub'
list_ucb_multipler = [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5]
list_lmd = [1e-9, 0.001, 0.01, 0.1, 1, 10, 100]
list_batch = [1, 15, 37, 224]
list_algorithms = [ "LinUCB", "Box A", "Box B"]

####################################################################################
####################################################################################
####################### Collect the Simulated Results ##############################
####################################################################################
####################################################################################

df_summary = export_summary_table(
    path_models=PATH_MODELS,
    seeds=list_seed_simulation,
    number_rounds=number_rounds,
    list_ucb_multipler=list_ucb_multipler,
    list_lmd = list_lmd,
    list_batch = list_batch,
    algorithms= list_algorithms,
    path_save=f'{PATH_TABLE}/df_summary.csv',
    with_union_bound=with_union_bound,
    use_parallel=True if n_jobs > 1 else False,
    n_jobs=n_jobs,
)

fig, ax, df_selected = plot_optimal_regret_curve_by_strategy(
    number_rounds=number_rounds,
    list_batch=list_batch,
    list_ucb_multipler=list_ucb_multipler,
    list_lmd=list_lmd,
    algorithms=["LinUCB", "Box A", "Box B"],
    seeds=list_seed_simulation,
    df_summary=df_summary,
    sampling=0.1,
    include_ci=True,
    path_save=f"{PATH_PICS}/optimal_regret_curve_by_strategy.pdf",
)

fig, ax, df_selected = plot_best_regret_curves(
    number_rounds=number_rounds,
    list_batch=list_batch,
    list_ucb_multipler=list_ucb_multipler,
    list_lmd = list_lmd,
    algorithms= list_algorithms,
    seeds=list_seed_simulation,
    df_summary=df_summary,
    sampling=0.1,
    include_ci=True,
    path_save=f"{PATH_PICS}/best_regret_curves.pdf",
)

figs, df_selected_by_c = plot_best_regret_by_c_per_strategy(
    number_rounds=number_rounds,
    list_batch=list_batch,
    list_ucb_multipler=list_ucb_multipler,
    list_lmd = list_lmd,
    algorithms= list_algorithms,
    seeds=list_seed_simulation,
    df_summary=df_summary,
    sampling=0.1,
    include_ci=True,
    path_save_dir=f"{PATH_PICS}",
)

dict_tables, latex_code = export_terminal_regret_tables(
    number_rounds=number_rounds,
    list_batch=list_batch,
    list_ucb_multipler=list_ucb_multipler,
    list_lmd = list_lmd,
    algorithms= list_algorithms,
    seeds=list_seed_simulation,
    df_summary=df_summary,
    path_excel=f"{PATH_TABLE}/df_performance.xlsx",
    path_latex_txt=f"{PATH_TABLE}/df_performance.tex",
)