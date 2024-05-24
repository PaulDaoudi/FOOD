import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from typing import List, Dict, Optional, Any


def plot_curves(analysis: ExperimentAnalysis,
                hps: List[str],
                metric: str,
                std_multiplier: float = 1,
                rolling_mean: float = 0.6,
                set_hyperparam: Dict[str, Any] = {},
                hyperparam_comparison: Optional[str] = None,
                to_plot: str = "final",
                label: str = "SAC",
                n_epochs: int = 2000,
                colors: Dict[str, Any] = {}) -> None:
    """
    Plots the learning curves from the experiment analysis.

    Args:
        analysis (ExperimentAnalysis): The experiment analysis object.
        hps (List[str]): List of hyperparameters to choose.
        metric (str): The metric to plot.
        std_multiplier (float, optional): Multiplier for standard deviation. Defaults to 1.
        rolling_mean (float, optional): Rolling mean factor. Defaults to 0.6.
        set_hyperparam (Dict[str, Any], optional): Specific hyperparameter to set. Defaults to {}.
        hyperparam_comparison (Optional[str], optional): Hyperparameter to compare. Defaults to None.
        to_plot (str, optional): Whether to plot 'final' or 'overall' performance. Defaults to "final".
        label (str, optional): Label for the plot. Defaults to "SAC".
        n_epochs (int, optional): Number of epochs to plot. Defaults to 2000.
        colors (Dict[str, Any], optional): Dictionary of colors for the plots. Defaults to {}.
    """
    # Get dataframe for all seeds
    group_by = [f'config/{hp}' for hp in hps if hp != 'repeat_run'] + ['epoch']
    dfs = analysis.trial_dataframes
    conf = analysis.get_all_configs()
    conf = {k: {f'config/{_k}': _v for _k, _v in v.items()} for k, v in conf.items()}
    df = pd.concat([dfs[k].assign(**conf[k]) for k in dfs.keys()])
    group = df.groupby(group_by)
    mean = group.mean()
    std = std_multiplier * group.std()

    # If overall or final
    if to_plot == "overall":
        plot_max_idx_init = mean[metric].idxmax()
        best_dict = {'mean': mean.loc[plot_max_idx_init], 'std': std.loc[plot_max_idx_init]}
    else:
        final_mean = mean.xs(chosen_max - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        final_std = std.xs(chosen_max - 1, axis=0, level=len(group_by) - 1, drop_level=False)
        plot_max_idx_init = final_mean[metric].idxmax()
        best_dict = {'mean': final_mean.loc[plot_max_idx_init], 'std': final_std.loc[plot_max_idx_init]}

    # If set_hyperamam
    if set_hyperparam:
        key, value = list(set_hyperparam.items())[0]
        idx_to_remove = hps.index(key)
        plot_max_idx = plot_max_idx_init[:idx_to_remove] + plot_max_idx_init[idx_to_remove + 1:]
        plot_max_idx = list(plot_max_idx)
        plot_max_idx.insert(idx_to_remove, value)
        label += f"_{key}_{value}"
        plot_max_idx = tuple(plot_max_idx)
    else:
        plot_max_idx = plot_max_idx_init

    # If hyperparam comparison
    if hyperparam_comparison is not None:
        hyperparams = hyperparam_comparison.split('-')
        idx_to_remove = hps.index(hyperparam_comparison)
        idx_but_two = plot_max_idx[:idx_to_remove] + plot_max_idx[idx_to_remove + 1:-1]
        hyperparam_list = list(mean.index.get_level_values(idx_to_remove).unique())xz
        for hyperparam in hyperparam_list:
            idxs = list(idx_but_two).copy()
            idxs.insert(idx_to_remove, hyperparam)
            idxs = tuple(idxs)
            plot_mean = mean.loc[idxs][metric][begin:n_epochs].ewm(alpha=rolling_mean).mean()
            plot_std = std.loc[idxs][metric][begin:n_epochs].ewm(alpha=rolling_mean).mean()
            if agent_name == 'DARC':
                plt.plot(plot_mean, label=r'$\sigma_{\mathrm{DARC}}$' + rf'$ = {hyperparam}$')
            elif agent_name == 'ANE':
                plt.plot(plot_mean, label=r'$\sigma_{\mathrm{ANE}}$' + rf'$ = {hyperparam}$')
            else:
                plt.plot(plot_mean, label=fr'$\alpha = {hyperparam}$')
            plt.fill_between(plot_mean.index,
                                plot_mean - plot_std,
                                plot_mean + plot_std,
                                alpha=0.2)

    else:
        # Plot for selected hyperparams
        idx_but_one = plot_max_idx[:-1]
        plot_mean = mean.loc[(idx_but_one)][metric]
        plot_std = std.loc[(idx_but_one)][metric]
        plot_mean = plot_mean.loc[plot_mean.index <= n_epochs].ewm(alpha=rolling_mean).mean()
        plot_std = plot_std.loc[plot_std.index <= n_epochs].ewm(alpha=rolling_mean).mean()
        plt.plot(plot_mean, label=label, color=colors[label])
        plt.fill_between(plot_mean.index,
                         plot_mean - plot_std,
                         plot_mean + plot_std,
                         alpha=0.2, color=colors[label])


def plot_all(env: str,
             agents: List[str],
             type_training: str = 'use_both',
             rolling_mean: float = 0.6,
             before_last: int = 1,
             verbose: int = 0,
             set_hyperparam: Dict[str, Any] = {},
             std_multiplier: float = 1,
             hyperparam_comparison: Optional[str] = None,
             init_path: str = "..",
             hps: List[str] = ['imit_coef'],
             metric: str = "mean_avg_return",
             mode: str = "max",
             to_plot: str = "final",
             n_epochs: int = 2000,
             colors: Dict[str, Any] = {}) -> None:
    """
    Plots learning curves for all specified agents in the given environment.

    Args:
        env (str): The environment name.
        agents (List[str]): List of agents to plot.
        type_training (str, optional): Type of training used. Defaults to 'use_both'.
        rolling_mean (float, optional): Rolling mean factor. Defaults to 0.6.
        before_last (int, optional): Number of checkpoints before the last to use. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 0.
        set_hyperparam (Dict[str, Any], optional): Specific hyperparameter to set. Defaults to {}.
        std_multiplier (float, optional): Multiplier for standard deviation. Defaults to 1.
        hyperparam_comparison (Optional[str], optional): Hyperparameter to compare. Defaults to None.
        init_path (str, optional): Initial path for the results. Defaults to "..".
        hps (List[str], optional): List of hyperparameters to choose. Defaults to ['imit_coef'].
        metric (str, optional): The metric to plot. Defaults to "mean_avg_return".
        mode (str, optional): Mode for selecting the best trial. Defaults to "max".
        to_plot (str, optional): Whether to plot 'final' or 'overall' performance. Defaults to "final".
        n_epochs (int, optional): Number of epochs to plot. Defaults to 2000.
        colors (Dict[str, Any], optional): Dictionary of colors for the plots. Defaults to {}.
    """
    assert to_plot in ["overall", "final"]
    dict_names = {
        'PPO': 'PPO',
        'PPOContinue': 'PPO',
        'PPOGAIL-SImit': 'FOOD (ours)',
        'PPOGAIL-SAImit': 'FOOD (ours)',
        'PPOGAIL-SASImit': 'FOOD (ours)',
        'PPODARC': 'DARC',
        'PPOANE': 'ANE',
    }

    # Loop over agents
    for agent in agents:

        # Get right path
        path = os.path.join(init_path, 'ray_results', env, type_training, agent)
        label = dict_names[agent]
        if verbose == 2:
                print('path:', path)
        main = sorted(glob.glob(f"{path}/*"), key=os.path.getmtime)[-before_last].split('/')[-1]
        experiment_checkpoint_path = os.path.join(path, main)
        if verbose:
            print('experiment_checkpoint_path:', experiment_checkpoint_path)

        # Get analysis
        analysis = ExperimentAnalysis(experiment_checkpoint_path, default_metric=metric, default_mode=mode)

        # Plot one curve
        auc = plot_curves(analysis,
                          hps,
                          metric,
                          std_multiplier=std_multiplier,
                          rolling_mean=rolling_mean,
                          set_hyperparam=set_hyperparam_,
                          hyperparam_comparison=hyperparam_comparison,
                          to_plot=to_plot,
                          label=label,
                          n_epochs=n_epochs,
                          colors=colors)
