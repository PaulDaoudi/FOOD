import os
import numpy as np
import yaml
from ray import tune
from ray.tune import run
from agents.common.names import get_global_env_name, update_config
from main import main


# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = '1'
os.environ["Timer"] = '1'


def trial_name(trial, hp_to_write):
    ti = 'repeat_run'
    identifier = ','.join([f'{hp}={trial.config[hp]}' for hp in hp_to_write]) + \
                 f',trial={trial.config[ti]},id={trial.trial_id}'
    return identifier


if __name__ == '__main__':

    envs = [
        # 'friction_HalfCheetah-v2',
        # 'MinitaurLinear',
        # 'broken_joint_Minitaur',
        # 'heavy_HalfCheetah-v2',
        # 'broken_joint_cheetah',
        # 'broken_joint_ant',
        # 'broken_leg_ant',
    ]

    for env_name in envs:

        # Retrieve hyper-params
        with open(os.path.join(os.getcwd(), 'ray_config', f'{env_name}.yaml')) as f:
            config_init = yaml.safe_load(f)
            np.random.seed(config_init['seed'])
            del config_init['seed']

        # Check type training and add config params
        assert config_init['type_training'] in [
            'use_both',
            'use_only_source',
            'use_only_target',
        ]
        config_init['env_name'] = env_name
        config_init['glob_env_name'] = get_global_env_name(env_name)
        config_init['orig_cwd'] = os.getcwd()
        config_init['device'] = 'cpu'

        # Loop over agents
        agents = config_init['agent']
        del config_init['agent']
        for agent in agents:
            config = config_init.copy()
            config['agent'] = agent

            # Remove unecessary grid searches
            update_config(config)

            # Ray preparation
            hps = [k for k, v in config.items() if type(v) is list]
            config_ray = {k: tune.grid_search(v) if type(v) is list else v for k, v in config.items()}
            config_ray['repeat_run'] = tune.grid_search(list(range(config['repeat_run'])))
            metric_columns = ['epoch', 'mean_avg_return', 'source_mean_avg_return', 'epoch_time']
            reporter = tune.CLIReporter(metric_columns=metric_columns,
                                        parameter_columns=hps)
            save_path = f'./ray_results/{config_ray["env_name"]}/{config["type_training"]}/{config_ray["agent"]}'

            analysis = run(
                main,
                config=config_ray,
                metric=config_ray['metric'],
                mode=config_ray['mode'],
                resources_per_trial={"cpu": 2, "gpu": 1 if config_ray['device'] == 'cuda' else 0},
                max_concurrent_trials=20,
                log_to_file=True,
                local_dir=save_path,
                trial_name_creator=lambda t: trial_name(t, hps),
                trial_dirname_creator=lambda t: trial_name(t, hps),
                progress_reporter=reporter,
                verbose=1)  # resume=True,
