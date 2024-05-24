# A Conservative Approach for Transfer in Few-Shot Off-Dynamics Reinforcement Learning

This is a reimplementation of the techniques discussed in the paper [A Conservative Approach for Transfer in Few-Shot Off-Dynamics Reinforcement Learning](https://arxiv.org/pdf/2312.15474).

## Create the conda virtual environment

```
conda create --name food python=3.8
conda activate food
pip install -e .
```

## Steps to launch it for a new environment

- Create the source and target environments in the `get_env.py` function.
- Train a PPO agent in the source environment and save it to a specific folder, for example in `your_env/model/PPO`. Careful, if you use this code and you normalize the environment, you also need to save the normalization class `ob_rms`.
- Gather trajectories in the target environment with the notebook `Gather_Target_Trajectories.ipynb` and save them in the folder `expert_trajectories`
- Add the environment in the global files `creation` and `confidence` in `envs`
- Add a config file in `ray_config`
- Modify the `main` file to include the new environment
- Enjoy :)

## Visualization

All the results are saved in a ray tune `Experimentanalysis`. You can plot them in the `Visualization.ipynb` notebook.

## License

I follow MIT License. Please see the [License](./LICENSE) file for more information.

## Credits

This code is built upon the [PPO Github](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). It also uses the [H2O](https://github.com/t6-thu/H2O) to build [DARC](https://arxiv.org/pdf/2006.13916), extracts one environment from [GARAT](https://github.com/Ricky-Zhu/GARAT/tree/master) and others from the official [DARC Code](https://github.com/google-research/google-research/tree/master/darc).

## Cite us

If you find this technique useful and you use it in a project, please cite it:
```
@inproceedings{daoudi2024conservative,
  title={A Conservative Approach for Transfer in Few-Shot Off-Dynamics Reinforcement Learning},
  author={Daoudi, Paul and Robu, Bogdan and Prieur, Christophe and Barlier, Merwan and Dos Santos, Ludovic},
  booktitle={Proceedings of the 2024 International Joint Conference on Artificial Intelligence},
  year={2024}
}
```

## Currently missing
- Proper check
- setup.py
