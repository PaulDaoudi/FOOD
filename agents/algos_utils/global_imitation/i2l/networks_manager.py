import os
from on_policy_sim2real.algo_utils.global_imitation.global_imitation_utils import expert_gen_inf, zipsame
from on_policy_sim2real.algo_utils.global_imitation.i2l.discriminator_model import I2LDiscriminator
from on_policy_sim2real.algo_utils.global_imitation.i2l.wcritic_model import Wcritic
from on_policy_sim2real.algo_utils.global_imitation.i2l.buffers.super_pq import SuperPQ
from on_policy_sim2real.algo_utils.global_imitation.i2l.misc.utils import flatten_completed_trajs


class NetworksManager:
    """
    Class that handles the airl-discriminator, the wasserstein critic and the priority buffers
    """
    def __init__(self, cfg, sim_envs, sim_rollouts, action_eval_fn, latest_trajs, device,
                 imit_only=False, **kwargs):
        self.cfg = cfg
        obs_dim = sim_envs.observation_space.shape[0]
        acs_dim = sim_envs.action_space.shape[0]
        # self.rl_agent = rl_agent

        self.discriminator = I2LDiscriminator(obs_dim, acs_dim, hidden_dim=64, device=device)
        # Wass-critic to discriminate b/w "state-only" data from the pq-buffer and the expert data
        self.wcritic = Wcritic(obs_dim, hidden_dim=64, device=device)

        # high level wrapper around a class that can manage multiple priority queues (if needed)
        self.super_pq = SuperPQ(count=cfg['num_pqs'],
                                capacity=cfg['pq_capacity'],
                                wcritic=self.wcritic,
                                refresh_rate=cfg['pq_refresh_rate'])

        # (infinite) generator to loop over expert states
        src_expert_data_filename = os.path.join(cfg['orig_cwd'],
                                                cfg['experts_dir'],
                                                "trajs_{}.pickle".format(cfg['dir_env']))
        self.expert_gen = expert_gen_inf(src_expert_data_filename,
                                         cfg['num_expert_trajs'],
                                         cfg['expert_batch_size'],
                                         subsample_frequency=1,
                                         drop_last=True)

        self.optim_params = {k: cfg[k] for k in ['expert_batch_size',
                                                 'airl_grad_steps',
                                                 'wcritic_grad_steps']}

        self.imit_only = imit_only
        self.sim_rollouts = sim_rollouts
        self.action_eval_fn = action_eval_fn
        self.latest_trajs = latest_trajs

    def update(self, iter_num):

        if not self.super_pq.is_empty:
            # perform multiple updates of the wcritic classifier using pq-buffer and expert data
            wcritic_loss = self.wcritic.update(iter_num,
                                               self.expert_gen,
                                               self.super_pq.random_select(),
                                               batch_size=self.optim_params['expert_batch_size'],
                                               num_grad_steps=self.optim_params['wcritic_grad_steps'])
        else:
            wcritic_loss = 0.

        completed_trajs = list(self.latest_trajs.values())[:-1]
        completed_trajs = flatten_completed_trajs(completed_trajs)
        assert len(completed_trajs) > 0, "No completed trajectory. Consider increasing cfg.num_steps"
        completed_trajs_scores = self.wcritic.assign_score(completed_trajs)

        # randomly select one of the pq-buffers, and add completed trajectories to it
        pqb = self.super_pq.random_select(ignore_empty=True)
        for traj, score in zipsame(completed_trajs, completed_trajs_scores):
            pqb.add_traj({**traj, 'score': score})

        # pqb.add_path() does a deepcopy, hence we can free some memory
        for i in list(self.latest_trajs.keys())[:-1]:
            del self.latest_trajs[i]

        # Update the entries in the pq-buffers using the latest critic
        if iter_num and iter_num % self.super_pq.refresh_rate == 0:
            self.super_pq.update()

        # Update rewards with values from the discriminator
        self.discriminator.predict_batch_rewards(self.sim_rollouts, imit_only=self.imit_only)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        discriminator_loss = self.discriminator.update(self.action_eval_fn,
                                                       self.super_pq.random_select(),
                                                       self.sim_rollouts,
                                                       num_grad_steps=self.optim_params['airl_grad_steps'])

        return {
            'discriminator_loss': discriminator_loss,
            'wcritic_loss': wcritic_loss
        }
