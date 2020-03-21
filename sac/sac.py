import numpy as np
import tensorflow as tf
import gym
import time
import os
import core
from utils.logx import EpochLogger

#for gpu in tf.config.experimental.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(gpu, True)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class SAC(tf.keras.Model):
    def __init__(self, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
        """
        Soft Actor-Critic (SAC)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: A function which takes in placeholder symbols 
                for state, ``x_ph``, and action, ``a_ph``, and returns the main 
                outputs from the agent's Tensorflow computation graph:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                               | given states.
                ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                               | states.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                               | the policy, of the action sampled by
                                               | ``pi``. Critical: must be differentiable
                                               | with respect to policy parameters all
                                               | the way through action sampling.
                ``q1``       (batch,)          | Gives one estimate of Q* for 
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q2``       (batch,)          | Gives another estimate of Q* for 
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
                function you provided to SAC.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        super(SAC, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.num_test_episodes = num_test_episodes
        self.polyak = polyak

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = self.env.action_space

        # Main outputs from computation graph
        self.policy, self.Q1, self.Q2 = actor_critic(obs_dim, act_dim, name='main', **ac_kwargs)
        _, self.Q1_targ, self.Q2_targ  = actor_critic(obs_dim, act_dim, name='target', **ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

        self.target_init()
        #print([np.array_equal(self.Q1.variables[i].numpy(), self.Q1_targ.variables[i].numpy()) for i in range(len(self.Q1.variables))])

    @tf.function
    def compute_loss_q(self, o, a, r, o2, d): 
        o_a = tf.concat([o, a], axis=1)
        q1 = self.Q1(o_a)
        q2 = self.Q2(o_a)

        # Bellman backup for Q functions
	# Target actions come from *current* policy
        a2_mu, a2_pi, logp_a2pi = self.policy(o2)
        o2_a2pi = tf.concat([o2, a2_pi], axis=-1)
	
        # Target Q-values
        q1_pi_targ = self.Q1_targ(o2_a2pi)
        q2_pi_targ = self.Q2_targ(o2_a2pi)
        q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)
        backup = tf.stop_gradient(r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2pi))

        q1_loss = 0.5 * tf.reduce_mean((backup-q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((backup-q2)**2)
        q_loss = q1_loss + q2_loss

        return q_loss, q1_loss, q2_loss, q1, q2

    @tf.function
    def compute_loss_pi(self, o):
        _, pi, logp_pi = self.policy(o)
        o_pi = tf.concat([o, pi], axis=-1)
        q1_pi = self.Q1(o_pi)
        q2_pi = self.Q2(o_pi)
        q_pi = tf.minimum(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = tf.reduce_mean(self.alpha * logp_pi - q_pi)

        return loss_pi, logp_pi

    @tf.function 
    def compute_apply_gradients(self, o, o2, a, r, d):
        main_pi_vars = self.policy.trainable_variables
        main_q_vars = self.Q1.trainable_variables + self.Q2.trainable_variables
        
        with tf.GradientTape() as tape:
            pi_loss, logp_pi = self.compute_loss_pi(o=o)
        pi_gradients = tape.gradient(pi_loss, main_pi_vars)
        self.policy_optimizer.apply_gradients(zip(pi_gradients, main_pi_vars))

        with tf.GradientTape() as tape:
           value_loss, q1_loss, q2_loss, q1vals, q2vals = self.compute_loss_q(o=o, o2=o2, a=a, r=r, d=d)
        v_gradients = tape.gradient(value_loss, main_q_vars)
        self.value_optimizer.apply_gradients(zip(v_gradients, main_q_vars))

        return pi_loss, q1_loss, q2_loss, q1vals, q2vals, logp_pi

    @tf.function
    def update_target(self): 
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        for v_main, v_targ in zip(self.Q1.variables, self.Q1_targ.variables):
            v_targ.assign(self.polyak*v_targ + (1-self.polyak)*v_main)

        for v_main, v_targ in zip(self.Q2.variables, self.Q2_targ.variables):
            v_targ.assign(self.polyak*v_targ + (1-self.polyak)*v_main)

    @tf.function 
    def target_init(self):
        # Initializing targets to match main variables
        for v_main, v_targ in zip(self.Q1.variables, self.Q1_targ.variables):
            v_targ.assign(v_main)

        for v_main, v_targ in zip(self.Q2.variables, self.Q2_targ.variables):
            v_targ.assign(v_main)

    def get_action(self, o, deterministic=False):
        mu, pi, _ = self.policy(tf.reshape(o, (1, -1)))
        if deterministic:
            return mu[0].numpy()
        else:
            return pi[0].numpy()

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                a = self.get_action(o, True)
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train(self):
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        total_steps = self.steps_per_epoch * self.epochs

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy.
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    outs = self.compute_apply_gradients(o=batch['obs1'], o2=batch['obs2'], a=batch['acts'], r=batch['rews'], d=batch['done'])
                    self.update_target()
                    self.logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                                 Q1Vals=outs[3], Q2Vals=outs[4], LogPi=outs[5])
            # End of epoch wrap-up
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True) 
                self.logger.log_tabular('Q2Vals', with_min_and_max=True) 
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ1', average_only=True)
                self.logger.log_tabular('LossQ2', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()
    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    magic_sac = SAC(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
    magic_sac.train()


