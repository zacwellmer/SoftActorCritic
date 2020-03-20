import numpy as np
import tensorflow as tf

EPS = 1e-8

@tf.function
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)
"""
Policies
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20

@tf.function
def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(input_tensor=2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi

class Policy(tf.keras.Model):
  def __init__(self, obs_dim, act_dim, name, hidden_sizes, activation, output_activation, action_space):
    super(Policy, self).__init__()
    self.action_space = action_space
    self.base = tf.keras.Sequential([tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=(obs_dim), name=name+'/input')])
    for i, hidden_size_i in enumerate(hidden_sizes):
      self.base.add(tf.keras.layers.Dense(hidden_size_i, activation=activation, name=name+'/h{}'.format(i)))
    

    self.mu = tf.keras.Sequential([tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=hidden_sizes[-1]),
        tf.keras.layers.Dense(act_dim, activation=output_activation, name=name+'mu')])
    self.log_std = tf.keras.Sequential([tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=hidden_sizes[-1]),
        tf.keras.layers.Dense(act_dim, activation=None, name=name+'log_std')])

  @tf.function
  def call(self, input_tensor, training=False):
    h = self.base(input_tensor)
    mu = self.mu(h)
    
    log_std = tf.clip_by_value(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)

    pi = mu + tf.random.normal(tf.shape(input=mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)     

    action_scale = self.action_space.high[0]
    mu *= action_scale
    pi *= action_scale
    return mu, pi, logp_pi

class Critic(tf.keras.Model):
  def __init__(self, obs_dim, act_dim, name, hidden_sizes, activation):
    super(Critic, self).__init__()
    self.base = tf.keras.Sequential([tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=(obs_dim + act_dim), name=name+'/input')])
    for i, hidden_size_i in enumerate(hidden_sizes):
      self.base.add(tf.keras.layers.Dense(hidden_size_i, activation=activation, name=name+'/h{}'.format(i)))
      
    self.Q = tf.keras.Sequential([tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=hidden_sizes[-1]),
        tf.keras.layers.Dense(1, activation=None, name=name+'/Q')])

  @tf.function
  def call(self, input_tensor, training=False):
    h = self.base(input_tensor)
    Q = self.Q(h) 
    return Q


"""
Actor-Critics
"""
def mlp_actor_critic(obs_dim, act_dim, name, hidden_sizes=(256), activation=tf.nn.relu, output_activation=None, action_space=None):
    policy = Policy(obs_dim, act_dim, name+'/pi', hidden_sizes, activation, output_activation, action_space)
    Q1 = Critic(obs_dim, act_dim, name+'/q1', hidden_sizes, activation)
    Q2 = Critic(obs_dim, act_dim, name+'/q2', hidden_sizes, activation)
    return policy, Q1, Q2
