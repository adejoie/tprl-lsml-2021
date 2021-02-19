import gym
import pybullet_envs
from gym.wrappers.monitoring import video_recorder
import torch
import numpy as np

discrete_envs = {
    'car':      'MountainCar-v0',
    'lunar':    'LunarLander-v2',
    'cartpole': 'CartPole-v0'
}
continuous_envs = {
    'car-continuous':   'MountainCarContinuous-v0',
    'lunar-continuous': 'LunarLanderContinuous-v2',
    'cheetah':          'HalfCheetahBulletEnv-v0',
    'hopper':           'HopperBulletEnv-v0',
    'walker':           'Walker2DBulletEnv-v0',
    'ant':              'AntBulletEnv-v0',
    'reacher':          'ReacherBulletEnv-v0',
    'pendulum':         'InvertedPendulumBulletEnv-v0',
    'double-pendulum':  'InvertedDoublePendulumBulletEnv-v0',
}
bullet_envs = {
    'cheetah',
    'hopper',
    'walker',
    'ant',
    'reacher',
    'pendulum',
    'doublependulum',
}


class GymEnvironment(object):
    '''
    :param env_name: short name of the gym env (e.g. 'car' for 'MountainCar-v0')
    '''

    def __init__(self, env_name):
        self._short_name = env_name
        self.is_bullet = env_name in bullet_envs
        if env_name in discrete_envs.keys():
            self.name = discrete_envs[env_name]
            self.is_discrete = True
        elif env_name in continuous_envs.keys():
            self.name = continuous_envs[env_name]
            self.is_discrete = False
        else:
            raise KeyError('env_name not valid!')

        self._env = gym.make(self.name)
        self.max_action = self.action_space.high if not self.is_discrete else None
        self.min_action = self.action_space.low if not self.is_discrete else None
        self.action_dim = self.action_space.shape[0] if not self.is_discrete else self.action_space.n
        self.observation_dim = self.state_space.shape[0]
        self.max_episode_steps = self._env.spec.max_episode_steps

        self.record = False
        self.recording = False
        self.video_name = './video'

    @property
    def action_space(self):
        # should be able to call env.action_space.sample()
        return self._env.action_space

    @property
    def state_space(self):
        return self._env.observation_space

    def reset(self, record=False):
        print(record)
        self.record = record
        print(self.record)
        return self._env.reset()

    def step(self, ac):
        # print(self.record)
        if self.record:
            if not self.recording:
                self.recording = True
                self.video_recorder = video_recorder.VideoRecorder(
                    env=self._env,
                    base_path=self.video_name)
            self.video_recorder.capture_frame()
        return self._env.step(ac)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        if self.recording:
            self.recording = False
            self.video_recorder.close()
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def process_state(self, ob):
        # to tensor of shape [1 , observation_dim]
        return torch.FloatTensor(ob).view(1, -1)

    def process_action(self, ac):
        # from tensor of shape [1]              if env is discrete
        # from tensor of shape [1, action_dim]  if env is continuous
        if self.is_discrete:
            return ac.item()
        else:
            return np.squeeze(ac.cpu().detach().numpy(), 0)

    def duplicate(self, n):
        return [GymEnvironment(self._short_name, device=self.device) for _ in range(n)]

    def setup_recording(self, name):
        self.video_name = name
