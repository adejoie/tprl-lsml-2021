import time


class Agent(object):
    def __init__(
        self,
        env,
    ):
        self.env = env
        self.eval_env = env.duplicate(1)[0]

    def update(self):
        pass

    def act(self, ob, eval=False):
        pass

    def get_episode_lengths(self):
        pass

    def get_episode_returns(self):
        pass

    def log_progress(self):
        pass

    def collect_rollouts(self, itr, render=False, bullet=False):
        pass

    def save(self, dir, itr):
        pass

    def eval(self, num_episodes=1, render=False):
        returns = []
        lengths = []
        for _ in range(num_episodes):
            episode_rewards = self._run_test_episode(render)
            returns.append(sum(episode_rewards))
            lengths.append(len(episode_rewards))
        if render and not self.eval_env.is_bullet:
            self.eval_env.close()
        return returns, lengths

    def _run_test_episode(self, render):
        if render and self.eval_env.is_bullet:
            self.eval_env.render(mode="human", close=False)
        ob = self.eval_env.reset(record=render)
        returns = []
        done = False
        while not done:
            if render and not self.eval_env.is_bullet:
                self.eval_env.render()
                # time.sleep(0.1)
            ac = self.act(ob, eval=True)
            ob, rew, done, _ = self.eval_env.step(ac)
            returns.append(rew)
        return returns