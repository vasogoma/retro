import os

from baselines.common.vec_env import (SubprocVecEnv, DummyVecEnv, VecFrameStack, VecEnvWrapper,
                                      VecMonitor)
from baselines.logger import configure
from baselines.ppo2 import ppo2
import gym
import numpy as np
import retro
import tensorflow as tf

import ssb64_train


class MultiAgentToSingleAgent(VecEnvWrapper):
    """Converts a vector of multi-agent environments into a vector of single-agent environments."""
    def __init__(self, venv, num_agents):
        super().__init__(venv)
        self.num_agents = num_agents
        self.num_envs = self.num_envs * self.num_agents
        self.action_space = self._get_action_space(venv.action_space, self.num_agents)

    def _get_action_space(self, multi_action_space, num_agents):
        if isinstance(multi_action_space, gym.spaces.MultiDiscrete):
            single_agent_length = len(multi_action_space.nvec) // num_agents
            nvec = multi_action_space.nvec[:single_agent_length]
            return gym.spaces.MultiDiscrete(nvec)
        else:
            raise NotImplementedError(f"Action space not implemented: {multi_action_space}")

    def reset(self):
        obs = self.venv.reset()
        obs = np.repeat(obs, repeats=self.num_agents, axis=0)
        return obs

    def step_async(self, actions):
        actions = np.reshape(actions, (self.venv.num_envs, -1))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = np.repeat(obs, repeats=self.num_agents, axis=0)
        rews = rews.reshape(self.num_envs)
        dones = np.repeat(dones, repeats=self.num_agents, axis=0)
        infos = np.repeat(infos, repeats=self.num_agents, axis=0)
        return obs, rews, dones, infos


def main():
    expdir = os.path.join("/home/wulfebw/experiments", "ssb64_006", "run_003")
    os.makedirs(expdir, exist_ok=True)
    monitor_filepath = os.path.join(expdir, "monitor.csv")
    movie_dir = os.path.join(expdir, "movies")
    os.makedirs(movie_dir, exist_ok=True)
    load_filepath = None
    # load_filepath = "/home/wulfebw/experiments/ssb64_004/run_006/checkpoints/00100"

    # This configures baselines logging.
    configure(dir=expdir)
    # Creating the session here prevents tf from using all the gpu memory, which
    # causes a failure in the emulator. I'm not sure why because when the emulator
    # is running with angrylion I thought it wasn't using any gpu memory, but
    # there's a lot here I don't understand so oh well.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1,
                                            gpu_options=gpu_options))

    def make_env(rank, grayscale=True, num_agents=2):
        retro.data.add_custom_integration("custom")
        state = "ssb64.pikachu.dk.dreamland.state"
        env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                                   use_restricted_actions=retro.Actions.MULTI_DISCRETE,
                                   inttype=retro.data.Integrations.CUSTOM,
                                   obs_type=retro.Observations.IMAGE,
                                   state=state,
                                   players=num_agents)
        env = ssb64_train.wrap_n64(env, grayscale=grayscale)
        env = ssb64_train.wrap_monitoring_n64(env, monitor_filepath=None, movie_dir=movie_dir)
        return env

    def make_vec_env(nenvs=4, recurrent=False, grayscale=True, frame_stack=4, num_agents=2):
        venv = SubprocVecEnv([
            lambda: make_env(rank, grayscale=grayscale, num_agents=num_agents)
            for rank in range(nenvs)
        ])
        # Uncomment this line in place of the one above for debugging.
        # venv = DummyVecEnv([lambda: make_env(0)])

        if not recurrent:
            # Perform the frame stack at the vectorized environment level as opposed to at
            # the individual environment level. I think this allows you to communicate fewer
            # images across processes.
            venv = VecFrameStack(venv, frame_stack)

        venv = MultiAgentToSingleAgent(venv, num_agents=num_agents)
        venv = VecMonitor(venv, filename=monitor_filepath)
        return venv

    network_name = "impala_cnn_lstm"
    recurrent = "lstm" in network_name
    grayscale = False
    frame_stack = 2
    num_agents = 2
    venv = make_vec_env(nenvs=16,
                        recurrent=recurrent,
                        grayscale=grayscale,
                        frame_stack=frame_stack,
                        num_agents=num_agents)
    ppo2.learn(network=network_name,
               env=venv,
               total_timesteps=int(10e6),
               nsteps=256,
               nminibatches=16,
               lam=0.95,
               gamma=0.999,
               noptepochs=2,
               log_interval=1,
               ent_coef=.01,
               lr=lambda f: f * 5e-4,
               cliprange=0.2,
               save_interval=10,
               load_path=load_filepath)


if __name__ == '__main__':
    main()
