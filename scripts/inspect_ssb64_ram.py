"""Notes:

1. for some reason the image observation space doesn't work
- just gives a segfault
- ok let's figure that out?
- what could be wrong?
- nah not yet

"""

import sys
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import retro
# import tensorflow as tf

# from baselines.common.atari_wrappers import (WarpFrame, FrameStack, ScaledFloatFrame, MaxAndSkipEnv)
# from baselines.common.policies import build_policy
# from baselines.common.retro_wrappers import (make_retro, wrap_deepmind_retro, MovieRecord,
#                                              RewardScaler)
# from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper
# from baselines.ppo2.model import Model
# from baselines.ppo2.runner import Runner

# from retro.examples.vec_frame_diff import VecFrameDiff


def main1():
    retro.data.add_custom_integration("custom")
    env = retro.RetroEnv(game="SuperSmashBros-N64",
                         inttype=retro.data.Integrations.CUSTOM,
                         obs_type=retro.Observations.IMAGE)
    action = env.action_space.sample()
    state = env.reset()
    plt.imshow(state)
    plt.savefig("color.png")
    plt.show()
    # state, reward, terminal, info = env.step(action)

    # player_list_pointer = 0x130D84
    # match_setting_pointer = 0xA50E8

    # print(state[match_setting_pointer:match_setting_pointer + 64])

    # import ipdb
    # ipdb.set_trace()


def load_starting_ram_for_state(state):
    env = retro.RetroEnv(game="SuperSmashBros-N64",
                         state=state,
                         inttype=retro.data.Integrations.CUSTOM,
                         obs_type=retro.Observations.RAM)
    action = env.action_space.sample()
    state = env.reset()
    state, reward, terminal, info = env.step(action)
    return state


def main2():
    retro.data.add_custom_integration("custom")
    player_list_pointer = 0x130D84
    match_setting_pointer = 0xA50E8
    player_size = 0xB50

    dreamland = load_starting_ram_for_state("ssb64.pikachu.level9dk.dreamland")
    np.save("save_states/dreamland.npy", dreamland)

    peaches = load_starting_ram_for_state("ssb64.pikachu.level9dk.peaches_castle")
    np.save("save_states/peaches.npy", peaches)

    peaches = load_starting_ram_for_state("kongo")
    np.save("save_states/kongo.npy", peaches)

    peaches = load_starting_ram_for_state("ssb64.pikachu.level9dk.sector_z")
    np.save("save_states/sector_z.npy", peaches)

    peaches = load_starting_ram_for_state("yoshi.level3dk.level3pikachu.level3fox.dreamland")
    np.save("save_states/4_player.npy", peaches)

    peaches = load_starting_ram_for_state(
        "ssb64.pikachu_6_dmg_16_lives.dk_45_dmg_17_lives.dreamland")
    np.save("save_states/dmg_lives.npy", peaches)

    peaches = load_starting_ram_for_state("ssb64.mario.level3samus.hyrule")
    np.save("save_states/dmg_lives_2.npy", peaches)

    ptr = player_list_pointer
    player_1_data = [hex(b) for b in dreamland[ptr:ptr + player_size]]
    player_2_data = [hex(b) for b in dreamland[ptr + player_size:ptr + 2 * player_size]]


def main3():
    retro.data.add_custom_integration("custom")
    env = retro.RetroEnv(game="SuperSmashBros-N64",
                         inttype=retro.data.Integrations.CUSTOM,
                         obs_type=retro.Observations.IMAGE)
    start = time.time()
    env.reset()
    num_steps = 200
    for i in range(num_steps):
        sys.stdout.write(f"\r{i+1} / {num_steps}")
        action = env.action_space.sample()
        env.step(action)
        env.render()
    end = time.time()
    print(end - start)


def main4():
    retro.data.add_custom_integration("custom")
    env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                               state="ssb64.pikachu.dk.dreamland.state",
                               use_restricted_actions=retro.Actions.MULTI_DISCRETE,
                               inttype=retro.data.Integrations.CUSTOM,
                               obs_type=retro.Observations.IMAGE)
    start = time.time()
    env.reset()
    num_steps = 20000

    action = np.array([0, 0, 0])

    for i in range(num_steps):
        sys.stdout.write(f"\r{i+1} / {num_steps}")
        obs, reward, done, info = env.step(action)

        print(reward)
        print(done)

        env.render()
    end = time.time()
    print(end - start)


class N64Interactive(retro.Interactive):
    """
    Interactive setup for retro games
    """
    def __init__(self, env):
        self._buttons = env.buttons
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def get_image(self, _obs, env):
        return env.render(mode='rgb_array')

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            'BUTTON': 'Z' in keys,
            'A': 'Z' in keys,
            'B': 'X' in keys,
            'C': 'C' in keys,
            'X': 'A' in keys,
            'Y': 'S' in keys,
            'Z': 'D' in keys,
            'L': 'Q' in keys,
            'R': 'W' in keys,
            'UP': 'UP' in keys,
            'DOWN': 'DOWN' in keys,
            'LEFT': 'LEFT' in keys,
            'RIGHT': 'RIGHT' in keys,
            'MODE': 'TAB' in keys,
            'SELECT': 'TAB' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'ENTER' in keys,
        }
        return [inputs[b] for b in self._buttons]


def main5():
    retro.data.add_custom_integration("custom")
    env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                               state="ssb64.pikachu.dk.dreamland.state",
                               use_restricted_actions=retro.Actions.ALL,
                               inttype=retro.data.Integrations.CUSTOM,
                               obs_type=retro.Observations.IMAGE)
    ie = N64Interactive(env)
    ie.run()


# def main6():
#     retro.data.add_custom_integration("custom")

#     def wrap_deepmind_n64(env, reward_scale=1 / 100.0, frame_stack=1):
#         env = MaxAndSkipEnv(env, skip=4)
#         env = WarpFrame(env, width=450, height=300, grayscale=False)
#         env = FrameStack(env, frame_stack)
#         env = ScaledFloatFrame(env)
#         env = RewardScaler(env, scale=reward_scale)
#         return env

#     def make_env():
#         retro.data.add_custom_integration("custom")
#         state = "ssb64.pikachu.level9dk.dreamland.state"
#         env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
#                                    use_restricted_actions=retro.Actions.MULTI_DISCRETE,
#                                    inttype=retro.data.Integrations.CUSTOM,
#                                    obs_type=retro.Observations.IMAGE,
#                                    state=state)
#         env = wrap_deepmind_n64(env)
#         return env

#     gpu_options = tf.GPUOptions(allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#     # env = make_env()
#     env = SubprocVecEnv([make_env] * 1)
#     # env = DummyVecEnv([make_env] * 1)

#     env.reset()
#     num_steps = 20000
#     # action = [np.array([0, 0, 0])]
#     # action = [env.action_space.sample() for _ in range(2)]
#     for i in range(num_steps):
#         sys.stdout.write(f"\r{i+1} / {num_steps}")
#         # action = env.action_space.sample()
#         action = [env.action_space.sample() for _ in range(1)]
#         obs, reward, done, info = env.step(action)

#         print(f"\nreward: {reward} done: {done}")
#         # input()
#         if (isinstance(done, bool) and done) or (isinstance(done, list) and all(done)):
#             env.reset()
#         # env.render()

#         if i % 50 == 0:
#             image = Image.fromarray((obs[0] * 255).astype(np.uint8))
#             image.save("/home/wulfebw/Desktop/color.png")

#             plt.imshow(obs[0, :, :, 0])

#             # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
#             # for j in range(1):
#             #     row = j // 2
#             #     col = j % 2
#             #     print(row)
#             #     print(col)
#             #     axs[row, col].imshow(obs[:, :])
#             plt.show()
#             plt.close()
#     end = time.time()
#     print(end - start)

#     return env


# def main7():
#     retro.data.add_custom_integration("custom")

#     def wrap_deepmind_n64(env, reward_scale=1 / 100.0, frame_stack=1, grayscale=False):
#         env = MaxAndSkipEnv(env, skip=4)
#         env = WarpFrame(env, width=150, height=100, grayscale=grayscale)
#         env = FrameStack(env, frame_stack)
#         env = ScaledFloatFrame(env)
#         env = RewardScaler(env, scale=1 / 100.0)
#         return env

#     def make_env():
#         retro.data.add_custom_integration("custom")
#         env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
#                                    use_restricted_actions=retro.Actions.MULTI_DISCRETE,
#                                    inttype=retro.data.Integrations.CUSTOM,
#                                    obs_type=retro.Observations.IMAGE)
#         env = wrap_deepmind_n64(env)
#         return env

#     gpu_options = tf.GPUOptions(allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#     nenvs = 2
#     # env = DummyVecEnv([make_env] * nenvs)
#     env = SubprocVecEnv([make_env] * nenvs)
#     network_name = "impala_cnn_lstm"
#     policy = build_policy(env, network_name)
#     recurrent = "lstm" in network_name
#     ob_space = env.observation_space
#     ac_space = env.action_space
#     nsteps = 10
#     nminibatches = 2
#     nbatch = nenvs * nsteps
#     nbatch_train = nbatch // nminibatches
#     model = Model(policy=policy,
#                   ob_space=ob_space,
#                   ac_space=ac_space,
#                   nbatch_act=nenvs,
#                   nbatch_train=nbatch_train,
#                   nsteps=nsteps,
#                   ent_coef=0.01,
#                   vf_coef=0.5,
#                   max_grad_norm=0.5,
#                   comm=None,
#                   mpi_rank_weight=1)
#     runner = Runner(env=env, model=model, nsteps=10, gamma=.99, lam=.95)

#     env.reset()
#     num_steps = 20000
#     action = [np.array([0, 0, 0]), np.array([0, 0, 0])]
#     for i in range(num_steps):
#         sys.stdout.write(f"\r{i+1} / {num_steps}")
#         action = [env.action_space.sample() for _ in range(nenvs)]
#         obs, reward, dones, info = env.step(action)
#         # env.reset(dones)
#         # env.render()

#         if i % 50 == 0:
#             if recurrent:
#                 fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))
#             else:
#                 fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 12))
#             for env_index in range(nenvs):
#                 if recurrent:
#                     axs[env_index].imshow(obs[env_index, :, :, :])
#                 else:
#                     for j in range(4):
#                         row = env_index * 2 + j // 2
#                         col = j % 2
#                         print(row)
#                         print(col)
#                         axs[row, col].imshow(obs[env_index, :, :, j])
#             plt.show()
#             plt.close()
#     end = time.time()
#     print(end - start)

#     return env


# class MultiAgentDummyVecEnv(DummyVecEnv):
#     def __init__(self, num_agents, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.buf_rews = np.zeros((self.num_envs, num_agents), dtype=np.float32)


# SSB64_IMAGE_MEAN = [0.39777202, 0.56584128, 0.43192356]


# class ImageNormalizer(gym.ObservationWrapper):
#     def __init__(self, env, mean):
#         super().__init__(env)
#         self.mean = np.array(mean, dtype=np.float32)
#         low = -self.mean.max()
#         high = 1 - self.mean.min()
#         self.observation_space = gym.spaces.Box(low=low,
#                                                 high=high,
#                                                 shape=env.observation_space.shape,
#                                                 dtype=np.float32)

#     def observation(self, observation):
#         return (observation - self.mean) * 2


# class MultiAgentToSingleAgent(VecEnvWrapper):
#     """Converts a vector of multi-agent environments into a vector of single-agent environments."""
#     def __init__(self, venv, num_agents):
#         super().__init__(venv)
#         self.num_agents = num_agents
#         self.num_envs = self.num_envs * self.num_agents
#         self.action_space = self._get_action_space(venv.action_space, self.num_agents)

#     def _get_action_space(self, multi_action_space, num_agents):
#         if isinstance(multi_action_space, gym.spaces.MultiDiscrete):
#             single_agent_length = len(multi_action_space.nvec) // num_agents
#             nvec = multi_action_space.nvec[:single_agent_length]
#             return gym.spaces.MultiDiscrete(nvec)
#         else:
#             raise NotImplementedError(f"Action space not implemented: {multi_action_space}")

#     def reset(self):
#         obs = self.venv.reset()
#         obs = np.repeat(obs, repeats=self.num_agents, axis=0)
#         return obs

#     def step_async(self, actions):
#         actions = np.reshape(actions, (self.venv.num_envs, -1))
#         return self.venv.step_async(actions)

#     def step_wait(self):
#         obs, rews, dones, infos = self.venv.step_wait()
#         obs = np.repeat(obs, repeats=self.num_agents, axis=0)
#         rews = rews.reshape(self.num_envs)
#         dones = np.repeat(dones, repeats=self.num_agents, axis=0)
#         infos = np.repeat(infos, repeats=self.num_agents, axis=0)
#         return obs, rews, dones, infos


# def main8():
#     retro.data.add_custom_integration("custom")

#     def wrap_deepmind_n64(env, reward_scale=1 / 100.0, frame_stack=1, normalize_observations=True):
#         env = MaxAndSkipEnv(env, skip=4)
#         env = WarpFrame(env, width=450, height=300, grayscale=False)
#         env = FrameStack(env, frame_stack)
#         env = ScaledFloatFrame(env)
#         if normalize_observations:
#             env = ImageNormalizer(env, mean=SSB64_IMAGE_MEAN)
#         env = RewardScaler(env, scale=reward_scale)
#         return env

#     def make_env():
#         retro.data.add_custom_integration("custom")
#         state = "ssb64.pikachu.dk.dreamland.state"
#         env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
#                                    use_restricted_actions=retro.Actions.MULTI_DISCRETE,
#                                    inttype=retro.data.Integrations.CUSTOM,
#                                    obs_type=retro.Observations.IMAGE,
#                                    state=state,
#                                    players=2)
#         env = wrap_deepmind_n64(env)
#         return env

#     gpu_options = tf.GPUOptions(allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#     num_envs = 1
#     num_agents = 2
#     # env = make_env()
#     # env = SubprocVecEnv([make_env] * num_envs)
#     # env = DummyVecEnv([make_env] * num_envs)
#     env = MultiAgentDummyVecEnv(num_agents=2, env_fns=[make_env] * num_envs)
#     env = MultiAgentToSingleAgent(env, num_agents=num_agents)

#     env.reset()
#     num_steps = 20000
#     # action = [np.array([0, 0, 0])]
#     # action = [env.action_space.sample() for _ in range(2)]
#     for i in range(num_steps):
#         sys.stdout.write(f"\r{i+1} / {num_steps}")
#         if isinstance(env, DummyVecEnv) or isinstance(env, MultiAgentDummyVecEnv):
#             action = env.action_space.sample()
#         else:
#             action = [env.action_space.sample() for _ in range(num_envs * num_agents)]
#         obs, reward, done, info = env.step(action)

#         print(f"\nreward: {reward} done: {done}")
#         # input()
#         if (isinstance(done, bool) and done) or (isinstance(done, list) and all(done)):
#             env.reset()
#         # env.render()

#         if i % 50 == 0:

#             if len(obs.shape) == 4:
#                 image = Image.fromarray((obs[0] * 255).astype(np.uint8))
#                 image.save("/home/wulfebw/Desktop/color.png")

#                 plt.imshow(obs[0, :, :])
#             elif len(obs.shape) == 3:
#                 plt.imshow(obs)

#             # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
#             # for j in range(1):
#             #     row = j // 2
#             #     col = j % 2
#             #     print(row)
#             #     print(col)
#             #     axs[row, col].imshow(obs[:, :])
#             plt.show()
#             plt.close()
#     end = time.time()
#     print(end - start)

#     return env


# def main9():
#     retro.data.add_custom_integration("custom")

#     def wrap_deepmind_n64(env, reward_scale=1 / 100.0, frame_stack=1, normalize_observations=True):
#         env = MaxAndSkipEnv(env, skip=4)
#         env = WarpFrame(env, width=450, height=300, grayscale=False)
#         env = ScaledFloatFrame(env)
#         if normalize_observations:
#             env = ImageNormalizer(env, mean=SSB64_IMAGE_MEAN)
#         env = RewardScaler(env, scale=reward_scale)
#         return env

#     def make_env():
#         retro.data.add_custom_integration("custom")
#         state = "ssb64.pikachu.dk.dreamland.state"
#         env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
#                                    use_restricted_actions=retro.Actions.MULTI_DISCRETE,
#                                    inttype=retro.data.Integrations.CUSTOM,
#                                    obs_type=retro.Observations.IMAGE,
#                                    state=state,
#                                    players=2)
#         env = wrap_deepmind_n64(env)
#         return env

#     gpu_options = tf.GPUOptions(allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#     num_envs = 1
#     num_agents = 2
#     # env = make_env()
#     # env = SubprocVecEnv([make_env] * num_envs)
#     # env = DummyVecEnv([make_env] * num_envs)
#     env = MultiAgentDummyVecEnv(num_agents=2, env_fns=[make_env] * num_envs)
#     env = MultiAgentToSingleAgent(env, num_agents=num_agents)
#     env = VecFrameDiff(env)

#     env.reset()
#     num_steps = 20000
#     # action = [np.array([0, 0, 0])]
#     # action = [env.action_space.sample() for _ in range(2)]
#     for i in range(num_steps):
#         sys.stdout.write(f"\r{i+1} / {num_steps}")
#         if isinstance(env, DummyVecEnv) or isinstance(env, MultiAgentDummyVecEnv):
#             action = env.action_space.sample()
#         else:
#             action = [env.action_space.sample() for _ in range(num_envs * num_agents)]
#         obs, reward, done, info = env.step(action)

#         print(f"\nreward: {reward} done: {done}")
#         # input()
#         if (isinstance(done, bool) and done) or (isinstance(done, list) and all(done)):
#             env.reset()
#         # env.render()

#         if i % 50 == 0:

#             if len(obs.shape) == 4:
#                 # image = Image.fromarray((obs[0] * 255).astype(np.uint8))
#                 # image.save("/home/wulfebw/Desktop/color.png")
#                 import ipdb
#                 ipdb.set_trace()
#                 # plt.imshow(obs[0, :, :])
#             elif len(obs.shape) == 3:
#                 plt.imshow(obs)

#             # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
#             # for j in range(1):
#             #     row = j // 2
#             #     col = j % 2
#             #     print(row)
#             #     print(col)
#             #     axs[row, col].imshow(obs[:, :])
#             plt.show()
#             plt.close()
#     end = time.time()
#     print(end - start)

#     return env


if __name__ == "__main__":
    main5()
