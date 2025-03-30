import os

#from baselines.bench import Monitor
#from baselines.common.atari_wrappers import (WarpFrame, FrameStack, ScaledFloatFrame, MaxAndSkipEnv)
#from baselines.common.retro_wrappers import (MovieRecord, RewardScaler)
#from baselines.common.vec_env import (SubprocVecEnv, DummyVecEnv, VecFrameStack)
#from baselines.common.wrappers import TimeLimit
#from baselines.logger import configure
#from baselines.ppo2 import ppo2
import gym
import numpy as np
import retro
import matplotlib.pyplot as plt

#import tensorflow as tf

#from retro.examples.vec_frame_diff import VecFrameDiff

SSB64_IMAGE_MEAN = [0.39777202, 0.56584128, 0.43192356]


class ImageNormalizer(gym.ObservationWrapper):
    def __init__(self, env, mean):
        super().__init__(env)
        self.mean = np.array(mean, dtype=np.float32)
        # expand to roughly [-1,1]?
        self.scale = 2
        self.observation_space = gym.spaces.Box(low=-self.mean.max(),
                                                high=1 - self.mean.min(),
                                                shape=env.observation_space.shape,
                                                dtype=np.float32)

    def observation(self, observation):
        return (observation - self.mean) * self.scale


# def wrap_n64(env,
#              reward_scale=1 / 100.0,
#              frame_skip=4,
#              width=150,
#              height=100,
#              grayscale=True,
#              normalize_observations=True):
#     env = MaxAndSkipEnv(env, skip=frame_skip)
#     env = WarpFrame(env, width=width, height=height, grayscale=grayscale)
#     env = ScaledFloatFrame(env)
#     if normalize_observations:
#         env = ImageNormalizer(env, mean=SSB64_IMAGE_MEAN)
#     env = RewardScaler(env, scale=1 / 100.0)
#     return env


# def wrap_monitoring_n64(env,
#                         max_episode_steps=5000,
#                         monitor_filepath=None,
#                         movie_dir=None,
#                         record_movie_every=10):
#     env = TimeLimit(env, max_episode_steps=max_episode_steps)
#     if monitor_filepath is not None:
#         env = Monitor(env, monitor_filepath, allow_early_resets=True)
#     if movie_dir is not None:
#         env = MovieRecord(env, movie_dir, k=record_movie_every)
#     return env


def main():
    expdir = os.path.join("/home/vasogoma/retro/experiments", "ssb64_005", "run_003")
    os.makedirs(expdir, exist_ok=True)
    monitor_filepath = os.path.join(expdir, "monitor.csv")
    movie_dir = os.path.join(expdir, "movies")
    os.makedirs(movie_dir, exist_ok=True)
    load_filepath = None
    # load_filepath = "/home/wulfebw/experiments/ssb64_004/run_006/checkpoints/00100"

    # This configures baselines logging.
    #configure(dir=expdir)
    # Creating the session here prevents tf from using all the gpu memory, which
    # causes a failure in the emulator. I'm not sure why because when the emulator
    # is running with angrylion I thought it wasn't using any gpu memory, but
    # there's a lot here I don't understand so oh well.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            # intra_op_parallelism_threads=1,
                                            # inter_op_parallelism_threads=1,
                                            # gpu_options=gpu_options))

    def make_env(rank, grayscale=True):
        retro.data.add_custom_integration("custom")
        #state = "ssb64.pikachu.dk.dreamland.state"
        state = "test2.state"
        env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                                   use_restricted_actions=retro.Actions.MULTI_DISCRETE,
                                   inttype=retro.data.Integrations.CUSTOM,
                                   state=state,
                                   players=2,
                                   obs_type=retro.Observations.IMAGE)
     #   env = wrap_n64(env, grayscale=grayscale)
     #   env = wrap_monitoring_n64(env, monitor_filepath=monitor_filepath, movie_dir=movie_dir)
        return env

    # def make_vec_env(nenvs=4, recurrent=False, grayscale=True, frame_stack=4, frame_diff=False):
    #     venv = SubprocVecEnv([lambda: make_env(rank, grayscale=grayscale) for rank in range(nenvs)])
    #     # Uncomment this line in place of the one above for debugging.
    #     # venv = DummyVecEnv([lambda: make_env(0)])

    #     if not recurrent:
    #         if frame_diff:
    #             venv = VecFrameDiff(venv)
    #         else:
    #             # Perform the frame stack at the vectorized environment level as opposed to at
    #             # the individual environment level. I think this allows you to communicate fewer
    #             # images across processes.
    #             venv = VecFrameStack(venv, frame_stack)
    #     return venv

    network_name = "impala_cnn"
    recurrent = "lstm" in network_name
    grayscale = False
    frame_stack = 2
    frame_diff = False

    env= make_env(0, grayscale=grayscale)
    i=0
    state = env.reset()
    import cv2

    #remove the old images
    for file in os.listdir("pics"):
        os.remove(f"pics/{file}")
    dqn_reward_accum = 0
    dqn_reward_accum2 = 0
    while(True):
        #env.reset()
        #env.step([1,1,1])
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        actions= [action1[0], action1[1], action1[2], action2[0], action2[1], action2[2]]
        #print(actions)
        i+=1
        state, reward, terminal, info = env.step(actions)
        dqn_reward_accum += reward[0]
        dqn_reward_accum2 += reward[1]
        # Update the window with the new image
        #print(f"i: {i}, reward: {reward}, reward accum p1 {dqn_reward_accum}, reward accum p2 {dqn_reward_accum2}")

        if i>1000:
            break
        if i%100==0:
            plt.imshow(state)
            plt.savefig(f"pics/color{i}.png")
        if terminal:
            plt.imshow(state)
            plt.savefig(f"pics/color{i}.png")
            print(f"reward: {reward}")
            break

        #env.step(env.action_space.sample())
        #env.reset()
    #convert the images to a video mp4
    return
    img_array = []
    for i in range(1,i):
        img_path = f"pics/color{i}.png"
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    # venv = make_vec_env(nenvs=16,
    #                     recurrent=recurrent,
    #                     grayscale=grayscale,
    #                     frame_stack=frame_stack,
    #                     frame_diff=frame_diff)
    # ppo2.learn(network=network_name,
    #            env=venv,
    #            total_timesteps=int(10e6),
    #            nsteps=256,
    #            nminibatches=8,
    #            lam=0.95,
    #            gamma=0.999,
    #            noptepochs=3,
    #            log_interval=1,
    #            ent_coef=.01,
    #            lr=lambda f: f * 5e-4,
    #            cliprange=0.2,
    #            save_interval=10,
    #            load_path=load_filepath)


if __name__ == '__main__':
    main()
