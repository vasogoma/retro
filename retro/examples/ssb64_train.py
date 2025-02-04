import datetime
import os
import numpy as np
import retro
import matplotlib.pyplot as plt
from baselines import deepq
import cv2
from baselines.common.vec_env import (SubprocVecEnv, DummyVecEnv, VecFrameStack, VecEnvWrapper,
                                      VecMonitor)
from abel import Abel
from abel_v0 import AbelV0
# Function called after every step in training (each frame)

dateStr= datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Callback function to stop training at specific conditions
def step_callback(lcl, _glb):
    global dateStr
    # stop training if reward exceeds 199
    #if lcl['t'] % 1000 == 0:
    #    print(lcl['episode_rewards'])
    #if len(lcl['episode_rewards']) >1:
    #    return True
    
    #if lcl['t'] % 1000 == 0:
    #    print(f"{lcl['t']}: {lcl['episode_rewards']}")

    # Mean result of the last 100 matches

    #Delete old bk2 files if reward is negative
    epinum = lcl['prev_episode_num']
    is_done = lcl['prev_done']
    filename_to_delete = lcl['filename_to_delete']
    steps = lcl['prev_step_count']

    # Delete the bk2 file if the reward is negative
    if is_done:
        print(f"Episode {epinum} is done: {is_done} and the reward is {lcl['episode_rewards'][-2]}, with file {filename_to_delete} and steps {steps}")
    if is_done and lcl['episode_rewards'][-2] < 100 and epinum > 0: # To visualize only very good replays
        if os.path.exists(filename_to_delete):
            os.remove(filename_to_delete)
        else:
            print(f"The file {filename_to_delete} does not exist")

    mean_reward = np.mean(lcl['episode_rewards'][-101:-1])
    if lcl['t'] % 10000 == 0:
        # Save the rewards to a csv file
        with open(f"rewards-{dateStr}.csv", "a") as f:
            f.write(f"{lcl['t']},{mean_reward}\n")
    if is_done and epinum > 0: 
        with open(f"episodes-{dateStr}.csv","a") as f:
            f.write(f"{epinum},{lcl['episode_rewards'][-2]},{filename_to_delete}\n")
    # Must do at least 100 matches
    if len(lcl['episode_rewards']) < 101:
        return False

    # If won more matches than it lost, stop training
    # 100,000,000
    if mean_reward>9.0:
        print(f"Mean reward: {mean_reward}")
        print(f"Training stopped at {lcl['t']} steps")
        return True # stop training when callback returns true (a condition has been met)
    return False

# Function to convert the images to a video
def convert_to_video(i):
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


# Function to create the environment for training
def make_env(mode, ai_level = 1, state = None):
    players = 2
    if mode == "ai":
        players = 1
    if state is not None and "-ai" in state:
        players = 1
    retro.data.add_custom_integration("custom")
    env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                                use_restricted_actions=retro.Actions.DISCRETE,
                                inttype=retro.data.Integrations.CUSTOM,
                                state=state,
                                players=players, # If AI mode, then 1 player, if VS mode 2 players
                                ai_level=ai_level,
                                record=True,
                                is_random_state=True if state is None else False,
                                obs_type=retro.Observations.RAM)
    return env

def make_vec_env(state, nenvs=4, recurrent=False, frame_stack=4, num_agents=2):
        venv = SubprocVecEnv([
            lambda: make_env(state)
            for rank in range(nenvs)
        ])
        # Uncomment this line in place of the one above for debugging.
        # venv = DummyVecEnv([lambda: make_env(0)])

        if not recurrent:
            # Perform the frame stack at the vectorized environment level as opposed to at
            # the individual environment level. I think this allows you to communicate fewer
            # images across processes.
            venv = VecFrameStack(venv, frame_stack)

        #venv = MultiAgentToSingleAgent(venv, num_agents=num_agents)
        venv = VecMonitor(venv, filename="monitor.csv")
        return venv

# Function to train a dqn model
def train_dqn(env,mode,ai_level=1):
    global dateStr
    load_prev = False
    path=f"tf_checkpoints-{dateStr}"
    os.makedirs(path)
    # Create a copy of the checkpoint
    if load_prev:
        load_path=f"tf_checkpoints-2025-01-15-00-19-50"
        os.system(f"cp -r {load_path}/* {path}")
    opponent = None
    if env.players == 2: # If VS mode, then create an opponent
        #opponent = Abel("Abel",player_num=2)
        opponent = AbelV0("Abel",player_num=2) #For curriculum learning Lvl 0
    act=deepq.learn(
        env,
        network='mlp',
        lr=8e-5,# Learning rate, following recommendation from https://github.com/vladfi1/phillip/blob/master/phillip/learner.py and https://arxiv.org/pdf/1710.02298
        adam_eps=1.5e-4, # Adam epsilon, following recommendation from same source as above
        total_timesteps=10000000,
        buffer_size=500000, #buffer of 1/200 
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        train_freq=1, # Re train after # of episodes. Change to 1 for faster training
        learning_starts=80000, # Start training after # of steps. ~80 episodes
        target_network_update_freq=32000,
        prioritized_replay=True, # Change to False to disable prioritized replay
        prioritized_replay_alpha=0.5,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=10000000,
        print_freq=10,
        checkpoint_freq=10000,
        callback=step_callback,
        param_noise=False, # Paramer noise for exploration
        load_path=path,
        load_from_previous_checkpoint=load_prev, # Set to True to load from a previous checkpoint, very first one is set to False
        metric_log_folder=f"metrics/{dateStr}",
        new_env_fn=make_env,
        opponent=opponent,
        mode=mode, # Set the mode to AI or VS
        ai_level=ai_level # Set the AI level
    )
    return act

# Run simulation
def run_sim(env):

    # Initialize Abel
    abel = Abel("Abel",player_num=1)
    #abel_v0 = AbelV0("AbelV0",player_num=1)

    i=0
    # Reset the environment
    state = env.reset()
    
    #remove the old images
    for file in os.listdir("pics"):
        os.remove(f"pics/{file}")
    #env.record_movie("test.bk2")
    while(True):
        #action1 = env.action_space.sample() # take a random action for player1
        #action2 = env.action_space.sample() # take a random action for player2
        action2 = abel.policy(state) # take an action for player2
        actions=[action2]
        i+=1
        state, reward, terminal, info = env.step(actions) # take the actions for that frame, and get the new state, reward, and terminal status
        # Get the image of the current state
        state_img= env.get_screen()
        #if i>1000: # stop after 1000 frames
        #    break
        if i%100==0: # print the state every 100 frames and save the image
            plt.imshow(state_img) # show the image of the state
            plt.savefig(f"pics/color{i}.png") # save the image
            print(state)
            print("------------------")
            print("Step: ", i)
            print(f"C: {state[0]}, P: ({state[1]},{state[2]}), V: ({state[3]},{state[4]}), MS: {state[5]}, MF: {state[6]}, D: {state[7]}, DMG: {state[8]}")
            print(f"C: {state[9]}, P: ({state[10]},{state[11]}), V: ({state[12]},{state[13]}), MS: {state[14]}, MF: {state[15]}, D: {state[16]}, DMG: {state[17]}")
            print("Action: ", action2)
        # Match ends if terminal is true (someone won)
        if terminal:
            print("Action: ", action2)
            plt.imshow(state_img) # show the image of the state
            plt.savefig(f"pics/color{i}.png") # save the image
            print(f"reward: {reward}") # print the reward
            break
    #env.stop_record()

def main():

    #To use later for multiprocessing
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #gpu_options = tf.GPUOptions(allow_growth=True)
    
    # Create the environment
    #env= make_env("ai",ai_level = 3,state="pikachu-pikachu-ai3.state") # AI3 mode
    #env= make_env("ai",ai_level = 1) # AI1 mode
    env= make_env("vs") # VS mode
    #frame_stack = 2
    #num_agents = 1
    #venv = make_vec_env('pikachu-pikachu-ai3.state',nenvs=2,
    #                    frame_stack=frame_stack,
    #                    num_agents=num_agents)
    # Train the dqn model
    act = train_dqn(env,"vs")
    #print(act)
    #run_sim(env)
    #convert the images to a video mp4
    return

if __name__ == '__main__':
    main()

#changes
# - Params: exploration rate for DQN
# -         wait for train
# -         buffer size
# -         learning rate
# -         gamma
# - Architectures
# - Reward function
# - selective training
# - more training steps
# step n-steps until action a is done with the animation