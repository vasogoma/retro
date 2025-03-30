"""
Code made by Valeria Gonzalez by adapting it from Wulfe's implementation: https://github.com/wulfebw/retro



"""

import datetime
import logging
import os
import numpy as np
import retro
import pandas as pd
from baselines import deepq
import cv2

from abel import Abel
from abel_v0 import AbelV0

# Function called after every step in training (each frame)
nameStr= datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def step_callback(lcl, _glb):
    """
    Callback function executed at each step during training.
    It logs episode information and determines when to stop training.

    Parameters:
    - lcl (dict): Local variables from the training loop.

    Returns:
    - bool: True if training should stop, False otherwise.
    """

    global nameStr
    # Mean result of the last 100 matches

    #Delete old bk2 files if reward is negative
    epinum = lcl['prev_episode_num']
    is_done = lcl['prev_done']
    filename_to_delete = lcl['filename_to_delete']
    steps = lcl['prev_step_count']

    # Delete the bk2 file if the reward is negative
    if is_done:
        logging.error(f"Episode {epinum} is done: {is_done} and the reward is {lcl['episode_rewards'][-2]}, with file {filename_to_delete} and steps {steps}")
    if is_done and lcl['episode_rewards'][-2] < 10 and epinum > 0: # To visualize only very good replays
        if os.path.exists(filename_to_delete):
            os.remove(filename_to_delete)

    mean_reward = np.mean(lcl['episode_rewards'][-101:-1])
    if lcl['t'] % 10000 == 0:
        # Save the rewards to a csv file
        with open(f"rewards-{nameStr}.csv", "a") as f:
            f.write(f"{lcl['t']},{mean_reward}\n")
    if is_done and epinum > 0: 
        with open(f"episodes-{nameStr}.csv","a") as f:
            f.write(f"{epinum},{lcl['episode_rewards'][-2]},{filename_to_delete}\n")
    # Must do at least 100 matches
    if len(lcl['episode_rewards']) < 101:
        return False

    # If won more matches than it lost, stop training
    # 100,000,000
    if mean_reward>19.0:
        logging.error(f"Mean reward: {mean_reward}")
        logging.error(f"Training stopped at {lcl['t']} steps")
        return True # stop training when callback returns true (a condition has been met)
    return False

# Function to convert the images to a video
def convert_to_video(i):
    """
    Converts a sequence of saved images into an MP4 video.
    """
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
def make_env(mode, ai_level = 1, state = None, char_list = "all"):
    """
    Creates a customized environment for training the AI

    Parameters:
    - mode (str): The mode of the game ("ai" for single-player, "vs" for multiplayer)
    - ai_level (int): The difficulty level of the AI opponent
    - state (str, optional): The initial game state to load
    - char_list (str): The list of characters available for selection

    Returns:
    - env (retro.n64_env.N64Env): The initialized game environment
    """
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
                                record=False,
                                char_list=char_list,
                                goal_dodge=False,
                                is_random_state=True if state is None else False,
                                obs_type=retro.Observations.RAM)
    return env

# Function to train a dqn model
def train_dqn(env,mode,ai_level=1,char_list='all',state=None,nsteps=1000000,opponent=None,run_name=None,load_model=None,exp_fr=0.3):
    """
    Trains a DQN model

    Parameters:
    - env (retro.n64_env.N64Env): The game environment
    - mode (str): The mode of the game ("ai" for single-player, "vs" for multiplayer)
    - ai_level (int): The difficulty level of the AI opponent
    - char_list (str): The list of characters available for selection
    - state (str, optional): The initial game state to load
    - nsteps (int): The total number of training steps
    - opponent (object, optional): An opponent AI to train against
    - run_name (str, optional): name for the training session
    - load_model (str, optional): Path to a pre-trained model checkpoint
    - exp_fr (float): Exploration fraction for the DQN training process

    Returns:
    - act (deepq Act object): The trained DQN model.
    """
    
    global nameStr
    if run_name is not None:
        nameStr = run_name
    path=f"tf_checkpoints-{nameStr}"
    os.makedirs(path)
    load_prev = False
    # Create a copy of the checkpoint
    if load_model is not None:
        load_path=f"tf_checkpoints-{load_model}"
        os.system(f"cp -r {load_path}/* {path}")
        load_prev = True
    act=deepq.learn(
        env,
        network='mlp',
        lr=1e-4,# Learning rate, following recommendation from https://github.com/vladfi1/phillip/blob/master/phillip/learner.py and https://arxiv.org/pdf/1710.02298
        adam_eps=1.5e-4, # Adam epsilon, following recommendation from same source as above
        total_timesteps=nsteps,
        buffer_size=int(nsteps/25), #buffer of 1/25: 4% 
        exploration_fraction=exp_fr,
        exploration_final_eps=0.02,
        train_freq=1, # Re train after # of episodes. Change to 1 for faster training
        learning_starts=int(nsteps/50), # Start training after 2% of steps. ~80 episodes
        target_network_update_freq=500,
        prioritized_replay=True, # Change to False to disable prioritized replay
        prioritized_replay_alpha=0.5,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        print_freq=10,
        checkpoint_freq=10000,
        callback=step_callback,
        param_noise=False, # Paramer noise for exploration
        load_path=path,
        load_from_previous_checkpoint=load_prev, # Set to True to load from a previous checkpoint, very first one is set to False
        metric_log_folder=f"metrics/{nameStr}",
        new_env_fn=make_env,
        opponent=opponent,
        mode=mode, # Set the mode to AI or VS
        ai_level=ai_level, # Set the AI level
        new_env_char_list=char_list,
        new_env_state=state
    )
    return act

# Run simulation using Abel for testing the agent
def test_ai(opponent, ai_level,char_list=["pikachu","mario", "dk","link","samus","kirby","yoshi", "fox" ], num_eps=200):
    """
    Simulates AI match against a given opponent to evaluate performance
    
    Parameters:
    - opponent: The AI opponent to test
    - ai_level: The difficulty level of the AI
    - char_list: List of characters available for selection
    - num_eps: Number of episodes to run
    
    Saves results to a CSV
    """
    
    rewards = []
    matches_data = []
    victory = False
    for i in range (num_eps):
        try:
            #Create the environment
            env = make_env("ai", char_list=char_list,ai_level=ai_level)

            j=0
            # Reset the environment
            state = env.reset()
            #remove the old images
            for file in os.listdir("pics"):
                os.remove(f"pics/{file}")
            #env.record_movie("test.bk2")
            cum_reward=0
            while(True):
                #Add randomness at the start to counter the deterministic behaviour of both agents
                if j < 10:
                    actions=[env.action_space.sample()] # take a random action for player
                else:
                    actions=[opponent.policy(state)] # take an action for player
                j+=1
                state, reward, terminal, info = env.step(actions) # take the actions for that frame, and get the new state, reward, and terminal status
                # Get the image of the current state
                state_img= env.get_screen()
                cum_reward+=reward
                # Match ends if terminal is true (someone won)
                if terminal:
                    print(f"Done with episode {i+1} reward: {cum_reward} after {j} steps")
                    rewards.append(cum_reward)
                    if cum_reward > 5:
                        victory = True
                    else:
                        victory = False
                    matches_data.append({
                        "player1":env.player1_name,
                        "player2":env.player2_name,
                        "victory":victory,
                        "reward":cum_reward
                    })
                    break
            env.close()
        except Exception as e:
            print(f"Error in episode {i+1}: {e}")
            continue
    print(f"rewards:{rewards}")
    print(f"Mean reward:{np.mean(rewards)}")
    matches_dataframe = pd.DataFrame(matches_data)
    matches_dataframe.to_csv(f"final_vs/{opponent.name}-ai{ai_level}_results.csv")

def run_vs(opponent1,opponent2,char_list=["pikachu","mario", "dk","link","samus","kirby","yoshi", "fox" ], num_eps=200):
    """
    Simulates AI vs AI battles and records results
    
    Parameters:
    - opponent1: player1
    - opponent2: player2
    - char_list: List of characters available for selection
    - num_eps: Number of episodes to run
    
    Saves results to a CSV file
    """
    rewards = []
    matches_data = []
    victory = False
    for i in range (num_eps):
        try:
            #Create the environment
            env = make_env("vs", char_list=char_list)

            j=0
            # Reset the environment
            state = env.reset()
            cum_reward=0
            while(True):
                #Add randomness at the start to counter the deterministic behaviour of both agents
                if j < 10:
                    actions=[env.action_space.sample(),env.action_space.sample()] # take a random action for player
                else:
                    actions=[opponent1.policy(state),opponent2.policy(state)] # take an action for player
                j+=1
                state, reward, terminal, info = env.step(actions) # take the actions for that frame, and get the new state, reward, and terminal status
                # Get the image of the current state
                state_img= env.get_screen()
                cum_reward+=reward[0]#Only player 1 reward
                # Match ends if terminal is true (someone won)
                if terminal:
                    print(f"Done with episode {i+1} reward: {cum_reward} after {j} steps")
                    rewards.append(cum_reward)
                    if cum_reward > 5:
                        victory = True
                    else:
                        victory = False
                    matches_data.append({
                        "player1":env.player1_name,
                        "player2":env.player2_name,
                        "victory":victory,
                        "reward":cum_reward
                    })
                    break
            env.close()
        except Exception as e:
            print(f"Error in episode {i+1}: {e}")
            continue
    print(f"rewards:{rewards}")
    print(f"Mean reward:{np.mean(rewards)}")
    matches_dataframe = pd.DataFrame(matches_data)
    matches_dataframe.to_csv(f"final_vs/{opponent1.name}-{opponent2.name}_results.csv")
    return np.mean(rewards), np.sum([1 for x in rewards if x > 5])

def main(command="train"):
    #To use for Abel testing
    if command == "test_abel":
        #test_abel(char_list=["mario"], num_eps=20)
        abel = Abel("Abel",player_num=1)
        test_ai(abel,1)
    #To use for training Kane
    elif command == "train":
        env= make_env("vs") # VS mode
        act = train_dqn(env,"vs",opponent=AbelV0("Abel",player_num=2))
    elif command == "train_pikachu":
        env= make_env("vs", state="pikachu-mario-vs.state") # VS mode
        act = train_dqn(env,"vs",state="pikachu-mario-vs.state", opponent=AbelV0("Abel",player_num=2))
    return

if __name__ == '__main__':
    with open("fault_handler.log", "w") as fobj:
        main()