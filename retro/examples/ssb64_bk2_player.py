import os
import numpy as np
import retro
import matplotlib.pyplot as plt
from baselines import deepq
import cv2

# Function to create the environment for training
def make_env(state,num_players=1):
    retro.data.add_custom_integration("custom")
    env = retro.n64_env.N64Env(game="SuperSmashBros-N64",
                                use_restricted_actions=retro.Actions.DISCRETE,
                                inttype=retro.data.Integrations.CUSTOM,
                                state=state,
                                players=num_players, # If AI mode, then 1 player, if VS mode 2 players
                                record=False,
                                is_random_state= False,
                                use_exact_keys=True,
                                obs_type=retro.Observations.RAM)
    return env


def main():
    create_vid=True # Set to True to create a video of the match, set to False to not create a video
    movie = retro.Movie('SuperSmashBros-N64-pikachu-samus-ai1-000000.bk2')
    # IF MANUAL RECORDING DONT DO THIS STEP
    movie.step()
    env=make_env(movie.get_state(),num_players=movie.players)
    env.initial_state = movie.get_state()
    env.reset()
    i=0
    total_reward = 0
    if create_vid:
        #remove the old video
        if os.path.exists("project.mp4"):
            os.remove("project.mp4")
        out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
    while movie.step():
        keys = []
        for p in range(movie.players):
            for e in range(env.num_buttons):
                keys.append(movie.get_key(e, p))
        #convert keys to discrete actions
        #keys = env.action_space.sample()
        state, reward, terminal, info = env.step(keys) # take the actions for that frame, and get the new state, reward, and terminal status
        # Get the image of the current state
        state_img= env.get_screen()
        if create_vid:
            #convert to bgr
            out.write(cv2.cvtColor(state_img, cv2.COLOR_RGB2BGR))
        i+=1
        total_reward += reward
        if i%100==0 or terminal: # print the state every 100 frames and save the image
            plt.imshow(state_img) # show the image of the state
            plt.savefig(f"pics/playcolor{i}.png") # save the image
            print(state)
            print("------------------")
            print("Step: ", i)
            print(f"C: {state[0]}, P: ({state[12]},{state[13]}), V: ({state[3]},{state[4]}), MS: {state[5]}, MF: {state[6]}, D: {state[7]}, DMG: {state[8]}")
            print(f"C: {state[9]}, P: ({state[10]},{state[11]}), V: ({state[12]},{state[13]}), MS: {state[14]}, MF: {state[15]}, D: {state[16]}, DMG: {state[17]}")
            print(f"reward: {reward}")
            print(f"total reward: {total_reward}")
            print(f"actions: {keys}")
            # Match ends if terminal is true (someone won)
        movie.step()
        movie.step() #To handle the steps used for debouncing button
    out.release()
    #Open the video in the system
    os.system("open project.mp4")
    # Match ends if terminal is true (someone won)
    #convert the images to a video mp4
    return

if __name__ == '__main__':
    main()
