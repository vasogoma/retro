import os
import matplotlib.pyplot as plt
from ssb64_train import make_env

# Define the save state directory
SAVE_STATE_DIR = "/home/vasogoma/retro/retro/data/custom"

# List of characters
characters = ["mario", "dk", "fox", "kirby", "link", "samus", "pikachu", "yoshi"]

# Ensure the directory exists
if not os.path.exists(SAVE_STATE_DIR):
    raise FileNotFoundError(f"The save state directory does not exist: {SAVE_STATE_DIR}")

def generate_save_state_file_paths():
    """
    Generates file paths for all possible character pairings.
    """
    save_state_paths = []
    for player1 in characters:
        for player2 in characters:
            #file_name = f"{player1}-{player2}-ai3.state"
            #file_name = f"{player1}-{player2}-ai1.state"
            file_name = f"{player1}-{player2}-vs.state"
            save_state_paths.append(file_name)
    return save_state_paths

# Run simulation
def run_sim(env, char1, char2):
    """
    Run a simulation for a given environment for testing.
    """
    i=0
    # Reset the environment
    state = env.reset()
    
    while(True):
        action1 = env.action_space.sample() # take a random action for player1
        action2 = env.action_space.sample() # take a random action for player2
        actions= [action1, action2] # combine the actions to take for both players
        #actions=action1
        i+=1
        state, reward, terminal, info = env.step(actions) # take the actions for that frame, and get the new state, reward, and terminal status
        # Get the image of the current state
        state_img= env.get_screen()
        if i>5: # stop after 5 frames
            plt.imshow(state_img) # show the image of the state
            #plt.savefig(f"pics/color-{char1}-{char2}-ai3.png") # save the image
            #plt.savefig(f"pics/color-{char1}-{char2}-ai1.png") # save the image
            plt.savefig(f"pics/color-{char1}-{char2}-vs.png") # save the image
            break

def main():
    # List all save state file paths
    save_state_files = generate_save_state_file_paths()

    # Ensure the pics directory exists and clear it
    if not os.path.exists("pics"):
        os.makedirs("pics")
    for file in os.listdir("pics"):
        os.remove(f"pics/{file}")

    # Iterate over each file and check its integrity
    for state_file in save_state_files:
        print(state_file)
        env = make_env("vs",state=state_file)
        run_sim(env, state_file.split("-")[0], state_file.split("-")[1])
        env.close()

if __name__ == '__main__':
    main()
