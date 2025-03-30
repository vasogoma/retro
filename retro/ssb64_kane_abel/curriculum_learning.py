"""
Code made by Valeria Gonzalez

Train and evaluate the Kane AI agents at different levels using DQN and curriculum learning. 
The training process involves multiple cycles where the AI competes against AI opponents at various difficulty levels.
The Abelv0 represents a very simplistic version of an opponent, where the enemy will not attack and its only logic is to avoid falling off the stage.
Level 1 represents an opponent that attacks and is mildly good at avoiding dying.
level 3 represents an opponent capable of attacking, defending and avoids fallings off.
The AI agents are trained in 1v1 matches, and their models are saved after each training cycle to track progress.
"""

import tensorflow as tf
import ssb64_train as ssb64
from abel_v0 import AbelV0
from abel import Abel

def train_lvl_0(steps):
    """
    Function to train the AI for level 0.
    Trains Kane with a very basic opponent, using AbelV0 as the opponent.

    Parameters:
    steps (int): Number of training steps

    Returns:
    tuple: A tuple containing the model name and the trained model
    """
    env= ssb64.make_env("vs") # VS mode
    act = ssb64.train_dqn(env,"vs",opponent=AbelV0("Abel",player_num=2),run_name="kane_lvl_0",nsteps=steps)
    return "kane_lvl_0",act #Return model name and the trained model

def train_lvl_ai(steps,model=None,ai_level=1,exp_fr=0.2):
    """
    Train the Kane AI against game's AI of the specified level.

    Parameters:
    steps (int): Number of training steps
    model (str, optional): Pre-trained model to load (default is None)
    ai_level (int): AI opponent's level
    exp_fr (float): Exploration factor controlling randomness in actions

    Returns:
    tuple: A tuple containing the model name and the trained model
    """
    env= ssb64.make_env("ai",ai_level=ai_level) 
    act = ssb64.train_dqn(env,"ai",ai_level=ai_level,run_name=f"kane_lvl_{ai_level}",load_model=model,nsteps=steps,exp_fr=exp_fr)
    return f"kane_lvl_{ai_level}",act

def main():
    steps= 250000
    #Train against level 0
    print("Training Kane lvl 0")
    v0_model_name, v0_model=train_lvl_0(steps)
    #save weights
    tf.saved_model.save(v0_model, f"{v0_model_name}")
    #Train against level 1
    print("Training Kane lvl 1")
    v1_model_name, v1_model=train_lvl_ai(steps,v0_model_name,1,0.2)
    tf.saved_model.save(v1_model, f"{v1_model_name}")
    #Train against level 2
    print("Training Kane lvl 3")
    v3_model_name, v3_model=train_lvl_ai(steps,"kane_lvl_1",3,0.1)
    tf.saved_model.save(v3_model, f"{v3_model_name}")

if __name__ == '__main__':
    main()
