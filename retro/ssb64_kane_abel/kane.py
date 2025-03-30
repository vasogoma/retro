"""
Code made by Valeria Gonzalez

Kane agent is a Reinforcement Learning agent that is trained usig DQN through Curriculum Learning and a League of Self-Play Agents. 
To be later on evalated against the game's AI and Abel (the rule based policy agent)
"""

import numpy as np
import tensorflow as tf

class Kane:
    """
    Kane class represents the Reinforcement Learning agent for training using DQN and evaluation.
    It loads a trained model, processes observations, and determines actions.

    Attributes:
    model_name (str): The name of the model checkpoint file.
    name (str): The name of the agent.
    player_num (int): The player number (either 1 or 2), defaults to 1.
    elo (int): The initial ELO rating of the agent, defaults to 400.
    model (tf.saved_model): The TensorFlow model loaded from the checkpoint.
    """
    def __init__(self, model_name,name, player_num=1,elo=400):
        """
        Initialize the Kane agent with the given model, name, player number, and ELO rating.

        Parameters:
        model_name (str): The name of the model checkpoint to load.
        name (str): The name of the agent.
        player_num (int): The player number (either 1 or 2), defaults to 1.
        elo (int): The initial ELO rating of the agent, defaults to 400.
        """
        #load model from tf_checkpoint\
        self.name=name
        self.model_name=model_name
        model = tf.saved_model.load(model_name)
        self.model = model #Assign to agent
        self.player_num = player_num
        self.elo = elo

    def convert_obs(self, obs):
        """
        Convert the observation based on the player number.
        For player 2, it swaps the observation data between player 1 and player 2.

        Parameters:
        obs (list or np.ndarray): The observation data.

        Returns:
        list: The modified observation if the player is 2, otherwise the original observation.
        """
        if self.player_num == 1:
            return obs 
        else:
            # Swap player 1 (ind. 0-19) and player 2 (ind. 20-39) data
            new_obs = []
            new_obs.extend(obs[20:40])
            new_obs.extend(obs[0:20])
            return new_obs
        
    def policy(self, obs):
        """
        Decide the action for the agent based on the given observation using the DQN policy.

        Parameters:
        obs (list or np.ndarray): The observation data from the environment.

        Returns:
        int: The action to be taken by the agent.
        """
        obs = self.convert_obs(obs)
        if type(obs)==np.ndarray:
            obs = tf.expand_dims(obs, axis=0)  # Add batch dimension
            obs = tf.cast(obs, dtype=tf.float32)
        elif type(obs)==list:
            if type(obs[0])==np.ndarray:
                obs=obs[0]
            obs = tf.expand_dims(obs, axis=0)  # Add batch dimension
            obs = tf.cast(obs, dtype=tf.float32)
        #convert to (1,40) shape
        q_values = self.model.q_network(obs)
        #Get action with highest value
        action = tf.argmax(q_values, axis=1)
        return action.numpy()[0]