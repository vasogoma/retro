"""
Code made by Valeria Gonzalez

Train and evaluate the AI agents using a self-play league system. To be run after curriculum learning has been completed.
It runs multiple training cycles where agents compete in 1v1 matches, update their Elo ratings, and the top performers are retrained using DQN.
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from kane import Kane
import ssb64_train as ssb64

#Keep 8 opponents so 4 1v1 matches can be played
#All 8 starting opponents are kane lvl_3 base agents
NUM_MATCHES_VS=5 #Number of matches to be played
NUM_STEPS_RETRAIN=100000 # Training steps for retraining

def update_elo(opponent1,opponent2,victory_count,opp_victory_count):
    """
    Update the ELO rating of both opponents based on match results.
    
    Parameters:
    opponent1 (Kane): First opponent
    opponent2 (Kane): Second opponent
    victory_count (int): Wins of opponent1
    opp_victory_count (int): Wins of opponent2
    """
    k=32 # ELO factor
    r1= 10**(opponent1.elo/400)
    r2= 10**(opponent2.elo/400)
    e1 = r1/(r1+r2) #Expected score for opponent1
    e2 = r2/(r1+r2) #Expected score for opponent2
    s1 = victory_count
    s2 = opp_victory_count
    opponent1.elo += k*(s1-e1) #Update ELO opponent1
    opponent2.elo += k*(s2-e2) #Update ELO opponent2
    print(f"Opponent 1 new elo: {opponent1.elo}")
    print(f"Opponent 2 new elo: {opponent2.elo}")

def play_matches(league):
    """
    Shuffle the league and play 1vs1 matches between all opponents (8)
    
    Parameters:
    league (list): List of Kane agents
    """
    #Get a random set of 2 opponents
    inds=[0,1,2,3,4,5,6,7]
    np.random.shuffle(inds) #Shuffle order to create random matchups
    #play 1v1 matches between all 8 opponents
    for i in range(0,8,2): #Iterate over the pair of opponets
        opponent1=league[inds[i]]
        opponent2=league[inds[i+1]]
        opponent1.player_num=1
        opponent2.player_num=2
        print(f"Playing match between {opponent1.name} and {opponent2.name}")
        mean_reward,victory_count=ssb64.run_vs(opponent1, opponent2, num_eps=NUM_MATCHES_VS)
        opp_victory_count= NUM_MATCHES_VS-victory_count #Calculate opponets victories
        print(f"Mean reward: {mean_reward}")
        print(f"Opponent 1 victory count: {victory_count}")
        print(f"Opponent 2 victory count: {opp_victory_count}")
        update_elo(opponent1,opponent2,victory_count,opp_victory_count)

def get_best_agents(league):
    """
    Get the 4 best agents based on their ELO
    
    Parameters:
    league (list): List of Kane agents
    
    Returns:
    list: The top 4 agents sorted by ELO rating
    """
    sorted_league=sorted(league,key=lambda x: x.elo,reverse=True)
    best_agents=sorted_league[:4]
    return best_agents


def train_new_agents(best_agents,cycle_num,cycle):
    """
    Train 4 new agents using the best agents from the last cycle
    
    Parameters:
    best_agents (list): List of top 4 agents
    cycle_num (int): The current cycle number
    cycle (int): The cycle index
    
    Returns:
    list: Updated list of best agents including newly trained ones
    """
    #Get a random set of 2 opponents
    inds=[0,1,2,3]
    np.random.shuffle(inds) #Shuffle for random matchups
    #Play 1v1 matches between all 8 opponents
    for i in range(0,4):
        j=i+1 if i+1<4 else 0 #Select the next opponent by cycle
        parent=best_agents[i]
        opponent=best_agents[j]
        opponent.player_num=2
        new_agent_name=f"kane_{cycle_num}_from_{parent.name}"
        if os.path.exists(new_agent_name+".txt"):
            print(f"Agent {new_agent_name} already exists, skipping")
        else:
            # Remove old training files if they exist
            if os.path.exists(f"rewards-{new_agent_name}.csv"):
                os.remove(f"rewards-{new_agent_name}.csv")
            if os.path.exists(f"episodes-{new_agent_name}.csv"):
                os.remove(f"episodes-{new_agent_name}.csv")
            if os.path.exists(f"tf_checkpoints-{new_agent_name}"):
                shutil.rmtree(f"tf_checkpoints-{new_agent_name}")
            if os.path.exists(f"metrics/{new_agent_name}"):
                shutil.rmtree(f"metrics/{new_agent_name}")
            if os.path.exists(f"{new_agent_name}"):
                shutil.rmtree(f"{new_agent_name}")

            print(f"Training agent {new_agent_name} vs {opponent.name}")
            env= ssb64.make_env("vs") # VS mode
            act = ssb64.train_dqn(env,"vs",opponent=opponent,run_name=new_agent_name,load_model=parent.model_name,nsteps=NUM_STEPS_RETRAIN,exp_fr=0.05)
            tf.saved_model.save(act, new_agent_name)
            env.close()

            with open(f"{new_agent_name}.txt","w") as f:
                f.write(f"{new_agent_name},{new_agent_name},{parent.elo}\n")
            print(f"New agent {new_agent_name} trained")
        
        new_opponent=Kane(new_agent_name, new_agent_name,player_num=1,elo=parent.elo)
        best_agents.append(new_opponent) #Add the new agent to the list of best agents
        cycle_num+=1
    return best_agents

def main():

    league=[]

    #Open a csv file to write the elo ratings to
    for i in range(5): #Run 5 training cycles
        print(f"Cycle {i} starting")
        #Play 1v1 matches between all 8 opponents randomly
        print("Playing matches")
        if os.path.exists(f"cycle-{i+1}-elo.csv"):
            print("Cycle already exists, skipping")
            continue
        #Check if file exists
        if os.path.exists(f"cycle-{i}-elo.csv"):
            print(f"File cycle-{i}-elo.csv already exists")
            with open(f"cycle-{i}-elo.csv","r") as f:
                lines=f.readlines()
                for line in lines[1:]: #To skip header
                    name,model_name,elo=line.split(",")
                    opponent = Kane(model_name,name, player_num=1,elo=float(elo))
                    league.append(opponent)
        else:    
            if i==0:
                for k in range(8):
                    opponent = Kane("kane_lvl_3",f"kane_baseline_{k}", player_num=1,elo=400)
                    league.append(opponent)
            play_matches(league)
            #Get the elo ratings
            print(f"Final elo ratings:")
            with open(f"cycle-{i}-elo.csv","w") as f:
                f.write("name,model,elo\n")
                for opponent in league:
                    print(f"{opponent.name}: {opponent.elo}")
                    f.write(f"{opponent.name},{opponent.model_name},{opponent.elo}\n")
        #Get the 4 best agents, then train them against each other to get 4 newer agents
        best_agents=get_best_agents(league)
        for agent in best_agents:
            print(f"Best agent: {agent.name} with elo: {agent.elo}")
        new_league=[]
        for agent in best_agents:
            new_league.append(agent)
        print("Training new agents")
        new_league=train_new_agents(new_league,4+4*(i+1),i)
        print("New agents trained")
        for agent in new_league:
            print(f"New agent: {agent.name} with elo: {agent.elo}")
        league=new_league
        print(f"Cycle {i} complete")
    print("Playing matches")
    play_matches(league)

    #Get the elo ratings
    print(f"Final elo ratings:")
    for opponent in league:
        print(f"{opponent.name}: {opponent.elo}")
        
    #Get the 4 best agents, then train them against each other to get 4 newer agents
    best_agents=get_best_agents(league)
    for agent in best_agents:
        print(f"Best agent: {agent.name} with elo: {agent.elo}")

    #Print to visually be sure I am done with this phase
    print("DONE DONE DONE DONE")

if __name__ == '__main__':
    main()