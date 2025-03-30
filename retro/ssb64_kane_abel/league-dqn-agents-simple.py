"""
Code made by Valeria Gonzalez

Simplified version of the league-dqn-agents.py code, which is used for testing for bugs that could cause the agent to have trouble training.
It runs multiple training cycles where agents compete in 1v1 matches, update their Elo ratings, and the top performers are retrained using DQN.
"""


import numpy as np
import tensorflow as tf
from kane import Kane
import ssb64_train as ssb64

#Keep 8 opponents so 4 1v1 matches can be played
#All 8 starting opponents are kane lvl_3 base agents
NUM_MATCHES_VS=5
NUM_STEPS_RETRAIN=100000

def update_elo(opponent1,opponent2,victory_count,opp_victory_count):
    # Do ELO calculation
    k=32
    r1= 10**(opponent1.elo/400)
    r2= 10**(opponent2.elo/400)
    e1 = r1/(r1+r2)
    e2 = r2/(r1+r2)
    s1 = victory_count
    s2 = opp_victory_count
    opponent1.elo += k*(s1-e1)
    opponent2.elo += k*(s2-e2)
    print(f"Opponent 1 new elo: {opponent1.elo}")
    print(f"Opponent 2 new elo: {opponent2.elo}")

def play_matches(league):
    #Get a random set of 2 opponents
    inds=[0,1,2,3,4,5,6,7]
    np.random.shuffle(inds)
    #play 1v1 matches between all 8 opponents
    for i in range(0,8,2):
        opponent1=league[inds[i]]
        opponent2=league[inds[i+1]]
        opponent1.player_num=1
        opponent2.player_num=2
        print(f"Playing match between {opponent1.name} and {opponent2.name}")
        mean_reward,victory_count=ssb64.run_vs(opponent1, opponent2, num_eps=NUM_MATCHES_VS)
        opp_victory_count= NUM_MATCHES_VS-victory_count
        print(f"Mean reward: {mean_reward}")
        print(f"Opponent 1 victory count: {victory_count}")
        print(f"Opponent 2 victory count: {opp_victory_count}")
        update_elo(opponent1,opponent2,victory_count,opp_victory_count)

def get_best_agents(league):
    #Get the 4 best agents
    sorted_league=sorted(league,key=lambda x: x.elo,reverse=True)
    best_agents=sorted_league[:4]
    return best_agents


def train_new_agents(best_agents,cycle_num):
    #Get a random set of 2 opponents
    inds=[0,1,2,3]
    np.random.shuffle(inds)
    #play 1v1 matches between all 8 opponents
    for i in range(0,4):
        j=i+1 if i+1<4 else 0
        parent=best_agents[i]
        opponent=best_agents[j]
        opponent.player_num=2
        new_agent_name=f"kane_{cycle_num}_from_{parent.name}"
        cycle_num+=1
        print(f"Training agent {new_agent_name} vs {opponent.name}")
        env= ssb64.make_env("vs") # VS mode
        act = ssb64.train_dqn(env,"vs",opponent=opponent,run_name=new_agent_name,load_model=parent.model_name,nsteps=NUM_STEPS_RETRAIN,exp_fr=0.05)
        tf.saved_model.save(act, new_agent_name)
        new_opponent=Kane(new_agent_name, new_agent_name,player_num=1,elo=parent.elo)
        print(f"New agent {new_agent_name} trained")
        best_agents.append(new_opponent)
        env.close()
    return best_agents

def main():
    league=[]
    for i in range(8):
        opponent = Kane("kane_lvl_3",f"kane_baseline_{i}", player_num=1,elo=400)
        league.append(opponent)

    for i in range(5):
        print(f"Cycle {i} starting")
        #play 1v1 matches between all 8 opponents randomly
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
        new_league=[]
        for agent in best_agents:
            new_league.append(agent)
        print("Training new agents")
        new_league=train_new_agents(new_league,4+4*(i+1))
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


if __name__ == '__main__':
    main()