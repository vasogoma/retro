"""
Code made by Wulfe, adapted by Valeria Gonzalez: https://github.com/wulfebw/retro



"""

import collections
from retro.data.ssb64_ram import SSB64RAM

STAGE_BOUNDARY_LEFT = -1900
STAGE_BOUNDARY_RIGHT = 2300
STAGE_BOUNDARY_Y = 0

class SSB64GameData:
    # Assume only two players.
    num_players = 2

    def __init__(self, penalize_taking_damage=True, reward_inflicting_damage=True):
        self.penalize_taking_damage = penalize_taking_damage
        self.reward_inflicting_damage = reward_inflicting_damage
        self.ram = None
        self.player_state = None
        self.reset()

    def reset(self):
        self.ram = SSB64RAM()
        self.player_state = collections.defaultdict(lambda: collections.defaultdict(int))

    
    def update(self, ram):
        self.ram.update(ram)
        for player_index in range(self.num_players):
            stock = self.ram.player_stock(player_index)
            # Stock should only decrease (handles an edge case with defaultdict usage / initialization).
            stock_change = 0
            if stock < self.player_state[player_index]["stock"]:
                stock_change= 1
            self.player_state[player_index]["stock"] = stock
            self.player_state[player_index]["stock_change"] = stock_change

            #load positions and velocities
            player_data= self.ram.player_data(player_index)
            self.player_state[player_index]["x"] = self.ram.player_position_x(player_data)
            self.player_state[player_index]["y"] = self.ram.player_position_y(player_data)


            damage = self.ram.player_damage(player_index)
            damage_change = damage - self.player_state[player_index]["damage"]
            self.player_state[player_index]["damage"] = damage
            if stock_change == 0:
                self.player_state[player_index]["damage_change"] = damage_change
            else:
                # When a stock is lost, don't count the change in damage.
                self.player_state[player_index]["damage_change"] = 0


    #START OF MY CODE
    def current_reward_dodge(self, player_index=0,steps=0):
        # Penalize loosing a live (stock).
        stock_change = self.player_state[player_index]["stock_change"]
        reward= 0
        if stock_change != 0:
            reward= -100
        if steps>5000:
            return 20
        # Penalize getting near the stage edges.
        #if self.player_state[player_index]['x'] < (STAGE_BOUNDARY_LEFT + 100) or self.player_state[player_index]['x'] > (STAGE_BOUNDARY_RIGHT -100):
        #    reward -= 1

        # Penalize falling off the stage.
        if self.player_state[player_index]['x'] < STAGE_BOUNDARY_LEFT or self.player_state[player_index]['x'] > STAGE_BOUNDARY_RIGHT or self.player_state[player_index]['y'] < STAGE_BOUNDARY_Y:
            reward -= 2
            return -20

        # Reward for being alive.
        reward += 0.2
        
        # clamp between -300 and 300
        reward= max(min(reward, 300), -300)
        # normalize between -1 and 1
        return reward/300
    
    def current_reward(self, player_index=0):
        # Penalize loosing a live (stock).
        stock_change = self.player_state[player_index]["stock_change"]
        if stock_change != 0:
            reward= -100

        # Penalize taking damage.
        if self.penalize_taking_damage:
            reward = -self.player_state[player_index]["damage_change"]*0.1
        else:
            reward = 0

        # Reward dealing damage or having other players lose a stock.
        for other_player_index in range(self.num_players):
            if other_player_index == player_index:
                continue
            if self.reward_inflicting_damage:
                reward += self.player_state[other_player_index]["damage_change"]*5
            if self.player_state[other_player_index]["stock_change"] != 0:
                return 20
                reward += 400
        
        # Penalize getting near the stage edges.
        if self.player_state[player_index]['x'] < (STAGE_BOUNDARY_LEFT + 100) or self.player_state[player_index]['x'] > (STAGE_BOUNDARY_RIGHT -100):
            reward -= .5

        # Penalize falling off the stage.
        if self.player_state[player_index]['x'] < STAGE_BOUNDARY_LEFT or self.player_state[player_index]['x'] > STAGE_BOUNDARY_RIGHT or self.player_state[player_index]['y'] < STAGE_BOUNDARY_Y:
            reward -= 1

        # Reward for being alive.
        reward += 0.01
        
        # clamp between -300 and 300
        reward= max(min(reward, 300), -300)
        # normalize between -1 and 1
        return reward/300
    #END OF MY CODE 
    
    def is_done(self):
        nonzero_stock_count = 0
        for player_index in range(self.num_players):
            if self.player_state[player_index]["stock"] > 0:
                nonzero_stock_count += 1
        return nonzero_stock_count == 1

    def lookup_all(self):
        return dict()
