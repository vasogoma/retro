"""
Code made by Valeria Gonzalez

Simplified version of the abel.py code, made to create an opponent for curriculum training that will not attack Kane, only avoid falling off the stage.
The agent will determine the action to take based on the character's positon, the enemy's position, and the character's specific abilities.
"""

# For using the actions, we can use the following. Sum the actions for combos:
DO_NOTHING = 0
UP=1 #jump
DOWN=2
LEFT=3
RIGHT=6
A=9 #normal attack
B=18 #special attack
Z=27 #shield

STAGE_BOUNDARY_LEFT = -1900
STAGE_BOUNDARY_RIGHT = 2300
STAGE_BOUNDARY_Y = 0

class AbelV0:
    def __init__(self, name,player_num=1,debug=False):
        self.name = name
        self.player_num = player_num
        self.debug = debug

    """ Basic characters: Mario = [0, 20], Fox = [1,21], DK = [2,22], Samus = [3,23], Link = [5,25], Yoshi = [6,26], Kirby = [8,28], Pikachu = [9,29] 
    Hidden characters: Luigi = [4,24], Captain Falcon = [7,27], Jigglypuff = [10,30], Ness = [11,31] (DO NOT USE!)
    [is_mario, is_fox, is_dk, is_samus, is_luigi, is_link, is_yoshi, is_falcon, is_kirby, is_pikachu, is_jigglypuff, is_ness, (0,11) (19,30)
    12, 32  = position_x, 
    13, 33 = position_y, 
    14, 34 = velocity_x, 
    15, 35 = velocity_y, 
    16, 36 = movement_state, 
    17, 37 = movement_frame, 
    18,19, 38,39 = direction """
    def convert_obs(self, obs):
        if len(obs)==1:
            obs = obs[0]
        #Check if abel is player 1 or 2, and set the variables accordingly
        if self.player_num == 1: #If abel is player 1
            self.abel_x = obs[12]
            self.abel_y = obs[13]
            self.abel_direction = obs[18]*-1+obs[19]
            self.abel_is_ranged = obs[1] or obs[3] or obs[9]
            self.abel_is_pikachu = obs[9]
            self.abel_is_mario = obs[0]
            
            #Enemy
            self.enemy_x = obs[32]
            self.enemy_y = obs[33]
            self.enemy_is_ranged = obs[21] or obs[23] or obs[29]
            self.enemy_direction = obs[38]*-1+obs[39]
        
        else: #If abel is player 2
            self.abel_x = obs[32]
            self.abel_y = obs[33]
            self.abel_direction = obs[38]*-1+obs[39]
            self.abel_is_ranged = obs[21] or obs[23] or obs[29]
            self.abel_is_pikachu = obs[29]
            self.abel_is_mario = obs[20]
            
            #Enemy
            self.enemy_x = obs[12]
            self.enemy_y = obs[13]
            self.enemy_is_ranged = obs[1] or obs[3] or obs[9]
            self.enemy_direction = obs[18]*-1+obs[19]

    def policy(self, obs):
        self.convert_obs(obs)
        generic_action = self.generic_policy()
        if generic_action is not None:
            return generic_action
        
    def generic_policy(self):

        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            if self.debug:
                print("Recover with teleport")
            action = UP + B #Recover with teleport
            return action
        
        # Step away from stage edges
        if self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            if self.debug: 
                print("Step away from left edge")
            action = RIGHT #Walk to the right
            return action
        elif self.abel_x > (STAGE_BOUNDARY_RIGHT - 100):
            if self.debug:
                print("Step away from right edge")
            action = LEFT #Walk to the left
            return action

        if self.abel_x > self.enemy_x and self.abel_direction > 0: # if abel is to the right of the enemy
            if self.debug:
                print("Turn to face enemy (to the left)")
            action = LEFT
            return action

        if self.abel_x < self.enemy_x and self.abel_direction < 0: # if abel is to the left of the enemy
            if self.debug:
                print("Turn to face enemy (to the right)")
            action = RIGHT
            return action
        
        #Distance to enemy
        distance_to_enemy = abs(self.abel_x - self.enemy_x)
        # Walk towards the enemy
        if distance_to_enemy > 500:
            if self.abel_x < self.enemy_x:
                if self.debug:
                    print("Walk towards the enemy (to the right)")
                action = RIGHT
                return action
            elif self.abel_x > self.enemy_x:
                if self.debug:
                    print("Walk towards the enemy (to the left)")
                action = LEFT
                return action
        
        # if dist is close and opponent is up jump
        dist_y= self.enemy_y - self.abel_y
        if distance_to_enemy < 100 and dist_y > 100:
            if self.debug:
                print("Jump")
            action = UP
            return action
        #if opponent is down, go down
        if dist_y < -100:
            if self.debug:
                print("Go down")
            action = DOWN
            return action
        return DO_NOTHING #No generic action taken