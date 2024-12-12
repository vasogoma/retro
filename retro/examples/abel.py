

# For using the actions, we can use the following. Sum the actions for combos:
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

class Abel:
    def __init__(self, name):
        self.name = name

#Basic characters: Mario = [0, 19], Fox = [1,20], DK = [2,21], Samus = [3,22], Link = [5,24], Yoshi = [6,25], Kirby = [8,27], , Pikachu = [9,28] 
#Hidden characters: Luigi = [4,23], Captain Falcon = [7,26], Jigglypuff = [10,29], Ness = [11,30] (DO NOT USE!)
#[is_mario, is_fox, is_dk, is_samus, is_luigi, is_link, is_yoshi, is_falcon, is_kirby, is_pikachu, is_jigglypuff, is_ness, (0,11) (19,30)
# 12, 31  = position_x, 
# 13, 32 = position_y, 
# 14, 33 = velocity_x, 
# 15, 34 = velocity_y, 
# 16, 35 = movement_state, 
# 17, 36 = movement_frame, 
# 18, 37 = direction
    def convert_obs(self, obs):
        #Abel
        self.abel_x = obs[31]
        self.abel_y = obs[32]
        self.abel_direction = obs[37]
        self.abel_is_ranged = obs[20] or obs[22] or obs[28]
        self.abel_is_pikachu = obs[28]
        self.abel_is_mario = obs[19]
        
        #Enemy
        self.enemy_x = obs[12]
        self.enemy_y = obs[13]
        self.enemy_is_ranged = obs[1] or obs[3] or obs[9]
        self.enemy_direction = obs[18]

    def policy(self, obs):
        self.convert_obs(obs)
        generic_action = self.generic_policy()
        if generic_action is not None:
            return generic_action
        if self.abel_is_pikachu:
            return self.pikachu_policy()
        elif self.abel_is_mario:
            return self.mario_policy()
        else:
            return self.default_policy()
        
    def generic_policy(self):
        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            print("Recover with teleport")
            action = UP + B #Recover with teleport
            return action
        
        # Step away from stage edges
        if self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            print("Step away from left edge")
            action = RIGHT #Walk to the right
            return action
        elif self.abel_x > (STAGE_BOUNDARY_RIGHT - 100):
            print("Step away from right edge")
            action = LEFT #Walk to the left
            return action

        if self.abel_x > self.enemy_x and self.abel_direction > 0: # if abel is to the right of the enemy
            print("Turn to face enemy (to the left)")
            action = LEFT
            return action

        if self.abel_x < self.enemy_x and self.abel_direction < 0: # if abel is to the left of the enemy
            print("Turn to face enemy (to the right)")
            action = RIGHT
            return action
        
        return None #No generic action taken
    
    def pikachu_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = DOWN + B #THUNDER default action

        # Close-range
        if distance_x < 100 and distance_y > 100: #Adjust y?
            if self.abel_y < self.enemy_y: #Enemy is above Pikachu
                print("Pikachu Thunder")
                action = DOWN + B #THUNDER
                return action
        
        if distance_x < 50: 
            print("Close Attack: Headbutt")
            action = A #HEADBUTT
            return action
            
        # Mid-range 
        # Consider y? 
        if 100 <= distance_x <= 300: 
            if distance_y < 100:
                if self.abel_x < self.enemy_x:
                    print("Advance Right with Vault Kick")
                    action = RIGHT + A #right VAULT KICK
                else:
                    print("Advance Left with Vault Kick")
                    action = LEFT + A #left VAULT KICK
            else:
                if self.abel_x < self.enemy_x:
                    print("Advance Right")
                    action = RIGHT #right
                else:
                    print("Advance Left")
                    action = LEFT #left 
         
        # Long-range (projectile)
        if distance_x > 300:
            print("Electric Shock")
            action = B #left ELECTRIC SHOCK
        
        print(f"Pos X Abel: {self.abel_x}")
        print(f"Pos X Enemy: {self.enemy_x}")
        print(f"Pos Y Abel: {self.abel_y}")
        print(f"Pos Y Enemy: {self.enemy_y}")
            
        return action
                
    def mario_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)

        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            print("Recover with Super Jump Punch")
            action = UP + B  # Recover with Super Jump Punch

        # Step away from stage edges
        elif self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            print("Step away from left edge")
            action = RIGHT  # Walk to the right
        elif self.abel_x > (STAGE_BOUNDARY_RIGHT - 100):
            print("Step away from right edge")
            action = LEFT  # Walk to the left

        # Close-range
        elif distance_x < 100 and distance_y < 100:  # Adjust y?
            if self.abel_y < self.enemy_y:  # Enemy is above Mario
                print("Close Attack: Spinning Uppercut")
                action = UP + A  # Spinning Uppercut
            elif distance_x < 50:
                print("Close Attack: Tornado Spin")
                action = DOWN + B  # Tornado Spin
            else: #CHOOSE A CASE FOR THIS ONE
                print("Close Attack: Punch")
                action = A #Hard Punch
        
        # Mid-range
        # Consider y?
        elif 100 <= distance_x <= 300:
            if self.abel_x < self.enemy_x:
                print("Mid Range Fireball (Right)")
                action = RIGHT + B  # Fireball right
            else:
                print("Mid Range Fireball (Left)")
                action = LEFT + B # Fireball left
        
        # Get closer to enemy
        else:
            if self.abel_x > self.enemy_x:
                action = LEFT
            elif self.abel_x < self.enemy_x:
                action = RIGHT

        return action


    def default_policy(self): #General behaviour, not character specific
        
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)


        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            print("Recover with teleport")
            action = UP + B #Recover with teleport
        
        # Step away from stage edges
        if self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            print("Step away from left edge")
            action = RIGHT #Walk to the right
        elif self.abel_x < (STAGE_BOUNDARY_RIGHT - 100):
            print("Step away from right edge")
            action = LEFT #Walk to the left
        
        # Get closer to enemy
        if distance_x > 300:
            if self.abel_x > self.enemy_x:
                action = LEFT
            elif self.abel_x < self.enemy_x:
                action = RIGHT

        #Attacks
        if self.abel_x > self.enemy_x: # if abel is to the right of the enemy
            if  (self.abel_x - self.enemy_x) > 100:
                if self.abel_is_ranged:
                    print("Ranged Attack Left")
                    action = LEFT + B
                else:
                    print("Get close to enemy left") 
                    action = LEFT
            else:
                print("Close Attack")
                action = LEFT + B
        if self.abel_x < self.enemy_x: # if abel is to the left of the enemy
            if  (self.enemy_x - self.abel_x) > 100:
                if self.abel_is_ranged:
                    print("Ranged Attack Right")
                    action = RIGHT + B
                else:
                    print("Get close to enemy right")
                    action = RIGHT
            else:
                print("Close Attack")
                action = RIGHT + B
        
        print(f"Pos X Abel: {self.abel_x}")
        print(f"Pos X Enemy: {self.enemy_x}")
        print(f"Pos Y Abel: {self.abel_y}")
        print(f"Pos Y Enemy: {self.enemy_y}")
        
        return action