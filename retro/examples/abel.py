

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

class Abel:
    def __init__(self, name,player_num=1):
        self.name = name
        self.player_num = player_num

    """ Basic characters: Mario = [0, 19], Fox = [1,20], DK = [2,21], Samus = [3,22], Link = [5,24], Yoshi = [6,25], Kirby = [8,27], , Pikachu = [9,28] 
    Hidden characters: Luigi = [4,23], Captain Falcon = [7,26], Jigglypuff = [10,29], Ness = [11,30] (DO NOT USE!)
    [is_mario, is_fox, is_dk, is_samus, is_luigi, is_link, is_yoshi, is_falcon, is_kirby, is_pikachu, is_jigglypuff, is_ness, (0,11) (19,30)
    12, 31  = position_x, 
    13, 32 = position_y, 
    14, 33 = velocity_x, 
    15, 34 = velocity_y, 
    16, 35 = movement_state, 
    17, 36 = movement_frame, 
    18, 37 = direction """

    def convert_obs(self, obs):
        if len(obs)==1:
            obs = obs[0]
        #Abel
        if self.player_num == 1:
            self.abel_x = obs[12]
            self.abel_y = obs[13]
            self.abel_direction = obs[18]
            self.abel_is_ranged = obs[1] or obs[3] or obs[9]
            self.abel_is_pikachu = obs[9]
            self.abel_is_mario = obs[0]
            self.abel_is_dk = obs[2]
            self.abel_is_link = obs[5]
            self.abel_is_samus = obs[3]
            self.abel_is_yoshi = obs[6]
            self.abel_is_kirby = obs[8]
            self.abel_is_fox = obs[1]
            
            #Enemy
            self.enemy_x = obs[31]
            self.enemy_y = obs[32]
            self.enemy_is_ranged = obs[20] or obs[22] or obs[28]
            self.enemy_direction = obs[37]
        
        else:
            self.abel_x = obs[31]
            self.abel_y = obs[32]
            self.abel_direction = obs[37]
            self.abel_is_ranged = obs[20] or obs[22] or obs[28]
            self.abel_is_pikachu = obs[28]
            self.abel_is_mario = obs[19]
            self.abel_is_dk = obs[21]
            self.abel_is_link = obs[24]
            self.abel_is_samus = obs[22]
            self.abel_is_yoshi = obs[25]
            self.abel_is_kirby = obs[27]
            self.abel_is_fox = obs[20]
            
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
        elif self.abel_is_dk:
            return self.dk_policy()
        elif self.abel_is_link:
            return self.link_policy()
        elif self.abel_is_samus:
            return self.samus_policy()
        elif self.abel_is_yoshi:
            return self.yoshi_policy()
        elif self.abel_is_kirby:
            return self.kirby_policy()
        elif self.abel_is_fox:
            return self.fox_policy()
        
        else:
            print("Unknown character")
        
    def generic_policy(self):
        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            #print("Generic: Recover with teleport")
            action = UP + B #Recover with teleport
            return action
        
        # Step away from stage edges
        if self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            #print("Generic: Step away from left edge")
            action = RIGHT #Walk to the right
            return action
        elif self.abel_x > (STAGE_BOUNDARY_RIGHT - 100):
            #print("Generic: Step away from right edge")
            action = LEFT #Walk to the left
            return action

        if self.abel_x > self.enemy_x and self.abel_direction > 0: # if abel is to the right of the enemy
            #print("Generic: Turn to face enemy (to the left)")
            action = LEFT
            return action

        if self.abel_x < self.enemy_x and self.abel_direction < 0: # if abel is to the left of the enemy
            #print("Generic: Turn to face enemy (to the right)")
            action = RIGHT
            return action
        
        return None #No generic action taken
    
    def pikachu_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = DOWN + B #THUNDER default action

        # Close-range
        if distance_x < 250 and distance_y > 100: 
            if self.abel_y < self.enemy_y: #Enemy is above Pikachu
                #print("Pikachu Thunder")
                action = DOWN + B #THUNDER
                return action
            if self.abel_y > self.enemy_y: #Enemy is bellow Pikachu
                action = DOWN #Go down
                return action
        
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            #print("Close Attack: Headbutt")
            action = A #HEADBUTT
            return action
            
        # Mid-range 
        if 250 <= distance_x <= 500: 
            if distance_y < 100:
                if self.abel_x < self.enemy_x:
                    #print("Advance Right with Vault Kick")
                    action = RIGHT + A #right VAULT KICK
                    return action
                else:
                    #print("Advance Left with Vault Kick")
                    action = LEFT + A #left VAULT KICK
                    return action
            else:
                if self.abel_x < self.enemy_x:
                    #print("Advance Right")
                    action = RIGHT #right
                    return action
                else:
                    #print("Advance Left")
                    action = LEFT #left 
                    return action
         
        # Long-range (projectile)
        if distance_x > 500:
            #print("Electric Shock")
            action = B #left ELECTRIC SHOCK
            return action
        
        #print(f"Pos X Abel: {self.abel_x}")
        #print(f"Pos X Enemy: {self.enemy_x}")
        #print(f"Pos Y Abel: {self.abel_y}")
        #print(f"Pos Y Enemy: {self.enemy_y}")
            
        return action
                
    def mario_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B # default action
        
        # Close-range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y:  # Enemy is above Mario
                #print("Close Attack: Spinning Uppercut")
                action = UP + A  # Spinning Uppercut
                return action
            else:
                #print("Go down")
                action = DOWN  # Go down
                return action
        
        # Closee-range same height
        if distance_x < 250 and distance_y < 100:
            #print("Close Attack: Tornado Spin")
            action = DOWN + B  # Tornado Spin
            return action
        
        # Mid-range
        if 250 <= distance_x <= 500:
            if self.abel_x < self.enemy_x:
                #print("Mid Range Fireball (Right)")
                action = RIGHT + B  # Fireball right
                return action
            else:
                #print("Mid Range Fireball (Left)")
                action = LEFT + B # Fireball left
                return action
        
        # Get closer to enemy
        else:
            if self.abel_x > self.enemy_x:
                action = LEFT
            elif self.abel_x < self.enemy_x:
                action = RIGHT

        return action
    
    def dk_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: # Enemy is above DK
                #print("DK :Close Attack: Helicopter punch")
                #print("Jump")
                action = UP
                return action
            else: #Enemy is bellow DK
                #print("DK :Go down")
                action = DOWN #Go down
                return action
            
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            #print("DK :Close Attack: Ground Pound")
            action = DOWN + B #Ground Pound
            return action
        
        # Outside of DK range of attacks
        if distance_x >= 400:
            if self.abel_x > self.enemy_x:
                #print(f"DK :Advance Left, {self.abel_x} > {self.enemy_x}, {distance_x}")
                action = LEFT
                return action
            else:
                #print(f"DK :Advance Right, {self.abel_x} < {self.enemy_x}, {distance_x}")
                action = RIGHT
                return action
        #print("DK :Default")
        return action
    
    def link_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Link
                #print("Link: Close Attack: Spin Attack")
                print("Link: Jump")
                action = UP
                return action
            elif self.abel_y > self.enemy_y: #Enemy is bellow Link
                #print("Link: Close Attack: Go down")
                action = DOWN
                return action
        
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            if self.abel_x > self.enemy_x: #Enemy is to the left of Link
                #print("Link: Close Attack: Left Dash Attack")
                action = LEFT + A
                return action
            else: #Enemy is to the Right of Link
                #print("Link: Close Attack: Right Dash Attack")
                action = RIGHT + A
                return action
        
        # Mid-range
        if 250 <= distance_x <= 500:
            if self.abel_x < self.enemy_x: #Enemy is to the right of Link
                #print("Link: Mid Range: Boomerang")
                action = RIGHT + B
                return action
            else: #Enemy is to the left of Link
                #print("Link: Mid Range: Boomerang")
                action = LEFT + B
                return action

        # Long-range   
        if distance_x > 500:
            if self.abel_x < self.enemy_x: #Enemy is to the right of Link
                #print("Link: Walk to the right")
                action = RIGHT
                return action
            else: #Enemy is to the left of Link
                #print("Link: Walk to the left")
                action = LEFT
                return action

        return action

    def samus_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Samus
                #print("Close Attack: Upward smash")
                action = UP + A
                return action
            elif self.abel_y > self.enemy_y: #Enemy is bellow Samus
                #print("Close Attack: Go down")
                action = DOWN
                return action
            
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            if self.abel_x < self.enemy_x: #Enemy is to the right of Samus
                #print("Close Attack: Right Forward Smash")
                action = RIGHT + A
                return action
            else: #Enemy is to the left of Samus
                #print("Close Attack: Left Forward Smash")
                action = LEFT + A
                return action

        # Long-range   
        if distance_x >= 500:
            #print("Mid Range: Charge shot")
            action = B
            return action
        
        # Get closer to enemy
        else:
            if self.abel_x < self.enemy_x:
                action = LEFT
            elif self.abel_x > self.enemy_x:
                action = RIGHT

        return action

    def yoshi_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Yoshi
                #print("Close Attack: Egg Throw")
                action = UP + B
                return action
            else: #Enemy is bellow Yoshi
                #print("Go Down")
                action = DOWN
                return action
        
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            #print("Close Attack: Ground Pound")
            action = DOWN + B
            return action

        # Mid-range
        if 250 <= distance_x <= 500:
            #print("Mid Range: Swallow")
            action = B
            return action

        # Long-range
        if distance_x > 500:
            if self.abel_x < self.enemy_x:
                #print("Walk to the right")
                action = RIGHT
                return action
            else:
                #print("Walk to the left")
                action = LEFT
                return action
        
        return action

    def kirby_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Kirby
                #print("Close Attack: Final Cutter")
                action = UP + B
                return action
            elif self.abel_y > self.enemy_y: #Enemy is bellow Kirby
                #print("Close Attack: Stone")
                action = DOWN + B
                return action
        
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            if self.abel_x < self.enemy_x:
                #print("Close Attack: Right Forward Smash")
                action = RIGHT + A
                return action
            else:
                #print("Close Attack: Left Forward Smash")
                action = LEFT + A
                return action

        # Mid-range
        if 250 <= distance_x <= 500:
            #print("Mid Range: Swallow")
            action = B

        # Long-range
        if distance_x > 500:
            if self.abel_x < self.enemy_x:
                #print("Walk to the right")
                action = RIGHT
                return action
            else:
                #print("Walk to the left")
                action = LEFT
                return action

        return action

    def fox_policy(self):
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Fox
                #print("Close Attack: Fire fox")
                action = UP + B
                return action
            elif self.abel_y > self.enemy_y:
                #print("Close Attack: Go down")
                action = DOWN
                return action
        
        # Close-range same height
        if distance_x < 250 and distance_y < 100:
            #print("Close Attack: Upward Air attack")
            action = UP + A
            return action
        
        #Long-range
        if distance_x >= 250:
            #print("Long Range: Laser")
            action = B
            return action
        
        return action

    """ def default_policy(self): #General behaviour, not character specific
        
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)

        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y:
            #print("Recover with teleport")
            action = UP + B #Recover with teleport
        
        # Step away from stage edges
        if self.abel_x < (STAGE_BOUNDARY_LEFT + 100):
            #print("Step away from left edge")
            action = RIGHT #Walk to the right
        elif self.abel_x < (STAGE_BOUNDARY_RIGHT - 100):
            #print("Step away from right edge")
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
                    #print("Ranged Attack Left")
                    action = LEFT + B
                else:
                    #print("Get close to enemy left") 
                    action = LEFT
            else:
                #print("Close Attack")
                action = LEFT + B
        if self.abel_x < self.enemy_x: # if abel is to the left of the enemy
            if  (self.enemy_x - self.abel_x) > 100:
                if self.abel_is_ranged:
                    #print("Ranged Attack Right")
                    action = RIGHT + B
                else:
                    #print("Get close to enemy right")
                    action = RIGHT
            else:
                #print("Close Attack")
                action = RIGHT + B
        
        #print(f"Pos X Abel: {self.abel_x}")
        #print(f"Pos X Enemy: {self.enemy_x}")
        #print(f"Pos Y Abel: {self.abel_y}")
        #print(f"Pos Y Enemy: {self.enemy_y}")
        
        return action """