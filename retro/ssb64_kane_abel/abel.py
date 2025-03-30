"""
Code made by Valeria Gonzalez

Define the Abel class for the rule based policy agent. The agent will determine the action to take based on the character's positon, the enemy's position, and the character's specific abilities.
"""

# FConstants the actions. Actions can be combined to create combos.
DO_NOTHING = 0
UP=1 #jump
DOWN=2 #Drop down
LEFT=3 #Move left
RIGHT=6 #Move right
A=9 #Normal attack
B=18 #Special attack
Z=27 #Shield

#Boundaries of the game stage
STAGE_BOUNDARY_LEFT = -1900
STAGE_BOUNDARY_RIGHT = 2300
STAGE_BOUNDARY_Y = 0

class Abel:
    def __init__(self, name,player_num=1):
        """
        Initializes an Abel character
        
        Parameters:
        name (str): Name of the character
        player_num (int): Player number (1 or 2), determines observation mapping
        """
        self.name = name
        self.player_num = player_num

    """ Basic characters: Mario = [0, 20], Fox = [1,21], DK = [2,22], Samus = [3,23], Link = [5,25], Yoshi = [6,26], Kirby = [8,28], Pikachu = [9,29] 
    Hidden characters (DO NOT USE!): Luigi = [4,24], Captain Falcon = [7,27], Jigglypuff = [10,30], Ness = [11,31] 
    [is_mario, is_fox, is_dk, is_samus, is_luigi, is_link, is_yoshi, is_falcon, is_kirby, is_pikachu, is_jigglypuff, is_ness, (0,11) (19,30)
    12, 32  = position_x, 
    13, 33 = position_y, 
    14, 34 = velocity_x, 
    15, 35 = velocity_y, 
    16, 36 = movement_state, 
    17, 37 = movement_frame, 
    18,19, 38,39 = direction """

    def convert_obs(self, obs):
        """
        Converts raw game observation data into useful attributes for decision-making.
        
        Parameters:
        obs (list): A list containing the game's observation data.
        """
        if len(obs)==1:
            obs = obs[0]
        #Abel
        if self.player_num == 1:
            #Self Position and direction
            self.abel_x = obs[12]
            self.abel_y = obs[13]
            self.abel_direction = obs[18]*-1+obs[19]
            # Character type (is_ranged is True if Abel has a ranged attack)
            self.abel_is_ranged = obs[1] or obs[3] or obs[9]
            self.abel_is_pikachu = obs[9]
            self.abel_is_mario = obs[0]
            self.abel_is_dk = obs[2]
            self.abel_is_link = obs[5]
            self.abel_is_samus = obs[3]
            self.abel_is_yoshi = obs[6]
            self.abel_is_kirby = obs[8]
            self.abel_is_fox = obs[1]
            #Enemy's position and direction
            self.enemy_x = obs[32]
            self.enemy_y = obs[33]
            self.enemy_is_ranged = obs[21] or obs[23] or obs[29]
            self.enemy_direction = obs[38]*-1+obs[39]
        
        else: # Abel is the second player, switch observation mappings
            #Self position and direction
            self.abel_x = obs[32]
            self.abel_y = obs[33]
            self.abel_direction = obs[38]*-1+obs[39]
            #Self Character type 
            self.abel_is_ranged = obs[21] or obs[23] or obs[29]
            self.abel_is_pikachu = obs[29]
            self.abel_is_mario = obs[20]
            self.abel_is_dk = obs[22]
            self.abel_is_link = obs[25]
            self.abel_is_samus = obs[23]
            self.abel_is_yoshi = obs[26]
            self.abel_is_kirby = obs[28]
            self.abel_is_fox = obs[21]
            #Enemys position and direction
            self.enemy_x = obs[12]
            self.enemy_y = obs[13]
            self.enemy_is_ranged = obs[1] or obs[3] or obs[9]
            self.enemy_direction = obs[18]*-1+obs[19]

    def policy(self, obs):
        """
        Determines the action to take based on game observations
        
        Parameters:
        obs (list): The game observation data
        
        Returns:
        int: The chosen action
        """
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
        """
        Implements generic recovery and positioning strategies.
        Returns None if no generic action is required.
        """
        # Recovery if falling off-stage
        if self.abel_y < STAGE_BOUNDARY_Y and not self.abel_is_yoshi:
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
        """
        Pikachu specific policy.
        - Uses Thunder if the opponent is above.
        - Moves down if the opponent is below.
        - Uses a Headbutt for close-range combat.
        - Uses Vault Kick for mid-range combat.
        - Uses Electric Shock as a long-range projectile attack.
        """
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
        """
        Mario specific policy.
        - Uses Spinning Uppercut if the opponent is close and above Mario.
        - Moves down if the opponent is below Mario.
        - Uses Tornado Spin for close-range combat at the same height.
        - Uses Fireball for mid-range combat (right or left depending on the enemy's position).
        - Moves towards the enemy if they are at a long distance.
        """
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
        """
        DK specific policy.
        - Uses Helicopter Punch (Jump) if the opponent is close and above DK.
        - Moves down if the opponent is below DK.
        - Uses Ground Pound for close-range combat at the same height.
        - Moves towards the opponent if they are at a long distance.
        """
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
        """
        Link specific policy.
        - Uses Spin Attack (Jump) if the opponent is close and above Link.
        - Moves down if the opponent is below Link.
        - Uses Left or Right Dash Attack for close-range combat at the same height.
        - Uses Boomerang for mid-range combat.
        - Walks towards the opponent if they are at a long distance.
        """
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Close range
        if distance_x < 250 and distance_y > 100:
            if self.abel_y < self.enemy_y: #Enemy is above Link
                #print("Link: Close Attack: Spin Attack")
                #print("Link: Jump")
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
        """
        Samus specific policy.
        - Uses Upward Smash if the opponent is close and above Samus.
        - Moves down if the opponent is below Samus.
        - Uses Right or Left Forward Smash for close-range combat at the same height.
        - Uses Charge Shot for long-range combat.
        - Moves towards the opponent if they are within a mid-range distance.
        """
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
        """
        Yoshi specific policy.
        - Recovers from falling of the stage by jumping
        - Uses Egg Throw (Up + B) if the opponent is close and above Yoshi.
        - Moves down if the opponent is below Yoshi.
        - Uses Ground Pound (Down + B) if the opponent is close and at the same height.
        - Uses Swallow (B) for mid-range combat.
        - Walks towards the opponent if they are at a long distance.
        """
        distance_x = abs(self.abel_x - self.enemy_x)
        distance_y = abs(self.abel_y - self.enemy_y)
        action = B

        # Recovery if falling off-stage (different from the others)
        if self.abel_y < STAGE_BOUNDARY_Y and not self.abel_is_yoshi:
            #print("Yoshi recovery: Recover with jump")
            action = A #Recover with jump
            return action

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
        """
        Kirby specific policy.
        Won't be able to use the full extent of the swallow ability due to the way button holding is being handled. 
        - Uses Final Cutter (Up + B) if the opponent is close and above Kirby.
        - Uses Stone (Down + B) if the opponent is below Kirby.
        - Uses Right or Left Forward Smash (A) for close-range combat at the same height.
        - Uses Swallow (B) for mid-range combat.
        - Walks towards the opponent if they are at a long distance.
        """
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
        """
        Fox specific policy.
        - Uses Fire Fox (Up + B) if the opponent is close and above Fox.
        - Moves down if the opponent is below Fox.
        - Uses Upward Air Attack (Up + A) for close-range combat at the same height.
        - Uses Laser (B) for long-range combat.
        """
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