"""This file contains a class that interprets the ram for SSB64.

Most of this information is from:
# https://github.com/Isotarge/ScriptHawk/blob/master/games/smash64.lua
"""

"""
Code adapted by Wulfe, from that version adapted by Valeria Gonzalez: https://github.com/wulfebw/retro



"""
import numpy as np

SSB64_CHARACTER_ORDERING = [
    "Mario",
    "Fox",
    "DK",
    "Samus",
    "Luigi",
    "Link",
    "Yoshi",
    "Falcon",
    "Kirby",
    "Pikachu",
    "Jigglypuff",
    "Ness",
]
SSB64_MAP_ORDERING = [
    "Peach's Castle",
    "Sector Z",
    "Kongo Jungle",
    "Planet Zebes",
    "Hyrule Castle",
    "Yoshi's Island",
    "Dream Land",
    "Saffron City",
    "Mushroom Kingdom",
]


# Read <u# (# byte unsigned little endian).
def convert_byte_list_to_int(byte_list):
    return int("".join(reversed([f"{byte:02x}" for byte in byte_list])), 16)

# read float big endian from byte list
def convert_byte_list_to_float(byte_list):
    return np.frombuffer(byte_list, dtype=np.float32)[0]

# read uint16 big endian from byte list
def convert_byte_list_to_uint16(byte_list):
    return np.frombuffer(byte_list, dtype=np.uint16)[0]

# read uint32 big endian from byte list
def convert_byte_list_to_uint32(byte_list):
    return np.frombuffer(byte_list, dtype=np.uint32)[0]

# read sint32 big endian from byte list
def convert_byte_list_to_sint32(byte_list):
    return np.frombuffer(byte_list, dtype=np.int32)[0]

class SSB64RAM:
    """Interprets the SSB64 ram."""

    rambase = 0x80000000
    addr_size = 4

    player_list_ptr = 0x130D84
    player_size = 0xB50
    #character_offset = 0x08

    match_settings_ptr = 0xA50E8
    match_settings_size = 0x1c8
    map_offset = 0x02
    players_bases = [0x20, 0x94, 0x108, 0x17C]
    damage_suboffset = 0x4C
    stock_suboffset = 0x8

    #Position
    position_ptr = 0x78 
    x_pos = 0x00   # 4 bytes
    y_pos = 0x04
    z_pos = 0x08

    #Velocity
    velocity_offset = 0x48
    x_vel = velocity_offset + 0x00   # 4 bytes
    y_vel = velocity_offset + 0x04
    z_vel = velocity_offset + 0x08
    
    character_idx= 0x08 # 1 Byte
    movement_state = 0x24 # 2 Bytes
    movement_frame= 0x1C # 4 Bytes
    direction= 0x44 # 4 Bytes

    def __init__(self):
        self.ram = None

    def update(self, ram):
        self.ram = ram

    def assert_valid_ram_address(self, addr):
        if addr < 0:
            raise ValueError(f"Address must be > 0: {addr}")
        if addr >= len(self.ram):
            raise ValueError(f"Address must be < size of ram: {addr}")
        #assert addr > 0, f"Address must be > 0: {addr}"
        #assert addr < len(self.ram), f"Address must be < size of ram: {addr}"

    def read_address(self, ptr):
        self.assert_valid_ram_address(ptr)
        addr_list = self.ram[ptr:ptr + self.addr_size]
        abs_addr = convert_byte_list_to_int(addr_list)
        rel_addr = abs_addr - self.rambase
        self.assert_valid_ram_address(rel_addr)
        return rel_addr

    @property
    def match_settings(self):
        match_settings_addr = self.read_address(self.match_settings_ptr)
        match_settings = self.ram[match_settings_addr:match_settings_addr +
                                  self.match_settings_size]
        return match_settings

    @property
    def match_map(self):
        match_map_index = self.match_settings[self.map_offset]
        assert match_map_index >= 0 and match_map_index < len(SSB64_MAP_ORDERING)
        return SSB64_MAP_ORDERING[match_map_index]

    def _assert_valid_player_index(self, player_index):
        assert player_index >= 0 and player_index <= 3

    def player_damage(self, player_index):
        self._assert_valid_player_index(player_index)
        dmg_index = self.players_bases[player_index] + self.damage_suboffset
        # Damage is stored as four bytes.
        dmg_bytes = self.match_settings[dmg_index:dmg_index + 4]
        dmg = convert_byte_list_to_int(dmg_bytes)
        return dmg

    def player_stock(self, player_index):
        self._assert_valid_player_index(player_index)
        # Stock is stored starting from 0 so add 1 to give the actual value.
        stock = self.match_settings[self.players_bases[player_index] + self.stock_suboffset]
        # Stock is unsigned, so when it wraps around to 255 + 1 = 256.
        # When this happens return a stock value of 0.
        if stock == 255:
            stock = 0
        else:
            stock += 1
        assert stock >= 0
        return stock

    # Player data is stored as a list of 4 players. Each player is 0xB50 bytes long.
    def player_data(self, player_index):
        self._assert_valid_player_index(player_index)
        players_addr = self.read_address(self.player_list_ptr)
        start = players_addr + self.player_size * player_index # Starting point of player data
        end = start + self.player_size # Ending point of player data
        player = self.ram[start:end] # Extract player data from ram
        return player

    # Get player character number
    def player_character(self, player_data):
        character = player_data[self.character_idx]
        return character
        assert character >= 0 and character < len(SSB64_CHARACTER_ORDERING)
        return SSB64_CHARACTER_ORDERING[character]
    
    # Get player position state in x
    def player_position_x(self, player_data):
        # This is the pointer to the position in the player data
        position_ptr_bytes = player_data[self.position_ptr:self.position_ptr + 4]
        position_ptr=convert_byte_list_to_uint32(position_ptr_bytes) - self.rambase
        #now get the x position
        x_pos_bytes= self.ram[position_ptr + self.x_pos :position_ptr+ self.x_pos + 4]
        x_pos= convert_byte_list_to_float(x_pos_bytes)
        return x_pos
    
    # Get player position state in y
    def player_position_y(self, player_data):
        # This is the pointer to the position in the player data
        position_ptr_bytes = player_data[self.position_ptr:self.position_ptr + 4]
        position_ptr=convert_byte_list_to_uint32(position_ptr_bytes) - self.rambase
        #now get the y position
        y_pos_bytes= self.ram[position_ptr + self.y_pos :position_ptr+ self.y_pos + 4]
        y_pos= convert_byte_list_to_float(y_pos_bytes)
        return y_pos
    
    # Get player position state in z
    # def player_position_z(self, player_data):
    #     # This is the pointer to the position in the player data
    #     position_ptr_bytes = player_data[self.position_ptr:self.position_ptr + 4]
    #     position_ptr=convert_byte_list_to_uint32(position_ptr_bytes) - self.rambase
    #     #now get the z position
    #     z_pos_bytes= self.ram[position_ptr + self.z_pos :position_ptr+ self.z_pos + 4]
    #     z_pos= convert_byte_list_to_float(z_pos_bytes)
    #     return z_pos
    
    # Get player velocity state in x
    def player_velocity_x(self, player_data):
        x_vel_bytes = player_data[self.x_vel:self.x_vel + 4]
        x_vel= convert_byte_list_to_float(x_vel_bytes)
        return x_vel
    
    # Get player velocity state in y
    def player_velocity_y(self, player_data):
        y_vel_bytes = player_data[self.y_vel:self.y_vel + 4]
        y_vel= convert_byte_list_to_float(y_vel_bytes)
        return y_vel
    
    # Get player velocity state in z
    # def player_velocity_z(self, player_data):
    #     z_vel_bytes = player_data[self.z_vel:self.z_vel + 4]
    #     z_vel= convert_byte_list_to_float(z_vel_bytes)
    #     return z_vel
    
    # Get player movement state
    def player_movement_state(self, player_data):
        return player_data[self.movement_state]

    # Get player movement frame
    def player_movement_frame(self, player_data):
        movement_frame_bytes = player_data[self.movement_frame:self.movement_frame + 4]
        movement_frame= convert_byte_list_to_uint32(movement_frame_bytes)
        return movement_frame
    
    # Get player direction
    def player_direction(self, player_data):
        direction_bytes = player_data[self.direction:self.direction + 4]
        direction= convert_byte_list_to_sint32(direction_bytes)
        return direction    

    #START OF MY CODE
    def player_observations_min(self, player_index):
        player_data = self.player_data(player_index)
        position_x = self.player_position_x(player_data)
        position_y = self.player_position_y(player_data)
        direction_left = self.player_direction(player_data)==-1
        direction_right = self.player_direction(player_data)==1
        return [position_x, position_y, direction_left, direction_right]

    #get all relevant observations for a player
    def player_observations(self, player_index):
        player_data = self.player_data(player_index)
        character = self.player_character(player_data)
        is_mario = character == 0
        is_fox = character == 1
        is_dk = character == 2
        is_samus = character == 3
        is_luigi = character == 4
        is_link = character == 5
        is_yoshi = character == 6
        is_falcon = character == 7
        is_kirby = character == 8
        is_pikachu = character == 9
        is_jigglypuff = character == 10
        is_ness = character == 11

        position_x = self.player_position_x(player_data)
        position_y = self.player_position_y(player_data)
        velocity_x = self.player_velocity_x(player_data)
        velocity_y = self.player_velocity_y(player_data)
        movement_state = self.player_movement_state(player_data)
        movement_frame = self.player_movement_frame(player_data)
        direction_left = self.player_direction(player_data)==-1
        direction_right = self.player_direction(player_data)==1
        return [is_mario, is_fox, is_dk, is_samus, is_luigi, is_link, is_yoshi, is_falcon, is_kirby, is_pikachu, is_jigglypuff, is_ness, 
                position_x, position_y, velocity_x, velocity_y, movement_state, movement_frame, direction_left, direction_right]
    
        #get all relevant observations for a player
    def player_observations_old(self, player_index):
        player_data = self.player_data(player_index)
        character = self.player_character(player_data)
        position_x = self.player_position_x(player_data)
        position_y = self.player_position_y(player_data)
        velocity_x = self.player_velocity_x(player_data)
        velocity_y = self.player_velocity_y(player_data)
        movement_state = self.player_movement_state(player_data)
        movement_frame = self.player_movement_frame(player_data)
        direction = self.player_direction(player_data)
        damage = self.player_damage(player_index)
        return [character, position_x, position_y, velocity_x, velocity_y, movement_state, movement_frame, direction,damage]
    #END OF MY CODE

def main():
    # filepath = "/home/wulfebw/programming/retro/save_states/dreamland.npy"
    # filepath = "/home/wulfebw/programming/retro/save_states/peaches.npy"
    # filepath = "/home/wulfebw/programming/retro/save_states/kongo.npy"
    # filepath = "/home/wulfebw/programming/retro/save_states/sector_z.npy"
    # filepath = "/home/wulfebw/programming/retro/save_states/4_player.npy"
    # filepath = "/home/wulfebw/programming/retro/save_states/dmg_lives.npy"
    filepath = "/home/wulfebw/programming/retro/save_states/dmg_lives_2.npy"
    ram = np.load(filepath)
    ssb64ram = SSB64RAM()
    ssb64ram.update(ram)


if __name__ == "__main__":
    main()
