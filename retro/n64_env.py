import gc
import gzip
import json
import os

import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import retro
import retro.data

gym_version = tuple(int(x) for x in gym.__version__.split('.'))

__all__ = ['RetroEnv']


import hashlib
import struct
from typing import Optional 
import numpy as np

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bt: bytes) -> int:
    
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt += b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def hash_seed(seed: Optional[int] = None, max_bytes: int = 8) -> int:
    """Any given evaluation is likely to have many PRNG's active at once.
    (Most commonly, because the environment is running in multiple processes.)
    There's literature indicating that having linear correlations between seeds of multiple PRNG's can correlate the outputs:
        http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
        http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
        http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme is likely not crypto-strength, but it should be good enough to get rid of simple correlations.)
    Args:
        seed: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    Returns:
        The hashed seed
    """

    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])

gym.utils.seeding.hash_seed = hash_seed

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL installed. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )
from gym.utils import seeding

gym_version = tuple(int(x) for x in gym.__version__.split('.'))

__all__ = ['RetroEnv']


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs
    )

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )

class SimpleImageViewer(object):
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = get_display(display)
        self.maxwidth = maxwidth

    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = get_window(
                width=width,
                height=height,
                display=self.display,
                vsync=False,
                resizable=True,
            )
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3
        )
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)  # draw
        self.window.flip()
        
class N64Env(gym.Env):
    """
    Nintendo 64 environment.

    We can't use the typical retro environment because n64 uses dynamic memory addresses.
    So we have to read and interpret the ram differently, which we handle in this class.
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60.0}

    def __init__(self,
                 game,
                 state=retro.State.DEFAULT,
                 scenario=None,
                 info=None,
                 use_restricted_actions=retro.Actions.FILTERED,
                 record=False,
                 players=1,
                 inttype=retro.data.Integrations.STABLE,
                 is_random_state=False,
                 use_exact_keys=False,
                 obs_type=retro.Observations.IMAGE):
        if not hasattr(self, 'spec'):
            self.spec = None
        self._obs_type = obs_type
        self.img = None
        self.ram = None
        self.viewer = None
        self.gamename = game
        self.statename = state
        self.initial_state = None
        self.players = players
        self.is_random_state = is_random_state
        self.step_count = 0
        self.use_exact_keys = use_exact_keys
        if game != "SuperSmashBros-N64":
            raise NotImplementedError("Only ssb64 supported so far")
        self.ssb64_game_data = retro.data.SSB64GameData()

        metadata = {}
        rom_path = retro.data.get_romfile_path(game, inttype)
        metadata_path = retro.data.get_file_path(game, 'metadata.json', inttype)

        if state == retro.State.NONE:
            self.statename = None
        elif state == retro.State.DEFAULT:
            self.statename = None
            try:
                if metadata_path:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    if 'default_player_state' in metadata and self.players <= len(
                            metadata['default_player_state']):
                        self.statename = metadata['default_player_state'][self.players - 1]
                    elif 'default_state' in metadata:
                        self.statename = metadata['default_state']
                    else:
                        self.statename = None
                else:
                    self.statename = None
            except (IOError, json.JSONDecodeError):
                pass

        if self.statename:
            if self.is_random_state:
                self.load_random_state(inttype)
            else:
                self.load_state(self.statename, inttype)

        self.data = retro.data.GameData()

        if info is None:
            info = 'data'

        if info.endswith('.json'):
            # assume it's a path
            info_path = info
        else:
            info_path = retro.data.get_file_path(game, info + '.json', inttype)

        if scenario is None:
            scenario = 'scenario'

        if scenario.endswith('.json'):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = retro.data.get_file_path(game, scenario + '.json', inttype)

        self.system = retro.get_romfile_system(rom_path)

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        self.em = retro.RetroEmulator(rom_path)
        self.em.configure_data(self.data)
        self.em.step()

        core = retro.get_system_info(self.system)
        self.buttons = core['buttons']
        self.num_buttons = len(self.buttons)

        try:
            assert self.data.load(
                info_path,
                scenario_path), 'Failed to load info (%s) or scenario (%s)' % (info_path,
                                                                               scenario_path)
        except Exception:
            del self.em
            raise

        self.button_combos = self.data.valid_actions()
        if use_restricted_actions == retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos**players)
        elif use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete([
                len(combos) if gym_version >= (0, 9, 6) else (0, len(combos) - 1)
                for combos in self.button_combos
            ] * players)
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)

        kwargs = {}
        if gym_version >= (0, 9, 6):
            kwargs['dtype'] = np.uint8

        if self._obs_type == retro.Observations.RAM:
            low=np.array([

                # Player 1
                # Character one hot encodings
                # - is_mario
                # - is_fox
                # - is_dk
                # - is_samus
                # - is_luigi
                # - is_link
                # - is_yoshi
                # - is_falcon
                # - is_kirby
                # - is_pikachu
                # - is_jigglypuff
                # - is_ness
                # X position
                # Y position
                # X velocity
                # Y velocity
                # movement state
                # movement frame
                # direction
                0,0,0,0,0,0,0,0,0,0,0,0,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                0,
                0,
                -1,
                
                #player 2
                0,0,0,0,0,0,0,0,0,0,0,0,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                np.finfo(np.float32).min,
                0,
                0,
                -1,
            ], dtype=np.float32)
            high=np.array([
                1,1,1,1,1,1,1,1,1,1,1,1,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                255,
                np.iinfo(np.int32).max,
                1,

                1,1,1,1,1,1,1,1,1,1,1,1,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                255,
                np.iinfo(np.int32).max,
                1,
                
                ], dtype=np.float32)
            print(low)
            print(high)
            self.observation_space = gym.spaces.Box(low=low,high=high, **kwargs)
        else:
            img = [self.get_screen(p) for p in range(players)]
            shape = img[0].shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, **kwargs)

        self.use_restricted_actions = use_restricted_actions
        self.movie = None
        self.movie_id = 0
        self.movie_path = None
        if record is True:
            self.auto_record()
        elif record is not False:
            self.auto_record(record)
        self.seed()
        if gym_version < (0, 9, 6):
            self._seed = self.seed
            self._step = self.step
            self._reset = self.reset
            self._render = self.render
            self._close = self.close

    def _update_obs(self):
        self.ram = self.get_ram()
        if self._obs_type == retro.Observations.RAM:
            try:
                player_1_obs = self.ssb64_game_data.ram.player_observations(0)
                player_2_obs = self.ssb64_game_data.ram.player_observations(1)
                return np.concatenate([player_1_obs, player_2_obs])
            except:
                print(f"Error getting player observations for scenario {self.statename} at step {self.step_count}")
                return None
            #return self.ram
        elif self._obs_type == retro.Observations.IMAGE:
            self.img = self.get_screen()
            return self.img
        else:
            raise ValueError('Unrecognized observation type: {}'.format(self._obs_type))

    def action_to_array(self, acts):
        actions = []
        for p in range(self.players):
            a=acts[p]
            action = 0
            if self.use_restricted_actions == retro.Actions.DISCRETE:
                for combo in self.button_combos:
                    current = a % len(combo)
                    a //= len(combo)
                    action |= combo[current]
            elif self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
                # # Is this entire thing just totally wrong?
                # I think so
                # maybe I should submit a pull request
                # ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                # for i in range(len(ap)):
                #     # I think this index should be modulo the number of button_combos?
                #     # It definitely goes beyond the length of the list.
                #     buttons = self.button_combos[i % len(self.button_combos)]
                #     action |= buttons[ap[i]]
                num_combos = len(self.button_combos)
                ap = a[num_combos * p:num_combos * (p + 1)]
                for i in range(len(ap)):
                    buttons = self.button_combos[i]
                    action |= buttons[ap[i]]
            else:
                ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    action |= int(ap[i]) << i
                if self.use_restricted_actions == retro.Actions.FILTERED:
                    action = self.data.filter_action(action)
            ap = np.zeros([self.num_buttons], np.uint8)
            for i in range(self.num_buttons):
                ap[i] = (action >> i) & 1
            actions.append(ap)
        return actions

    def step(self, a):
        if self.img is None and self.ram is None:
            raise RuntimeError('Please call env.reset() before env.step()')
        print(f"Step {self.step_count} with action {a}")
        save_p = [] # Save button pressed
        save_ap = [] # Save the actions for each player
        if self.use_exact_keys:
            for p in range(self.players):
                ap= a[12*p:12*(p+1)]
                #convert to ints
                ap = [int(i) for i in ap]
                #print(f"Player {p} action: {ap}")
                self.em.set_button_mask(ap, p)
                save_p.append(p)
                save_ap.append(ap)
        else:
            for p, ap in enumerate(self.action_to_array(a)):
                #ap[0] = 0 # Action 'A' (normal attack)
                #ap[1] = 0 # Action 'B' (special attack)
                #ap[2] = 0 # SELECT (DO NOT USE!)
                #ap[3] = 0 # START (DO NOT USE!)
                #ap[4] = 0 # UP (Jump) 
                #ap[5] = 0 # DOWN (Crouch)
                #ap[6] = 0 # LEFT
                #ap[7] = 0 # RIGHT
                #ap[8] = 0 # R or ?? Grab or ??  (DO NOT USE!)
                #ap[9] = 0 # R or ?? Grab or ?? (DO NOT USE!)
                #ap[10] = 0 # L Taunt (DO NOT USE!)
                #ap[11] = 0 # Z (Shield)
                #print(f"Player {p} action: {ap}")
                if self.movie:
                    for i in range(self.num_buttons):
                        self.movie.set_key(i, ap[i], p)
                self.em.set_button_mask(ap, p)
                save_p.append(p)
                save_ap.append(ap)

        if self.movie:
            self.movie.step()
        self.em.step()
        self.data.update_ram()

        #Make a second step where we release the buttons 'B' 'A' and 'UP'
        for j in range(2):
            for i in range(len(save_p)):
                p = save_p[i]
                ap = save_ap[i]
                ap[0] = 0
                ap[1] = 0
                ap[4] = 0
                self.em.set_button_mask(ap, p)
                #print(f"Player {p} action: {ap}")

            if self.movie:
                self.movie.step()
            self.em.step()
            self.data.update_ram()

        ob = self._update_obs()
        #print(f"Step {self.step_count}")
        #print(f"Player 1: {self.ssb64_game_data.ram.player_observations(0)}")
        #print(f"Player 2: {self.ssb64_game_data.ram.player_observations(1)}")

        if ob is None:
            return ob, 0, True, {}
        rew, done, info = self.compute_step()
        self.step_count += 1
        return ob, rew, bool(done), dict(info)

    def reset(self):
        self.step_count = 0
        if self.is_random_state:
            self.load_random_state()
        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
            self.record_movie(
                os.path.join(self.movie_path,
                             '%s-%s-%06d.bk2' % (self.gamename, rel_statename, self.movie_id)))
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.ssb64_game_data.reset()
        self.data.update_ram()
        self.ram = self.get_ram()
        self.ssb64_game_data.update(self.ram)
        obs= self._update_obs()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2=0
        #seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer:
                self.viewer.close()
            return

        img = self.get_screen() if self.img is None else self.img
        if mode == "rgb_array":
            return img
        elif mode == "human":
            return img
            if self.viewer is None:
                #from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
                
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if hasattr(self, 'em'):
            del self.em

    def get_action_meaning(self, act):
        actions = []
        for p, action in enumerate(self.action_to_array(act)):
            actions.append([self.buttons[i] for i in np.extract(action, np.arange(len(action)))])
        if self.players == 1:
            return actions[0]
        return actions

    def get_ram(self):
        blocks = []
        for offset in sorted(self.data.memory.blocks):
            arr = np.frombuffer(self.data.memory.blocks[offset], dtype=np.uint8)
            blocks.append(arr)
        return np.concatenate(blocks)

    def get_screen(self, player=0):
        img = self.em.get_screen()
        # OpenGL returns the image flipped and I'm not sure how to fix it there.
        img = np.flipud(img)

        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > img.shape[1]:
            w = img.shape[1]
        else:
            w += x
        if not h or y + h > img.shape[0]:
            h = img.shape[0]
        else:
            h += y
        if x == 0 and y == 0 and w == img.shape[1] and h == img.shape[0]:
            return img
        return img[y:h, x:w]

    def load_state(self, statename, inttype=retro.data.Integrations.DEFAULT):
        if type(statename) is not str:
            self.initial_state = statename
            print(f"Loaded state from memory")
            self.statename = "Playback"
        else:
            print(f"Loading state {statename}")
            if not statename.endswith('.state'):
                statename += '.state'
            # open the state file and read it into memory
            with gzip.open(retro.data.get_file_path(self.gamename, statename, inttype), 'rb') as fh:
                self.initial_state = fh.read()
            self.statename = statename

    def load_random_state(self, inttype=retro.data.Integrations.DEFAULT):
        characters = ["mario", "dk", "fox", "kirby", "link", "samus", "pikachu", "yoshi"]
        player1= characters[np.random.randint(0, len(characters))]
        player2= characters[np.random.randint(0, len(characters))]
        #statename = f"{player1}-{player2}-ai3.state"
        statename = f"{player1}-{player2}-ai1.state"
        
        # open the state file and read it into memory
        with gzip.open(retro.data.get_file_path(self.gamename, statename, inttype), 'rb') as fh:
            self.initial_state = fh.read()
        print(f"Loaded random state {statename}")
        self.statename = statename

    def compute_step(self):
        """Specific to ssb64 for now."""
        self.ssb64_game_data.update(self.ram)
        if self.players > 1:
            # Make the reward a numpy array so that certain wrappers work with it.
            reward = np.array([self.ssb64_game_data.current_reward(p) for p in range(self.players)])
        else:
            reward = self.ssb64_game_data.current_reward()
        done = self.ssb64_game_data.is_done()
        return reward, done, self.ssb64_game_data.lookup_all()

    def record_movie(self, path):
        self.movie = retro.Movie(path, True, self.players)
        self.movie.configure(self.gamename, self.em)
        if self.initial_state:
            self.movie.set_state(self.initial_state)

    def stop_record(self):
        self.movie_path = None
        self.movie_id = 0
        if self.movie:
            self.movie.close()
            self.movie = None

    def auto_record(self, path=None):
        if not path:
            path = os.getcwd()
        self.movie_path = path
