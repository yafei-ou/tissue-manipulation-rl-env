import random
import numpy as np
import gym
from gym import spaces
from typing import *

import Sofa.Simulation, Sofa.Gui, Sofa.Core, SofaRuntime
import os, time
import numpy as np
from Sofa.constants import Key


class SofaEnv(gym.Env):

    def __init__(self, action_space_type: str = "continuous", action_size: Union[Tuple, int] = None, obs_size: int = None, obs_sequence_length: int = 10, render_mode: str = "human") -> None:

        self.render_mode = render_mode
        self.action_size = action_size
        self.obs_size = obs_size

        self.rng = random.Random()
        self.nprng = np.random.default_rng()

        if action_space_type == "continuous":
            self.action_space_type = 0 # continuous action space
            self.action_space = spaces.Box(-1.0, 1.0, action_size)
        elif action_space_type == "discrete":
            self.action_space_type = 1 # discrete action space
            self.action_space = spaces.MultiDiscrete(action_size)
        elif action_space_type == "single_discrete":
            self.action_space_type = 2 # discrete action space
            self.action_space = spaces.Discrete(action_size)

        self.observation_space = spaces.Box( low=-np.inf, high=np.inf, shape=(obs_size*obs_sequence_length, ))
        self.sequence_length = obs_sequence_length
        self.count_finish = None
        self.padding = None

        plugins = ["SofaComponentAll"]
        for p in plugins:
            SofaRuntime.importPlugin(p)

        self.sf_root = Sofa.Core.Node("root")

    def seed(self, seed: int = None):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed=seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def reset(self) -> None:
        self.count_finish = 0
        self.padding = np.array([[0.0 for _i in range(self.obs_size)] for _ in range(self.sequence_length)])

    def _pad_obs(self, obs) -> np.ndarray:
        
        self.padding = np.concatenate((self.padding, obs), axis=0)[1:]
        states = self.padding.reshape(1, -1)
        return states

    def _test_sofa(self):
        Sofa.Simulation.init(self.sf_root)
        # Find out the supported GUIs
        print ("Supported GUIs are: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(self.sf_root, __file__)
        Sofa.Gui.GUIManager.SetDimension(720, 720)
        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(self.sf_root)

    def _step_simulation(self, t):
        for _ in range(int(t/self.sf_root.dt.value)):
            Sofa.Simulation.animate(self.sf_root, self.sf_root.dt.value)