from copy import deepcopy
from mcts import mcts
import minesweeper as ms
import gym
import numpy as np
import itertools

"""
Additional helper function for converting 
gym spaces to lists for library utilization
"""
def get_space_list(space):

    """
    Converts gym `space`, constructed from `types`, to list `space_list`
    """

    # -------------------------------- #

    types = [
        gym.spaces.multi_binary.MultiBinary,
        gym.spaces.discrete.Discrete,
        gym.spaces.multi_discrete.MultiDiscrete,
        gym.spaces.dict.Dict,
        gym.spaces.tuple.Tuple,
    ]

    if type(space) not in types:
        raise ValueError(f'input space {space} is not constructed from spaces of types:' + '\n' + str(types))

    # -------------------------------- #

    if type(space) is gym.spaces.multi_binary.MultiBinary:
        return [
            np.reshape(np.array(element), space.n)
            for element in itertools.product(
                *[range(2)] * np.prod(space.n)
            )
        ]

    if type(space) is gym.spaces.discrete.Discrete:
        return list(range(space.n))

    if type(space) is gym.spaces.multi_discrete.MultiDiscrete:
        return [
            np.array(element) for element in itertools.product(
                *[range(n) for n in space.nvec]
            )
        ]

    if type(space) is gym.spaces.dict.Dict:

        keys = space.spaces.keys()
        
        values_list = itertools.product(
            *[get_space_list(sub_space) for sub_space in space.spaces.values()]
        )

        return [
            {key: value for key, value in zip(keys, values)}
            for values in values_list
        ]

        return space_list

    if type(space) is gym.spaces.tuple.Tuple:
        return [
            list(element) for element in itertools.product(
                *[get_space_list(sub_space) for sub_space in space.spaces]
            )
        ]

    # -------------------------------- #

"""
Minesweeper State is a class used for wrapping minesweeper gym env
and passing to MCTS library.
"""
class MinesweeperState(): 
    def __init__(self): 
        self.env = ms.MinesweeperEnv() 
        self.actions = self.env.action_space
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
    
    def getPossibleActions(self): 
        possible = []

        # Include all valid actions
        for action in get_space_list(self.actions): 
            if self.env.valid_actions[action[0], action[1]]:
                 possible.append((action[0], action[1]))
        return possible
    
    def takeAction(self, action): 
        copy = deepcopy(self)
        next_state, reward, done, info = copy.env.step(action)
        copy.reward = reward
        copy.done = done
        copy.state = next_state
        return copy
    
    def isTerminal(self):
        return self.done
    
    def getReward(self): 
        return self.reward

"""
Hashable action function for identification.
"""
class Action():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


""" 
Run MCTS on current state.
Produces optimal action. 
"""
if __name__=="__main__":
    initialState = MinesweeperState()
    searcher = mcts(timeLimit=1000)
    action = searcher.search(initialState=initialState)

    print(action)