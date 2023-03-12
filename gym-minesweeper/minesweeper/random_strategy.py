import minesweeper as ms
import numpy as np 

import ExplorationHelpers as ExpHelpers

# initialize params here 
NUM_ITERS = 20000000

wins = []

env = ms.MinesweeperEnv() 
actions = env.action_space
state = env.reset()

for i in range(NUM_ITERS):
    
    if i % 100000 == 0: 
        print(i)
        print("Game length:", len(wins))
        print("Average win:", np.mean(wins))
        print("Average win (last 10k):", np.mean(wins[-10000:]))

    action_x, action_y = ExpHelpers.select_random_action(env, actions)

    next_state, reward, done, info = env.step((action_x, action_y))

    if done: 
        if reward > 0: wins.append(1)
        else: wins.append(0)
        state = env.reset()
    else: 
        state = next_state


print("Game length:", len(wins))
print("Average win:", np.mean(wins))