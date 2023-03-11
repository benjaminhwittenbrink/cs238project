import minesweeper as ms
import numpy as np 

import ExplorationHelpers as ExpHelpers

# initialize params here 
NUM_ITERS = 100000000
GAMMA = 0.9
ALPHA = 0.1

# epsilon parameters
EPSILON = 1
MAX_EPS = 1
MIN_EPS = 1e-2
EPS_DECAY = 1e-3

wins = []
Q = {}

env = ms.MinesweeperEnv() 
actions = env.action_space
state = env.reset()

for i in range(NUM_ITERS):
    
    if i % 10000 == 0: 
        print(i)
        print("State space size:", len(Q))
        print("Game length:", len(wins))
        print("Average win:", np.mean(wins))
        print("eps:", EPSILON)

    state_str = ms.board2str(state)
    if state_str not in Q: 
        Q[state_str] = np.zeros((ms.BOARD_SIZE, ms.BOARD_SIZE))

    action_x, action_y = ExpHelpers.epsilon_greedy_exploration(env, actions, Q[state_str], EPSILON)

    next_state, reward, done, info = env.step((action_x, action_y))

    curr_Q = Q[state_str][action_x, action_y]
    update = reward + GAMMA * (np.max(Q[state_str]) - curr_Q)
    Q[state_str][action_x, action_y] = curr_Q + ALPHA * update

    if done: 
        if reward > 0: wins.append(1)
        else: wins.append(0)
        state = env.reset()
    else: 
        state = next_state
    
    EPSILON = ExpHelpers.epsilon_decay_schedule(i, MAX_EPS, MIN_EPS, EPS_DECAY)


print("Game length:", len(wins))
print("Average win:", np.mean(wins))
np.save("Q.npy", Q)
