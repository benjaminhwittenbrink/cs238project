import minesweeper as ms
import numpy as np 
from tqdm import tqdm 
import pickle

import ExplorationHelpers as ExpHelpers

# initialize params here 
NUM_ITERS = 10_000_000
GAMMA = 0.9
ALPHA = 0.1

# epsilon parameters
EPSILON = 1
MAX_EPS = 1
MIN_EPS = 1e-2
EPS_DECAY = 1e-5


wins = []
Q = {}

env = ms.MinesweeperEnv() 
actions = env.action_space
state = env.reset()

for i in tqdm(range(NUM_ITERS)):
    
    if i % 100_000 == 0: 
        print(i)
        print("State space size:", len(Q))
        print("Game length:", len(wins))
        print("Average win (last 100k):", np.mean(wins[-100_000:]))
        print("Average win (last 10k):", np.mean(wins[-10_000:]))
        print("eps:", EPSILON)

        if i % 1_000_000 == 0: 
            pickle.dump(Q, open(f"q_learning_data_55_p2/Q_{i}_p2.pkl", "wb"))
        np.save(f"q_learning_data_55_p2/wins_{i}_p2.npy", wins)
        wins = []

    done = False 
    while not done: 
        state_str = ms.board2str(state)
        if state_str not in Q: 
            Q[state_str] = np.zeros((ms.BOARD_SIZE, ms.BOARD_SIZE))
        
        Q_valid_actions = np.copy(Q[state_str])
        Q_valid_actions[state != -2] = - np.inf 

        action_x, action_y = ExpHelpers.epsilon_greedy_exploration(env, actions, Q_valid_actions, EPSILON)
        next_state, reward, done, info = env.step((action_x, action_y))
        next_state_str = ms.board2str(next_state)

        curr_Q = Q[state_str][action_x, action_y]
        update = reward + GAMMA * (np.max(Q[next_state_str]) - curr_Q)
        Q[state_str][action_x, action_y] = curr_Q + ALPHA * update

        if done: 
            if reward > 0: wins.append(1)
            else: wins.append(0)
            state = env.reset()
        else: 
            state = next_state
        
    
    if i > 500_000: 
        EPSILON = ExpHelpers.epsilon_decay_schedule(i - 500_000, MAX_EPS, MIN_EPS, EPS_DECAY)


print("Game length:", len(wins))
print("Average win:", np.mean(wins))

