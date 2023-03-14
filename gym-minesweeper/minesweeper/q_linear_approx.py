import minesweeper as ms
import numpy as np 
import time 
from tqdm import tqdm 

import ExplorationHelpers as ExpHelpers

# initialize params here 
NUM_ITERS = 20_000_000
GAMMA = 0.9
ALPHA = 0.001

# epsilon parameters
EPSILON = 0.2
# MAX_EPS = 1
# MIN_EPS = 1e-2
# EPS_DECAY = 1e-5

rng = np.random.RandomState(123)
# linear
W = rng.uniform(low=-1e-5,high=1e-5,size = [ms.CELLS, ms.CELLS * ms.POSSIBLE_VALUES])
B = np.zeros([ms.CELLS])

I = np.eye(ms.POSSIBLE_VALUES)

wins = []

env = ms.MinesweeperEnv() 
actions = env.action_space
state = env.reset()
# note need to add 2 bc our state starts at -2, indexing starts at 0
state_vec = np.reshape(I[(state + 2).flatten()], [-1])
start_time = time.time()
for i in tqdm(range(NUM_ITERS)):
    
    if i % 100_000 == 0: 
        i_time = time.time()
        print(i)
        print("Time taken:", i_time - start_time)
        print("W matrix:", W[0])
        print("B vec", B)
        print("Game length:", len(wins))
        print("Average win (last 10k):", np.mean(wins[-10_000:]))
        print("eps:", EPSILON)
        np.save(f"weights/W_{i}.npy", W)
        np.save(f"weights/B_{i}.npy", B)
        wins = []
        start_time = i_time 

    done = False 
    while not done: 

        Q_vec = W.dot(state_vec)
        Q = Q_vec.reshape(ms.BOARD_SIZE, ms.BOARD_SIZE)

        action_x, action_y = ExpHelpers.epsilon_greedy_exploration(env, actions, Q, EPSILON)

        next_state, reward, done, info = env.step((action_x, action_y))

        next_state_vec = np.reshape(I[(next_state + 2).flatten()], [-1])
        next_Q_vec = W.dot(next_state_vec)
        
        # max action we haven't chosen
        max_Q = np.max(next_Q_vec)
        delta = reward + GAMMA * max_Q - Q[action_x, action_y]
        grad_W = state_vec
        grad_B = np.ones(ms.CELLS)
        
        W[action_x * ms.BOARD_SIZE + action_y] += ALPHA * delta * grad_W
        B[action_x * ms.BOARD_SIZE + action_y] += ALPHA * delta 

        if done: 
            if reward > 0: wins.append(1)
            else: wins.append(0)
            state = env.reset()
            state_vec = np.reshape(I[(state + 2).flatten()], [-1])
        else: 
            state = next_state
            state_vec = next_state_vec
    
    # #if i > 500_000: 
    # EPSILON = ExpHelpers.epsilon_decay_schedule(i, MAX_EPS, MIN_EPS, EPS_DECAY)
