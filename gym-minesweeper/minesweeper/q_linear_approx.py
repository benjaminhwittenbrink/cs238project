import minesweeper as ms
import numpy as np 
import time 
from tqdm import tqdm 

import ExplorationHelpers as ExpHelpers

# initialize params here 
NUM_ITERS = 10_000_000
GAMMA = 0.9
ALPHA = 0.01
# epsilon parameters
EPSILON = 0.1

rng = np.random.RandomState(123)
# linear
# W = rng.uniform(low=-1e-5,high=1e-5,size = [ms.CELLS, ms.CELLS * ms.POSSIBLE_VALUES])
# B = np.zeros([ms.CELLS])
# warm start 
W = np.load("linear_weights_55/W_9900000.npy")
B = np.load("linear_weights_55/B_9900000.npy")

I = np.eye(ms.POSSIBLE_VALUES)
def convert_state_to_features(state):
    # note need to add 2 bc our state starts at -2, indexing starts at 0
    return np.reshape(I[(state + 2).flatten()], [-1])


wins = []

env = ms.MinesweeperEnv() 
actions = env.action_space
state = env.reset()
state_vec = convert_state_to_features(state)

start_time = time.time()
for i in tqdm(range(NUM_ITERS)):
    
    if i % 100_000 == 0: 
        i_time = time.time()
        print(i)
        print("Time taken:", i_time - start_time)
        print(f"W matrix: {W[0, 0]}; B matrix: {B[0]}")
        print("Game length:", len(wins))
        print("Average win (last 10k):", np.mean(wins[-10_000:]))
        print("eps:", EPSILON)
        np.save(f"linear_weights_55_warm/W_{i}.npy", W)
        np.save(f"linear_weights_55_warm/B_{i}.npy", B)
        np.save(f"linear_weights_55_warm/wins_{i}.npy", wins)
        if i != 0: np.save(f"linear_weights_55_warm/Q_{i}.npy", Q)
        wins = []
        start_time = i_time 

    done = False 
    while not done: 

        # calculate Q 
        Q_vec = W.dot(state_vec) + B
        Q = Q_vec.reshape(ms.BOARD_SIZE, ms.BOARD_SIZE)
        # filter out visited states
        Q[state != ms.CLOSED] = - np.inf 

        # get and take epsilon-greedy action
        action_x, action_y = ExpHelpers.epsilon_greedy_exploration(env, actions, Q, EPSILON)
        next_state, reward, done, info = env.step((action_x, action_y))
        next_state_vec = convert_state_to_features(next_state)
        
        # calculate next Q 
        next_Q_vec = W.dot(next_state_vec) + B
        
        # max action we haven't chosen
        max_Q = np.max(next_Q_vec[state.flatten() == ms.CLOSED])

        # calculate delta 
        delta = reward + GAMMA * max_Q - Q[action_x, action_y]
        
        # calculate gradients and perform update rule
        grad_W = state_vec
        grad_B = 1
        W[action_x * ms.BOARD_SIZE + action_y] += ALPHA * delta * grad_W
        B[action_x * ms.BOARD_SIZE + action_y] += ALPHA * delta * grad_B

        if done: 
            if reward > 0: wins.append(1)
            else: wins.append(0)
            state = env.reset()
            state_vec = convert_state_to_features(state)
        else: 
            state = next_state
            state_vec = next_state_vec
