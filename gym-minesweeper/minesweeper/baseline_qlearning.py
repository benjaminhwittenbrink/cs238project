import minesweeper as ms
import numpy as np 

# initialize params here 
NUM_ITERS = 10
eps = 0.1
GAMMA = 0.9
ALPHA = 0.2

wins = []

env = ms.MinesweeperEnv() 
states = env.observation_space
actions = env.action_space
state = env.reset()
Q = np.zeros((ms.BOARD_SIZE, ms.BOARD_SIZE, ms.BOARD_SIZE, ms.BOARD_SIZE))

action = None

def select_random_action(env, actions):
    next_action = actions.sample()
    while not env.valid_actions[next_action[0], next_action[1]]:
        next_action = actions.sample()
    return next_action

def epsilon_greedy_exploration(env, Q, random = False):

    rand = np.random.randint(1, 100) / 100 
    if random or rand < eps: 
        action = select_random_action(env, actions)
    else: 
        action = np.unravel_index(Q.argmax(), Q.shape)
    
    return action 


for i in range(NUM_ITERS):
    
    if action is None: 
        action = epsilon_greedy_exploration(env, Q, True)
    else: 
        action = epsilon_greedy_exploration(env, Q)

    next_state, reward, done, info = env.step(action)

    # TODO: convert the states to an index 
    print(state)
    print(next_state)
    state_x, state_y = state
    action_x, action_y = action

    curr_Q = Q[state_x, state_y, action_x, action_y]
    update = reward + GAMMA * (np.max(Q[state_x, state_y]) - curr_Q)
    
    Q[state_x, state_y, action_x, action_y] = curr_Q + ALPHA * update

    if done: 
        if reward > 0: wins.append(1)
        else: wins.append(0)
        state = env.reset()
    else: 
        state = next_state

np.save("Q.npy", Q)

# greedy explore
# def epsilon_greedy_exploration(s, epsilon): 
#     rand = np.random.randint(1, 100)/100 
#     if rand < epsilon: 
#         a = np.random.randint(1, action_space)
#     else: 
#         subset = df[(df['s'].values == s)]
#         a = df['a'][subset['r'].idxmax()]
#     return a

