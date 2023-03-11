
import numpy as np

def epsilon_greedy_exploration(env, actions, action_values, eps = 1):
    
    rand = np.random.uniform(0, 1)
    if rand < eps: 
        action = select_random_action(env, actions)
    else: 
        action = np.unravel_index(action_values.argmax(), action_values.shape)
    
    return action

def select_random_action(env, actions):
    next_action = actions.sample()
    while not env.valid_actions[next_action[0], next_action[1]]:
        next_action = actions.sample()
    return next_action

def epsilon_decay_schedule(i, max_eps, min_eps, eps_decay_rate):
    return min_eps + (max_eps - min_eps)*np.exp(-eps_decay_rate * i) 
