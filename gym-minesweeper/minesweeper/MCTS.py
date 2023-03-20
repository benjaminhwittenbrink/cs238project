from copy import deepcopy
import numpy as np
import random
import minesweeper as ms

GAMMA = 0.9 

class MCTS:
    def __init__(self, env, exploration_param=1.0, widening_param=0.5):
        self.env = env
        self.exploration_param = exploration_param
        self.widening_param = widening_param
        self.Q = {}
        self.N = {}
        #self.N_widening = {}
        
    def run_mcts_search(self, state_str, num_simulations=10):
        for _ in range(num_simulations):
            self.simulate(state_str, self.search(state_str, self.env))

        return self.get_best_action(state_str)

    def search(self, state, env):
        if state not in self.Q:  # if state doesn't exist 
            self.Q[state] = {}
            self.N[state] = {}
            #self.N_widening[state] = 0

            for action in env.get_valid_actions(state):
                self.Q[state][action] = 0
                self.N[state][action] = 0

            # if self.N_widening[state] < (env.board_size ** 2) ** self.widening_param:
            #     action = random.choice(list(self.Q[state].keys()))
            #     self.N_widening[state] += 1
            #     return action 
            # else:
            return self.get_best_action(state)

        else: ## returns best action in the state 
            return self.get_best_action(state)

    def simulate(self, state_str, action):

        visited_states = [state_str]
        visited_actions = [action]

        env_copy = deepcopy(self.env)
        new_state, reward, done, _ = env_copy.step(action)
        rewards = [reward]

        while not done:
            new_state_str = ms.board2str(new_state)
            next_action = self.search(new_state_str, env_copy)
            visited_states.append(new_state_str)
            visited_actions.append(next_action)

            new_state, reward, done, _ = env_copy.step(next_action)
            rewards.append(reward)

        q = 0 
        for j in range(len(visited_states) - 1, -1, -1): 
            q = rewards[j] + GAMMA * q
            self.update(visited_states[j], visited_actions[j], q)

        return q

    def update(self, state, action, q):
        self.N[state][action] += 1
        self.Q[state][action] += ((q - self.Q[state][action]) / self.N[state][action])

    def get_best_action(self, state):
        best_action = None
        best_score = -np.inf
        
        N_sum = sum(self.N[state].values())

        for action in self.Q[state].keys():
            if self.N[state][action] == 0:
                score = np.inf
                return action 
            else:
                UCB1 = self.exploration_param * np.sqrt( np.log( N_sum ) / self.N[state][action] )
                score = self.Q[state][action] + UCB1 

            if score > best_score:
                best_action = action
                best_score = score

        return best_action