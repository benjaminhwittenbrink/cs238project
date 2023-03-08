import minesweeper.minesweeper as ms

env = ms.MinesweeperEnv() 


# greedy explore
# def epsilon_greedy_exploration(s, epsilon): 
#     rand = np.random.randint(1, 100)/100 
#     if rand < epsilon: 
#         a = np.random.randint(1, action_space)
#     else: 
#         subset = df[(df['s'].values == s)]
#         a = df['a'][subset['r'].idxmax()]
#     return a

