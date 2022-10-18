# import random
# from typing import Dict, List, Tuple

# import numpy as np
# from connect4.utils import Integer, get_pts, get_valid_actions


# class AIPlayer:
#     def evaluation(self,state):
#         my_player_number = self.player_number
#         v1 = get_pts(my_player_number, state[0])
#         v2 = get_pts(3-my_player_number,state[0])
#         return v1**2-(0.5 + 1.5*(self.get_number_of_filled_cells(state[0])/(state[0].shape[0]*state[0].shape[1])))*v2**2
#     def __init__(self, player_number: int, time: int):
#         """
#         :param player_number: Current player number
#         :param time: Time per move (seconds)
#         """
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}:ai'.format(player_number)
#         self.time = time
#         self.depth = 2
#         self.counter = 0
#         # Do the rest of your implementation here

    


#         # Do the rest of your implementation here
#         # raise NotImplementedError('Whoops I don\'t know what to do')
    
#     def get_number_of_filled_cells( self, board : np.array ) -> int :
#         """
#             returns the number of filled cells on the board
#         """
#         total_cells = board.shape[0] * board.shape[1]
#         zero_cells = np.count_nonzero(board==0)
#         return total_cells - zero_cells

#     def get_player_number( self, board : np.array ) -> int : 

#         return self.player_number
#         # filled_cells = get_number_of_filled_cells(board)
#         # return 1 when (filled_cells % 2 == 0) else 2

#     def apply_action( self, action : Tuple[int, bool],  state : Tuple[np.array, Dict[int, Integer]], player : int ) -> Tuple[np.array, Dict[int, Integer]] :
#         """
#             returns the new state after applying the action on the given state
#             player: player number playing the action
#         """
#         m, n = state[0].shape
#         my_player_number = self.player_number
#         is_popout = action[1]
#         column = action[0]
#         next_state = (state[0]+0,state[1]) # to be returned by this function
#         if is_popout:
#             next_state[1][player] = Integer(next_state[1][player].get_int()-1)
#             # next_state[1][player].decrement() # players pop out moves decreases
#             next_state[0][0][column] = 0  # first value in column will become zero
#             # shift values in the columns
#             for row in range(m-1,0,-1):
#                 next_state[0][row][column] = next_state[0][row-1][column] 

#         else:
#             empty_column = True
#             for row in range(m):
#                 if(state[0][row][column] != 0 ):
#                     # first non zero value in this column
#                     next_state[0][row-1][column] = player
#                     empty_column = False
#                     break
#             if( empty_column ):
#                 next_state[0][m-1][column] = player
#         return next_state



#     def expectation_node(self, state : Tuple[np.array, Dict[int, Integer]]) -> int :
#         """
#             returns the mean of value of all children
#         """
#         adversary_number = 3 - self.player_number
#         valid_actions  = get_valid_actions(adversary_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ):
#             v1 = get_pts(self.player_number, state[0])
#             v2 = get_pts(3-self.player_number,state[0])
#             return self.evaluation(state)
#         sum_of_children = 0
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, adversary_number)
#             self.depth -= 1
#             child_value, _ = self.expectimax_node(next_state)
#             self.depth += 1
#             sum_of_children += child_value
        
#         return sum_of_children / total_number_of_valid_actions
#     def evaluation_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         my_player_number = 3-self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # print(state)
#             # return (get_pts(my_player_number, state[0]), None)
#             v1 = get_pts(my_player_number, state[0])
#             v2 = get_pts(3-my_player_number,state[0])
#             return self.evaluation(state)
#         best_value, best_action = None, None       
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             self.depth -= 1
#             child_value,_ = self.minimax_node(next_state)
#             self.depth += 1
#             if( best_value is None ):
#                 # best_action = action
#                 best_value = child_value
#             elif( child_value < best_value ):
#                 best_value = child_value
#                 # best_action = action
#         return best_value



#     def expectimax_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         my_player_number = self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # print(state)
#             # return (get_pts(my_player_number, state[0]), None)
            
#             return (self.evaluation(state),None)
#         best_value, best_action = None, None       
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             child_value = self.expectation_node(next_state)
#             if( best_value is None ):
#                 best_action = action
#                 best_value = child_value
#             elif( child_value > best_value ):
#                 best_value = child_value
#                 best_action = action
#         return (best_value, best_action)
#     def minimax_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         my_player_number = self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # print(state)
#             # return (get_pts(my_player_number, state[0]), None)
#             v1 = get_pts(self.player_number, state[0])
#             v2 = get_pts(3-self.player_number,state[0])
#             return (self.evaluation(state),None)
#         best_value, best_action = None, None       
#         for action in valid_actions:
#             # self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             # self.depth -= 1
#             child_value = self.evaluation_node(next_state)
#             # self.depth += 1
#             if( best_value is None ):
#                 best_action = action
#                 best_value = child_value
#             elif( child_value > best_value ):
#                 best_value = child_value
#                 best_action = action
#         return (best_value, best_action)
#     def get_minimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move based on
#         the Expecti max algorithm.
#         This will play against the random player, who chooses any valid move
#         with equal probability
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         # Do the rest of your implementation here
#         # self.depth = 5
#         ans = self.minimax_node(state)[1]
#         # print(self.counter,self.depth)
#         while self.counter < 1000 and self.depth < 200:
#             self.depth += 1
#             ans = self.minimax_node(state)[1]
#             # print(self.counter,self.depth)
#         self.counter = 0

#         return ans 
        


#     def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move based on
#         the Expecti max algorithm.
#         This will play against the random player, who chooses any valid move
#         with equal probability
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         # Do the rest of your implementation here
#         # self.depth = 5
#         ans = self.expectimax_node(state)[1]
#         # print(self.counter,self.depth)
#         while self.counter < 1500 and self.depth < 200:
#             self.depth += 1
#             ans = self.expectimax_node(state)[1]
#             # print(self.counter,self.depth)
#         self.counter = 0

#         return ans 
#         # raise NotImplementedError('Whoops I don\'t know what to do')
#     def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move
#         This will play against either itself or a human player
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         ans = self.get_minimax_move(state)
#         # print(self.counter)
#         # self.counter = 0
#         print(ans)
#         return ans
#         # raise NotImplementedError('Whoops I don\'t know what to do')


## Vaibhav Implementation
# alpha beta pruning done 
import random
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer

import copy
import random
import time
import queue

# DEPTH = 4

# import multiprocessing.pool
# import functools

# def timeout(max_timeout):
#     """Timeout decorator, parameter in seconds."""
#     def timeout_decorator(item):
#         """Wrap the original function."""
#         @functools.wraps(item)
#         def func_wrapper(*args, **kwargs):
#             """Closure for function."""
#             pool = multiprocessing.pool.ThreadPool(processes=1)
#             async_result = pool.apply_async(item, args, kwargs)
#             # raises a TimeoutError if execution exceeds max_timeout
#             return async_result.get(max_timeout)
#         return func_wrapper
#     return timeout_decorator

class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time

        # Do the rest of your implementation here
        self.opponent = 3 - self.player_number
        # self.prev_states = []

    def get_next_state(self, action, player, state) -> Tuple[np.array, Dict[int, Integer]]:
        ret_state = copy.deepcopy(state)
        column_num, is_popout = action[0], action[1]
        if is_popout:
            ret_state[0][1:, column_num] = ret_state[0][:-1, column_num]
            ret_state[0][0, column_num] = 0
            ret_state[1][player].decrement()
        else:
            for i in range(self.row):
                if ret_state[0][i, column_num] != 0:
                    ret_state[0][i-1, column_num] = player
                    break
            if(ret_state[0][self.row - 1, column_num] == 0):
                ret_state[0][self.row - 1, column_num] = player
        return ret_state

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        # global DEPTH
        # global prev_states
        DEPTH = 2
        print(DEPTH)
        self.row = len(state[0])
        self.col = len(state[0][0])

        start = time.time()

        def get_evaluation():
            first_eval = get_pts(self.player_number, self.temp_state[0])
            second_eval = get_pts(self.opponent, self.temp_state[0])
            return (first_eval) - 3 * (second_eval)

        def max_value(depth, pn, ismax, alpha, beta):
            if(time.time() - start > self.time - 0.2):
                return -1, -1

            valid_actions = get_valid_actions(pn, self.temp_state)
            # change this
            valid_actions.sort(key= lambda x : abs(x[0] - (self.col - 1) // 2))
            if(depth == DEPTH or len(valid_actions) == 0):
                score = get_evaluation()
                # print(depth, self.temp_state[0], ismax, score, None)
                return score, None

            # curr = -1e9 if ismax else 0
            curr = -1e15 if ismax else 1e15
            action = None
            for act in valid_actions:
                col_act, isp = act[0], act[1]
                if(isp):
                    # continue
                    # if(depth == 0 and np.array2string(self.temp_state[0]) in prev_states):
                        # print("already popped this", prev_states)
                    #     continue
                    curr_col = self.temp_state[0][-1, col_act]
                    self.temp_state[0][1:, col_act] = self.temp_state[0][0:-1, col_act]
                    self.temp_state[0][0, col_act] = 0
                    self.temp_state[1][pn].decrement()
                    sco, n_ = max_value(depth + 1, 3 - pn, not ismax, alpha, beta)

                    # print(sco, depth, "exp")
                    self.temp_state[0][0:-1, col_act] = self.temp_state[0][1:, col_act]
                    self.temp_state[0][-1, col_act] = curr_col
                    self.temp_state[1][pn].increment()
                    if(ismax):
                        alpha = max(alpha, sco)

                        if alpha >= beta:
                            return alpha, act

                        if(sco > curr):
                            curr = sco
                            action = act
                    else:
                        # curr += sco
                        beta = min(beta, sco)

                        if alpha >= beta:
                            return beta, act

                        if(sco < curr):
                            curr = sco
                            action = act

                else:
                    upd = -1
                    for i in range(self.row):
                        if(self.temp_state[0][i, col_act] != 0): 
                            self.temp_state[0][i-1, col_act] = pn
                            upd = i-1
                            break
                    if(upd == -1):
                        self.temp_state[0][self.row - 1, col_act] = pn
                        upd = self.row-1

                    sco, n_ = max_value(depth + 1, 3 - pn, not ismax, alpha, beta)
                    # print(sco, depth, "exp")
                    self.temp_state[0][upd, col_act] = 0
                    if(ismax):
                        alpha = max(alpha, sco)
                        if alpha >= beta:
                            return alpha, act
                        if(sco > curr):
                            curr = sco
                            action = act
                    else:
                        # curr += sco
                        beta = min(beta, sco)
                        if alpha >= beta:
                            return beta, act
                        if(sco < curr):
                            curr = sco
                            action = act

            # if(not ismax): curr /= len(valid_actions)
            # print(depth, self.temp_state[0], ismax, curr, action)
            return curr, action

        # start writing stuff from here

        # deterministic

        # beginning game

        # mid game

        # opponent handle

        # score improvement

        # endgame

        # popout handle

        


        # minimax
        self.temp_state = state

        # a = - np.log(1 - 0.6) / 0.3 # 0.8 is the final probab, and 0.6 is the game stage at this probab
        # def get_proba():
        #     cutoff = 1 - np.exp(-a * self.progress(state[0]))
        #     return random.random() < cutoff

        
        # curr_pts = get_pts(self.player_number, state[0]) - get_pts(self.opponent, state[0])
        
        # best_score_det = -1e15
        # best_action_det = None
        # if(get_proba() and np.array2string(state[0]) not in prev_states):
        #     print("deterministically popping fn")
        #     for act in get_valid_actions(self.player_number, state):
        #         if(act[1] == True):
        #             col_act, isp = act[0], act[1]
        #             curr_col = self.temp_state[0][-1, col_act]
        #             self.temp_state[0][1:, col_act] = self.temp_state[0][0:-1, col_act]
        #             self.temp_state[0][0, col_act] = 0
        #             self.temp_state[1][self.player_number].decrement()
        #             new_pts = get_pts(self.player_number, state[0]) - get_pts(self.opponent, self.temp_state[0])
        #             if(new_pts > best_score_det):
        #                 best_score_det = new_pts
        #                 best_action_det = act
        #             # print(sco, depth, "exp")
        #             self.temp_state[0][0:-1, col_act] = self.temp_state[0][1:, col_act]
        #             self.temp_state[0][-1, col_act] = curr_col
        #             self.temp_state[1][self.player_number].increment()
        #     # print(best_score)
        #     if(best_score_det > curr_pts):
        #         print("deterministically popping was valid, let's see ahead")
        #         # prev_states.append(np.array2string(state[0]))
        #         # if(len(prev_states) > 4):
        #         #     prev_states.pop(0)
        #         # return best_action

        best_score = -1e9
        best_action = None
        max_depth = 200
        while(DEPTH <= max_depth):
            self.temp_state = state
            ok = time.time()

            try:
                action = max_value(0, self.player_number, True, -1e15, 1e15)
            except Exception as e:
                print(repr(e))
            if(time.time() - start > self.time - 0.2):
                break
            # if(best_score < action[0]):
            best_score = action[0]
            best_action = action[1]
            # print("Depth {}. Best {}. Move {}. Time = {}.".format(DEPTH, best_score, best_action, time.time() - ok))
            DEPTH += 2
        end = time.time()

        # minmax_state = self.get_next_state(best_action, self.player_number, state)
        # sc = get_pts(self.player_number, minmax_state[0]) - get_pts(self.player_number, minmax_state[0])

        # if(sc > best_score_det):
        #     print("selected minmax move over det pop")
        #     return best_action
        # else: 
        #     print("selected det_pop over minmax")
        #     return best_action_det

        # if(best_action[1]):
        #     print("popping", prev_states)
        #     prev_states.append(np.array2string(state[0]))
        #     if(len(prev_states) > 4):
        #         prev_states.pop(0)

        # print(prev_states)  


        print('ai is player {}. he chose move {}. time taken = {}. depth = {}.'.format(self.player_number, best_action, end - start, DEPTH))
        return best_action

    temp_state = None
    row = -1
    col = -1

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        # global DEPTH
        DEPTH = 2
        self.row = len(state[0])
        self.col = len(state[0][0])

        start = time.time()

        def get_evaluation():
            first_eval = get_pts(self.player_number, self.temp_state[0])
            second_eval = get_pts(self.opponent, self.temp_state[0])
            return (first_eval) - 7 * (second_eval)

        def exp_value(depth, pn, ismax):
            if(time.time() - start > self.time - 0.2):
                return -1, -1

            valid_actions = get_valid_actions(pn, self.temp_state)
            valid_actions.sort(key= lambda x : abs(x[0] - (self.col - 1) // 2))
            if(depth == DEPTH or len(valid_actions) == 0):
                score = get_evaluation()
                # print(depth, self.temp_state[0], ismax, score, None)
                return score, None

            curr = -1e9 if ismax else 0
            action = None
            for act in valid_actions:
                col_act, isp = act[0], act[1]
                if(isp):
                    curr_col = self.temp_state[0][-1, col_act]
                    self.temp_state[0][1:, col_act] = self.temp_state[0][0:-1, col_act]
                    self.temp_state[0][0, col_act] = 0
                    self.temp_state[1][pn].decrement()
                    sco, n_ = exp_value(depth + 1, 3 - pn, not ismax)
                    # print(sco, depth, "exp")
                    self.temp_state[0][0:-1, col_act] = self.temp_state[0][1:, col_act]
                    self.temp_state[0][-1, col_act] = curr_col
                    self.temp_state[1][pn].increment()
                    if(ismax):
                        if(sco > curr):
                            curr = sco
                            action = act
                    else:
                        curr += sco

                else:
                    upd = -1
                    for i in range(self.row):
                        if(self.temp_state[0][i, col_act] != 0): 
                            self.temp_state[0][i-1, col_act] = pn
                            upd = i-1
                            break
                    if(upd == -1):
                        self.temp_state[0][self.row - 1, col_act] = pn
                        upd = self.row-1

                    sco, n_ = exp_value(depth + 1, 3 - pn, not ismax)
                    # print(sco, depth, "exp")
                    self.temp_state[0][upd, col_act] = 0
                    if(ismax):
                        if(sco > curr):
                            curr = sco
                            action = act
                    else:
                        curr += sco

            if(not ismax): curr /= len(valid_actions)
            # print(depth, self.temp_state[0], ismax, curr, action)
            return curr, action

        best_score = -1e9
        best_action = None
        max_depth = 200
        while(DEPTH <= max_depth):
            self.temp_state = state
            ok = time.time()

            try:
                action = exp_value(0, self.player_number, True)
            except Exception as e:
                print(repr(e))
            if(time.time() - start > self.time - 0.2):
                break
            # if(best_score < action[0]):
            best_score = action[0]
            best_action = action[1]
            # print("Depth {}. Best {}. Move {}. Time = {}.".format(DEPTH, best_score, best_action, time.time() - ok))
            DEPTH += 2
        end = time.time()

        # print('ai is player {}. he chose move {}. time taken = {}. depth = {}.'.format(self.player_number, best_action, end - start, DEPTH))
        return best_action