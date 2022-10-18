# import random
# from tracemalloc import start
# import numpy as np
# from typing import List, Tuple, Dict
# from connect4.utils import get_pts, get_valid_actions, Integer
# import time

# # 0 based indexing always

# class AIPlayer:
#     def __init__(self, player_number: int, time: int):
#         """
#         :param player_number: Current player number
#         :param time: Time per move (seconds)
#         """
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}:ai'.format(player_number)
#         self.time = time
#         self.moveCount = 0
#         self.columns = -1
#         self.rows = -1
#         self.state_count_expectimax = 0
#         self.state_count_intelligent = 0
#         # Do the rest of your implementation here

#     def state_update(self, state, move, isPop, player_number):
#         board = state[0]
#         new_board = board.copy()
#         rows = len(board)
#         if not isPop and board[0][move] :
#             return -1
#         elif isPop and (board[-1][move]==0 or state[1][player_number].get_int()==0 or 1+move%2!=player_number):
#             return -1
#         elif isPop :
#             first_filled = 0
#             for i in range(rows):
#                 if board[i][move]==0:
#                     new_board[i][move] = 0
#                 else :
#                     new_board[i][move] = 0
#                     first_filled = i
#                     break
#             for i in range(first_filled, rows-1):
#                 new_board[i+1][move] = board[i][move]
#             return (new_board, {player_number: Integer(state[1][player_number].get_int()-1), 3-player_number: state[1][3-player_number]})
#         else :
#             first_filled = 0
#             for i in range(rows):
#                 if board[i][move]==0:
#                     continue
#                 else :
#                     first_filled = i
#                     break
#             new_board[first_filled-1][move] = player_number
#             return (new_board, state[1])

#     def get_expectimax_score(self, state, player_number : int, max_depth : int, step, limit) -> int :
#         move_scores = {}
#         self.state_count_expectimax += 1
#         if max_depth == 0 or limit <= 1:
#             return self.evaluate(state, step)
#         l = get_valid_actions(player_number, state)
#         number_of_moves = len(l)
#         if not l : 
#             return self.evaluate(state, step)
#         for move in l :
#             new_state = self.state_update(state, move[0], move[1], player_number)
#             move_scores["push{}".format(move[0])] = self.get_expectimax_score(new_state,3-player_number, max_depth-1, step + 1, limit//number_of_moves)
#             # if new_state_pop != -1:
#             #     move_scores["pop{}".format(col)] = self.get_expectimax_score(new_state_pop,3-player_number, max_depth-1, step + 1)
#         retval = -1e9
#         if player_number == self.player_number:
#             for move in move_scores:
#                 retval = max(retval, move_scores[move])
#             return retval
#         else :
#             retval = 0
#             for move in move_scores:
#                 retval += move_scores[move]
#             retval /= len(l)
#             return retval

#     def get_minimax_score(self, state, player_number, max_depth, step, limit):
#         move_scores = {}
#         columns = len(state[0])
#         self.state_count_intelligent += 1
#         if max_depth == 0 or limit <= 1:
#             return self.evaluate(state, step)
#         l = get_valid_actions(player_number, state)
#         number_of_moves = len(l)
#         if not l : 
#             return self.evaluate(state, step)
#         for move in l :
#             new_state = self.state_update(state, move[0], move[1], player_number)
#             move_scores["push{}".format(move[0])] = self.get_minimax_score(new_state,3-player_number, max_depth-1, step + 1, limit//number_of_moves)
#         # for col in range(columns) :
#         #     new_state_push = self.state_update(state, col, 0, player_number)
#         #     new_state_pop = self.state_update(state, col, 1, player_number)
#         #     if new_state_push != -1:
#         #         move_scores["push{}".format(col)] = self.get_minimax_score(new_state_push,3-player_number, max_depth-1,step + 1)
#         #     if new_state_pop != -1:
#         #         move_scores["pop{}".format(col)] = self.get_minimax_score(new_state_pop,3-player_number, max_depth-1,step + 1)
#         retval = -1e9
#         if player_number == self.player_number:
#             retval = -1e9
#             for move in move_scores:
#                 retval = max(retval, move_scores[move])
#             return retval
#         else :
#             retval = 1e9
#             for move in move_scores:
#                 retval = min(retval, move_scores[move])
#             return retval     

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
#         self.columns = len(state[0][0])
#         self.rows = len(state[0])
#         self.state_count_intelligent = 0
#         columns = len(state[0][0])
#         move = -1
#         isPop = 0
#         maax = -1e9
#         # for depth in range(10):
#         #     for col in range(columns) :
#         #         new_state_push = self.state_update(state, col, 0, self.player_number)
#         #         new_state_pop = self.state_update(state, col, 1, self.player_number)
#         #         if new_state_push != -1:
#         #             val = self.get_expectimax_score(new_state_push, 3-self.player_number, depth)
#         #             if val > maax :
#         #                 maax = val
#         #                 move = col
#         #                 isPop = 0 
#         #         if new_state_pop != -1:
#         #             val = self.get_expectimax_score(new_state_pop, 3-self.player_number, depth)
#         #             if val > maax :
#         #                 maax = val
#         #                 move = col
#         #                 isPop = 1
#         #     move = -1
#         #     isPop = 0
#         #     maax = -1e9
#         for col in range(columns) :
#             new_state_push = self.state_update(state, col, 0, self.player_number)
#             new_state_pop = self.state_update(state, col, 1, self.player_number)
#             if new_state_push != -1:
#                 val = self.get_minimax_score(new_state_push, 3-self.player_number, 10, self.moveCount, 1000)
#                 if val > maax :
#                     maax = val
#                     move = col
#                     isPop = 0 
#             if new_state_pop != -1:
#                 val = self.get_minimax_score(new_state_pop, 3-self.player_number, 10, self.moveCount, 1000)
#                 if val > maax :
#                     maax = val
#                     move = col
#                     isPop = 1
#         print("States explored", self.state_count_intelligent)
#         print((move, isPop))
#         if isPop: 
#             self.moveCount -= 1
#         else:
#             self.moveCount += 1
#         return (move, isPop)
#         raise NotImplementedError('Whoops I don\'t know what to do')

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
#         self.columns = len(state[0][0])
#         self.rows = len(state[0])
#         self.state_count_expectimax = 0
#         columns = len(state[0][0])
#         move = -1
#         isPop = 0
#         maax = -1e9
#         # for depth in range(10):
#         #     for col in range(columns) :
#         #         new_state_push = self.state_update(state, col, 0, self.player_number)
#         #         new_state_pop = self.state_update(state, col, 1, self.player_number)
#         #         if new_state_push != -1:
#         #             val = self.get_expectimax_score(new_state_push, 3-self.player_number, depth)
#         #             if val > maax :
#         #                 maax = val
#         #                 move = col
#         #                 isPop = 0 
#         #         if new_state_pop != -1:
#         #             val = self.get_expectimax_score(new_state_pop, 3-self.player_number, depth)
#         #             if val > maax :
#         #                 maax = val
#         #                 move = col
#         #                 isPop = 1
#         #     move = -1
#         #     isPop = 0
#         #     maax = -1e9
#         for col in range(columns) :
#             new_state_push = self.state_update(state, col, 0, self.player_number)
#             new_state_pop = self.state_update(state, col, 1, self.player_number)
#             if new_state_push != -1:
#                 val = self.get_expectimax_score(new_state_push, 3-self.player_number, 10, self.moveCount, 1000)
#                 if val > maax :
#                     maax = val
#                     move = col
#                     isPop = 0 
#             if new_state_pop != -1:
#                 val = self.get_expectimax_score(new_state_pop, 3-self.player_number, 10, self.moveCount, 1000)
#                 if val > maax :
#                     maax = val
#                     move = col
#                     isPop = 1
#         print("States explored", self.state_count_expectimax)
#         print(move, isPop)
#         if isPop: 
#             self.moveCount -= 1
#         else:
#             self.moveCount += 1
#         return (move, isPop)
#         raise NotImplementedError('Whoops I don\'t know what to do')

#     def evaluate(self, state: Tuple[np.array, Dict[int, Integer]], step) -> int :
#         pts1 = get_pts(self.player_number, state[0])
#         pts2 = get_pts(3-self.player_number, state[0])
#         pops1 = state[1][self.player_number].get_int()
#         pops2 = state[1][3-self.player_number].get_int()
#         cells = self.rows * self.columns 
#         def weight(x, n):
#             return 1/(1 + np.exp(-(x-n/4)))
        
#         # pop_diff = pops1 - pops2
#         # pop_factor = pts1/max(1, state[1][self.player_number].get_int())-pts2/max(1, state[1][3-self.player_number].get_int())
#         # pop_factor = 1/max(1, )
#         # pop_factor = 
#         w = weight(step, cells)
#         return 2.25*(pts1) - (w+2)*(pts2) + (pts1/(1 + pops1) - pts2/(1 + pops2))*2*w
#         # return pts1 - pts2





from copy import deepcopy
import random
from typing import Dict, List, Tuple
import time
import numpy as np
from connect4.utils import Integer, get_pts, get_valid_actions
from queue import PriorityQueue
import traceback
class AIPlayer:
    def evaluation(self,state):
        my_player_number = self.player_number
        v1 = get_pts(my_player_number, state[0])
        v2 = get_pts(3-my_player_number,state[0])
        frc = (self.get_number_of_filled_cells(state[0])/(state[0].shape[0]*state[0].shape[1]))
        # return v1**2-(1 + 1.5*(self.get_number_of_filled_cells(state[0])/(state[0].shape[0]*state[0].shape[1])))*v2**2
        # return v1**2-(1 + 0.25*frc)*v2**2
        return v1 - v2
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.depth = 4
        self.counter = 0
        self.store_action = PriorityQueue()
        # Do the rest of your implementation here

    


        # Do the rest of your implementation here
        # raise NotImplementedError('Whoops I don\'t know what to do')
    def key(self, action : Tuple[int, bool]):
        x = action[0]*2+action[1]
        return x
    # def store(self, action : Tuple[int, bool], val):
    #     x = action[0]*2+action[1]
    #     self.store_action[x] = val
    # def check(self,action):
    #     if action[0]*2+action[1] in self.store_action:
    #         return (True, self.store_action[action[0]*2+action[1]])
    #     else:
    #         return (False, 0)

    def get_number_of_filled_cells( self, board : np.array ) -> int :
        """
            returns the number of filled cells on the board
        """
        total_cells = board.shape[0] * board.shape[1]
        zero_cells = np.count_nonzero(board==0)
        return total_cells - zero_cells

    def get_player_number( self, board : np.array ) -> int : 

        return self.player_number
        # filled_cells = get_number_of_filled_cells(board)
        # return 1 when (filled_cells % 2 == 0) else 2

    def apply_action( self, action : Tuple[int, bool],  state : Tuple[np.array, Dict[int, Integer]], player : int ) -> Tuple[np.array, Dict[int, Integer]] :
        """
            returns the new state after applying the action on the given state
            player: player number playing the action
        """
        m, n = state[0].shape
        my_player_number = self.player_number
        is_popout = action[1]
        column = action[0]
        # next_state = (state[0]+0,state[1]) # to be returned by this function
        next_state = deepcopy(state)
        if is_popout:
            next_state[1][player] = Integer(next_state[1][player].get_int()-1)
            # next_state[1][player].decrement() # players pop out moves decreases
            next_state[0][0][column] = 0  # first value in column will become zero
            # shift values in the columns
            for row in range(m-1,0,-1):
                next_state[0][row][column] = next_state[0][row-1][column] 

        else:
            empty_column = True
            for row in range(m):
                if(state[0][row][column] != 0 ):
                    # first non zero value in this column
                    next_state[0][row-1][column] = player
                    empty_column = False
                    break
            if( empty_column ):
                next_state[0][m-1][column] = player
        return next_state



    def expectation_node(self, state : Tuple[np.array, Dict[int, Integer]]) -> int :
        """
            returns the mean of value of all children
        """
        if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
            raise Exception("Time out")       
        adversary_number = 3 - self.player_number
        valid_actions  = get_valid_actions(adversary_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or self.depth == 0:
            v1 = get_pts(self.player_number, state[0])
            v2 = get_pts(3-self.player_number,state[0])
            return self.evaluation(state)
        sum_of_children = 0
        for action in valid_actions:
            self.counter += 1
            next_state = self.apply_action(action, state, adversary_number)
            self.depth -= 1
            child_value, _ = self.expectimax_node(next_state)
            self.depth += 1
            sum_of_children += child_value
        
        return sum_of_children / total_number_of_valid_actions
    def evaluation_node( self, state : Tuple[np.array, Dict[int, Integer]],store , alpha , beta) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.intelligent_st + self.time ) - time.time() < 0.5 ):
            print(self.counter)
            raise Exception("Time out")
        my_player_number = 3-self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # print(state)
            # return (get_pts(my_player_number, state[0]), None)
            v1 = get_pts(my_player_number, state[0])
            v2 = get_pts(3-my_player_number,state[0])
            return self.evaluation(state)
        best_value = None
        explored = {-1}
        storec = PriorityQueue()
        itt = 0
        while not store.empty():
            self.counter += 1
            itt += 1
            k = store.get()
            action = k[1]
            explored.add(action)
            if itt < 40:
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                child_value = self.minimax_node(next_state, k[2], alpha, beta)[0]
                self.depth += 1
                storec.put((-child_value,k[1],k[2]))
                if( best_value is None ):
                    # best_action = action
                    best_value = child_value
                elif( child_value < best_value ):
                    best_value = child_value
                    # best_action = action
                if (beta is None) :
                    beta = best_value
                elif (best_value < beta):
                    beta = best_value
                if (alpha is not None):
                    if beta <= alpha:
                        break
        if( alpha is not None):
            if (beta is not None):
                if  beta <= alpha:
                    return best_value
        while not storec.empty():
            t = storec.get()
            store.put(t)
        for action in valid_actions:
            if action not in explored:
                explored.add(action)
                newpq = PriorityQueue()
                self.counter += 1
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                child_value = self.minimax_node(next_state, newpq, alpha, beta)[0]
                store.put((child_value,action,newpq))
                self.depth += 1
                if( best_value is None ):
                    # best_action = action
                    best_value = child_value
                elif( child_value < best_value ):
                    best_value = child_value
                    # best_action = action
                if (beta is None) :
                    beta = best_value
                elif (best_value < beta):
                    beta = best_value
                if (alpha is not None):
                    if beta <= alpha:
                        break
        return best_value



    def expectimax_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
            raise Exception("Time out")

        my_player_number = self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # print(state)
            # return (get_pts(my_player_number, state[0]), None)
            # print(valid_actions,self.depth)
            return (self.evaluation(state),None)
        best_value, best_action = None, None       
        for action in valid_actions:
            self.counter += 1
            next_state = self.apply_action(action, state, my_player_number)
            self.depth -= 1
            child_value = self.expectation_node(next_state)
            self.depth += 1
            if( best_value is None ):
                best_action = action
                best_value = child_value
            elif( child_value > best_value ):
                best_value = child_value
                best_action = action
        return (best_value, best_action)
    def minimax_node( self, state : Tuple[np.array, Dict[int, Integer]],store , alpha = None, beta = None) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.intelligent_st + self.time ) - time.time() < 0.5 ):
            print(self.counter)
            raise Exception("Time out")
        my_player_number = self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # if( total_number_of_valid_actions == 0 ):
            #     print("herewego")
            return (self.evaluation(state),None,valid_actions)
        best_value, best_action = None, None
        explored = {-1}
        # storec = deepcopy(store)
        storec = PriorityQueue()
        itt = 0
        while not store.empty():
            self.counter += 1
            itt += 1
            k = store.get()
            # print(k)
            action = k[1]
            explored.add(action)
            if itt <= 30:
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                child_value = self.evaluation_node(next_state, k[2], alpha, beta)
                self.depth += 1
                storec.put((-child_value,k[1],k[2]))
                if( best_value is None ):
                    best_action = action
                    best_value = child_value
                elif( child_value > best_value ):
                    best_value = child_value
                    best_action = action
                if (alpha is None) :
                    alpha = best_value
                elif (best_value > alpha):
                    alpha = best_value
                if (beta is not None):
                    if beta <= alpha:
                        break
        while not storec.empty():
            t = storec.get()
            store.put(t)
        if( alpha is not None):
            if (beta is not None):
                if  beta <= alpha:
                    # print(len(valid_actions)+1-len(explored))
                    return (best_value, best_action,valid_actions)

        for action in valid_actions:
            # print(action,self.depth)
            self.counter += 1
            if action not in explored:
                explored.add(action)
                newpq = PriorityQueue()
                next_state = self.apply_action(action, state, my_player_number)
                # dec = False
                # if self.depth > 2:
                #     self.depth -= 1

                self.depth -= 1
                child_value = self.evaluation_node(next_state, newpq, alpha, beta)
                self.depth += 1
                store.put((-child_value,action,newpq))

                if( best_value is None ):
                    best_action = action
                    best_value = child_value
                elif( child_value > best_value ):
                    best_value = child_value
                    best_action = action
                if (alpha is None) :
                    alpha = best_value
                elif (best_value > alpha):
                    alpha = best_value
                if (beta is not None):
                    if beta <= alpha:
                        break

                
        # if best_action is None:
        #     print(best_value, len(valid_actions))
        return (best_value, best_action,valid_actions)
    def get_minimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        self.depth = 4
        self.intelligent_st = time.time()
        ans = self.minimax_node(state,self.store_action)
        print(self.counter,self.depth,time.time()-self.intelligent_st)
        # while self.counter < 5000 and self.depth < 100 and time.time()-st < self.time/10:
        while self.depth < 100:
            self.counter = 0
            self.depth += 1
            try:
                ans = self.minimax_node(state,self.store_action)
                print(self.counter,self.depth,time.time()-self.intelligent_st)
            except Exception as e:
                # print(e.)
                # print(traceback.format_exc())
                print(e)
                break
            # if self.depth %  == 0:
            # print(self.counter,self.depth,time.time()-st)
            self.counter = 0
        # print(ans)
        # time.sleep(1)
        return ans[1] 
        


    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        self.depth = 1
        self.expectimax_st = time.time()        
        # ans = self.expectimax_node(state)
        # print(ans)
        # print(self.counter,self.depth)
        # while self.counter < 1000 and self.depth < 100:
        while self.depth < 100:
            self.depth += 1
            self.counter = 0
            try:
                ans = self.expectimax_node(state)
            except:
                print("Time about to end !!!")
                break
            # if x[1] is not None:
            #     ans = x
            # else:
            #     break
            # print(ans)
            # print(self.counter,self.depth)
        # self.counter = 0
        # time.sleep(1)
        print(ans)
        return ans[1]
        # raise NotImplementedError('Whoops I don\'t know what to do')
    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        ans = self.get_minimax_move(state)
        # print(self.counter)
        # self.counter = 0
        # print(ans)
        return ans
        # raise NotImplementedError('Whoops I don\'t know what to do')
