import random
from tracemalloc import start
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer
import time

# 0 based indexing always

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
        self.moveCount = 0
        self.columns = -1
        self.rows = -1
        self.start_time = 0
        self.state_count_expectimax = 0
        self.state_count_intelligent = 0
        self.noOf_intelligent_nodes = {}
        # Do the rest of your implementation here

    def state_update(self, state, move, isPop, player_number):
        board = state[0]
        new_board = board.copy()
        rows = len(board)
        if not isPop and board[0][move] :
            return -1
        elif isPop and (board[-1][move]==0 or state[1][player_number].get_int()==0 or 1+move%2!=player_number):
            return -1
        elif isPop :
            first_filled = 0
            for i in range(rows):
                if board[i][move]==0:
                    new_board[i][move] = 0
                else :
                    new_board[i][move] = 0
                    first_filled = i
                    break
            for i in range(first_filled, rows-1):
                new_board[i+1][move] = board[i][move]
            return (new_board, {player_number: Integer(state[1][player_number].get_int()-1), 3-player_number: state[1][3-player_number]})
        else :
            first_filled = 0
            for i in range(rows):
                if board[i][move]==0:
                    continue
                else :
                    first_filled = i
                    break
            new_board[first_filled-1][move] = player_number
            return (new_board, state[1])

    def get_minimax_score(self, state, player_number, max_depth, step, limit):
        move_scores = {}
        if step in self.noOf_intelligent_nodes:
            self.noOf_intelligent_nodes[step] += 1
        else :
            self.noOf_intelligent_nodes[step] = 1 
        self.state_count_intelligent += 1
        if max_depth == 0 or limit <= 1:
            return self.evaluate_minimax(state, step)
        l = get_valid_actions(player_number, state)

        if not l : 
            return self.evaluate_minimax(state, step)
        
        # shuffle l 
        random.shuffle(l)
        index = 0
        sum = 0
        minimum = 1e9
        sum = 0

        for move in l :
            new_state = self.state_update(state, move[0], move[1], player_number)
            move_scores[index] = self.evaluate_minimax(new_state, step+1)
            minimum = min(minimum, move_scores[index])
            index += 1
        for move in move_scores:
            sum += max(5, move_scores[move]-minimum)
        index = 0
        for move in l:
            new_state = self.state_update(state, move[0], move[1], player_number)
            move_scores[index] = self.get_minimax_score(new_state,3-player_number, max_depth-1, step + 1, (limit*max(5, move_scores[index]-minimum))//sum)
            index += 1
        retval = -1e9
        if player_number == self.player_number:
            retval = -1e9
            for move in move_scores:
                retval = max(retval, move_scores[move])
            return retval
        else :
            retval = 1e9
            for move in move_scores:
                retval = min(retval, move_scores[move])
            return retval  

    def get_minimax_score_ab(self, state, player_number, max_depth, step, alpha, beta):

        if time.time() - self.start_time > self.time-1:
            return -1e18
        l = get_valid_actions(player_number, state)
        if max_depth == 0 or not l:
            return self.evaluate_minimax(state, step)
        random.shuffle(l)
        if player_number == self.player_number:
            v = -1e9
            for move in l:
                new_state = self.state_update(state, move[0], move[1], player_number)
                score = self.get_minimax_score_ab(new_state, 3-player_number, max_depth-1, step+1, alpha, beta)
                if score == -1e18:
                    return score
                v = max(v, score) 
                if v >= beta :
                    return v
                alpha = max(alpha, v)   
            return v
        
        else :
            v = 1e9
            for move in l:
                new_state = self.state_update(state, move[0], move[1], player_number)
                score = self.get_minimax_score_ab(new_state, 3-player_number, max_depth-1, step+1, alpha, beta)
                v = min(v, score) 
                if score == -1e18:
                    return score
                if v <= alpha :
                    return v
                beta = min(beta, v)
            return v
    
    def iterative_deepening_search(self, state: Tuple[np.array, Dict[int, Integer]], max_depth) -> Tuple[int, bool]:

        l = get_valid_actions(self.player_number, state)
        random.shuffle(l)

        best_move = -1
        isPop = 0

        alpha = -1e9
        beta = 1e9
        maax = -1e9

        for move in l:
            new_state = self.state_update(state, move[0], move[1], self.player_number)
            score = self.get_minimax_score_ab(new_state, 3-self.player_number, max_depth-1, 1, alpha, beta)
            if score == -1e18:
                return (-1, -1)
            if score > maax:
                maax = score
                best_move = move[0]
                isPop = move[1]
            alpha = max(alpha, maax)   
        return (best_move, isPop)

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:

        self.start_time = time.time()

        best_move = -1
        isPop = 0

        for depth in range(3-self.player_number, 500, 2):
            (a, b) = self.iterative_deepening_search(state, depth)
            if a+1:
                (best_move, isPop) = (a, b)
                print("Upto depth", depth, "best column is", best_move, "isPop is", isPop)
            else :
                break
        print(best_move, isPop, "time taken", time.time()-self.start_time)
        return (best_move, isPop)

    def evaluate_minimax(self, state: Tuple[np.array, Dict[int, Integer]], step) -> int :
        pts1 = get_pts(self.player_number, state[0])
        pts2 = get_pts(3-self.player_number, state[0])
        pops1 = state[1][self.player_number].get_int()
        pops2 = state[1][3-self.player_number].get_int()
        n = self.rows * self.columns 

        self_cnt = 0
        opp_cnt = 0

        board = state[0]

        # use np here 
        for i in range(self.rows):
            for j in range(self.columns):
                self_cnt += board[i][j] == self.player_number
                opp_cnt += board[i][j] == 3-self.player_number
        x = self_cnt + opp_cnt
        def weight(x, n):
            return 1/(1 + np.exp(-(x-n/4)))
        
        def f1(x):
            if x <= n/4 and x >= 0:
                return 1
            return 0
        def f2(x):
            if x <= 3*n/4 and x > n/4:
                return 1
            return 0
        def f3(x):
            if x > 3*n/4: 
                return 1
            return 0

        return f1(x)*(20*pops1 + pts1 - 7*pts2) + f2(x)*(pts1*pts1 - 5*pts2*pts2) + f3(x)*(pts1 - 3*pts2)

    def get_expectimax_score(self, state, player_number : int, max_depth : int, step) -> int :

        if time.time() - self.start_time > self.time-1:
            return -1e18
        move_scores = {}
        self.state_count_expectimax += 1

        l = get_valid_actions(player_number, state)

        random.shuffle(l)

        if max_depth == 0 or not l:
            return self.evaluate_expectimax(state, step)

        index = 0
        for move in l :
            new_state = self.state_update(state, move[0], move[1], player_number)
            move_scores[index] = self.get_expectimax_score(new_state, 3-player_number, max_depth-1, step+1)
            if move_scores[index] == -1e18:
                return -1e18
            index += 1

        retval = -1e9
        if player_number == self.player_number:
            for move in move_scores:
                retval = max(retval, move_scores[move])
            return retval
        else :
            retval = 0
            for move in move_scores:
                retval += move_scores[move]
            retval /= len(l)
            return retval

    def iterative_deepening_search_expectimax(self, state: Tuple[np.array, Dict[int, Integer]], max_depth) -> Tuple[int, bool]:
        l = get_valid_actions(self.player_number, state)
        random.shuffle(l)

        best_move = -1
        isPop = 0

        maax = -1e9

        for move in l:
            new_state = self.state_update(state, move[0], move[1], self.player_number)
            score = self.get_expectimax_score(new_state, 3-self.player_number, max_depth-1, 1)
            if score == -1e18:
                return (-1, -1)
            if score > maax:
                maax = score
                best_move = move[0]
                isPop = move[1] 
        return (best_move, isPop)

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:

        self.start_time = time.time()

        best_move = -1
        isPop = 0

        for depth in range(3-self.player_number, 500, 2):
            (a, b) = self.iterative_deepening_search_expectimax(state, depth)
            if a+1:
                (best_move, isPop) = (a, b)
                print("Upto depth", depth, "best column is", best_move, "isPop is", isPop)
            else :
                break
        print(best_move, isPop, "time taken", time.time()-self.start_time)
        return (best_move, isPop)
    
    def evaluate_expectimax(self, state, step) :
        pts1 = get_pts(self.player_number, state[0])
        pts2 = get_pts(3-self.player_number, state[0])
        pops1 = state[1][self.player_number].get_int()
        pops2 = state[1][3-self.player_number].get_int()
        n = self.rows * self.columns 

        self_cnt = 0
        opp_cnt = 0

        board = state[0]

        # use np here 
        for i in range(self.rows):
            for j in range(self.columns):
                self_cnt += board[i][j] == self.player_number
                opp_cnt += board[i][j] == 3-self.player_number
        x = self_cnt + opp_cnt
        def weight(x, n):
            return 1/(1 + np.exp(-(x-n/4)))
        
        def f1(x):
            if x <= n/4 and x >= 0:
                return 1
            return 0
        def f2(x):
            if x <= 3*n/4 and x > n/4:
                return 1
            return 0
        def f3(x):
            if x > 3*n/4: 
                return 1
            return 0

        return f1(x)*(20*pops1 + pts1 - 7*pts2) + f2(x)*(pts1*pts1 - 5*pts2*pts2) + f3(x)*(pts1 - 3*pts2)
