import math
import random
import copy
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer, get_row_score, get_diagonals_primary, get_diagonals_secondary
from typing import List, Tuple, Dict, Union

import numpy as np

import time

win_pts = [0, 0, 2, 10, 30]

iter=0
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
        self.iter=0
        self.start=0
        self.transposition_table = {}
        # Do the rest of your implementation here

    def get_row_score(player_number: int, row: Union[np.array, List[int]]):
        score = 0
        n = len(row)
        j = 0
        while j < n:
            if row[j] == player_number:
                count = 0
                while j < n and row[j] == player_number:
                    count += 1
                    j += 1
                k = len(win_pts) - 1
                score += win_pts[count % k] + (count // k) * win_pts[k]
            else:
                j += 1
        return score


    def get_diagonals_primary(board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for j in range(max(0, k - m + 1), min(n, k + 1)):
                i = k - j
                diag.append(board[i, j])
            yield diag


    def get_diagonals_secondary(board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for x in range(max(0, k - m + 1), min(n, k + 1)):
                j = n - 1 - x
                i = k - x
                diag.append(board[i][j])
            yield diag

    def score(self, player_number: int, board: np.array, pop) -> int:
        """
        :return: Returns the total score of player (with player number) on the board
        """
        score = 0
        m, n = board.shape
        # center_array = [int(i) for i in list(board[:, n//2])]
        # center_count = center_array.count(player_number)
        # score += center_count

        # score in rows
        for i in range(m):
            score += get_row_score(player_number, board[i])
        # score in columns
        for j in range(n):
            # if pop!=0:
            #     if player_number==1:
            #         if j % 2 == 0:
            #             score += get_row_score(player_number, board[:, j])+ 7
            #         else:
            #             score += get_row_score(player_number, board[:, j])+ 5
            #     else:
            #         if j % 2 == 1:
            #             score += get_row_score(player_number, board[:, j])+ 7
            #         else:
            #             score += get_row_score(player_number, board[:, j])+ 5
            # else:
            center_advantage = [int(i) for i in list(board[:, j])].count(player_number)
            score += 4 * center_advantage * j * (n - j) / (n * n)
            score+=get_row_score(player_number, board[:, j])
        # scores in diagonals_primary
        for diag in get_diagonals_primary(board):
            score += get_row_score(player_number, diag)
        # scores in diagonals_secondary
        for diag in get_diagonals_secondary(board):
            score += get_row_score(player_number, diag)
        return score

    def apply(self,state,action,curr_player_number):
        if action==None:
            return state
        if (action[1]):
            col=action[0]
            rows=state[0].shape[0]
            newboard=state[0].copy()
            for i in range(rows-1,0,-1):
                newboard[i][col]=newboard[i-1][col]
            newboard[0][col]= 0
            dict={}
            i1=Integer(state[1][3-curr_player_number].get_int())
            dict[3 - curr_player_number]=i1
            i2=Integer(state[1][curr_player_number].get_int()-1)
            dict[curr_player_number]=i2
            return (newboard,dict)
        else:
            col=action[0]
            rows=state[0].shape[0]
            newboard=state[0].copy()
            for i in range(rows-1,-1,-1):
                if (newboard[i][col]==0):
                    newboard[i][col]=curr_player_number
                    break
            return (newboard,state[1])

    def alphabeta(self, state, depth, s, a ,b):
        # str_of_state = str(state) + str(depth)
        # print(hash(str_of_state))
        # if str_of_state in self.transposition_table:
            # print(f'Found a copy of state')
            # return self.transposition_table[str_of_state]
        if time.time() - self.start > self.time - 0.7:
            return -math.inf, None

        actions= get_valid_actions(self.player_number if s==1 else 3 - self.player_number, state)
        if depth==0 or len(actions)==0:
            return self.evaluation(state,0), None
        if (s==1):
            maxpoints= -math.inf
        else:
            maxpoints= math.inf
        #maxpoints=get_pts(self.player_number,state[0])-get_pts(3-self.player_number,state[0])
        act= None

        for action in actions:
            newstate=self.apply(state ,action,self.player_number if s==1 else 3 - self.player_number)
            points, news=self.alphabeta(newstate,depth-1,1-s,a,b)
            if points == -math.inf:
                return points, news
            # if (get_pts(3-self.player_number,newstate[0])+25<get_pts(3-self.player_number,state[0])):
            #     if (s==1):
            #         points+=10
            # if (s==1):
            #     points+= action[0] * (state[0].shape[1] - action[0])/2
            # print(f'extra score : {j * (n - j)}')
            # points=get_pts(self.player_number,newstate[0])-get_pts(3-self.player_number,newstate[0])
            if points>maxpoints and s==1:
                maxpoints=points
                if (maxpoints>a):
                    if (maxpoints>=b):
                        return maxpoints, act
                    a=maxpoints
                act=action
            if points<maxpoints and s==0:
                maxpoints=points
                if (maxpoints<b):
                    if (maxpoints<a):
                        return a,act
                    b=maxpoints
                act=action
        # self.transposition_table[str_of_state] = maxpoints, act
        return maxpoints, act

    def minimax(self, state, depth, s):
        actions=get_valid_actions(self.player_number, state)
        if (s==1):
            maxpoints= -math.inf
        else:
            maxpoints= math.inf
        #maxpoints=get_pts(self.player_number,state[0])-get_pts(3-self.player_number,state[0])
        act=None
        if len(actions)==0 or depth==0:
            return self.evaluation(state,0), act
        for action in actions:
            newstate=self.apply(state ,action,self.player_number)
            points,news=self.minimax(newstate,depth-1,1-s)
            #if (action[1]):
                # if (get_pts(3-self.player_number,newstate[0])+15<get_pts(3-self.player_number,state[0])):
                #     if (s==1):
                #         points+=5
                #     else:
                #         points-=5
            #points=get_pts(self.player_number,newstate[0])-get_pts(3-self.player_number,newstate[0])
            if points>maxpoints and s==1:
                maxpoints=points
                act=action
            if points<maxpoints and s==0:
                maxpoints=points
                act=action
        return maxpoints,act

    def numberemptycell(self,board):
        col=board.shape[1]
        rows=board.shape[0]
        number=0
        for row in board:
            for cell in row:
                if cell==0:
                    number+=1
        return number/(rows*col)

    def evaluation(self,state,depth):
        #print(state)
        # score=(self.score(self.player_number,state[0],state[1][3-self.player_number].get_int())-1.5*get_pts(3-self.player_number,state[0]))
        # return score

        # x = fraction of filled cells
        board = state[0]
        rows, cols =board.shape
        number=0
        for row in board:
            for cell in row:
                if cell==0:
                    number+=1
        x = 1 - number/(rows*cols)
        # x = 1 - self.numberemptycell(state[0])


        pts1 = get_pts(self.player_number, state[0])
        pts2 = get_pts(3-self.player_number, state[0])
        pops1 = state[1][self.player_number].get_int()
        pops2 = state[1][3-self.player_number].get_int()        
        def f1(x):
            if x <= 1/4 and x >= 0:
                return 1
            return 0
        def f2(x):
            if x <= 3/4 and x > 1/4:
                return 1
            return 0
        def f3(x):
            if x > 3/4: 
                return 1
            return 0

        return f1(x)*(pops1+pts1 - 7*pts2) + f2(x)*(pts1 - 5*pts2) + f3(x)*(pts1 - 3*pts2)

    def evaluationexpecti(self,state,depth):
        #print(state)
        score=(get_pts(self.player_number,state[0])-get_pts(3-self.player_number,state[0]))
        return score

    def maxval(self, state, depth):
        actions= get_valid_actions(self.player_number, state)
        #maxpoints=self.evaluation(state,depth)
        maxpoints= -math.inf
        final_action_to_take= None
        if len(actions)==0 or depth==0:
            return self.evaluationexpecti(state,depth), final_action_to_take
        for action in actions:
            # points= 0
            newstate=self.apply(state ,action, self.player_number)
            points= self.randmove(newstate,depth-1)
            #print(points)
            if points> maxpoints:
                maxpoints= points
                final_action_to_take= action
            self.iter+=1
        return maxpoints, final_action_to_take
        
    def randmove(self, state,depth):
        actions=get_valid_actions(3-self.player_number, state)
        exppoints=0
        # act= None
        if len(actions)==0 or depth==0:
            return self.evaluationexpecti(state,depth)
        for action in actions:
            newstate=self.apply(state ,action, 3-self.player_number)
            points, act= self.maxval(newstate,depth-1)
            #print(points)
            exppoints+=points
            self.iter+=1
        exppoints/=len(actions)
        return exppoints


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
        #point,move=self.minimax(state,4,1)
        # Ai2=AIPlayer2(3-self.player_number,self.time-0.1)

        self.start= time.time()

        bestmove=None
        print(f'player is {self.player_number}')
        if self.player_number==1:
            # point,move=self.alphabeta(state,int(max(1,6*(1-self.numberemptycell(state[0])))),1,-10000,10000)
            # point, bestmove=self.alphabeta(state,1,1,-10000,10000)

            for depth in range(1,5000):
                # point,move=self.alphabeta(state,int(max(1,6*(1-self.numberemptycell(state[0])))),1,-10000,10000)
                point, move = self.alphabeta(state, depth,1, -math.inf,math.inf)
                # print(depth)
                if point != -math.inf:
                    bestmove= move
                    # print(f'Depth reached {depth-1}, time taken {time.time() - self.start}')
                else:
                    print(f'Depth reached {depth-1}, time taken {time.time() - self.start}')
                    return bestmove

            # #return Ai2.get_expectimax_move(state)
            # point,move=self.maxval(state,4)
            # #return Ai2.get_intelligent_move(state)
            # return move
        else:
            #Ai2=AIPlayer2(self.player_number,self.time - 0.5)
            point,move=self.maxval(state,4)
            #return Ai2.get_intelligent_move(state)
            return move
        # #print(point)
        print(f'Depth till end, time taken {time.time() - self.start}')
        return bestmove

        # Do the rest of your implementation here
        raise NotImplementedError('Whoops I don\'t know what to do')

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
        
        point,move=self.maxval(state,4)
        # print(point)
        # print(self.iter)
        return move

        raise NotImplementedError('Whoops I don\'t know what to do')

class AIPlayer2:
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
            # move_scores[index] = self.get_minimax_score(new_state,3-player_number, max_depth-1, step + 1, limit//number_of_moves)
        # for col in range(columns) :
        #     new_state_push = self.state_update(state, col, 0, player_number)
        #     new_state_pop = self.state_update(state, col, 1, player_number)
        #     if new_state_push != -1:
        #         move_scores["push{}".format(col)] = self.get_minimax_score(new_state_push,3-player_number, max_depth-1,step + 1)
        #     if new_state_pop != -1:
        #         move_scores["pop{}".format(col)] = self.get_minimax_score(new_state_pop,3-player_number, max_depth-1,step + 1)
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

    # def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
    #     """
    #     Given the current state of the board, return the next move
    #     This will play against either itself or a human player
    #     :param state: Contains:
    #                     1. board
    #                         - a numpy array containing the state of the board using the following encoding:
    #                         - the board maintains its same two dimensions
    #                             - row 0 is the top of the board and so is the last row filled
    #                         - spaces that are unoccupied are marked as 0
    #                         - spaces that are occupied by player 1 have a 1 in them
    #                         - spaces that are occupied by player 2 have a 2 in them
    #                     2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
    #     :return: action (0 based index of the column and if it is a popout move)
    #     """
    #     self.columns = len(state[0][0])
    #     self.rows = len(state[0])
    #     self.state_count_intelligent = 0
    #     columns = len(state[0][0])
    #     best_move = -1
    #     isPop = 0
    #     maax = -1e9
    #     # for depth in range(10):
    #     #     for col in range(columns) :
    #     #         new_state_push = self.state_update(state, col, 0, self.player_number)
    #     #         new_state_pop = self.state_update(state, col, 1, self.player_number)
    #     #         if new_state_push != -1:
    #     #             val = self.get_expectimax_score(new_state_push, 3-self.player_number, depth)
    #     #             if val > maax :
    #     #                 maax = val
    #     #                 move = col
    #     #                 isPop = 0 
    #     #         if new_state_pop != -1:
    #     #             val = self.get_expectimax_score(new_state_pop, 3-self.player_number, depth)
    #     #             if val > maax :
    #     #                 maax = val
    #     #                 move = col
    #     #                 isPop = 1
    #     #     move = -1
    #     #     isPop = 0
    #     #     maax = -1e9
    #     # for col in range(columns) :
    #     #     new_state_push = self.state_update(state, col, 0, self.player_number)
    #     #     new_state_pop = self.state_update(state, col, 1, self.player_number)
    #     #     if new_state_push != -1:
    #     #         val = self.get_minimax_score(new_state_push, 3-self.player_number, 100, 1, 1000)
    #     #         print(val)
    #     #         if val > maax :
    #     #             maax = val
    #     #             move = col
    #     #             isPop = 0 
    #     #     if new_state_pop != -1:
    #     #         val = self.get_minimax_score(new_state_pop, 3-self.player_number, 100, 1, 1000)
    #     #         print(val)
    #     #         if val > maax :
    #     #             maax = val
    #     #             move = col
    #     #             isPop = 1
    #     move_scores = {}
    #     l = get_valid_actions(self.player_number, state)

    #     if not l : 
    #         return (-1, 0)
        
    #     # shuffle l 
    #     random.shuffle(l)
    #     index = 0
    #     minimum = 1e9
    #     sum = 0

    #     for move in l :
    #         new_state = self.state_update(state, move[0], move[1], self.player_number)
    #         move_scores[index] = self.evaluate_minimax(new_state, 1)
    #         minimum = min(minimum, move_scores[index])
    #         index += 1
    #     for move in move_scores:
    #         sum += max(5, move_scores[move]-minimum)
    #     index = 0
    #     for move in l:
    #         new_state = self.state_update(state, move[0], move[1], self.player_number)
    #         move_scores[index] = self.get_minimax_score(new_state,3-self.player_number, 100, 1, (10000*max(5, move_scores[index]-minimum))//sum)
    #         print("Column", move[0], end = ' ')
    #         if move[1]:
    #             print("with Pop has score", move_scores[index])
    #         else :
    #             print("without Pop has score", move_scores[index])
    #         if move_scores[index] > maax :
    #             maax = move_scores[index]
    #             best_move = move[0]
    #             isPop = move[1]
    #         index += 1

    #     print("The chosen move has score", maax)
    #     print("Nodes at each depth")
    #     print(self.noOf_intelligent_nodes)
    #     print("States explored", self.state_count_intelligent)
    #     print((best_move, isPop))
    #     if isPop: 
    #         self.moveCount -= 1
    #     else:
    #         self.moveCount += 1
    #     return (best_move, isPop)
    #     raise NotImplementedError('Whoops I don\'t know what to do')
    def get_minimax_score_ab(self, state, player_number, max_depth, step, alpha, beta):

        if time.time() - self.start_time > self.time-0.2:
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

        for depth in range(1, 200):
            (a, b) = self.iterative_deepening_search(state, depth)
            if a+1:
                (best_move, isPop) = (a, b)
            else :
                print("Upto depth", depth - 1, "best column is", best_move, "isPop is", isPop)
                break
        # print(best_move, isPop, "time taken", time.time()-self.start_time)
        # h2   = ("170.20.10.5", 20010)
        # a2   = ("170.20.10.2", 20012)
        # h1   = ("170.20.10.2", 20014)
        # a1   = ("170.20.10.5", 20016)
        
        # if self.player_number == 2:
        #     me,other = a2,h2
        
        # else:
        #     me,other = a1,h1
        # UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # UDPClientSocket.bind(me)
        # msg = f"{best_move}{'P' if isPop == True else ''}"
        # bytesToSend   = str.encode(msg)
        # UDPClientSocket.sendto(bytesToSend, other)
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

        return f1(x)*(pops1+pts1 - 7*pts2) + f2(x)*(pts1 - 5*pts2) + f3(x)*(pts1 - 3*pts2)

        # pop_diff = pops1 - pops2
        # pop_factor = pts1/max(1, state[1][self.player_number].get_int())-pts2/max(1, state[1][3-self.player_number].get_int())
        # pop_factor = 1/max(1, )
        # pop_factor = 
        # w = weight(step, cells)
        # return 2.25*(pts1*pts1) - (w+15+2*step%2)*(pts2*pts2) + (pts1*pts1/(1 + pops1) - pts2*pts2/(1 + pops2))*2*w
        # return (pts1*(1+opp_cnt) - 3*pts2*(1+self_cnt))/(step**2)
        # return pts1 - 10*pts2

    def get_expectimax_score(self, state, player_number : int, max_depth : int, step) -> int :

        if time.time() - self.start_time > self.time-0.2:
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

        for depth in range(1, 200):
            (a, b) = self.iterative_deepening_search_expectimax(state, depth)
            if a+1:
                (best_move, isPop) = (a, b)
                print("Upto depth", depth, "best column is", best_move, "isPop is", isPop)
            else :
                break
        # print(best_move, isPop, "time taken", time.time()-self.start_time)
        return (best_move, isPop)
    
    def evaluate_expectimax(self, state, step) :
        pts1 = get_pts(self.player_number, state[0])
        pts2 = get_pts(3-self.player_number, state[0])
        pops1 = state[1][self.player_number].get_int()
        pops2 = state[1][3-self.player_number].get_int()
        cells = self.rows * self.columns 
        def weight(x, n):
            return 1/(1 + np.exp(-(x-n/4)))
        
        # pop_diff = pops1 - pops2
        # pop_factor = pts1/max(1, state[1][self.player_number].get_int())-pts2/max(1, state[1][3-self.player_number].get_int())
        # pop_factor = 1/max(1, )
        # pop_factor = 
        w = weight(step, cells)
        # return 2.25*(pts1) - (w+2+2*step%2)*(pts2) + (pts1/(1 + pops1) - pts2/(1 + pops2))*2*w
        return pts1 - 3 * pts2