import math
import random
import copy
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer, get_row_score, get_diagonals_primary, get_diagonals_secondary
import time

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
        # Do the rest of your implementation here

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

    def apply(self,state,action,number):
        if action==None:
            return state
        if (action[1]):
            col=action[0]
            rows=state[0].shape[0]
            newboard=state[0].copy()
            for i in range(rows-1,0,-1):
                newboard[i][col]=newboard[i-1][col]
            newboard[0][col]=0
            dict={}
            i1=Integer(state[1][3-self.player_number].get_int())
            dict[3-self.player_number]=i1
            i2=Integer(state[1][self.player_number].get_int()-1)
            dict[self.player_number]=i2
            return (newboard,dict)
        else:
            col=action[0]
            rows=state[0].shape[0]
            newboard=state[0].copy()
            for i in range(rows-1,-1,-1):
                if (newboard[i][col]==0):
                    newboard[i][col]=number
                    break
            return (newboard,state[1])

    def alphabeta(self, state, depth, s, a ,b):
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
            if time.time() - self.start > self.time - 0.5:
                return -math.inf, None
            newstate=self.apply(state ,action,self.player_number)
            points,news=self.alphabeta(newstate,depth-1,1-s,a,b)
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
        return maxpoints,act

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
            if (action[1]):
                if (get_pts(3-self.player_number,newstate[0])+15<get_pts(3-self.player_number,state[0])):
                    if (s==1):
                        points+=5
                    else:
                        points-=5
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
        score=(self.score(self.player_number,state[0],state[1][3-self.player_number].get_int())-1.5*get_pts(3-self.player_number,state[0]))
        return score

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

        self.start= time.time()

        bestmove= None
        print(f'player is{self.player_number}')

        for depth in range(1,15):
            # point,move=self.alphabeta(state,int(max(1,6*(1-self.numberemptycell(state[0])))),1,-10000,10000)
            point, move = self.alphabeta(state, depth,1, -math.inf,math.inf)
            # print(depth)
            if move != None:
                bestmove= move
            else:
                print(f'Depth reached {depth-1}, time taken {time.time() - self.start}')
                return bestmove
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