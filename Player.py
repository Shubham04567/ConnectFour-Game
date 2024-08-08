import numpy as np
import math
import copy


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opposite_player_number =2 if player_number == 1 else 1
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
       
    #Checking the winning condition for given player in given state
    def terminalstate(self,board,player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int32)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int32))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    # Maximize function for alpha beta algorithm which maximize the score of max player
    def maximize(self,board,alpha,beta,depth):
        
        if self.terminalstate(board,self.player_number):
            return (100789562, None)
        elif depth==0:
            return (self.evaluation_function(board),None)
        
        v = -1009766890
        valid_pos=[]
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)
                for j in range(5,-1,-1):
                    if col[j]==0:
                        valid_pos.append([j,i])
                        break
        
        move=None
        lst = []
        for child_node in valid_pos:
            x, y = child_node
            new_board=copy.deepcopy(board)
            new_board[x][y]=self.player_number
            a=v
            val = self.minimize(new_board, alpha, beta, depth-1)[0]
            v=max(v, val )
            lst.append((val,child_node))
            if a!=v:                                                    #update the move by best move
                move=child_node[1]
            if v >= beta:                                               #applying alpha pruning 
                return (v,move)
            alpha = max(alpha, v)
        return (v,move)
        
    #Minimize function to minimize the score by min player
    def minimize(self,board,alpha,beta,depth):
        
        if self.terminalstate(board,self.opposite_player_number):
            return (-1009766890, None)
        if depth==0:
            return (self.evaluation_function(board),None)
        
        v = 100789562
        valid_pos=[]
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)
                for j in range(5,-1,-1):
                    if col[j]==0:
                        valid_pos.append([j,i])
                        break
                    
        move = None
        lst = []
        for child_node in valid_pos:
            new_board=copy.deepcopy(board)
            x, y = child_node
            new_board[x][y]=self.opposite_player_number
            a = v
            val = self.maximize(new_board, alpha, beta, depth-1)[0]
            v=min(val,v )
            # if a != beta:
            #     move.append(child_node[1])
            lst.append((val,child_node))
            if v <= alpha:                                                  #applying condition for beta pruning
                return (v,move)
            beta = min(beta, v)
        return (v,None)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha=-10000
        beta=10000
        depth=5                                                    
        result=self.maximize(board, alpha, beta, depth)
        return result[1]
    
    
    ## Expectimaximize function to maximize the value
    def expecti_maximize(self,board,alpha,beta,depth):
        
        if self.terminalstate(board,self.player_number):
            return (100789562, None)
        elif depth==0:
            return (self.evaluation_function(board),None)
        
        v = -1009766890
        valid_pos=[]
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)
                for j in range(5,-1,-1):
                    if col[j]==0:
                        valid_pos.append([j,i])
                        break
        move=None
        lst = []
        for child_node in valid_pos:
            x, y = child_node
            new_board=copy.deepcopy(board)
            new_board[x][y]=self.player_number
            a=v
            val = self.expecti_minimize(new_board, alpha, beta, depth-1)[0]
            v=max(v, val )
            lst.append((val,child_node))
            
            if a!=v:                                                                       #update the move by best move
                move=child_node[1]
            
            alpha = max(alpha, v)                                                          #maximixe the value

        return (v,move)
    
    def expecti_minimize(self,board,alpha,beta,depth):
        
        if self.terminalstate(board,self.opposite_player_number):
            return (-1009766890, None)
        elif depth==0:
            return (self.evaluation_function(board),None)
        
        v = 100789562
        valid_pos=[]
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)
                for j in range(5,-1,-1):
                    if col[j]==0:
                        valid_pos.append([j,i])
                        break
                    
        move = None
        lst = []
        for child_node in valid_pos:
            new_board=copy.deepcopy(board)
            x, y = child_node
            new_board[x][y]=self.opposite_player_number
            val = self.expecti_maximize(new_board, alpha, beta, depth-1)[0]
            lst.append((val,child_node))
            # beta = min(beta, v)
        s=0
        for i in range(len(lst)):
            s = s + (lst[i][0])/(len(lst))
        
        return (s,None)  

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        # raise NotImplementedError('Whoops I don\'t know what to do')
        alpha=-1009766890
        beta=100789562
        depth=4
        result=self.expecti_maximize(board, alpha, beta, depth)
       
        return result[1]

    def evaluation_function(self,board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the  row filled
                - spaces that lastare unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        score=0
        
        #checking for horizontal winnig chance of ai player
        for row in range(board.shape[0]):
            for col in range(4):
                if board[row][col]==board[row][col+1]==board[row][col+2]==board[row][col+3]==self.player_number:
                    score+=50
                    
        for row in range(board.shape[0]):
            for col in range(5):
                if board[row][col]==board[row][col+1]==board[row][col+2]==self.player_number:
                    score+=25
        
        for row in range(board.shape[0]):
            for col in range(6):
                if board[row][col]==board[row][col+1]==self.player_number:
                    score+=10
                    
        #checking for vertical winnig chance of ai player
        for col in range(board.shape[1]):
            for row in range(3):
                if board[row][col]==board[row+1][col]==board[row+2][col]==board[row+3][col]==self.player_number:
                    score+=50
                    
        for col in range(board.shape[1]):
            for row in range(4):
                if board[row][col]==board[row+1][col]==board[row+2][col]==self.player_number:
                    score+=25
        
        for col in range(board.shape[1]):
            for row in range(5):
                if board[row][col]==board[row+1][col]==self.player_number:
                    score+=10
                
                
        #pattern for blocking the opposition win
        pattern_a = ['{0}{0}{0}{1}'.format(self.opposite_player_number,self.player_number), '{1}{0}{0}{0}'.format(self.opposite_player_number,self.player_number), '{0}{0}{1}{0}'.format(self.opposite_player_number,self.player_number), '{0}{1}{0}{0}'.format(self.opposite_player_number,self.player_number)]
        to_str = lambda a: ''.join(a.astype(str))
        
        for row in board:
            for pattern in pattern_a:
                if pattern in to_str(row):
                    score+=75

        for row in board.T:
            for pattern in pattern_a:
                if pattern in to_str(row):
                    ############
                    score+=75
        # print("ksd",score)                                 ##############
        pattern_b = ['{0}{1}{0}'.format(self.opposite_player_number,self.player_number),'{0}{0}{1}'.format(self.opposite_player_number,self.player_number), '{1}{0}{0}'.format(self.opposite_player_number,self.player_number)]
        for row in board:
            for pattern in pattern_b:
                if pattern in to_str(row):
                    score+=25
        
        for row in board.T:
            for pattern in pattern_b:
                if pattern in to_str(row):
                    score+=25
                    
        pattern_c = '{1}{0}{0}{0}{1}'.format(self.opposite_player_number,self.player_number)
        for row in board:
            if pattern_c in to_str(row):
                score+=80
                
        for row in board.T:
            if pattern_c in to_str(row):
                score+=80  
                
                
        #decreasing the evaluation function by increasing the human score
        
        #checking for horizontal chance of winning of human player
        for row in range(board.shape[0]):
            for col in range(4):
                if board[row][col]==board[row][col+1]==board[row][col+2]==board[row][col+3]==self.opposite_player_number:
                    score-=400
                    
        for col in range(board.shape[1]):
            for row in range(3):
                if board[row][col]==board[row+1][col]==board[row+2][col]==board[row+3][col]==self.opposite_player_number:
                    score-=400
                    
        #checking the condition for verticle chance of winning of human player
        for row in range(board.shape[0]):
            for col in range(5):
                if board[row][col]==board[row][col+1]==board[row][col+2]==self.opposite_player_number:
                    score-=150
        
        for col in range(board.shape[1]):
            for row in range(4):
                if board[row][col]==board[row+1][col]==board[row+2][col]==self.opposite_player_number:
                    score-=150
                    
        #checking for terminal condition of winning of ai player
        pattern_d = '{0}{0}{0}{0}'.format(self.player_number)         
        #                     score+=200
        score+=self.check_diagonal(board,pattern_d,200)
             
        #pattern for checking opposite player to win at terminal condition               
        p='{0}{0}{0}{0}'.format(self.opposite_player_number)
        #                     score-=500
        score+=self.check_diagonal(board,p, -500)
                          
        #pattern favourable to winning of opposite player  
        pattern_e = ['{0}{0}{0}0'.format(self.opposite_player_number),'0{0}{0}{0}'.format(self.opposite_player_number),'{0}{0}0{0}'.format(self.opposite_player_number),'{0}0{0}{0}'.format(self.opposite_player_number)]       
        #                         score-=400
        score+=self.check_diagonal_many(board,pattern_e, -400)
                            
        pattern_f = ['{0}{0}{0}0'.format(self.player_number) , '0{0}{0}{0}'.format(self.player_number), '{0}0{0}{0}'.format(self.player_number), '{0}{0}0{0}'.format(self.player_number)]       
    
        #                         score+=75
        score+=self.check_diagonal_many(board,pattern_f, 75)
                                
        pattern_g = ['{0}{0}{0}{1}'.format(self.opposite_player_number,self.player_number), '{1}{0}{0}{0}'.format(self.opposite_player_number,self.player_number), '{0}{0}{1}{0}'.format(self.opposite_player_number,self.player_number), '{0}{1}{0}{0}'.format(self.opposite_player_number,self.player_number)]
                  
        #                         score+=200
        score+=self.check_diagonal_many(board,pattern_g, 200)
                    
        return score
    
    def check_diagonal(self, board, pattern, point):
        score=0
        to_str = lambda a: ''.join(a.astype(str))
        # pattern_d = '{0}{0}{0}{0}'.format(self.player_number)         
        for op in [None, np.fliplr]:
                op_board = op(board) if op else board
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int32)
                if pattern in to_str(root_diag):
                    score+=point

                for i in range(1,3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int32))
                        if pattern in diag:
                            score+=point
        return score

    def check_diagonal_many(self, board, pattern, point):
        score=0
        to_str = lambda a: ''.join(a.astype(str))
        # pattern_d = '{0}{0}{0}{0}'.format(self.player_number)         
        for op in [None, np.fliplr]:
                op_board = op(board) if op else board
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int32)
                for patt in pattern:
                    if patt in to_str(root_diag):
                        score+=point

                for i in range(1,3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int32))
                        for patt in pattern:
                            if patt in diag:
                                score+=point
        return score

class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_pos=[]
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)
    
        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))                     
            move = int(input('Enter your move: '))

        return move

