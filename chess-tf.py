import tensorflow as tf
import numpy
import chess

#base class for interacting with board
#TODO add tree searching
class baseBoard(chess.Board):
    
    def __init__(self):
        self.baseBoard = chess.Board()
        self.parsedBoard = self.baseBoard
        self.moveStack = []
        self.trainingData = []
        self.legalMoveList = self.getLegalMoveList()
        
    def getLegalMoveList(self):
        a = list(enumerate(self.baseBoard.legal_moves))
        b = [x[1] for x in a]
        c = []
        i = 0
        for item in b:
            c.append(str(b[i]))
            i += 1
        return c

    def parseMove(self,fromSquare,toSquare = None):
        frT = type(fromSquare)
        toT = type(toSquare)
        if(toSquare == None and frT == str):
            move = fromSquare
        elif(frT == str and toT == str):
            move = fromSquare + toSquare
        else:
            return "Invalid Move Representation"
        return move
    
    def playMove(self,move):
        if(move in self.legalMoveList):
            self.baseBoard.push_uci(move)
        elif(move + 'q' in self.legalMoveList):
            self.baseBoard.push_uci(move + 'q')
        else:
            return 'illegal move'
        
        self.updateAttributes()
        if(self.baseBoard.is_game_over()):
            return self.baseBoard.result()       

        return self.parsedBoard

    def addTrainingData(self,move):
        a = [self.parsedBoard,move]
        if(self.playMove(move) != 'illegal move'):
            self.trainingData.append(a)
            return self.parsedBoard
        else:
            return 'illegal move'
        
        

    def reset(self):
        self.baseBoard.reset()
        self.updateAttributes()
        return self.parsedBoard

    def updateAttributes(self):
        self.legalMoveList = self.getLegalMoveList()
        self.parsedBoard = self.baseBoard
        
board = baseBoard()

