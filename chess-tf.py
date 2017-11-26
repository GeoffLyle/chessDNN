import tensorflow as tf
import numpy
import chess

class baseBoard():
    
    def __init__(self):
        self.boardState = chess.Board()
        self.moveStack = []
        self.legalMoveList = [] 
        self.legalMoveList = self.getLegalMoveList()
        self.cunt = 1
    def getLegalMoveList(self):
        a = list(enumerate(self.boardState.legal_moves))
        b = [x[1] for x in a]
        c = []
        i = 0
        for item in b:
            c.append(str(b[i]))
            i += 1
        return c
board = baseBoard()
