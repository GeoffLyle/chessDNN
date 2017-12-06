import tensorflow as tf
import numpy as np
import chess

# base class for interacting with board
# TODO add tree searching
# TODO create default board parsing


class baseBoard(chess.Board):

    def __init__(self):
        self.baseBoard = chess.Board()
        self.parsedBoard = self.parseBoard()
        self.moveStack = []
        self.trainingData = []
        self.legalMoveList = self.getLegalMoveList()

    def parseBoard(self):
        self.parsedBoard = self.baseBoard

    def getLegalMoveList(self):
        a = list(enumerate(self.baseBoard.legal_moves))
        b = [x[1] for x in a]
        c = []
        i = 0
        for item in b:
            c.append(str(b[i]))
            i += 1
        return c

    def parseMove(self, fromSquare, toSquare=None):
        frT = type(fromSquare)
        toT = type(toSquare)
        if(toSquare is None and frT == str):
            move = fromSquare
        elif(frT == str and toT == str):
            move = fromSquare + toSquare
        else:
            return "Invalid Move Representation"
        return move

    def playMove(self, move):
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

    def addTrainingData(self, move):
        a = [self.parsedBoard, move]
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
        self.parseBoard()

# TODO add options to change parsing


class parsedBoard(baseBoard):

    def parseBoard(self):
        b0 = np.zeros((12, 8, 8))
        i = 0
        while(i < 8):
            j = 0
            while(j < 8):
                k = str(self.baseBoard.piece_at(i * 8 + j))
                if(self.baseBoard.turn):
                    if(k == "P"):
                        b0[0][i][j] = 1
                    elif(k == "B"):
                        b0[1][i][j] = 1
                    elif(k == "N"):
                        b0[2][i][j] = 1
                    elif(k == "R"):
                        b0[3][i][j] = 1
                    elif(k == "Q"):
                        b0[4][i][j] = 1
                    elif(k == "K"):
                        b0[5][i][j] = 1
                    elif(k == "p"):
                        b0[6][i][j] = 1
                    elif(k == "b"):
                        b0[7][i][j] = 1
                    elif(k == "n"):
                        b0[8][i][j] = 1
                    elif(k == "r"):
                        b0[9][i][j] = 1
                    elif(k == "q"):
                        b0[10][i][j] = 1
                    elif(k == "k"):
                        b0[11][i][j] = 1
                else:
                    if(k == "P"):
                        b0[6][i][j] = 1
                    elif(k == "B"):
                        b0[7][i][j] = 1
                    elif(k == "N"):
                        b0[8][i][j] = 1
                    elif(k == "R"):
                        b0[9][i][j] = 1
                    elif(k == "Q"):
                        b0[10][i][j] = 1
                    elif(k == "K"):
                        b0[11][i][j] = 1
                    elif(k == "p"):
                        b0[0][i][j] = 1
                    elif(k == "b"):
                        b0[1][i][j] = 1
                    elif(k == "n"):
                        b0[2][i][j] = 1
                    elif(k == "r"):
                        b0[3][i][j] = 1
                    elif(k == "q"):
                        b0[4][i][j] = 1
                    elif(k == "k"):
                        b0[5][i][j] = 1
                j = j + 1
            i = i + 1
        return b0


board = parsedBoard()
