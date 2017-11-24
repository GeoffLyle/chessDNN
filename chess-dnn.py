import chess
import numpy as np
import time
import tensorflow as tf
import pickle
import random
from random import shuffle

board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")


print(board)

def parseBoard(isWhite):
    b0 = np.zeros((12,8,8))
    i = 0
    while(i<8):
        j = 0
        while(j<8):
            k = str(board.piece_at(i*8+j))
            if(isWhite==True):
                if(k=="P"):
                    b0[0][i][j] = 1
                elif(k=="B"):
                    b0[1][i][j] = 1
                elif(k=="N"):
                    b0[2][i][j] = 1
                elif(k=="R"):
                    b0[3][i][j] = 1
                elif(k=="Q"):
                    b0[4][i][j] = 1
                elif(k=="K"):
                    b0[5][i][j] = 1
                elif(k=="p"):
                    b0[6][i][j] = 1
                elif(k=="b"):
                    b0[7][i][j] = 1
                elif(k=="n"):
                    b0[8][i][j] = 1
                elif(k=="r"):
                    b0[9][i][j] = 1
                elif(k=="q"):
                    b0[10][i][j] = 1
                elif(k=="k"):
                    b0[11][i][j] = 1
            else:
                if(k=="P"):
                    b0[6][i][j] = 1
                elif(k=="B"):
                    b0[7][i][j] = 1
                elif(k=="N"):
                    b0[8][i][j] = 1
                elif(k=="R"):
                    b0[9][i][j] = 1
                elif(k=="Q"):
                    b0[10][i][j] = 1
                elif(k=="K"):
                    b0[11][i][j] = 1
                elif(k=="p"):
                    b0[0][i][j] = 1
                elif(k=="b"):
                    b0[1][i][j] = 1
                elif(k=="n"):
                    b0[2][i][j] = 1
                elif(k=="r"):
                    b0[3][i][j] = 1
                elif(k=="q"):
                    b0[4][i][j] = 1
                elif(k=="k"):
                    b0[5][i][j] = 1
            j = j + 1
        i = i +1
    return b0


def getBoard():
    a = parseBoard(board.turn)
    a = a.flatten()
    return a

def getLegalMoveList():
    a = list(enumerate(board.legal_moves))
    b = [x[1] for x in a]
    c = []
    i = 0
    for item in b:
        c.append(str(b[i]))
        i += 1
    return c
'''
functional programming you retard
rewrite this as playmove, parse, save, pop, return
that way it's a finite equation and can't fuck up something else
'''
    
def playLegalMove(iteration):
    a = getLegalMoveList()
    b = len(a)
    if(iteration >= b):
        return False
    board.push_uci(a[iteration])
    return True
    
def parseMove(iteration):
    a = playLegalMove(iteration)
    if(a == False):
        return False
    b = getBoard()
    board.pop()
    return b



def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def initWeights(shape):
    w = []
    shapeLen = len(shape)
    i = 1
    while(i < shapeLen):
        w.append(np.random.randn(shape[i-1], shape[i]))
        i += 1
    return w

def forwardPass(data, weights):
    net = data
    net = np.dot(net, weights[0])
    net = sigmoid(net)
    net = np.dot(net, weights[1])
    net = sigmoid(net)
    net = np.dot(net, weights[2])
    net = sigmoid(net)
    net = np.dot(net, weights[3])
    net = sigmoid(net)
    net = net[0] - net [1]
    
    return net

def layerOutputPass(data, weights):
    out = []
    h1 = np.dot(data, weights[0])
    h1a = sigmoid(h1)
    h2 = np.dot(h1a, weights[1])
    h2a = sigmoid(h2)
    h3 = np.dot(h2a, weights[2])
    h3a = sigmoid(h3)
    ol = np.dot(h3a, weights[3])
    ola = sigmoid(ol)
    #out.append(h1)
    out.append(h1a)
    #out.append(h2)
    out.append(h2a)
    #out.append(h3)
    out.append(h3a)
    #out.append(ol)
    out.append(ola)

    return out



def derivative(output):
    return output * (1 - output)

def calcOError(target,output):
    return (target - output) * derivative(output)

def calcNError(errorSignal,weight,output):
    return (weight*errorSignal) * derivative(output)

def backPropError(data,w,target):
    outs = layerOutputPass(data,w)
    errors = []
    nodes = list(reversed(outs))
    weights = list(reversed(w))
    i = 0
    err = []
    while(i<2):
        calc = (calcOError(target[i],nodes[0][i]))
        err.append(calc)
        i+=1
    errors.append(err)
    i = 0
    err = []
    while(i<20):
        total = 0
        j = 0
        while(j<2):
            a = calcNError(errors[0][j],weights[0][i][j],nodes[1][i])
            total += a
            j+=1
        err.append(a)
        i+=1
    errors.append(err)

    i = 0
    err = []
    while(i<100):
        total = 0
        j = 0
        while(j<20):
            a = calcNError(errors[1][j],weights[1][i][j],nodes[2][i])
            total += a
            j+=1
        err.append(a)
        i+=1
    errors.append(err)

    i = 0
    err = []
    while(i<200):
        total = 0
        j = 0
        while(j<100):
            a = calcNError(errors[2][j],weights[2][i][j],nodes[3][i])
            total += a
            j+=1
        err.append(a)
        i+=1
    errors.append(err)

    return errors

def calcNewWeight(weight,error,inValue):
    a = 0.1 * error * inValue
    return weight + a

def updateWeights(error,w,data):
    outs = layerOutputPass(data,w)
    nodes = list(reversed(outs))
    weights = list(reversed(w))
    newWeights = weights

    i = 0
    while(i<2):
        j = 0
        while(j<20):
            a = weights[0][j][i]
            b = error[0][i]
            c = nodes[1][j]
            nw = calcNewWeight(a,b,c)
            newWeights[0][j][i] = nw
            j+=1
        i+=1
    i = 0
    while(i<20):
        j = 0
        while(j<100):
            a = weights[1][j][i]
            b = error[1][i]
            c = nodes[2][j]
            nw = calcNewWeight(a,b,c)
            newWeights[1][j][i] = nw
            j+=1
        i+=1
    i = 0
    while(i<100):
        j = 0
        while(j<200):
            a = weights[2][j][i]
            b = error[2][i]
            c = nodes[3][j]
            nw = calcNewWeight(a,b,c)
            newWeights[2][j][i] = nw
            j+=1
        i+=1
    i = 0
    while(i<200):
        j = 0
        while(j<768):
            a = weights[3][j][i]
            b = error[3][i]
            c = data[j]
            nw = calcNewWeight(a,b,c)
            newWeights[3][j][i] = nw
            j+=1
        i+=1
    w1 = list(reversed(newWeights))
    return w1


'''
rewrite this shit too
this is literally retarded
'''
def chooseMove(w):
    evalList = []
    i = 0
    while(type(parseMove(i)) != bool):
    
        data = parseMove(i)
        out = forwardPass(data,w)
        evalList.append(out)
        i += 1

    m = max(evalList)
    chosen = [i for i, j in enumerate(evalList) if j == m]
    return chosen

def saveW(w):
    with open('outfile', 'wb') as fp:
        pickle.dump(w, fp)

def loadW():
    with open ('outfile', 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def loss(predicted, target):
    return np.square(target - predicted).sum()

def getRandomTrainingData():
    moves = []
    board.reset()
    i = 0
    while(i<50):
        if(board.is_game_over()==True):
            return moves
        l = getLegalMoveList()
        n = len(l)
        iteration = random.randint(0,n-1)
        m = parseMove(iteration)
        moves.append(m)
        playLegalMove(iteration)
        i+=1
    return moves
            
            

    
#w = initWeights([768,200,100,20,2])
w = loadW()
def train(iterations):
    board.reset()
    i = 0
    whiteMoves = []
    blackMoves = []
    elapsed = time.time()
    games = 0
    whiteWins = 0
    blackWins = 0
    w = loadW()
    while(i<iterations):
        

        if(board.is_game_over()==True):
            games += 1
            if(board.result() == "1-0"):
                whiteWins +=1
                trainingData = whiteMoves
                j = 0
                for count in range(10):
                    trainingData = shuffle(trainingData)
                    while(True):
                        try:
                            g = backPropError(trainingData[j],w,[1,0])
                            newW = updateWeights(g,w,trainingData[j])
                            w = newW
                            j+=1
                        except IndexError:
                            board.reset()
                            #board.set_fen('8/8/2k5/5Q2/3K4/8/8/8 w - - 6 4')
                            whiteMoves = []
                            blackMoves = []
                            break
                        
            elif(board.result() == "0-1"):
                blackWins += 1
                trainingData = blackMoves
                j = 0
                for count in range(10):
                    shuffle(trainingData)
                    while(True):
                        try:
                            g = backPropError(trainingData[j],w,[1,0])
                            newW = updateWeights(g,w,trainingData[j])
                            w = newW
                            j+=1
                        except IndexError:
                            board.reset()
                            #board.set_fen('8/8/2k5/5Q2/3K4/8/8/8 w - - 6 4')
                            whiteMoves = []
                            blackMoves = []
                            break
            else:
                trainingData = getRandomTrainingData()
                j = 0
                for count in range(2):
                    shuffle(trainingData)
                    while(True):
                        try:
                            g = backPropError(trainingData[j],w,[1,0])
                            newW = updateWeights(g,w,trainingData[j])
                            w = newW
                            j+=1
                        except IndexError:
                            board.reset()
                            #board.set_fen('8/8/2k5/5Q2/3K4/8/8/8 w - - 6 4')
                            whiteMoves = []
                            blackMoves = []
                            break
            
        
        m = chooseMove(w)
        m = m[0]
        if(board.turn == True):
            playLegalMove(m)
            a = getBoard()
            whiteMoves.append(a)
        else:
            playLegalMove(m)
            a = getBoard()
            blackMoves.append(a)
        if(i % 100 == 0):
            a = time.time()
            b = a - elapsed
            print(str(i) + " took " + str(b))
            elapsed = time.time()
            
        i+=1
    saveW(w)
    print("Games: " + str(games) + "\n" + "White Wins: " + str(whiteWins) + "\n" + "Black Wins: " + str(blackWins))

def play(speed):
    board.reset()
    while(board.is_game_over() == False):
        m = chooseMove(w)
        m = m[0]
        playLegalMove(m)
        print(board)
        print(" ")
        time.sleep(speed)
    print(board.result())

'''

 
grad_y_pred = 2.0 * (output of forward pass - label)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)


a[37] = '('
a = "cunt"
>>> a = a + "bitch"
>>> print(a)



'''


"""
       

def playMove(f,t):
    move = chess.square_name(f) + chess.square_name(t)
    Qmove = chess.square_name(f) + chess.square_name(t) + 'q'
    isLegal = chess.Move.from_uci(move) in board.legal_moves
    isQLegal = chess.Move.from_uci(Qmove) in board.legal_moves
    if(isLegal == True):
        board.push_uci(move)
        return True
    elif(isQLegal == True):
        board.push_uci(Qmove)
        return True
    else:
        return False






last_time = time.time()
i = 0
inc = 0
while(i < 64):
    f = chess.square_name(i)
    j = 0
    while(j<64):
        t = chess.square_name(j)
        j = j + 1
        chess.Move.from_uci(f+t) in board.legal_moves
        inc = inc + 1
    i = i + 1
print('loop took {} seconds'.format(time.time()-last_time))
print(inc)
"""
  
"""


board.turn
#chess.square_name(43)
#chess.Move.from_uci("a8a1") in board.legal_moves


#propogate
#if not legal just go down the probability list
#save inputs + output with chosen move set to 1 and all others set to 0
#play move
#after one side has won backprop over saved training data for that side
"""
