# Based on:
# Soduku resolution with Recocido Simulado - Brice de Soras
#https://github.com/bdscloud/Sudoku-Simulated-Annealing-Solution

import numpy as np
import matplotlib.pyplot as plt
import random
from math import exp

sudokuInitial = np.array([[0,0,0,0,3,0,6,0,0],
                         [5,0,0,9,0,0,4,0,0],
                         [0,8,0,6,0,7,0,0,9],
                         [0,7,0,0,0,0,8,0,1],
                         [0,5,0,0,8,0,0,2,0],
                         [3,0,8,0,0,0,0,5,0],
                         [1,0,0,8,0,4,0,9,0],
                         [0,0,2,0,0,6,0,0,5],
                         [0,0,9,0,1,0,0,0,0]])

def set_seed(seed):
    random.seed(seed)

def copySudoku(sudoku):
    sudokuCopy = []
    for i in range(len(sudoku)):
        sudokuCopy.append(list(sudoku[i]))
    return sudokuCopy

## function that returns a square of the sudoku, determined by it's position (i,j)
## example : square(1,1) returns the top left square
def square(sudo,i,j):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku (in our exercice 4)
    sudoSquareLen = int(sudoLen ** (1 / 2.0))

    A = [] #  empty regular list
    for t in range(sudoSquareLen):
        X = sudo[(i-1)*sudoSquareLen+t][(j-1)*sudoSquareLen:(j-1)*sudoSquareLen+sudoSquareLen]
        A.append(X)
    return np.array(A)

## function that puts in a list all the couples (u,v) of the numbers that are not fixed in a square (i,j)
## Returns a list of the coordinates (u,v) of the numbers that have to be found in a square (i,j)
def nonFixedPosition(sudo,i,j):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku
    sudoSquareLen = int(sudoLen ** (1 / 2.0))
    A = [] #local positions
    #B = [] #global positions
    sq = square(sudo,i,j)
    for u in range(sudoSquareLen):
        for v in range(sudoSquareLen):
            if(sq[u][v] == 0):
                A.append([u,v])
    return A

## function that returns a random position (u,v) in a square (i,j) amongs all the non fixed numbers of this square
def randomNonFixedNumber(sudo,i,j):
    rd = random.randint(0, len(nonFixedPosition(sudo,i,j)) - 1)
    return nonFixedPosition(sudo,i,j)[rd]

## function that returns the list of all the numbers that are not present in the square (i,j)
def listPossibleNumbers(sudo,i,j):
    # lenth of the sudoku
    sudoLen = len(sudo)
    A = [] ##list of all the possible numbers
    B = [] ##list of all the non possible numbers
    sq = square(sudo,i,j)
    for u in range(len(sq)):
        for v in range(len(sq)):
            if (sq[u,v] != 0):
                B.append(sq[u,v])
    for t in range(sudoLen):
        if (t+1 not in (B)):
            A.append(t+1)
    return A

## function that fills a square (i,j) of the sudoku by putting numbers that respect the 3rd rule
def sudokuFill3rdRule1(sudo, i,j):
    sq = square(sudo, i,j)
    noFixedPos = nonFixedPosition(sudo,i,j)
    for x in range(len(noFixedPos)) :
        sq[noFixedPos[x][0], noFixedPos[x][1]] = listPossibleNumbers(sudokuInitial,i,j)[x]
    return sq

## function that fills the sudoku by putting numbers that respect the 3rd rule
def sudokuFill3rdRule(sudo):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku
    sudoSquareLen = int(sudoLen ** (1 / 2.0))
    A = []
    B = []
    for i in range(1, sudoSquareLen + 1):
        for j in range(1, sudoSquareLen + 1):
            A.append(sudokuFill3rdRule1(sudo,i,j))
    for i in range(3):
        B.append(np.concatenate((A[0][i], A[1][i], A[2][i]), axis=None).tolist())
    for i in range(3):
        B.append(np.concatenate((A[3][i], A[4][i], A[5][i]), axis=None).tolist())
    for i in range(3):
        B.append(np.concatenate((A[6][i], A[7][i], A[8][i]), axis=None).tolist())
    return B


## function that fills the sudoku by putting numbers that respect the 3rd rule
def sudokuFill3rdRule_2(sudo):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku
    sudoSquareLen = int(sudoLen ** (1 / 2.0))
    A = []
    B = []
    for i in range(1, sudoSquareLen + 1):
        for j in range(1, sudoSquareLen + 1):
            A.append(sudokuFill3rdRule1(sudo, i, j))

    for i in range(sudoSquareLen):
        for j in range(sudoSquareLen):
            B.append(np.concatenate((A[sudoSquareLen * i][j], A[sudoSquareLen * i + 1][i], A[sudoSquareLen * i + 2][j]),
                                    axis=None).tolist())

    return B

## this function calculates the cost of all lines
def costLine(sudo):
    # lenth of the sudoku
    sudoLen = len(sudo)
    A = []
    for i in range(sudoLen):
        line = sudo[i]
        cost = 0
        for j in range(len(line)):
            if line.count(j+1) == 0:
                cost += 1
        A.append(cost)
    return sum(A)

## this function calculates the cost of all lines
def costColumn(sudo):
    # lenth of the sudoku
    sudoLen = len(sudo)
    transposesudokuGrid = np.transpose(sudo)
    A = []
    for i in range(sudoLen):
        line = transposesudokuGrid[i]
        cost = 0
        for j in range(len(line)):
            if line.tolist().count(j+1) == 0:
                cost += 1
        A.append(cost)
    return sum(A)

## this function calculates the global cost
def costGlobal(sudo):
    return costColumn(sudo) + costLine(sudo)

## function that return the coordinates (u,v) (0->15) from the square(i,j) (1->4) and position in square (0->3)
def ChangeCoordinates(sudo, i,j,u,v):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku
    sudoSquareLen = int(sudoLen ** (1 / 2.0))
    return [(i-1)*sudoSquareLen+u , (j-1)*sudoSquareLen+v]


def swapRandomCells(sudo):
    # lenth of the sudoku
    sudoLen = len(sudo)
    # lenth of a square of the sudoku
    sudoSquareLen = int(sudoLen ** (1 / 2.0))
    newSudo = copySudoku(sudo)

    # choose a random square and get the non-fixed positions in this square
    rd1 = random.randint(1, sudoSquareLen)
    rd2 = random.randint(1, sudoSquareLen)
    # square = square(sudoku, rd1, rd2)
    A = nonFixedPosition(sudokuInitial, rd1, rd2)  ##list of non fixed positions in a random square of the sudoku

    # chose two random cells among the non fixed ones in the random square
    rd3 = random.randint(0, len(A) - 1)
    cell_1_coordinates = A[rd3]
    rd4 = random.randint(0, len(A) - 1)
    while rd4 == rd3:
        rd4 = random.randint(0, len(A) - 1)
    cell_2_coordinates = A[rd4]

    # in the whole sudoku
    coordinates_cell_1 = ChangeCoordinates(sudo, rd1, rd2, cell_1_coordinates[0], cell_1_coordinates[1])
    coordinates_cell_2 = ChangeCoordinates(sudo, rd1, rd2, cell_2_coordinates[0], cell_2_coordinates[1])

    # swap
    tmp = newSudo[coordinates_cell_1[0]][coordinates_cell_1[1]]
    newSudo[coordinates_cell_1[0]][coordinates_cell_1[1]] = newSudo[coordinates_cell_2[0]][coordinates_cell_2[1]]
    newSudo[coordinates_cell_2[0]][coordinates_cell_2[1]] = tmp

    return newSudo


def recocidoSimulado(sudo, temp= 500, alpha = 0.98, iter = 500):

    sudoCopy = copySudoku(sudo)
    costs = []
    changeProbs = []
    compteur = 0

    while temp > 1:
        changeProb = 0
        #param = 10
        for i in range(iter):

            if costGlobal(sudoCopy) == 0:
                print("Sudoku solved")
                #plt.plot(np.arange(len(costs)), costs)
                #plt.show()
                #print("Final cost : " + str(costGlobal(sudoCopy)))
                #print("Temperature changed " + str(compteur) + " times")
                return sudoCopy

            sudoAfter = swapRandomCells(sudoCopy)

            delta = costGlobal(sudoAfter) - costGlobal(sudoCopy)

            if (delta <= 0):
                sudoCopy = sudoAfter
            else:
                uniformValue = random.uniform(0, 1)
                if (uniformValue <= exp(-30*delta / temp)):
                    changeProb += 1
                    sudoCopy = sudoAfter

            costs.append(costGlobal(sudoCopy))
        changeProbs.append(changeProb)
        compteur += 1
        temp = alpha * temp

    #plt.plot(np.arange(len(costs)), costs)
    #plt.show()

    percents = list(map(lambda x: float(x) / iter * 100, changeProbs))
    #plt.plot(np.arange(len(percents)), percents)
    #plt.show()

    #print("Final cost : " + str(costGlobal(sudoCopy)))
    #print("Temperature changed " + str(compteur) + " times")
    return sudoCopy, costs, percents




def graph_soduku(solution):
    lines = np.arange(1, 10)
    columns = np.arange(1, 10)
    fig, ax = plt.subplots(figsize=(15,8))
    im = ax.imshow(solution)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(lines)))
    ax.set_yticks(np.arange(len(columns)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(lines)
    ax.set_yticklabels(columns)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(lines)):
        for j in range(len(columns)):
            text = ax.text(j, i, solution[i][j],
                           ha="center", va="center", color="w", weight="bold", fontsize=20)

    #ax.set_title("Sudoku resolved", fontsize=30)
    fig.tight_layout()
    return plt

def print_solution_verb(solution):
    print("Final cost : " + str(costGlobal(solution)))
    # lenth of the sudoku
    sudoLen = len(solution)

    for i in range(sudoLen):
        line = solution[i]
        for j in range(sudoLen):
            if line.count(j + 1) == 0:
                print("Line " + str(i + 1) + " : missing : " + str(j + 1))
    for i in range(sudoLen):
        column = np.transpose(solution)[i]
        for j in range(sudoLen):
            if column.tolist().count(j + 1) == 0:
                print("Column " + str(i + 1) + " : missing : " + str(j + 1))

# sudokuFilled = sudokuFill3rdRule(sudokuInitial)
# solution, costs, percents = recocidoSimulado(sudokuFilled, temp= 250, alpha = 0.97, iter = 100)
# print_solution_verb(solution)



