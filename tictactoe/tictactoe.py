"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    counter = 0

    for i in range(0, len(board[0])):
        for j in range(0, len(board[0])):
            if board[i][j] == X:
                counter += 1
            if board[i][j] == O:
                counter -= 1

    if counter == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    for i in range(0, len(board[0])):
        for j in range(0, len(board[0])):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]

    # Check if move is valid
    if i not in [0, 1, 2] or j not in [0, 1, 2]:
        raise ValueError
    elif board[i][j] != EMPTY:
        raise ValueError

    result_board = copy.deepcopy(board)
    result_board[i][j] = player(board)

    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Checking rows
    for row in board:
        if row.count(X) == 3:
            return X
        if row.count(O) == 3:
            return O

    # Checking columns, this is a stupid but fast way to do this, kind of brute forcing it
    if board[0][0] == board[1][0] == board[2][0] and board[0][0] != EMPTY:
        return board[0][0]
    if board[0][1] == board[1][1] == board[2][1] and board[0][1] != EMPTY:
        return board[0][1]
    if board[0][2] == board[1][2] == board[2][2] and board[0][2] != EMPTY:
        return board[0][2]


    # Checking diagonal, again a stupid way to do this, brute force
    if board[0][0] == board[1][1] == board[2][2] == X or board[2][0] == board[1][1] == board[0][2] == X:
        return X
    if board[0][0] == board[1][1] == board[2][2] == O or board[2][0] == board[1][1] == board[0][2] == O:
        return O

    # If no winner return none
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or not actions(board):
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Using Alpha Beta pruning
    # No move possible if game is over
    if terminal(board):
        return None

    # Best for Alpha and best for Beta
    MAXv = float("-inf")
    MINv = float("inf")

    # Return best move for each player
    if player(board) == X:
        return max_value(board, MAXv, MINv)[1]
    else:
        return min_value(board, MAXv, MINv)[1]


def min_value(board, MAXv, MINv):
    """
    Helper function to return min value for min player
    """
    if terminal(board):
        return [utility(board), None]

    # Best minimum value
    move = None
    v = float('inf')

    # Loop through possible moves
    for action in actions(board):
        test = max_value(result(board, action), MAXv, MINv)[0]
        # Update Beta, best minimum value
        MINv = min(MINv, test)

        # Find lowest value move
        if test < v:
            v = test
            move = action
        # Prune, stop searching if no better move is possible
        if MAXv >= MINv:
            break

    return [v, move]


def max_value(board, MAXv, MINv):
    """
    Helper function to return max value for max player
    """
    if terminal(board):
        return [utility(board), None]

    # Best maximum value
    move = None
    v = float('-inf')

    # Loop through all possible moves
    for action in actions(board):
        test = min_value(result(board, action), MAXv, MINv)[0]
        # Update Alpha, best maximum value
        MAXv = max(MAXv, test)

        # Find highest value move
        if test > v:
            v = test
            move = action
        # Prune, stop searching is no better move is possible
        if MAXv >= MINv:
            break

    return [v, move]
