import numpy as np

# TODO: Connect 4

# class State:

#     def __init__ (self, width: int = 7, height: int = 6):
#         """ Initializes an empty state """
#         self.dimensions = {"h": height, "w": width}
#         self.maps = np.zeros(height, width) # 1 represents the current player and -1 the other player
#         self.moves = np.zeros(height * width, dtype = "int32")
#         self.heights = np.zeros((width), dtype="int32")
#         self.turn = 0

#     def getValidMoves (self) -> np.array:
#         """ Returns a list of valid moves """
#         return np.array([1 if (self.CheckIfMoveIsValid(i) == True) else 0 for i in range(self.dimensions["w"])])
    
#     def CheckIfMoveIsValid (self, move: int) -> bool:
#         """ Checks if a move is valid in the current gamestate """
#         return (self.maps[0][move] == 0)
    
#     def isWonBy (self, index: int) -> int: # Returns 1 if its won, 0 if its a draw and -1 if its nether
#         """ Checks if the game is won by the player specified by the index """
#         m = self.maps[index] * ((-1) if (self.turn % 2 == 0) else 1)

#         # Horizontal
#         for row in m: 
#             for i in range(self.dimensions["w"] - 3):
#                 if (np.sum(row[i:i + 4]) == 4): return 1

#         # Vertical
#         for col in [m[:, i] for i in range(self.dimensions["w"])]:
#             for i in range(self.dimensions["h"] - 3):
#                 if (np.sum(col[i:i + 4]) == 4): return 1


#         # Diagonal filters:
#         filters  =  [np.array([[1, 0, 0, 0],
#                                [0, 1, 0, 0],
#                                [0, 0, 1, 0],
#                                [0, 0, 0, 1]]),

#                      np.array([[0, 0, 0, 1],
#                                [0, 0, 1, 0],
#                                [0, 1, 0, 0],
#                                [1, 0, 0, 0]])]
        
#         # Diagonals
#         for f in filters:
#             for i in range(self.dimensions["h"] - 3):
#                 for j in range(self.dimensions["w"] - 3):
#                     if (np.sum(np.multiply(f, m[i:i + 4, i:i + 4])) == 4): return 1

#         if (np.sum(self.heights) == self.dimensions["w"] * self.dimensions["h"]): return 0
        
#         return -1 # If the game isnt won or drewed
    
#     def playMove (self, move: int):
#         """ Plays a given move on the current gamestate """
#         self.maps[-self.heights[move] - 1][move] = 1 if (self.turn % 2 == 0) else -1
#         self.moves[self.turn] = move
#         self.heights[move] += 1
#         self.turn += 1
    
#     def undoLastMove (self):
#         """ Undoes the last move played """
#         # TODO: Redo this
#         self.turn -= 1
#         move = self.moves[self.turn] # Note that this does not need to be overwriten
#         self.heights[move] -= 1
#         self.maps[self.turn % 2][-self.heights[move] - 1][move] = 0

#     def __str__ (self):
#         """ Returns a string representation of the state """
#         rows = []
#         for i in range(self.dimensions["h"]):
#             string = "|"
#             for j in range(self.dimensions["w"]):
#                 if (self.maps[i][j] == 1):
#                     string += "x|"
#                 elif (self.maps[i][j] == -1):
#                     string += "o|"
#                 else: 
#                     string += " |"
#             rows.append(string + "\n")
#         lastRow = "".join([" " + str(i) for i in range(self.dimensions["w"])]) + "\n"
#         return "".join(rows) + lastRow

# X and O 
class State:
    
    def __init__ (self):
        self.map = np.zeros((3, 3))
        self.turn = 0
        self.currentPlayer = 1
    
    def playMove (self, move: int):
        self.map[move // 3][move % 3] = self.currentPlayer
        self.currentPlayer *= -1
        self.turn += 1
        
    def getValidMoves (self):
        return [1 if self.map[move // 3][move % 3] == 0 else 0 for move in range(9)]
    
    def isWonBy (self, player: int):
        m = self.map * player
        
        for row in m:
            if (np.sum(row) == 3): return 1
        
        for col in [m[:, i] for i in range(self.dimensions["w"])]:
            if (np.sum(col) == 3): return 1
        
        filters = [np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]),
                   
                   np.array([[0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]])]
        
        for f in filters:
            if (np.sum(np.multiply(f, m)) == 3): return 1
        
        if (self.turn == 8): return 0
        
        return None
    
    def __str__ (self):
        string = ""
        for row in self.map:
            for cell in row:
                if (cell == 0): string += ":"
                else: string += "x" if cell == 1 else "o"
            string += "\n"
        return string
    
if __name__ == "__main__":
    S = State()
    print(str(S))
    print(S.getValidMoves())
    S.playMove(1)
    S.playMove(3)
    print(str(S))