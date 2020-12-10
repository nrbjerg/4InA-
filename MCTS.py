import numpy as np 
from State import State
from copy import deepcopy as dc

def USBscore (parent, child) -> float:
    """ USB score with """
    return child.prior * np.sqrt(parent.visits / (child.visits + 1)) - child.value()

class Node:
    
    def __init__ (self, S: State, prior, turn: int):
        """ Initializes a node for monte carlo tree search """
        self.visits = 0
        self.turn = turn
        self.prior = prior
        self.totalValue = 0
        self.children = {}
        self.state = S

    def isExpanded (self):
        """ Check if the node has been expanded """
        return len(self.children.keys()) > 0
    
    def value(self):
        """ The value of the node """
        if (self.visits == 0): return 0
        else:
            return self.totalValue / self.visits
    
    def selectAction (self, t: float):
        """ Select an action based on the childrens visit counts and the temperature (t) """
        # Number of visits and the valid actions
        visits = np.array([child.visits for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        
        # Chose an action based on the temperature
        if (t == 0):
            action = actions[np.argmax((visits))]
        elif (t == np.inf):
            action = np.random.choice(actions)
        else:
            distribution = np.power(visits, (1 / t))
            action = np.random.choice(actions, p=(distribution / np.sum(distribution)))
        
        return action
               
    def selectChild (self):
        """ Selects a child based on the UCB score """
        best = {"s": -np.inf, "a": None, "c": None}
        
        # Find the child and action with the best UCB score 
        for action, child in self.children.items():
            score = USBscore(self, child)
            if (score > best["s"]):
                best = {"s": score, "a": action, "c": child}
        
        return (best["a"], best["c"]) # Return a tuple of the best action (int) and the best child (node)
    
    def expand (self, propablities):
        """ Expand the node and keep track of the prior policy from the neural network """
        for a, prob in enumerate(propablities):
            if (prob != 0):
                self.children[a] = Node(S = dc(self.State).playMove(a), prior=prob, turn = self.turn + 1)
    
    def __repr__(self):
        """ Debugger pretty print node info """
        return "{} Prior: {0:.2f} Count: {} Value: {}".format(self.state.__str__(), self.prior, self.visit_count, self.value())

class MCTS:
    
    def __init__(self, model):
        self.node = None
        self.model = model
        self.paths = []
        
    def backpropagate (self, score, terminalTurn: int):
        """ Back propagate the value through the last path """
        for node in reversed(self.paths[-1]):
            node.totalValue += score if (terminalTurn % 2 != (node.state.turn - 1) % 2) else -score
            node.visits += 1
    
    def expand (self):
        """ Expands the current node """
        probs, value = self.model.predict(self.node.state)
        probs = probs * self.node.state.getValidMoves() # Mask (then the moves that are illeagal have a 0 prob)
        self.node.expand(probs)
        
    def run (self, initialState: State, n: int) -> Node:
        """ Runs the MCTS algoritm """
        root = Node(initialState, 0, turn = 0)
        self.node = root
        
        # Expand root
        self.expand()
        
        for _ in range(n):
            self.node = root
            path = [self.node] # Search path
            
            # Select until we reach a leaf node
            while self.node.isExpanded():
                action, self.node = self.node.selectChild()
                path.append(self.node)
            
            # We are now at a leaf node
            leafState = self.node.state
            value = leafState.isWonBy((leafState.turn - 1) % 2) # Check if the previus move won the game

            if (value == -1): # Then the games hasn't ended
                # Expand
                self.expand()

            # Backpropagate
            self.paths.append(path)
            self.backpropagate(leafState.turn)
                
        return root
    
    def getAction (self, S: State, n: int):
        """ Computes the best action """
        root = self.run(S, n)
        return root.selectAction(0)