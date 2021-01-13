from typing import List
from numba import njit, int32, float32
import numpy as np
from state import checkIfGameIsWon, makeMove, getAvailableMoves
from torch import from_numpy, Tensor
from model import Net
import torch
# TODO: I think this inteire file needs an overhaul (TO USE THE NEURAL NETWORK MORE.)
# See https://web.stanford.edu/~surag/posts/alphazero.html

@njit()
def UCBscore (childPrior: float, parentVisits: int, childVisits: int, childValue: float) -> float:
    """ Computes the UCB score """
    return childValue + childPrior * np.sqrt(parentVisits / (childVisits + 1))

# TODO: JIT THIS CLASS 
class Node:
    
    def __init__ (self, S: np.array, prior: float):
        """ Initializes a node for monte carlo tree search """
        self.visits = 0
        self.prior = prior
        self.totalValue = 0
        self.children = {}
        self.state = S

    def isExpanded (self):
        """ Check if the node has been expanded """
        return len(self.children.keys()) > 0
    
    @property
    def probabilities (self) -> np.array:
        """ Gives the probabilities of each of the children """
        probs = np.zeros((1, self.state.shape[1]), dtype = "float32")
        for key in range(self.state.shape[1]):
            if (key in self.children.keys()):
                probs[0][key] = self.children[key].visits / self.visits
        return probs
    
    @property
    def value(self):
        """ The value of the node """
        if (self.visits == 0): return 0
        else:
            return self.totalValue / self.visits
    
    def selectAction (self, t: float):
        """ Select an action based on the children's visit counts and the temperature (t) """
        # Number of visits and the valid actions
        visits = np.array([child.visits for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        
        # Chose an action based on the temperature
        try:
            if (t == 0):
                action = actions[np.argmax((visits))]
            else:
                distribution = np.power(visits, (1 / t))
                action = np.random.choice(actions, p=(distribution / np.sum(distribution)))
        except ValueError:
            action = 0
                
        return action
               
    def selectChild (self):
        """ Selects a child based on the UCB score """
        best = {"s": -np.inf, "a": None, "c": None}
        
        # Find the child and action with the best UCB score 
        for action, child in self.children.items():
            score = UCBscore(child.prior, self.visits, child.visits, child.value)
            if (score > best["s"]):
                best = {"s": score, "a": action, "c": child}
        
        return (best["a"], best["c"]) # Return a tuple of the best action (int) and the best child (node)
        
    def __repr__(self):
        """ Debugger pretty print node info """
        return "{0}\nPrior: {1:.2f} \nVisits: {2} \nValue: {3}".format(self.state, self.prior, self.visits, self.value)

def expand (node: Node, probabilities: np.array) -> None:
    """ Simpely expands the node with new propabilities """
    for a, prob in enumerate(probabilities.transpose()):
        if (prob[0] != 0.0):
            s = -makeMove(node.state, a) # NOTE: This will also copy the state, also the - makes sure that the next perspective is from the next player
            node.children[a] = Node(S = s, prior=prob[0])

def backpropagate (path: List[Node], score: int) -> None:
    """ Back propagate the value through the last path """
    n = len(path)
    for i in reversed(range(n)):
        if (i % 2 != n % 2):
            path[i].totalValue += score 
        else: 
            path[i].totalValue -= score
        path[i].visits += 1  
        
def MontecarloTreeSearch (state: np.array, n: int, model: torch.nn.Module) -> Node:
    """ Performs Monte-Carlo tree search on the state """
    with torch.no_grad(): # This will not train the neural network therefore we don't need the gradients
        
        def getModelOutputs (state: np.array, model: torch.nn.Module) -> np.array:
            """ Get propabilities form the neural network """
            probs, value = model(from_numpy(state).view(1, 1, state.shape[1], state.shape[0])) # Convert to 4d tensor
            return probs.numpy() * getAvailableMoves(state), value.numpy() # Remove illegal moves, by giving them 0.0 propabilities
        
        # Initialize root node
        root = Node(state, 0.0)

        # Expand root
        expand(root, getModelOutputs(state, model)[0])
            
        for _ in range(n): # Perform the n iterations
            state *= -1 # Switch the view
            node = root
            path = [node] # Search path
            
            # Select until we reach a leaf node
            while node.isExpanded():
                action, node = node.selectChild()
                # print(node.state)
                path.append(node)
            
            # We are now at a leaf node
            leafState = node.state
            value = checkIfGameIsWon(leafState) # Check if the previous move won the game

            if (value == -1): # Then the games hasn't ended
                # Expand the leaf node
                probs, predictedValue = getModelOutputs(leafState, model)
                expand(node, probs)
                backpropagate(path, predictedValue) # Update the visit counts
            else: 
                # Backpropagate the value of the leaf node back up the tree
                backpropagate(path, value)
                
        return root

def getAction (S: np.array, n: int, model: torch.nn.Module) -> int:
    """ Computes the best action (gives the index of the best move) """
    root = MontecarloTreeSearch(S, n, model)
    return root.selectAction(0)

if __name__ == "__main__":
    S = np.zeros((6, 7), dtype = "float32")
    root = MontecarloTreeSearch(S, 100, Net())
    