# 4inA-
An implementation of Alpha Zero, for connect 4. Using: NumPy, Numba & PyTorch.

The project is currently being upgraded for parallel MCTS for training & evaluation purposes, as a result it doesn't currently function correctly.

### Let me play!

First make sure that you have the dependencies installed:
* Numpy
* Numba
* PyTorch (Have only been tested with GPU support (you might need to change some setting in the config.py file, if no GPU is available))

To play against the model simply run `main.py` after installing the dependencies.

## Training & Evaluation:
The models was trained over 200 iterations, one iteration is: 
100 games, with 24 roolouts per turn, followed by 100 epochs of learning and 98 evaluation games against the current best network with different unique starting positions. These iterations took about 10 minutes per iteration, using my laptop with a Geforce GTX 1650 gpu. 

## Current state of the project
The project still needs some work before it's completely complete, for example the code for parallel MCTS (including the code in the parallel.py file and the function createDatasetWithParallelMCTS) does not yet function correctly...
Also it could be very cool to have visualization options for visualizing the progress (maybe the winrate of each model against the first model?)
