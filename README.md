# 4inA-
An implementation of Alpha Zero, for connect 4. The project is implemented using pytorch aswell.

### Let me play!
To play the model simpely run `main.py` after installing the dependencies (pytorch, tqdm & numpy).


## Training & Evaluation:
The models was trained over 200 iterations, one iteration is: 
10 games, with 24 roolouts per turn, followed by 10 epochs of learning and atlast 98 evaluation games against the current best network. These iterations took about 10 minutes per iteration, using my laptop with a Geforce GTX 1650 gpu. 

## TODO:
* Look for ways to speedup the MCTS algorithm
    * Maybe run multiple mcts at a time with a single neural network for evalation and creating training dataset. (This could improve GPU saturation)
* Create vizualisation of the learning rates, eloes & winrates against the first model.
* Create additional comments & documentation