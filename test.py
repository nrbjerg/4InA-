import numpy as np
import math 

def sigmoid (x: float) -> float:
    return 1 / (1 + math.exp(x))

reward = 1
n = 12
for i in range(n):
    reward = 1 - (sigmoid(-i // 2) / 2 - 0.2) if (i % 2) == 0 else -reward
    if (i >= 10): reward = 0
    print(reward)
    