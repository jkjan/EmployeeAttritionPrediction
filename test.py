import numpy as np
import scipy.stats as ss

arr = [[1, 2, 3],
       [2, 6, 10],
       [1, 2, 3]]

mean = np.mean(arr, axis=0)
print(mean)
std = np.std(arr, axis=0)
print(std)
z_score = (arr - mean) / std
print(z_score)
