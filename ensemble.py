# -*- coding: utf-8 -*-

# Compare ensemble classifier to a base classifier
from scipy.misc import comb
import math
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier-k)
            for k in range(k_start, n_classifier+1)] 
    return sum(probs)
    
print(ensemble_error(n_classifier=11, error=0.25))

# Compute the error rates for a range of different base errors
import numpy as np
import matplotlib.pyplot as plt

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, label='Base error', 
         linestyle='--', linewidth=2)

plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=5)
plt.show()

# argmax + bincount functions
import numpy as np
print(np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])))

# Implement the majority voting for predicted class label using probability
ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]
               ])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])    
print(np.argmax(p))