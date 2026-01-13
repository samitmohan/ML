import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    X, y = np.asarray(X), np.asarray(y)
    n = X.shape[0]
    if replacements:
        subset_size = n
    else:
        subset_size = n//2
    np.random.seed(seed)
    subsets = []
    for i in range(n_subsets):
        indices = np.random.choice(n, size = subset_size, replace=replacements)
        
        X_subset = X[indices].tolist()
        Y_subset = y[indices].tolist()
        subsets.append((X_subset, Y_subset))
    return subsets

def main():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    n_subsets = 3
    replacements = False
    print(get_random_subsets(X, y, n_subsets, replacements, seed=42))

main()

'''
[([[3, 4], [9, 10]], [2, 5]),
 ([[7, 8], [3, 4]], [4, 2]),
 ([[3, 4], [1, 2]], [2, 1])]
'''