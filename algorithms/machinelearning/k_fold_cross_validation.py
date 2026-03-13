# https://www.deep-ml.com/problems/18
# https://aman.ai/primers/ai/cross-validation/

import numpy as np

"""
Notes
K-Fold Cross-Validation is a resampling technique used to evaluate machine learning models by partitioning the dataset into multiple folds.
How it Works
    The dataset is split into k equal (or almost equal) parts called folds.
    Each fold is used once as a test set, while the remaining k-1 folds form the training set.
    The process is repeated k times, ensuring each fold serves as a test set exactly once.
Why : Reduces bias introduced by a single training/testing split.

Implementation Steps
    Shuffle the data if required.
    Split the dataset into k equal (or nearly equal) folds.
    Iterate over each fold, using it as the test set while using the remaining data as the training set.
    Return train-test indices for each iteration.

By implementing this function, you will learn how to split a dataset for cross-validation, a crucial step in model evaluation.

# Using Library
kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_seed)
folds = [] # This list will store the (train_list, test_list) tuples
    for train_index_np, test_index_np in kf.split(X, y):
        train_list = list(train_index_np)
        test_list = list(test_index_np)

        folds.append((train_list, test_list)) # Your previous example had (test_list, train_list)

    return folds
"""


# X = features/rows, y = labels(outputs/columns)
def k_fold_cross_validation( X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None): 
    samples = X.shape[0]  # input
    if samples != y.shape[0]:
        raise ValueError("Different size")

    indices = np.arange(samples)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    folds = []
    fold_size = samples // k
    fold_remainder = samples % k
    # curr_posn = next fold should start in indices arr
    curr_posn = 0
    # kfolds implement
    for i in range(k):
        curr_fold_size = fold_size + (1 if i < fold_remainder else 0)
        start_idx, end_idx = curr_posn, curr_posn + curr_fold_size
        test_data = indices[start_idx:end_idx]
        train_data = np.concatenate((indices[:start_idx], indices[end_idx:]))
        train_list, test_list = list(train_data), list(test_data)
        folds.append((train_list, test_list))
        curr_posn = end_idx
    return folds

    # X_train, y_train = X[train_list], y[train_list]
    # model = model()
    # train and test


def main():
    print(
        k_fold_cross_validation(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            k=5,
            shuffle=False,
        )
    )


main()

"""
Output-:

[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), 
([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), 
([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), 
([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), 
([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]

Slice your actual X and y data using these indices (X_train = X[train_list], y_train = y[train_list], etc.).
Initialize a fresh model.
Train the model on X_train, y_train.
Evaluate its performance on X_test, y_test.
Collect these k performance scores and average them.
"""
