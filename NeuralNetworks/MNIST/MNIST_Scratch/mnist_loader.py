"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data. For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``. In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""
# Image has 28 * 28 = 784 pixels.

import gzip
import pickle
from pathlib import Path
import sys
import numpy as np


def _vectorized_result(j: int) -> np.ndarray:
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data( data_path = "data/mnist.pkl.gz",):
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries:
    - The first entry (images): A numpy ndarray of shape (50000, 784),
      representing 50,000 28x28 pixel images flattened to 784 values.
    - The second entry (labels): A numpy ndarray of shape (50000,),
      containing the digit values (0...9) for the corresponding images.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains 10,000 images, so their shapes are (10000, 784) and (10000,).

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """

    data_file = Path(data_path)

    # debugging
    print(f"DEBUG: Current Working Directory: {Path.cwd()}")
    print(f"DEBUG: Data path argument received: {data_path}")
    print(f"DEBUG: Resolved path `data_file`: {data_file.resolve()}")


    if not data_file.exists():
        print(f"Error: Data file not found at {data_file.resolve()}") # Use resolve() here too for clarity
        print(
            "Please ensure 'mnist.pkl.gz' is in a 'data' "
            "subdirectory directly accessible from the current working directory, "
            "or that the provided data_path is absolute and correct."
        )
        sys.exit(1)

    with gzip.open(data_file, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


def load_data_wrapper( data_path = "data/mnist.pkl.gz",):
    """
    Return a tuple containing ``(training_data, validation_data, test_data)``.
    Based on ``load_data``, but the format is more convenient for use
    in our implementation of neural networks.

    In particular:
    - ``training_data``: A list containing 50,000 2-tuples ``(x, y)``.
      ``x`` is a 784-dimensional numpy.ndarray (shape (784, 1)) containing
      the input image. ``y`` is a 10-dimensional numpy.ndarray (shape (10, 1))
      representing the one-hot encoded unit vector corresponding to the
      correct digit for ``x``.
    - ``validation_data`` and ``test_data``: Lists containing 10,000
      2-tuples ``(x, y)``. In each case, ``x`` is a 784-dimensional
      numpy.ndarray (shape (784, 1)) containing the input image, and
      ``y`` is the corresponding classification (an integer digit value 0-9).
    """
    # Unpack the raw data from load_data
    tr_d, va_d, te_d = load_data(data_path) # pass data_path explicitly

    # Process training data: images (x) reshaped, labels (y) one-hot encoded
    training_data = [
        (np.reshape(x, (784, 1)), _vectorized_result(y))
        for x, y in zip(tr_d[0], tr_d[1])
    ]

    # Process validation and test data: images (x) reshaped, labels (y) as integers
    validation_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(va_d[0], va_d[1])]
    test_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(te_d[0], te_d[1])]

    return training_data, validation_data, test_data

# This is for debugging
if __name__ == "__main__":
    mnist_data_path = Path(__file__).parent / "data" / "mnist.pkl.gz"

    print("\n--- Testing load_data from mnist_loader.py ---")
    print(f"Attempting to load data using explicit path: {mnist_data_path}")

    try:
        raw_tr, raw_va, raw_te = load_data(mnist_data_path)
        print("\n--- Raw Data Loaded Successfully ---")
        print(f"Training images shape: {raw_tr[0].shape}")
        print(f"Training labels shape: {raw_tr[1].shape}")
        # ... rest of prints
    except Exception as e:
        print(f"Error loading raw data: {e}")
        print("Hint: Ensure 'mnist.pkl.gz' is downloaded and in the 'data' folder.")


    print("\n--- Testing load_data_wrapper from mnist_loader.py ---")
    print(f"Attempting to load wrapped data using explicit path: {mnist_data_path}")
    try:
        wrapped_tr, wrapped_va, wrapped_te = load_data_wrapper(mnist_data_path)
        print("\n--- Wrapped Data Loaded Successfully ---")
        print(f"Number of training examples: {len(wrapped_tr)}")
    except Exception as e:
        print(f"Error loading wrapped data: {e}")