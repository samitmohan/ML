# Label Smoothing (epsilon = 0.1 in the paper)
# Instead of hard targets [0, 0, 1, 0], use soft targets [0.033, 0.033, 0.9, 0.033]
# Penalizes overconfident predictions, improves generalization

import numpy as np


def label_smoothing(target_ids: np.ndarray, vocab_size: int, epsilon: float = 0.1) -> np.ndarray:
    # target_ids: (seq_len,) integer class labels
    # Returns: (seq_len, vocab_size) smoothed probability distribution
    seq_len = len(target_ids)
    smooth = np.full((seq_len, vocab_size), epsilon / vocab_size)
    for i, t in enumerate(target_ids):
        smooth[i, t] += (1.0 - epsilon)
    return smooth


def cross_entropy_with_smoothing(probs: np.ndarray, targets: np.ndarray, vocab_size: int,
                                  epsilon: float = 0.1) -> float:
    smooth_targets = label_smoothing(targets, vocab_size, epsilon)
    # KL divergence: sum(q * log(q/p)) but we use cross-entropy form: -sum(q * log(p))
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))
    loss = -np.sum(smooth_targets * log_probs) / len(targets)
    return loss


def main():
    np.random.seed(42)
    vocab_size = 10
    targets = np.array([3, 7, 1])

    smooth = label_smoothing(targets, vocab_size, epsilon=0.1)
    print(f"Hard target for class 3: {np.eye(vocab_size)[3]}")
    print(f"Smooth target for class 3: {smooth[0]}")
    print(f"Sum of smooth target: {smooth[0].sum():.4f}")

    # Simulate model output
    probs = np.random.dirichlet(np.ones(vocab_size), size=len(targets))
    loss = cross_entropy_with_smoothing(probs, targets, vocab_size)
    print(f"\nCross-entropy loss with label smoothing: {loss:.4f}")

if __name__ == "__main__":
    main()
