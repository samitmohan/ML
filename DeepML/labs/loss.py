import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray):
    '''
    logits: [N, C] softmax probabilities
    target: [N]   class indices (int64)
    '''
    preds, target = np.array(preds), np.array(target)
    N, C = preds.shape # 4,3

    eps = 1e-12

    correct_class = np.clip(preds[np.arange(N), target], eps, 1.0)
    loss = -np.mean(np.log(correct_class))

    grad_loss = np.zeros_like(preds)
    grad_loss[np.arange(N), target] = -1.0 / correct_class # 1/x log derivative
    grad_loss = grad_loss / N # mean
    return loss, grad_loss

def main():
    preds = [ [ 0.1, 0.7, 0.2 ], [ 0.8, 0.1, 0.1 ], [ 0.2, 0.3, 0.5 ], [ 0.05, 0.05, 0.9 ] ]
    target = [ 1, 0, 2, 2 ]
    print(loss_function(preds, target))

main()
