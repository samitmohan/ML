# https://www.deep-ml.com/problems/87
# explanation: https://www.youtube.com/watch?v=IWvTU6swl_E
# Just give GD momentum, track last momentum (running average), step size and update accordingy.

# momentum -> how much it needs to go down, velocity -> how much it should go down, mhat and vhat to balance it so it doesn't change only a little when starting
import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    mt = beta1 * m + (1-beta1) * grad
    vt = beta2 * v + (1-beta2) * (grad**2)
    m_hat = mt/(1-beta1**t)
    v_hat = vt/(1-beta2**t)
    parameter = parameter - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))
	return np.round(parameter,5), np.round(mt,5), np.round(vt,5)
