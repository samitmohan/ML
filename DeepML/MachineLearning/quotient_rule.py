# https://www.deep-ml.com/problems/312?returnTo=paths
import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    g, h = np.array(g_coeffs, dtype=float), np.array(h_coeffs, dtype=float)
    g_x = np.polyval(g,x)
    h_x = np.polyval(h,x)
    g_prime, h_prime = np.polyder(g), np.polyder(h)

    g_prime_x = np.polyval(g_prime,x)
    h_prime_x = np.polyval(h_prime,x)

    qr = (g_prime_x * h_x - g_x * h_prime_x) / (h_x**2)
    return qr
