# https://www.deep-ml.com/problems/127?returnTo=paths
'''
Docstring for ml.deepml.math.captain_redbeard
Captain Redbeard, the most daring pirate of the seven seas, has uncovered a mysterious ancient map. Instead of islands, it shows a strange wavy curve, and the treasure lies at the lowest point of the land! (watch out for those tricky local mins)

The land's height at any point x
x is given by:

f(x) = x^4 - 3x^3 + 2
'''
def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    def grad(x):
        return 4*x**3 - 9*x**2 # derivative
    x = start_x
    learning_rate = 0.01
    tolerance = 1e-6
    max_iters = 10_000
    for _ in range(max_iters):
        gradient = grad(x)
        if abs(gradient) < tolerance:
            break  # we found the minima
        x -= learning_rate * gradient
    return round(x,6)
