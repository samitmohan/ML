# https://www.deep-ml.com/problems/308?returnTo=paths
import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
    grad = np.array(gradient, dtype=float)
    magnitude = np.linalg.norm(grad)
    # edge case
    if magnitude == 0.0:
        zero_vec = [0.0] * len(gradient)
        return {
                "magnitude": 0.0,
                "direction": zero_vec,
                "descent_direction": zero_vec

                }

    direction = (grad / magnitude).tolist() 
    descent_dir = (- grad / magnitude).tolist()
    return {
            "magnitude": float(magnitude),
            "direction": direction,
            "descent_direction": descent_dir,
            }
