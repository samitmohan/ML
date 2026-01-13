# https://www.deep-ml.com/problems/55?from=Linear%20Algebra
import numpy as np
def translate_object(points, tx, ty):
    points_np = np.asarray(points)
    return [(i+tx, j+ty) for i, j in points_np]

def main():
    points = [[0, 0], [1, 0], [0.5, 1]]
    tx, ty = 2, 3

    print(translate_object(points, tx, ty))

main()

